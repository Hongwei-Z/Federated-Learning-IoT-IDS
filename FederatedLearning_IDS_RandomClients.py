#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本地随机将数据分成N个客户端，使用Flower进行联邦学习。
dataset.csv 已包含:
   - Client_ID: 表示客户端身份 (1~N)
   - Label: 攻击 or 正常标签 (或多分类标签)
   - 其他特征列(数值型)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import flwr as fl
from flwr.common import Context, parameters_to_ndarrays
import warnings
warnings.simplefilter('ignore')

# 配置GPU内存
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 根据是否有GPU设置设备名称
device_name = "/GPU:0" if gpus else "/CPU:0"

print("Available devices:", tf.config.list_physical_devices())

# ======== 全局变量记录多轮训练指标 ============
TRAIN_METRICS = {
    "round": [],
    "loss": [],
    "accuracy": [],
}

VAL_METRICS = {
    "round": [],
    "loss": [],
    "accuracy": [],
}

# ============ 数据加载与划分 ============
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    return df

# Extract features and labels
def extract_features_and_labels(df: pd.DataFrame):
    label_col = "Label"
    exclude_cols = ["Label", "Client_ID"]

    feat_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feat_cols].values
    y = df[label_col].values
    client_ids = df["Client_ID"].values

    return X, y, client_ids, feat_cols

def split_data_by_client_id(X, y, client_ids, test_size=0.2, random_state=42):
    unique_clients = np.unique(client_ids)
    client_data = {}
    for cid in unique_clients:
        mask = (client_ids == cid)
        X_c = X[mask]
        y_c = y[mask]
        X_train, X_val, y_train, y_val = train_test_split(
            X_c, y_c, test_size=test_size, random_state=random_state
        )
        client_data[cid] = (X_train, y_train, X_val, y_val)
    return client_data

# ============ 模型设计 ============
def create_cnn_model(input_dim: int, num_classes: int) -> Sequential:
    with tf.device(device_name):
        model = Sequential([
            layers.Input(shape=(input_dim, 1)),
            layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax")
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
    return model

def create_rnn_model(input_dim: int, num_classes: int) -> Sequential:
    with tf.device(device_name):
        model = Sequential([
            layers.Input(shape=(input_dim, 1)),
            layers.GRU(32),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
    return model

# ============ Flower客户端 ============
class IDSClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model
        self.X_train = np.expand_dims(X_train, axis=-1)
        self.y_train = y_train
        self.X_val   = np.expand_dims(X_val, axis=-1)
        self.y_val   = y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # 调整训练轮数
        epochs = config.get("epochs", 5)
        batch_size = config.get("batch_size", 32)
        history = self.model.fit(
            self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0
        )
        train_loss = history.history["loss"][-1]
        train_acc  = history.history["accuracy"][-1]
        metrics = {"train_loss": float(train_loss), "train_accuracy": float(train_acc)}
        return self.model.get_weights(), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)

        # y_pred_probs = self.model.predict(self.X_val)
        # y_pred = np.argmax(y_pred_probs, axis=1)
        #
        # precision = precision_score(self.y_val, y_pred, average="weighted", zero_division=0)
        # recall = recall_score(self.y_val, y_pred, average="weighted", zero_division=0)
        # f1 = f1_score(self.y_val, y_pred, average="weighted", zero_division=0)
        #
        # return loss, len(self.X_val), {
        #     "loss": loss,
        #     "accuracy": accuracy,
        #     "precision": precision,
        #     "recall": recall,
        #     "f1_score": f1
        # }

        return loss, len(self.X_val), {"loss": float(loss), "accuracy": float(accuracy)}

# ============ 自定义聚合函数 ============
def fit_metrics_aggregation_fn(fit_metrics):
    if not fit_metrics:
        return {}
    losses = [metrics["train_loss"] for _, metrics in fit_metrics]
    accs = [metrics["train_accuracy"] for _, metrics in fit_metrics]
    avg_loss = float(np.mean(losses))
    avg_acc = float(np.mean(accs))
    round_no = len(TRAIN_METRICS["round"]) + 1
    TRAIN_METRICS["round"].append(round_no)
    TRAIN_METRICS["loss"].append(avg_loss)
    TRAIN_METRICS["accuracy"].append(avg_acc)
    return {"train_loss": avg_loss, "train_accuracy": avg_acc}

def evaluate_metrics_aggregation_fn(eval_metrics):
    if not eval_metrics:
        # return {"accuracy": 0.0, "loss": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        return {"accuracy": 0.0, "loss": 0.0}

    losses = [metrics["loss"] for _, metrics in eval_metrics]
    accs = [metrics["accuracy"] for _, metrics in eval_metrics]
    # precisions = [metrics["precision"] for _, metrics in eval_metrics]
    # recalls = [metrics["recall"] for _, metrics in eval_metrics]
    # f1_scores = [metrics["f1_score"] for _, metrics in eval_metrics]

    avg_loss = float(np.mean(losses))
    avg_acc = float(np.mean(accs))
    # avg_precision = float(np.mean(precisions))
    # avg_recall = float(np.mean(recalls))
    # avg_f1 = float(np.mean(f1_scores))

    round_no = len(VAL_METRICS["round"]) + 1
    VAL_METRICS["round"].append(round_no)
    VAL_METRICS["loss"].append(avg_loss)
    VAL_METRICS["accuracy"].append(avg_acc)
    return {"accuracy": avg_acc, "loss": avg_loss}
    # return {
    #     "accuracy": avg_acc,
    #     "loss": avg_loss,
    #     "precision": avg_precision,
    #     "recall": avg_recall,
    #     "f1_score": avg_f1
    # }

# ============ 自定义策略 ============
class MyFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, num_rounds, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.final_parameters = None

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if server_round == self.num_rounds:
            self.final_parameters = aggregated_parameters
        return aggregated_parameters, metrics

# ============ 主函数入口 ============
def main():
    parser = argparse.ArgumentParser(description="Federated IDS with Random Clients")
    parser.add_argument("--data", type=str, default="dataset.csv", help="CSV数据文件")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "rnn"], help="模型类型: cnn或rnn")
    parser.add_argument("--rounds", type=int, default=5, help="联邦学习轮数")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"数据文件 {args.data} 不存在！")
        return

    df = load_data(args.data)
    X, y, client_ids, feat_cols = extract_features_and_labels(df)
    client_data = split_data_by_client_id(X, y, client_ids)

    MODEL_TYPE = args.model
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))

    clist = sorted(client_data.keys())
    def client_fn(context: Context):
        try:
            idx = int(context.node_id) % len(clist)
        except Exception:
            idx = 0
        c_id = clist[idx]
        X_train, y_train, X_val, y_val = client_data[c_id]
        if MODEL_TYPE == "rnn":
            m = create_rnn_model(input_dim, num_classes)
        else:
            m = create_cnn_model(input_dim, num_classes)
        return IDSClient(m, X_train, y_train, X_val, y_val).to_client()

    strategy = MyFedAvg(
        num_rounds=args.rounds,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(clist),
        min_evaluate_clients=len(clist),
        min_available_clients=len(clist),
        on_fit_config_fn=lambda rnd: {"epochs": 5, "batch_size": 32},
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    config = fl.server.ServerConfig(num_rounds=args.rounds)

    num_clients = len(clist)
    gpu_fraction = 1.0 / num_clients if num_clients > 0 else 1.0

    result = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=config,
        strategy=strategy,
        client_resources={'num_cpus': 1, 'num_gpus': gpu_fraction},
    )

    final_params = strategy.final_parameters
    if final_params is None:
        print("未返回全局模型参数，无法进行全局评估。")
        print("历史指标：")
        print("训练指标：", TRAIN_METRICS)
        print("验证指标：", VAL_METRICS)
        return

    weights = parameters_to_ndarrays(final_params)

    if MODEL_TYPE == "rnn":
        global_model = create_rnn_model(input_dim, num_classes)
    else:
        global_model = create_cnn_model(input_dim, num_classes)
    global_model.set_weights(weights)

    X_expanded = np.expand_dims(X, axis=-1)
    final_loss, final_acc = global_model.evaluate(X_expanded, y, verbose=0)
    print(f"\n[最终全局模型评估] Loss={final_loss:.4f}, Accuracy={final_acc:.4f}")

    y_pred = np.argmax(global_model.predict(X_expanded), axis=1)
    print("\n分类报告:\n", classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    print("混淆矩阵:\n", cm)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(TRAIN_METRICS["round"], TRAIN_METRICS["loss"], "o-", label="Train Loss")
    plt.plot(VAL_METRICS["round"], VAL_METRICS["loss"], "x-", label="Val Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss Over Rounds")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(TRAIN_METRICS["round"], TRAIN_METRICS["accuracy"], "o-", label="Train Acc")
    plt.plot(VAL_METRICS["round"], VAL_METRICS["accuracy"], "x-", label="Val Acc")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Train/Val Accuracy Over Rounds")
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

if __name__ == "__main__":
    main()
