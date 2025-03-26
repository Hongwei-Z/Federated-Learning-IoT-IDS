#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from keras import layers, Sequential
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import warnings

warnings.simplefilter('ignore')

# 初始化 logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# GPU 配置
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logging.error(e)
device_name = "/GPU:0" if gpus else "/CPU:0"
logging.info(f"Available devices: {tf.config.list_physical_devices()}")

# 全局存储训练/验证指标
GLOBAL_TRAIN_METRICS = {}
GLOBAL_VAL_METRICS = {}

# ============ 公共工具函数 ============
def make_dirs_for_results(dataset_name, model_name, is_federated=True):
    mode = "federated" if is_federated else "local"
    folder = os.path.join("results", mode, dataset_name, model_name)
    os.makedirs(folder, exist_ok=True)
    return folder

# ============ 数据加载与预处理 ============
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def extract_features_and_labels(df: pd.DataFrame):
    label_col = "Label"
    exclude_cols = ["Label", "Client_ID"]
    feat_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feat_cols].values
    y = df[label_col].values
    client_ids = df["Client_ID"].values
    return X, y, client_ids, feat_cols

def split_data_by_client_id(X, y, client_ids, test_size=0.2, random_state=42):
    client_data = {}
    for cid in np.unique(client_ids):
        mask = (client_ids == cid)
        X_c, y_c = X[mask], y[mask]
        X_train, X_val, y_train, y_val = train_test_split(X_c, y_c, test_size=test_size, random_state=random_state)
        client_data[cid] = (X_train, y_train, X_val, y_val)
    return client_data

# ============ 模型构建函数 ============
# 默认超参数配置，可通过命令行或配置文件扩展
DEFAULT_MODEL_PARAMS = {
    "cnn": {"filters": 32, "kernel_size": 3, "pool_size": 2, "dense_units": 64},
    "rnn": {"units": 32, "dense_units": 64},
    "mlp": {"dense_units1": 64, "dense_units2": 32},
    "transformer": {"embed_dim": 8, "num_heads": 2, "ff_dim": 32, "dense_units": 32},
}

def create_cnn_model(input_dim: int, num_classes: int, params: dict) -> Sequential:
    with tf.device(device_name):
        model = Sequential([
            layers.Input(shape=(input_dim, 1)),
            layers.Conv1D(filters=params.get("filters", 64), kernel_size=params.get("kernel_size", 3), activation="relu"),
            layers.MaxPooling1D(pool_size=params.get("pool_size", 2)),
            layers.Conv1D(filters=params.get("filters_2", 128), kernel_size=params.get("kernel_size", 3), activation="relu"),
            layers.MaxPooling1D(pool_size=params.get("pool_size", 2)),
            layers.Flatten(),
            layers.Dense(params.get("dense_units", 128), activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def create_rnn_model(input_dim: int, num_classes: int, params: dict) -> Sequential:
    with tf.device(device_name):
        model = Sequential([
            layers.Input(shape=(input_dim, 1)),
            layers.GRU(params.get("units", 32)),
            layers.Dense(params.get("dense_units", 64), activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def create_mlp_model(input_dim: int, num_classes: int, params: dict) -> Sequential:
    with tf.device(device_name):
        model = Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(params.get("dense_units1", 64), activation="relu"),
            layers.Dense(params.get("dense_units2", 32), activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.dropout1(self.att(inputs, inputs), training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.dropout2(self.ffn(out1), training=training)
        return self.layernorm2(out1 + ffn_output)

def create_transformer_model(input_dim: int, num_classes: int, params: dict) -> tf.keras.Model:
    with tf.device(device_name):
        inputs = layers.Input(shape=(input_dim, 1))
        x = layers.Dense(params.get("embed_dim", 8))(inputs)
        x = TransformerBlock(
            embed_dim=params.get("embed_dim", 8),
            num_heads=params.get("num_heads", 2),
            ff_dim=params.get("ff_dim", 32),
            rate=0.1
        )(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(params.get("dense_units", 32), activation="relu")(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def create_random_forest_model(random_state=42):
    return RandomForestClassifier(n_estimators=100, random_state=random_state)

def create_decision_tree_model(random_state=42):
    return DecisionTreeClassifier(random_state=random_state)

def create_keras_model(model_type: str, input_dim: int, num_classes: int, params: dict = None):
    params = params or DEFAULT_MODEL_PARAMS.get(model_type, {})
    if model_type == "cnn":
        return create_cnn_model(input_dim, num_classes, params)
    elif model_type == "rnn":
        return create_rnn_model(input_dim, num_classes, params)
    elif model_type == "mlp":
        return create_mlp_model(input_dim, num_classes, params)
    elif model_type == "transformer":
        return create_transformer_model(input_dim, num_classes, params)
    else:
        raise ValueError(f"未知的 Keras 模型类型: {model_type}")

# ============ Flower 客户端 ============
class IDSClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_val, y_val, model_type):
        self.model = model
        self.model_type = model_type
        if model_type in ["cnn", "rnn", "transformer"]:
            self.X_train = np.expand_dims(X_train, axis=-1)
            self.X_val = np.expand_dims(X_val, axis=-1)
        else:
            self.X_train = X_train
            self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 32)
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        metrics = {
            "train_loss": float(history.history["loss"][-1]),
            "train_accuracy": float(history.history["accuracy"][-1])
        }
        return self.model.get_weights(), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        return loss, len(self.X_val), {"loss": float(loss), "accuracy": float(accuracy)}

# ============ 自定义聚合策略 ============
def fit_metrics_aggregation_fn(fit_metrics):
    if not fit_metrics:
        return {}
    losses = [m["train_loss"] for _, m in fit_metrics]
    accs = [m["train_accuracy"] for _, m in fit_metrics]
    return {"train_loss": float(np.mean(losses)), "train_accuracy": float(np.mean(accs))}

def evaluate_metrics_aggregation_fn(eval_metrics):
    if not eval_metrics:
        return {"accuracy": 0.0, "loss": 0.0}
    losses = [m["loss"] for _, m in eval_metrics]
    accs = [m["accuracy"] for _, m in eval_metrics]
    return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(accs))}

class MyFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, num_rounds, model_key, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.model_key = model_key
        self.final_parameters = None

    def aggregate_fit(self, server_round: int, results, failures):
        agg_params, metrics = super().aggregate_fit(server_round, results, failures)
        fit_dicts = [r.metrics for _, r in results]
        avg_train_loss = np.mean([d["train_loss"] for d in fit_dicts])
        avg_train_acc = np.mean([d["train_accuracy"] for d in fit_dicts])
        GLOBAL_TRAIN_METRICS.setdefault(self.model_key, {"round": [], "loss": [], "accuracy": []})
        GLOBAL_TRAIN_METRICS[self.model_key]["round"].append(server_round)
        GLOBAL_TRAIN_METRICS[self.model_key]["loss"].append(avg_train_loss)
        GLOBAL_TRAIN_METRICS[self.model_key]["accuracy"].append(avg_train_acc)
        if server_round == self.num_rounds:
            self.final_parameters = agg_params
        return agg_params, metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        if not results:
            return None
        total_loss = 0.0
        total_acc = 0.0
        total_examples = 0
        for _, eval_res in results:
            loss = eval_res.loss
            num_examples = eval_res.num_examples
            metrics = eval_res.metrics
            total_loss += loss * num_examples
            total_acc += metrics.get("accuracy", 0.0) * num_examples
            total_examples += num_examples
        avg_loss = total_loss / total_examples
        avg_acc = total_acc / total_examples

        GLOBAL_VAL_METRICS.setdefault(self.model_key, {"round": [], "loss": [], "accuracy": []})
        GLOBAL_VAL_METRICS[self.model_key]["round"].append(server_round)
        GLOBAL_VAL_METRICS[self.model_key]["loss"].append(avg_loss)
        GLOBAL_VAL_METRICS[self.model_key]["accuracy"].append(avg_acc)

        # 返回两个值：聚合后的损失和指标字典
        return avg_loss, {"loss": avg_loss, "accuracy": avg_acc}



# ============ Flower 联邦训练 ============
def run_federated_training(model_type, client_data, input_dim, num_classes,
                           num_rounds=50, epochs_per_round=1, batch_size=32, model_params=None):
    client_ids = sorted(client_data.keys())
    num_clients = len(client_ids)

    def client_fn_builder(cid):
        X_train, y_train, X_val, y_val = client_data[cid]
        model = create_keras_model(model_type, input_dim, num_classes, model_params)
        return IDSClient(model, X_train, y_train, X_val, y_val, model_type).to_client()

    def client_fn(fl_ctx: fl.common.Context):
        idx = int(fl_ctx.node_id) % num_clients
        cid = client_ids[idx]
        return client_fn_builder(cid)

    strategy = MyFedAvg(
        num_rounds=num_rounds,
        model_key=model_type,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda r: {"epochs": epochs_per_round, "batch_size": batch_size},
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0 if not gpus else 0.01},
    )

    if strategy.final_parameters is None:
        logging.warning("联邦结束后未能获取到全局模型参数!")
        return None
    final_weights = parameters_to_ndarrays(strategy.final_parameters)
    global_model = create_keras_model(model_type, input_dim, num_classes, model_params)
    global_model.set_weights(final_weights)
    return global_model

# ============ 非联邦训练 ============
def run_local_training_keras(model_type, X, y, input_dim, num_classes,
                             epochs=50, batch_size=32, val_ratio=0.2, model_params=None):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=42)
    if model_type in ["cnn", "rnn", "transformer"]:
        X_train_ = np.expand_dims(X_train, axis=-1)
        X_val_ = np.expand_dims(X_val, axis=-1)
    else:
        X_train_, X_val_ = X_train, X_val

    model = create_keras_model(model_type, input_dim, num_classes, model_params)
    history = model.fit(X_train_, y_train, validation_data=(X_val_, y_val),
                        epochs=epochs, batch_size=batch_size, verbose=0)
    final_loss, final_acc = model.evaluate(X_val_, y_val, verbose=0)
    y_pred = np.argmax(model.predict(X_val_), axis=1)
    cm = confusion_matrix(y_val, y_pred)
    return model, history, (final_loss, final_acc), (y_val, y_pred, cm)

def run_local_training_sklearn(model_type, X, y, val_ratio=0.2):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=42)
    model = create_random_forest_model() if model_type == "random_forest" else create_decision_tree_model()
    model.fit(X_train, y_train)
    acc = model.score(X_val, y_val)
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    return model, acc, (y_val, y_pred, cm)

# ============ 绘图与报告保存 ============
def save_metrics_plot(train_metrics, val_metrics, dataset_name, model_name, save_dir):
    rounds_t, train_loss, train_acc = train_metrics["round"], train_metrics["loss"], train_metrics["accuracy"]
    rounds_v, val_loss, val_acc = val_metrics["round"], val_metrics["loss"], val_metrics["accuracy"]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rounds_t, train_loss, "o-", label="Train Loss")
    plt.plot(rounds_v, val_loss, "x-", label="Val Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title(f"Train/Val Loss - {model_name}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rounds_t, train_acc, "o-", label="Train Acc")
    plt.plot(rounds_v, val_acc, "x-", label="Val Acc")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"Train/Val Accuracy - {model_name}")
    plt.legend()

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_{model_name}_metrics_{timestamp}.png"))
    plt.close()

def save_confusion_matrix(cm, dataset_name, model_name, save_dir, title_extra=""):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name} {title_extra}")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_{model_name}_cm_{title_extra}_{timestamp}.png"))
    plt.close()

def save_classification_report(y_true, y_pred, dataset_name, model_name, save_dir, title_extra=""):
    cr = classification_report(y_true, y_pred)
    filename = os.path.join(save_dir, f"{dataset_name}_{model_name}_report_{title_extra}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(cr)

# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser(description="Federated & Local IDS Demo (优化版)")
    parser.add_argument("--datasets", nargs="+", default=["multi.csv"], help="数据集文件名列表")
    parser.add_argument("--federated_rounds", type=int, default=50, help="联邦训练轮数")
    parser.add_argument("--local_epochs", type=int, default=0, help="本地训练 epochs (Keras)")
    parser.add_argument("--model_params", type=str, default="", help="以JSON格式传入的模型参数配置")
    args = parser.parse_args()

    # 解析模型参数配置（支持 json 格式字符串）
    import json
    model_params_cfg = {}
    if args.model_params:
        try:
            model_params_cfg = json.loads(args.model_params)
        except Exception as e:
            logging.warning("解析 model_params 失败，使用默认配置。")
    ALL_MODEL_TYPES = ["cnn", "mlp", "transformer", "random_forest", "decision_tree"]
    for dataset_csv in args.datasets:
        if not os.path.exists(dataset_csv):
            logging.warning(f"数据文件 {dataset_csv} 不存在，跳过。")
            continue
        logging.info(f"================ 处理数据集: {dataset_csv} ================")
        df = load_data(dataset_csv)
        X, y, client_ids, feat_cols = extract_features_and_labels(df)
        input_dim = X.shape[1]
        num_classes = len(np.unique(y))
        client_data = split_data_by_client_id(X, y, client_ids)
        
        # 联邦训练（仅适用于 Keras 模型）
        for model_type in ALL_MODEL_TYPES:
            if model_type in ["random_forest", "decision_tree"]:
                logging.info(f"[联邦-跳过] model={model_type}")
                total_acc, total_samples = 0.0, 0
                for cid, (X_tr, y_tr, X_val, y_val) in client_data.items():
                    m = create_random_forest_model() if model_type=="random_forest" else create_decision_tree_model()
                    m.fit(X_tr, y_tr)
                    acc = m.score(X_val, y_val)
                    total_acc += acc * len(X_val)
                    total_samples += len(X_val)
                logging.info(f"  -> 平均准确率: {total_acc / total_samples if total_samples else 0:.4f}")
                continue
            else:
                logging.info(f"[联邦训练] model={model_type}, Rounds={args.federated_rounds}")
                GLOBAL_TRAIN_METRICS[model_type] = {"round": [], "loss": [], "accuracy": []}
                GLOBAL_VAL_METRICS[model_type] = {"round": [], "loss": [], "accuracy": []}
                # 获取特定模型参数配置（若有），否则使用默认
                params = model_params_cfg.get(model_type, None)
                global_model = run_federated_training(
                    model_type=model_type,
                    client_data=client_data,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    num_rounds=args.federated_rounds,
                    epochs_per_round=1,
                    batch_size=32,
                    model_params=params
                )
                fed_save_dir = make_dirs_for_results(os.path.basename(dataset_csv), model_type, is_federated=True)
                if global_model is not None:
                    if GLOBAL_TRAIN_METRICS.get(model_type) and GLOBAL_VAL_METRICS.get(model_type):
                        save_metrics_plot(GLOBAL_TRAIN_METRICS[model_type], GLOBAL_VAL_METRICS[model_type],
                                          os.path.basename(dataset_csv), model_type, fed_save_dir)
                    if model_type in ["cnn", "rnn", "transformer"]:
                        X_eval = np.expand_dims(X, axis=-1)
                    else:
                        X_eval = X
                    final_loss, final_acc = global_model.evaluate(X_eval, y, verbose=0)
                    logging.info(f"  -> 全局模型评估: Loss={final_loss:.4f}, Acc={final_acc:.4f}")
                    y_pred = np.argmax(global_model.predict(X_eval), axis=1)
                    cm = confusion_matrix(y, y_pred)
                    save_confusion_matrix(cm, os.path.basename(dataset_csv), model_type, fed_save_dir, title_extra="Fed")
                    save_classification_report(y, y_pred, os.path.basename(dataset_csv), model_type, fed_save_dir, title_extra="Fed")
        
        # 非联邦训练
        logging.info(f"[非联邦训练] 数据集 {dataset_csv}, 样本数: {len(X)}")
        for model_type in ALL_MODEL_TYPES:
            logging.info(f"  -> 训练模型: {model_type}")
            local_save_dir = make_dirs_for_results(os.path.basename(dataset_csv), model_type, is_federated=False)
            params = model_params_cfg.get(model_type, None)
            if model_type in ["cnn", "rnn", "mlp", "transformer"]:
                model, history, (final_loss, final_acc), (y_val, y_pred, cm) = run_local_training_keras(
                    model_type, X, y, input_dim, num_classes,
                    epochs=args.local_epochs, batch_size=32, val_ratio=0.2, model_params=params
                )
                logging.info(f"    - 验证: Loss={final_loss:.4f}, Acc={final_acc:.4f}")
                # 绘制训练曲线
                hist_dict = history.history
                epochs_range = range(1, len(hist_dict["loss"])+1)
                plt.figure(figsize=(12, 5))
                plt.subplot(1,2,1)
                plt.plot(epochs_range, hist_dict["loss"], "o-", label="Train Loss")
                if "val_loss" in hist_dict:
                    plt.plot(epochs_range, hist_dict["val_loss"], "x-", label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Local Loss - {model_type}")
                plt.legend()
                plt.subplot(1,2,2)
                plt.plot(epochs_range, hist_dict["accuracy"], "o-", label="Train Acc")
                if "val_accuracy" in hist_dict:
                    plt.plot(epochs_range, hist_dict["val_accuracy"], "x-", label="Val Acc")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(f"Local Accuracy - {model_type}")
                plt.legend()
                plt.tight_layout()
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(os.path.join(local_save_dir, f"{dataset_csv}_{model_type}_local_history_{timestamp}.png"))
                plt.close()
                save_confusion_matrix(cm, os.path.basename(dataset_csv), model_type, local_save_dir, title_extra="Local")
                save_classification_report(y_val, y_pred, os.path.basename(dataset_csv), model_type, local_save_dir, title_extra="Local")
            elif model_type in ["random_forest", "decision_tree"]:
                model, val_acc, (y_val, y_pred, cm) = run_local_training_sklearn(model_type, X, y, val_ratio=0.2)
                logging.info(f"    - 验证准确率: {val_acc:.4f}")
                save_confusion_matrix(cm, os.path.basename(dataset_csv), model_type, local_save_dir, title_extra="Local")
                save_classification_report(y_val, y_pred, os.path.basename(dataset_csv), model_type, local_save_dir, title_extra="Local")
            else:
                logging.warning(f"未知模型类型: {model_type}，跳过")
    logging.info("======= 全部流程结束 =======")

if __name__ == "__main__":
    main()
