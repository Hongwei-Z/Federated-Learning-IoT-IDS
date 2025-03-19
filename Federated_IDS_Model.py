#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Federated Learning for Smart Home IoT Intrusion Detection
This script:
- Uses Flower for federated learning
- Expects dataset.csv to contain:
  - Client_ID: Representing client identity (1~N)
  - Label: Attack or normal traffic labels (or multi-class labels)
  - Other numerical feature columns
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras import layers, Sequential
import flwr as fl
from flwr.common import Context, parameters_to_ndarrays
import time


# Define hyperparameters
EPOCHS = 10
ROUNDS = 5
BATCH_SIZE = 128
DATASET_PATH = "multi.csv"


# Configure GPU memory
def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    device_name = "/GPU:0" if gpus else "/CPU:0"
    print("Available devices:", tf.config.list_physical_devices())
    return device_name


# Global variables to track metrics across training rounds
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


# ============ Data Loading and Processing ============
def load_data(csv_path: str):
    # Load data from CSV file
    df = pd.read_csv(csv_path)
    return df


def extract_features_and_labels(df: pd.DataFrame):
    # Extract features and labels from dataframe
    label_col = "Label"
    exclude_cols = ["Label", "Client_ID"]

    feat_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feat_cols].values
    y = df[label_col].values
    client_ids = df["Client_ID"].values

    return X, y, client_ids, feat_cols


def split_data_by_client_id(X, y, client_ids, test_size=0.2, random_state=42):
    # Split data by client ID into training and validation sets
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


# ============ Model Design ============
def create_cnn_model(input_dim: int, num_classes: int, device_name: str) -> Sequential:
    # Create a CNN model
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


# ============ Flower Client ============
class IDSClient(fl.client.NumPyClient):
    # Client for Federated IDS

    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model
        self.X_train = np.expand_dims(X_train, axis=-1)
        self.y_train = y_train
        self.X_val = np.expand_dims(X_val, axis=-1)
        self.y_val = y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        epochs = config.get("epochs", EPOCHS)
        batch_size = config.get("batch_size", BATCH_SIZE)

        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        train_loss = history.history["loss"][-1]
        train_acc = history.history["accuracy"][-1]
        metrics = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc)
        }

        return self.model.get_weights(), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        return loss, len(self.X_val), {"loss": float(loss), "accuracy": float(accuracy)}


# ============ Custom Aggregation Functions ============
def fit_metrics_aggregation_fn(fit_metrics):
    # Aggregate training metrics from clients
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

    print(f'Training round {round_no}: loss={avg_loss:.4f}, accuracy={avg_acc:.4f}')
    return {"train_loss": avg_loss, "train_accuracy": avg_acc}


def evaluate_metrics_aggregation_fn(eval_metrics):
    # Aggregate evaluation metrics from clients
    if not eval_metrics:
        return {"accuracy": 0.0, "loss": 0.0}

    losses = [metrics["loss"] for _, metrics in eval_metrics]
    accs = [metrics["accuracy"] for _, metrics in eval_metrics]

    avg_loss = float(np.mean(losses))
    avg_acc = float(np.mean(accs))

    round_no = len(VAL_METRICS["round"]) + 1
    VAL_METRICS["round"].append(round_no)
    VAL_METRICS["loss"].append(avg_loss)
    VAL_METRICS["accuracy"].append(avg_acc)

    print(f'Validation round {round_no}: loss={avg_loss:.4f}, accuracy={avg_acc:.4f}')
    return {"accuracy": avg_acc, "loss": avg_loss}


# ============ Custom Federated Strategy ============
class EnhancedFedAvg(fl.server.strategy.FedAvg):
    # Enhanced FedAvg strategy

    def __init__(self, num_rounds, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.final_parameters = None

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if server_round == self.num_rounds:
            self.final_parameters = aggregated_parameters
        return aggregated_parameters, metrics


# ============ Visualization Functions ============
def plot_metrics():
    # Plot training and validation metrics
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(TRAIN_METRICS["round"], TRAIN_METRICS["loss"], "o-", label="Train Loss")
    plt.plot(VAL_METRICS["round"], VAL_METRICS["loss"], "x-", label="Val Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Train / Val Loss Over Rounds")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(TRAIN_METRICS["round"], TRAIN_METRICS["accuracy"], "o-", label="Train Acc")
    plt.plot(VAL_METRICS["round"], VAL_METRICS["accuracy"], "x-", label="Val Acc")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Train / Val Accuracy Over Rounds")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm):
    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


# ============ Evaluation Functions ============
def evaluate_global_model(global_model, X, y):
    # Evaluate the global model performance
    X_expanded = np.expand_dims(X, axis=-1)
    final_loss, final_acc = global_model.evaluate(X_expanded, y, verbose=0)
    print(f"\n[Final Global Model Evaluation] Loss={final_loss:.4f}, Accuracy={final_acc:.4f}")

    y_pred = np.argmax(global_model.predict(X_expanded), axis=1)
    print("\nClassification Report:\n", classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:\n", cm)

    return cm


# ============ Main Function ============
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Federated Learning Based IoT IDS")
    parser.add_argument("--data", type=str, default=DATASET_PATH, help="CSV data file path")
    parser.add_argument("--model", type=str, default="cnn", help="Model type: cnn")
    parser.add_argument("--rounds", type=int, default=ROUNDS, help="Number of fl rounds")
    args = parser.parse_args()

    # Verify data file exists
    if not os.path.exists(args.data):
        print(f"Data file {args.data} does not exist!")
        return

    # Configure GPU
    device_name = configure_gpu()

    # Load and process data
    df = load_data(args.data)
    X, y, client_ids, feat_cols = extract_features_and_labels(df)
    client_data = split_data_by_client_id(X, y, client_ids)

    # Model configuration
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))

    # Setup client selection function
    clist = sorted(client_data.keys())

    def client_fn(context: Context):
        try:
            idx = int(context.node_id) % len(clist)
        except Exception:
            idx = 0
        c_id = clist[idx]
        X_train, y_train, X_val, y_val = client_data[c_id]

        m = create_cnn_model(input_dim, num_classes, device_name)
        return IDSClient(m, X_train, y_train, X_val, y_val).to_client()

    # Configure federated learning strategy
    strategy = EnhancedFedAvg(
        num_rounds=args.rounds,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(clist),
        min_evaluate_clients=len(clist),
        min_available_clients=len(clist),
        on_fit_config_fn=lambda rnd: {"epochs": EPOCHS, "batch_size": BATCH_SIZE},
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    config = fl.server.ServerConfig(num_rounds=args.rounds)

    # Calculate resources per client
    num_clients = len(clist)
    gpu_fraction = 1.0 / num_clients if num_clients > 0 else 1.0

    # Start federated simulation
    result = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=config,
        strategy=strategy,
        client_resources={'num_cpus': 1, 'num_gpus': gpu_fraction},
    )

    # Evaluate final global model
    final_params = strategy.final_parameters
    if final_params is None:
        print("No global model parameters returned, cannot perform global evaluation.")
        print("Historical metrics:")
        print("Training metrics:", TRAIN_METRICS)
        print("Validation metrics:", VAL_METRICS)
        return

    # Convert parameters to weights and create model
    weights = parameters_to_ndarrays(final_params)
    global_model = create_cnn_model(input_dim, num_classes, device_name)
    global_model.set_weights(weights)

    # Evaluate and visualize results
    cm = evaluate_global_model(global_model, X, y)
    plot_metrics()
    plot_confusion_matrix(cm)


if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training time: {training_time:.4f} seconds")
