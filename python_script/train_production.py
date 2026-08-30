"""Run the selected production training configuration from params.yaml."""

import json
from pathlib import Path

import pandas as pd
import torch
import yaml

try:
    from . import train_resnet18 as training
except ImportError:
    import train_resnet18 as training


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
OUTPUT_DIR = PROJECT_ROOT / "models" / "production"
DVC_METRICS_PATH = OUTPUT_DIR / "dvc_metrics.json"


def load_params():
    with PARAMS_PATH.open("r", encoding="utf-8") as params_file:
        return yaml.safe_load(params_file)


def save_dvc_metrics(run_name):
    with (OUTPUT_DIR / f"{run_name}_test_summary.json").open(
        "r", encoding="utf-8"
    ) as summary_file:
        summary = json.load(summary_file)
    history = pd.read_csv(OUTPUT_DIR / f"{run_name}_metrics.csv")

    metrics = {
        "best_validation_accuracy": float(history["validation_accuracy"].max()),
        "test_loss": float(summary["test_loss"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "test_macro_precision": float(summary["macro_precision"]),
        "test_macro_recall": float(summary["macro_recall"]),
        "test_macro_f1": float(summary["macro_f1"]),
    }
    with DVC_METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")


def main():
    params = load_params()
    settings = params["training"]
    mlflow_settings = params["mlflow"]
    if settings["architecture"] != "resnet18":
        raise ValueError("The production trainer currently supports resnet18")

    training.OUTPUT_DIR = OUTPUT_DIR
    training.SEED = int(settings["seed"])
    training.BATCH_SIZE = int(settings["batch_size"])
    training.NUM_WORKERS = int(settings["number_of_workers"])
    training.LEARNING_RATE = float(settings["learning_rate"])
    training.MIN_DELTA = float(settings["minimum_improvement"])
    training.MLFLOW_ENABLED = bool(mlflow_settings["enabled"])
    training.MLFLOW_TRACKING_URI = str(mlflow_settings["tracking_uri"])
    training.MLFLOW_EXPERIMENT_NAME = str(mlflow_settings["experiment_name"])
    training.MLFLOW_REGISTERED_MODEL_NAME = mlflow_settings.get("registered_model_name")

    train_frame = pd.read_csv(training.SPLIT_DIR / "train.csv")
    label_table = train_frame[["label_id", "label"]].drop_duplicates()
    class_names = label_table.sort_values("label_id")["label"].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attempt = {
        "name": str(settings["run_name"]),
        "epochs": int(settings["epochs"]),
        "early_stopping": bool(settings["early_stopping"]),
        "patience": int(settings["patience"]),
    }
    training.train_attempt(attempt, device, class_names)
    save_dvc_metrics(attempt["name"])


if __name__ == "__main__":
    main()
