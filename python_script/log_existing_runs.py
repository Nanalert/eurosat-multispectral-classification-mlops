"""Import existing EuroSAT training runs into a running MLflow server."""

import json
import subprocess
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature
from torch import nn
from torchvision import models

try:
    from .train_cnn import SmallCNN
    from .train_deepfreqnet import DeepFreqNet
    from .training_metadata import dataset_summary
except ImportError:
    from train_cnn import SmallCNN
    from train_deepfreqnet import DeepFreqNet
    from training_metadata import dataset_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SCRIPT_DIR = PROJECT_ROOT / "python_script"
SPLIT_DIR = PROJECT_ROOT / "data" / "split"

# Change these settings if your server or experiment uses another name.
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "eurosat-rgb-classification"
SKIP_EXISTING_RUNS = True
RUN_NAMES_TO_IMPORT = []  # Empty means import all runs.
LOG_COMPLETE_MODELS = True

# Set this to a run name such as "resnet2" to register only that model.
REGISTER_MODEL_RUN = None
REGISTERED_MODEL_NAME = "EuroSAT_RGB_Classifier"


RUNS = [
    {
        "run_name": "cnn1",
        "architecture": "SmallCNN",
        "checkpoint": "models/cnn/cnn1/best_cnn.pt",
        "history": "models/cnn/cnn1/cnn_metrics.csv",
        "reports": "reports/cnn1",
        "metadata": None,
        "learning_rate": 0.001,
        "epoch_limit": 50,
        "early_stopping": True,
        "patience": 10,
        "normalization": "EuroSAT",
        "augmentation": "horizontal flip, vertical flip, 90-degree rotation",
        "pretrained": False,
        "class_weights": False,
        "settings_source": "reconstructed",
    },
    {
        "run_name": "cnn2",
        "architecture": "SmallCNN",
        "checkpoint": "models/cnn/cnn2/best_cnn.pt",
        "history": "models/cnn/cnn2/cnn_metrics.csv",
        "reports": "reports/cnn2",
        "metadata": None,
        "learning_rate": 0.001,
        "epoch_limit": 200,
        "early_stopping": True,
        "patience": 50,
        "normalization": "EuroSAT",
        "augmentation": "horizontal flip, vertical flip, 90-degree rotation",
        "pretrained": False,
        "class_weights": False,
        "settings_source": "reconstructed",
    },
    {
        "run_name": "cnn3",
        "architecture": "SmallCNN",
        "checkpoint": "models/cnn/cnn3/best_cnn.pt",
        "history": "models/cnn/cnn3/cnn_metrics.csv",
        "reports": "reports/cnn3",
        "metadata": None,
        "learning_rate": 0.001,
        "epoch_limit": 500,
        "early_stopping": True,
        "patience": 50,
        "normalization": "EuroSAT",
        "augmentation": "horizontal flip, vertical flip, 90-degree rotation",
        "pretrained": False,
        "class_weights": False,
        "settings_source": "reconstructed",
    },
    {
        "run_name": "resnet1",
        "architecture": "ResNet18",
        "checkpoint": "models/resnet/resnet1/resnet1.pt",
        "history": "models/resnet/resnet1/resnet1_metrics.csv",
        "reports": "reports/resnet1",
        "metadata": None,
        "learning_rate": 0.0001,
        "epoch_limit": 50,
        "early_stopping": True,
        "patience": 10,
        "normalization": "ImageNet",
        "augmentation": "horizontal flip, vertical flip, 90-degree rotation",
        "pretrained": True,
        "class_weights": False,
        "settings_source": "requested configuration",
    },
    {
        "run_name": "resnet2",
        "architecture": "ResNet18",
        "checkpoint": "models/resnet/resnet2/resnet2.pt",
        "history": "models/resnet/resnet2/resnet2_metrics.csv",
        "reports": "reports/resnet2",
        "metadata": None,
        "learning_rate": 0.0001,
        "epoch_limit": 200,
        "early_stopping": True,
        "patience": 50,
        "normalization": "ImageNet",
        "augmentation": "horizontal flip, vertical flip, 90-degree rotation",
        "pretrained": True,
        "class_weights": False,
        "settings_source": "requested configuration",
    },
    {
        "run_name": "resnet3",
        "architecture": "ResNet18",
        "checkpoint": "models/resnet/resnet3/resnet3.pt",
        "history": "models/resnet/resnet3/resnet3_metrics.csv",
        "reports": "reports/resnet3",
        "metadata": None,
        "learning_rate": 0.0001,
        "epoch_limit": 500,
        "early_stopping": False,
        "patience": 50,
        "normalization": "ImageNet",
        "augmentation": "horizontal flip, vertical flip, 90-degree rotation",
        "pretrained": True,
        "class_weights": False,
        "settings_source": "requested configuration",
    },
    {
        "run_name": "deepfreqnet1",
        "architecture": "DeepFreqNet",
        "checkpoint": "models/deepfreqnet/deepfreqnet1/deepfreqnet1.pt",
        "history": "models/deepfreqnet/deepfreqnet1/deepfreqnet_metrics1.csv",
        "reports": "reports/deepfreqnet1",
        "metadata": "models/deepfreqnet/deepfreqnet1/deepfreqnet_metadata1.json",
        "learning_rate": 0.0001,
        "epoch_limit": 50,
        "early_stopping": True,
        "patience": 10,
        "normalization": "EuroSAT",
        "augmentation": "flips, 90-degree rotation, zoom, shear",
        "pretrained": False,
        "class_weights": True,
        "settings_source": "saved metadata",
    },
    {
        "run_name": "deepfreqnet2",
        "architecture": "DeepFreqNet",
        "checkpoint": "models/deepfreqnet/deepfreqnet2/deepfreqnet.pt",
        "history": "models/deepfreqnet/deepfreqnet2/deepfreqnet_metrics.csv",
        "reports": "reports/deepfreqnet2",
        "metadata": "models/deepfreqnet/deepfreqnet2/deepfreqnet_metadata.json",
        "learning_rate": 0.0001,
        "epoch_limit": 200,
        "early_stopping": True,
        "patience": 10,
        "normalization": "EuroSAT",
        "augmentation": "flips, 90-degree rotation, zoom, shear",
        "pretrained": False,
        "class_weights": True,
        "settings_source": "saved metadata",
    },
    {
        "run_name": "deepfreqnet3",
        "architecture": "DeepFreqNet",
        "checkpoint": "models/deepfreqnet/deepfreqnet3/deepfreqnet.pt",
        "history": "models/deepfreqnet/deepfreqnet3/deepfreqnet_metrics.csv",
        "reports": "reports/deepfreqnet3",
        "metadata": "models/deepfreqnet/deepfreqnet3/deepfreqnet_metadata.json",
        "learning_rate": 0.0001,
        "epoch_limit": 500,
        "early_stopping": False,
        "patience": 10,
        "normalization": "EuroSAT",
        "augmentation": "flips, 90-degree rotation, zoom, shear",
        "pretrained": False,
        "class_weights": True,
        "settings_source": "saved metadata",
    },
]


def create_model(architecture, number_of_classes):
    if architecture == "SmallCNN":
        return SmallCNN(number_of_classes)
    if architecture == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, number_of_classes)
        return model
    if architecture == "DeepFreqNet":
        return DeepFreqNet(number_of_classes)
    raise ValueError(f"Unsupported architecture: {architecture}")


def get_git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def validate_files(run):
    required = (run["checkpoint"], run["history"], run["reports"])
    missing = [path for path in required if not (PROJECT_ROOT / path).exists()]
    if run["metadata"] and not (PROJECT_ROOT / run["metadata"]).exists():
        missing.append(run["metadata"])
    if missing:
        raise FileNotFoundError(f"{run['run_name']} is missing: {missing}")


def already_imported(client, experiment_id, run_name):
    matches = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.import_key = 'historical-{run_name}'",
        max_results=1,
    )
    return bool(matches)


def log_epoch_metrics(history):
    for row in history.itertuples():
        mlflow.log_metrics(
            {
                "train_loss": float(row.train_loss),
                "train_accuracy": float(row.train_accuracy),
                "validation_loss": float(row.validation_loss),
                "validation_accuracy": float(row.validation_accuracy),
            },
            step=int(row.epoch),
        )


def log_complete_model(model, run, number_of_classes):
    input_example = np.zeros((1, 3, 64, 64), dtype=np.float32)
    with torch.inference_mode():
        output_example = model(torch.from_numpy(input_example)).numpy()
    signature = infer_signature(input_example, output_example)
    registered_name = (
        REGISTERED_MODEL_NAME if run["run_name"] == REGISTER_MODEL_RUN else None
    )
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="model",
        registered_model_name=registered_name,
        signature=signature,
        input_example=input_example,
        code_paths=[str(PYTHON_SCRIPT_DIR)],
        serialization_format="pickle",
        metadata={
            "architecture": run["architecture"],
            "number_of_classes": number_of_classes,
            "input_format": "normalized NCHW float32 tensor",
        },
    )


def import_run(run, experiment_id, client, class_names, dataset_info, git_commit):
    validate_files(run)
    run_name = run["run_name"]
    if SKIP_EXISTING_RUNS and already_imported(client, experiment_id, run_name):
        print(f"Skipping {run_name}: already imported")
        return

    checkpoint_path = PROJECT_ROOT / run["checkpoint"]
    history_path = PROJECT_ROOT / run["history"]
    reports_path = PROJECT_ROOT / run["reports"]
    history = pd.read_csv(history_path)
    model = create_model(run["architecture"], len(class_names))
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    best_row = history.loc[history["validation_accuracy"].idxmax()]
    test_rows = history.dropna(subset=["test_accuracy"])
    report_summary_path = reports_path / f"{run_name}_test_summary.json"
    report_summary = json.loads(report_summary_path.read_text(encoding="utf-8"))

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "import_key": f"historical-{run_name}",
                "historical_import": "true",
                "architecture": run["architecture"],
                "dataset": "EuroSAT_RGB",
                "settings_source": run["settings_source"],
            }
        )
        if git_commit:
            mlflow.set_tag("git_commit", git_commit)

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        mlflow.log_params(
            {
                "architecture": run["architecture"],
                "total_parameters": parameter_count,
                "number_of_classes": len(class_names),
                "input_channels": 3,
                "image_size": "64x64",
                "optimizer": "Adam",
                "learning_rate": run["learning_rate"],
                "loss_function": "CrossEntropyLoss",
                "batch_size": 64,
                "seed": 42,
                "epoch_limit": run["epoch_limit"],
                "completed_epochs": len(history),
                "early_stopping": run["early_stopping"],
                "patience": run["patience"],
                "normalization": run["normalization"],
                "augmentation": run["augmentation"],
                "pretrained": run["pretrained"],
                "class_weights": run["class_weights"],
            }
        )
        log_epoch_metrics(history)
        final_metrics = {
            "best_epoch": float(best_row["epoch"]),
            "best_validation_accuracy": float(best_row["validation_accuracy"]),
            "best_validation_loss": float(best_row["validation_loss"]),
            "test_loss": float(report_summary["test_loss"]),
            "test_accuracy": float(report_summary["test_accuracy"]),
            "test_macro_precision": float(report_summary["macro_precision"]),
            "test_macro_recall": float(report_summary["macro_recall"]),
            "test_macro_f1": float(report_summary["macro_f1"]),
        }
        if not test_rows.empty:
            final_metrics["original_test_accuracy"] = float(
                test_rows.iloc[0]["test_accuracy"]
            )
        mlflow.log_metrics(final_metrics)

        mlflow.log_dict(run, "configuration/run_configuration.json")
        mlflow.log_dict(dataset_info, "configuration/dataset_summary.json")
        mlflow.log_artifact(history_path, artifact_path="training")
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoint")
        mlflow.log_artifacts(reports_path, artifact_path="evaluation")
        if run["metadata"]:
            mlflow.log_artifact(
                PROJECT_ROOT / run["metadata"], artifact_path="metadata"
            )
        if LOG_COMPLETE_MODELS:
            log_complete_model(model, run, len(class_names))

    print(f"Imported {run_name}")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        client = MlflowClient()
        client.search_experiments(max_results=1)
    except Exception as error:
        raise ConnectionError(
            f"Cannot connect to MLflow at {TRACKING_URI}. Is the server running?"
        ) from error

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    train_frame = pd.read_csv(SPLIT_DIR / "train.csv")
    label_table = train_frame[["label_id", "label"]].drop_duplicates()
    class_names = label_table.sort_values("label_id")["label"].tolist()
    dataset_info = dataset_summary(SPLIT_DIR, PROJECT_ROOT)
    git_commit = get_git_commit()

    selected_runs = [
        run
        for run in RUNS
        if not RUN_NAMES_TO_IMPORT or run["run_name"] in RUN_NAMES_TO_IMPORT
    ]
    for run in selected_runs:
        import_run(
            run,
            experiment.experiment_id,
            client,
            class_names,
            dataset_info,
            git_commit,
        )

    print(f"Open the MLflow UI: {TRACKING_URI}")


if __name__ == "__main__":
    main()
