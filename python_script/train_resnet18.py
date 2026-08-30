"""Fine-tune and evaluate pretrained ResNet18 on EuroSAT RGB."""

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from .evaluation_reports import evaluate_and_save_reports
    from .mlflow_tracking import MLflowTracker
    from .training_metadata import (
        dataset_summary,
        model_summary,
        optimizer_summary,
        runtime_summary,
        save_metadata,
        utc_now,
    )
except ImportError:
    from evaluation_reports import evaluate_and_save_reports
    from mlflow_tracking import MLflowTracker
    from training_metadata import (
        dataset_summary,
        model_summary,
        optimizer_summary,
        runtime_summary,
        save_metadata,
        utc_now,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "split"
OUTPUT_DIR = PROJECT_ROOT / "models" / "resnet" / "resnet4"

# Change these training settings as needed.
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 0.001
MIN_DELTA = 0.00000000000000000000000000000001

# Set this to False when you want to train without MLflow.
MLFLOW_ENABLED = True
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "eurosat-rgb-classification"
MLFLOW_REGISTERED_MODEL_NAME = None

ATTEMPTS = [
    {"name": "resnet1", "epochs": 50, "early_stopping": True, "patience": 10},
    {"name": "resnet2", "epochs": 200, "early_stopping": True, "patience": 50},
    {"name": "resnet3", "epochs": 500, "early_stopping": False, "patience": 50},
]

# Pretrained ResNet18 expects ImageNet normalization.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class RandomRotate90:
    def __call__(self, image):
        return image.rotate(random.choice((0, 90, 180, 270)))


class EuroSATDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.data = pd.read_csv(csv_path)
        required = {"image_path", "label", "label_id"}
        if not required.issubset(self.data.columns):
            raise ValueError(f"{csv_path} must contain {sorted(required)}")
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        with Image.open(PROJECT_ROOT / row["image_path"]) as image:
            image = self.transform(image.convert("RGB"))
        return image, int(row["label_id"])


def create_dataloaders(device):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotate90(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    datasets = {
        "train": EuroSATDataset(SPLIT_DIR / "train.csv", train_transform),
        "validation": EuroSATDataset(
            SPLIT_DIR / "validation.csv", evaluation_transform
        ),
        "test": EuroSATDataset(SPLIT_DIR / "test.csv", evaluation_transform),
    }
    return {
        name: DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=name == "train",
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
            persistent_workers=NUM_WORKERS > 0,
        )
        for name, dataset in datasets.items()
    }


def run_epoch(model, dataloader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            predictions = model(images)
            loss = loss_function(predictions, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions.argmax(1) == labels).sum().item()

    size = len(dataloader.dataset)
    return total_loss / size, total_correct / size


def train_attempt(attempt, device, class_names):
    started_at = utc_now()
    started_timer = time.perf_counter()
    name = attempt["name"]
    epochs = attempt["epochs"]
    early_stopping = attempt["early_stopping"]
    patience = attempt["patience"]
    model_path = OUTPUT_DIR / f"{name}.pt"
    metrics_path = OUTPUT_DIR / f"{name}_metrics.csv"
    metadata_path = OUTPUT_DIR / f"{name}_metadata.json"
    checkpoint_path = OUTPUT_DIR / f"{name}_training_checkpoint.pt"
    number_of_classes = len(class_names)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dataloaders = create_dataloaders(device)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, number_of_classes)
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    stopping_text = "off" if not early_stopping else str(patience)
    print(f"\nStarting {name}: epochs={epochs}, patience={stopping_text}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tracker = MLflowTracker(
        enabled=MLFLOW_ENABLED,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        run_name=name,
        parameters={
            "architecture": "ResNet18",
            "pretrained_weights": str(models.ResNet18_Weights.DEFAULT),
            "seed": SEED,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "number_of_workers": NUM_WORKERS,
            "learning_rate": LEARNING_RATE,
            "optimizer": optimizer.__class__.__name__,
            "loss_function": loss_function.__class__.__name__,
            "early_stopping": early_stopping,
            "patience": patience,
            "minimum_improvement": MIN_DELTA,
            "normalization_mean": MEAN,
            "normalization_standard_deviation": STD,
            "number_of_classes": number_of_classes,
        },
        tags={"architecture": "resnet18", "dataset": "EuroSAT_RGB"},
    )
    tracker.start()
    best_accuracy = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    first_epoch = 1
    checkpoint_settings = {
        "epochs": epochs,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "early_stopping": early_stopping,
        "patience": patience,
        "minimum_improvement": MIN_DELTA,
    }

    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if checkpoint["settings"] != checkpoint_settings:
            raise ValueError(
                f"Checkpoint settings do not match the current run: {checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_state = checkpoint["best_state"]
        best_accuracy = checkpoint["best_accuracy"]
        best_epoch = checkpoint["best_epoch"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        history = checkpoint["history"]
        first_epoch = checkpoint["epoch"] + 1
        print(f"Resuming {name} from epoch {first_epoch}")

    for epoch in range(first_epoch, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, dataloaders["train"], loss_function, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, dataloaders["validation"], loss_function, device
        )
        print(
            f"{name} | Epoch {epoch:03d}/{epochs} | "
            f"train loss {train_loss:.4f}, accuracy {train_accuracy:.4f} | "
            f"validation loss {validation_loss:.4f}, "
            f"accuracy {validation_accuracy:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "test_loss": None,
                "test_accuracy": None,
            }
        )
        pd.DataFrame(history).to_csv(metrics_path, index=False)
        if validation_accuracy > best_accuracy + MIN_DELTA:
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "epoch": epoch,
                "settings": checkpoint_settings,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            },
            checkpoint_path,
        )
        tracker.log_epoch(
            epoch,
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy,
        )

        if early_stopping and epochs_without_improvement >= patience:
            print(
                f"{name}: early stopping after {patience} epochs without improvement."
            )
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)

    report = evaluate_and_save_reports(
        model=model,
        dataloader=dataloaders["test"],
        loss_function=loss_function,
        device=device,
        class_names=class_names,
        output_dir=model_path.parent,
        run_name=name,
        history=history,
    )
    test_loss = report["test_loss"]
    test_accuracy = report["test_accuracy"]
    history[best_epoch - 1]["test_loss"] = test_loss
    history[best_epoch - 1]["test_accuracy"] = test_accuracy
    pd.DataFrame(history).to_csv(metrics_path, index=False)

    metadata = {
        "schema_version": 1,
        "status": "completed",
        "source_script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "run_name": name,
        "model": model_summary(
            model,
            "ResNet18",
            (3, 64, 64),
            pretrained_weights=str(models.ResNet18_Weights.DEFAULT),
        ),
        "loss": {
            "name": loss_function.__class__.__name__,
            "class_weights": None,
            "reduction": loss_function.reduction,
        },
        "optimizer": optimizer_summary(optimizer),
        "training": {
            "seed": SEED,
            "configured_epochs": epochs,
            "completed_epochs": len(history),
            "batch_size": BATCH_SIZE,
            "number_of_workers": NUM_WORKERS,
            "early_stopping": {
                "enabled": early_stopping,
                "patience": patience,
                "minimum_improvement": MIN_DELTA,
                "monitor": "validation_accuracy",
                "mode": "max",
                "stopped_early": early_stopping and len(history) < epochs,
            },
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "duration_seconds": time.perf_counter() - started_timer,
        },
        "preprocessing": {
            "image_mode": "RGB",
            "image_size": [64, 64],
            "normalization_mean": list(MEAN),
            "normalization_standard_deviation": list(STD),
            "normalization_source": "ImageNet pretrained weights",
            "training_augmentation": [
                "random horizontal flip",
                "random vertical flip",
                "random rotation by 0, 90, 180, or 270 degrees",
            ],
            "validation_augmentation": None,
            "test_augmentation": None,
        },
        "dataset": dataset_summary(SPLIT_DIR, PROJECT_ROOT),
        "runtime": runtime_summary(device),
        "results": {
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        },
        "artifacts": {
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "epoch_metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
            "evaluation_report_directory": str(
                model_path.parent.relative_to(PROJECT_ROOT)
            ),
            "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
        },
    }
    save_metadata(metadata_path, metadata)
    tracker.finish(
        model=model,
        final_metrics={
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_macro_precision": report["macro_precision"],
            "test_macro_recall": report["macro_recall"],
            "test_macro_f1": report["macro_f1"],
        },
        output_directory=model_path.parent,
        number_of_classes=number_of_classes,
        registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
        artifact_stem=name,
    )
    checkpoint_path.unlink(missing_ok=True)

    print(f"{name} best epoch: {best_epoch}")
    print(f"{name} best validation accuracy: {best_accuracy:.4f}")
    print(f"{name} final test loss: {test_loss:.4f}, accuracy: {test_accuracy:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved metadata: {metadata_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_frame = pd.read_csv(SPLIT_DIR / "train.csv")
    label_table = train_frame[["label_id", "label"]].drop_duplicates()
    class_names = label_table.sort_values("label_id")["label"].tolist()

    for attempt in ATTEMPTS:
        train_attempt(attempt, device, class_names)


if __name__ == "__main__":
    main()
