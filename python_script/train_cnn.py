"""Train and evaluate a small CNN on EuroSAT RGB."""

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
from torchvision import transforms

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
MODEL_PATH = PROJECT_ROOT / "models" / "cnn4" / "best_cnn.pt"
METRICS_PATH = PROJECT_ROOT / "models" / "cnn4" / "cnn_metrics.csv"
METADATA_PATH = PROJECT_ROOT / "models" / "cnn4" / "cnn_metadata.json"

# Change these training settings as needed.
SEED = 42
EPOCHS = 500
BATCH_SIZE = 64
NUM_WORKERS = 0
LEARNING_RATE = 0.0001
EARLY_STOPPING = False
PATIENCE = 50
MIN_DELTA = 0.00000000000000000000000000000001

# Set this to False when you want to train without MLflow.
MLFLOW_ENABLED = True
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "eurosat-rgb-classification"
MLFLOW_REGISTERED_MODEL_NAME = None

# EuroSAT RGB statistics calculated in EDA.ipynb.
MEAN = (0.344377, 0.380292, 0.407771)
STD = (0.202657, 0.136891, 0.115544)


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


class SmallCNN(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, number_of_classes)

    def forward(self, images):
        features = self.features(images)
        return self.classifier(torch.flatten(features, 1))


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


def main():
    started_at = utc_now()
    started_timer = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = create_dataloaders(device)

    train_frame = pd.read_csv(SPLIT_DIR / "train.csv")
    label_table = train_frame[["label_id", "label"]].drop_duplicates()
    class_names = label_table.sort_values("label_id")["label"].tolist()
    number_of_classes = len(class_names)
    model = SmallCNN(number_of_classes).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tracker = MLflowTracker(
        enabled=MLFLOW_ENABLED,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        run_name=MODEL_PATH.parent.name,
        parameters={
            "architecture": "SmallCNN",
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "number_of_workers": NUM_WORKERS,
            "learning_rate": LEARNING_RATE,
            "optimizer": optimizer.__class__.__name__,
            "loss_function": loss_function.__class__.__name__,
            "early_stopping": EARLY_STOPPING,
            "patience": PATIENCE,
            "minimum_improvement": MIN_DELTA,
            "normalization_mean": MEAN,
            "normalization_standard_deviation": STD,
            "number_of_classes": number_of_classes,
        },
        tags={"architecture": "cnn", "dataset": "EuroSAT_RGB"},
    )
    tracker.start()
    best_accuracy = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model, dataloaders["train"], loss_function, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, dataloaders["validation"], loss_function, device
        )
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
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
        pd.DataFrame(history).to_csv(METRICS_PATH, index=False)
        tracker.log_epoch(
            epoch,
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy,
        )

        if validation_accuracy > best_accuracy + MIN_DELTA:
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if EARLY_STOPPING and epochs_without_improvement >= PATIENCE:
            print(f"Early stopping after {PATIENCE} epochs without improvement.")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)

    report = evaluate_and_save_reports(
        model=model,
        dataloader=dataloaders["test"],
        loss_function=loss_function,
        device=device,
        class_names=class_names,
        output_dir=MODEL_PATH.parent,
        run_name="cnn",
        history=history,
    )
    test_loss = report["test_loss"]
    test_accuracy = report["test_accuracy"]
    history[best_epoch - 1]["test_loss"] = test_loss
    history[best_epoch - 1]["test_accuracy"] = test_accuracy
    pd.DataFrame(history).to_csv(METRICS_PATH, index=False)

    metadata = {
        "schema_version": 1,
        "status": "completed",
        "source_script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "model": model_summary(model, "SmallCNN", (3, 64, 64)),
        "loss": {
            "name": loss_function.__class__.__name__,
            "class_weights": None,
            "reduction": loss_function.reduction,
        },
        "optimizer": optimizer_summary(optimizer),
        "training": {
            "seed": SEED,
            "configured_epochs": EPOCHS,
            "completed_epochs": len(history),
            "batch_size": BATCH_SIZE,
            "number_of_workers": NUM_WORKERS,
            "early_stopping": {
                "enabled": EARLY_STOPPING,
                "patience": PATIENCE,
                "minimum_improvement": MIN_DELTA,
                "monitor": "validation_accuracy",
                "mode": "max",
                "stopped_early": EARLY_STOPPING and len(history) < EPOCHS,
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
            "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "epoch_metrics_path": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "evaluation_report_directory": str(
                MODEL_PATH.parent.relative_to(PROJECT_ROOT)
            ),
            "metadata_path": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    save_metadata(METADATA_PATH, metadata)
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
        output_directory=MODEL_PATH.parent,
        number_of_classes=number_of_classes,
        registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Final test loss: {test_loss:.4f}, accuracy: {test_accuracy:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
