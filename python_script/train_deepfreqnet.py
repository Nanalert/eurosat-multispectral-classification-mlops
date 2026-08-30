"""Train a DeepFreqNet-inspired model adapted for EuroSAT RGB."""

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
OUTPUT_DIR = PROJECT_ROOT / "models" / "deepfreqnet" / "deepfreqnet5"
MODEL_PATH = OUTPUT_DIR / "deepfreqnet.pt"
METRICS_PATH = OUTPUT_DIR / "deepfreqnet_metrics.csv"
CLASS_METRICS_PATH = OUTPUT_DIR / "deepfreqnet_class_metrics.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "deepfreqnet_confusion_matrix.csv"
METADATA_PATH = OUTPUT_DIR / "deepfreqnet_metadata.json"

# EuroSAT training settings. The paper did not report one universal setup.
SEED = 42
EPOCHS = 500
BATCH_SIZE = 64
NUM_WORKERS = 0
LEARNING_RATE = 0.0001
EARLY_STOPPING = False
PATIENCE = 10
MIN_DELTA = 0.000001
USE_CLASS_WEIGHTS = True

# Set this to False when you want to train without MLflow.
MLFLOW_ENABLED = True
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "eurosat-rgb-classification"
MLFLOW_REGISTERED_MODEL_NAME = None

# EuroSAT RGB statistics calculated in EDA.ipynb.
# MEAN = (0.344377, 0.380292, 0.407771)
# STD = (0.202657, 0.136891, 0.115544)

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


class MultiKernelBlock(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch_1x1 = nn.Conv2d(input_channels, 64, kernel_size=1)
        self.branch_3x3 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.branch_5x5 = nn.Conv2d(input_channels, 64, kernel_size=5, padding=2)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features):
        branches = [
            self.activation(self.branch_1x1(features)),
            self.activation(self.branch_3x3(features)),
            self.activation(self.branch_5x5(features)),
        ]
        return torch.cat(branches, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.main = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features):
        combined = self.main(features) + self.shortcut(features)
        return self.activation(self.batch_norm(combined))


class DeepFreqNet(nn.Module):
    """Paper-inspired architecture adapted from 224x224 to 64x64 input."""

    def __init__(self, number_of_classes):
        super().__init__()
        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.multi_kernel_block = MultiKernelBlock(32)
        self.multi_kernel_pool = nn.MaxPool2d(2)
        self.depthwise_separable_block = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1, groups=192),
            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.residual_block = ResidualBlock(128, 256)
        self.residual_pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, number_of_classes),
        )

    def forward(self, images):
        features = self.initial_block(images)
        features = self.multi_kernel_pool(self.multi_kernel_block(features))
        features = self.depthwise_separable_block(features)
        features = self.residual_pool(self.residual_block(features))
        return self.classifier(features)


def create_dataloaders(device):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotate90(),
            transforms.RandomAffine(
                degrees=0,
                scale=(0.9, 1.1),
                shear=(-10, 10),
                fill=(88, 97, 104),
            ),
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
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()

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
    label_table = label_table.sort_values("label_id")
    class_names = label_table["label"].tolist()
    number_of_classes = len(class_names)

    model = DeepFreqNet(number_of_classes).to(device)
    if USE_CLASS_WEIGHTS:
        class_counts = train_frame["label_id"].value_counts().sort_index()
        weights = len(train_frame) / (number_of_classes * class_counts.to_numpy())
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        class_weights = None
    training_loss_function = nn.CrossEntropyLoss(weight=class_weights)
    evaluation_loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tracker = MLflowTracker(
        enabled=MLFLOW_ENABLED,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        run_name=OUTPUT_DIR.name,
        parameters={
            "architecture": "DeepFreqNet-inspired",
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "number_of_workers": NUM_WORKERS,
            "learning_rate": LEARNING_RATE,
            "optimizer": optimizer.__class__.__name__,
            "training_loss_function": training_loss_function.__class__.__name__,
            "evaluation_loss_function": evaluation_loss_function.__class__.__name__,
            "class_weights": USE_CLASS_WEIGHTS,
            "early_stopping": EARLY_STOPPING,
            "patience": PATIENCE,
            "minimum_improvement": MIN_DELTA,
            "normalization_mean": MEAN,
            "normalization_standard_deviation": STD,
            "number_of_classes": number_of_classes,
        },
        tags={"architecture": "deepfreqnet", "dataset": "EuroSAT_RGB"},
    )
    tracker.start()
    best_accuracy = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Using device: {device}")
    print(f"Trainable parameters: {parameter_count:,}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            dataloaders["train"],
            training_loss_function,
            device,
            optimizer,
        )
        validation_loss, validation_accuracy = run_epoch(
            model, dataloaders["validation"], evaluation_loss_function, device
        )
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
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
        loss_function=evaluation_loss_function,
        device=device,
        class_names=class_names,
        output_dir=OUTPUT_DIR,
        run_name="deepfreqnet",
        history=history,
    )
    test_loss = report["test_loss"]
    test_accuracy = report["test_accuracy"]
    macro_precision = report["macro_precision"]
    macro_recall = report["macro_recall"]
    macro_f1 = report["macro_f1"]
    history[best_epoch - 1]["test_loss"] = test_loss
    history[best_epoch - 1]["test_accuracy"] = test_accuracy
    pd.DataFrame(history).to_csv(METRICS_PATH, index=False)
    metadata = {
        "schema_version": 1,
        "status": "completed",
        "source_script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "model": model_summary(model, "DeepFreqNet-inspired", (3, 64, 64)),
        "architecture_notes": {
            "source": "adapted from the user-provided DeepFreqNet paper description",
            "adaptation": "native 64x64 input, 10 classes, 512-unit dense layer",
            "frequency_operation_present": False,
            "dropout_enabled": False,
        },
        "loss": {
            "training": training_loss_function.__class__.__name__,
            "evaluation": evaluation_loss_function.__class__.__name__,
            "class_weights_enabled": USE_CLASS_WEIGHTS,
            "class_weight_formula": "total_samples / (number_of_classes * class_count)",
            "class_weights_by_label_id": (
                class_weights.detach().cpu().tolist()
                if class_weights is not None
                else None
            ),
            "reduction": training_loss_function.reduction,
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
            "normalization_source": "EuroSAT EDA",
            "training_augmentation": [
                "random horizontal flip",
                "random vertical flip",
                "random rotation by 0, 90, 180, or 270 degrees",
                "random zoom from 0.9 to 1.1",
                "random shear from -10 to 10 degrees",
                "affine fill colour RGB (88, 97, 104)",
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
            "test_macro_precision": macro_precision,
            "test_macro_recall": macro_recall,
            "test_macro_f1": macro_f1,
        },
        "artifacts": {
            "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "epoch_metrics_path": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "class_metrics_path": str(CLASS_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "confusion_matrix_path": str(
                CONFUSION_MATRIX_PATH.relative_to(PROJECT_ROOT)
            ),
            "evaluation_report_directory": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
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
            "test_macro_precision": macro_precision,
            "test_macro_recall": macro_recall,
            "test_macro_f1": macro_f1,
        },
        output_directory=OUTPUT_DIR,
        number_of_classes=number_of_classes,
        registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Final test loss: {test_loss:.4f}, accuracy: {test_accuracy:.4f}")
    print(
        f"Macro precision: {macro_precision:.4f}, "
        f"recall: {macro_recall:.4f}, F1: {macro_f1:.4f}"
    )
    print(f"Saved model and reports in: {OUTPUT_DIR}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
