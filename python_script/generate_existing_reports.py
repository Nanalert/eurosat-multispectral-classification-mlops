"""Generate standardized reports for all existing EuroSAT checkpoints."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from .evaluation_reports import (
        evaluate_and_save_reports,
        save_experiment_comparison,
    )
    from .train_cnn import SmallCNN
    from .train_deepfreqnet import DeepFreqNet
except ImportError:
    from evaluation_reports import evaluate_and_save_reports, save_experiment_comparison
    from train_cnn import SmallCNN
    from train_deepfreqnet import DeepFreqNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = PROJECT_ROOT / "data" / "split" / "test.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
BATCH_SIZE = 128
NUM_WORKERS = 0
EUROSAT_MEAN = (0.344377, 0.380292, 0.407771)
EUROSAT_STD = (0.202657, 0.136891, 0.115544)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


EXPERIMENTS = [
    {
        "run_name": "cnn1",
        "architecture": "SmallCNN",
        "checkpoint": "models/cnn/cnn1/best_cnn.pt",
        "history": "models/cnn/cnn1/cnn_metrics.csv",
        "normalization": "eurosat",
    },
    {
        "run_name": "cnn2",
        "architecture": "SmallCNN",
        "checkpoint": "models/cnn/cnn2/best_cnn.pt",
        "history": "models/cnn/cnn2/cnn_metrics.csv",
        "normalization": "eurosat",
    },
    {
        "run_name": "cnn3",
        "architecture": "SmallCNN",
        "checkpoint": "models/cnn/cnn3/best_cnn.pt",
        "history": "models/cnn/cnn3/cnn_metrics.csv",
        "normalization": "eurosat",
    },
    {
        "run_name": "resnet1",
        "architecture": "ResNet18",
        "checkpoint": "models/resnet/resnet1/resnet1.pt",
        "history": "models/resnet/resnet1/resnet1_metrics.csv",
        "normalization": "imagenet",
    },
    {
        "run_name": "resnet2",
        "architecture": "ResNet18",
        "checkpoint": "models/resnet/resnet2/resnet2.pt",
        "history": "models/resnet/resnet2/resnet2_metrics.csv",
        "normalization": "imagenet",
    },
    {
        "run_name": "resnet3",
        "architecture": "ResNet18",
        "checkpoint": "models/resnet/resnet3/resnet3.pt",
        "history": "models/resnet/resnet3/resnet3_metrics.csv",
        "normalization": "imagenet",
    },
    {
        "run_name": "deepfreqnet1",
        "architecture": "DeepFreqNet",
        "checkpoint": "models/deepfreqnet/deepfreqnet1/deepfreqnet1.pt",
        "history": "models/deepfreqnet/deepfreqnet1/deepfreqnet_metrics1.csv",
        "normalization": "eurosat",
    },
    {
        "run_name": "deepfreqnet2",
        "architecture": "DeepFreqNet",
        "checkpoint": "models/deepfreqnet/deepfreqnet2/deepfreqnet.pt",
        "history": "models/deepfreqnet/deepfreqnet2/deepfreqnet_metrics.csv",
        "normalization": "eurosat",
    },
    {
        "run_name": "deepfreqnet3",
        "architecture": "DeepFreqNet",
        "checkpoint": "models/deepfreqnet/deepfreqnet3/deepfreqnet.pt",
        "history": "models/deepfreqnet/deepfreqnet3/deepfreqnet_metrics.csv",
        "normalization": "eurosat",
    },
]


class TestDataset(Dataset):
    def __init__(self, transform):
        self.data = pd.read_csv(TEST_CSV)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        with Image.open(PROJECT_ROOT / row["image_path"]) as image:
            image = self.transform(image.convert("RGB"))
        return image, int(row["label_id"])


def create_model(architecture, number_of_classes):
    if architecture == "SmallCNN":
        return SmallCNN(number_of_classes)
    if architecture == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, number_of_classes)
        return model
    if architecture == "DeepFreqNet":
        return DeepFreqNet(number_of_classes)
    raise ValueError(f"Unknown architecture: {architecture}")


def create_test_loader(normalization, device):
    if normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = EUROSAT_MEAN, EUROSAT_STD
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    return DataLoader(
        TestDataset(transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )


def main():
    test_frame = pd.read_csv(TEST_CSV)
    label_table = test_frame[["label_id", "label"]].drop_duplicates()
    class_names = label_table.sort_values("label_id")["label"].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    comparison_rows = []

    for experiment in EXPERIMENTS:
        run_name = experiment["run_name"]
        checkpoint_path = PROJECT_ROOT / experiment["checkpoint"]
        history_path = PROJECT_ROOT / experiment["history"]
        print(f"Generating reports for {run_name}...")

        model = create_model(experiment["architecture"], len(class_names))
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        dataloader = create_test_loader(experiment["normalization"], device)
        report = evaluate_and_save_reports(
            model=model,
            dataloader=dataloader,
            loss_function=nn.CrossEntropyLoss(),
            device=device,
            class_names=class_names,
            output_dir=REPORT_DIR / run_name,
            run_name=run_name,
            history=history_path,
        )

        history = pd.read_csv(history_path)
        best_row = history.loc[history["validation_accuracy"].idxmax()]
        comparison_rows.append(
            {
                "run_name": run_name,
                "architecture": experiment["architecture"],
                "completed_epochs": int(len(history)),
                "best_epoch": int(best_row["epoch"]),
                "best_validation_accuracy": float(best_row["validation_accuracy"]),
                **report,
            }
        )
        del model, dataloader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_experiment_comparison(comparison_rows, REPORT_DIR)
    print(f"Saved all reports in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
