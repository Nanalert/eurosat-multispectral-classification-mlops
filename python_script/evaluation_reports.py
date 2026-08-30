"""Reusable classification evaluation, CSV reporting, and plotting helpers."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _save_figure(figure, path):
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _history_frame(history):
    if history is None:
        return None
    if isinstance(history, (str, Path)):
        return pd.read_csv(history)
    if isinstance(history, pd.DataFrame):
        return history.copy()
    return pd.DataFrame(history)


def _plot_training_curves(history, path, run_name):
    frame = _history_frame(history)
    if frame is None or frame.empty:
        return

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(frame["epoch"], frame["train_accuracy"], label="Train")
    axes[0].plot(frame["epoch"], frame["validation_accuracy"], label="Validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(frame["epoch"], frame["train_loss"], label="Train")
    axes[1].plot(frame["epoch"], frame["validation_loss"], label="Validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.suptitle(f"{run_name} training history")
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_class_metrics(frame, path, run_name):
    positions = np.arange(len(frame))
    width = 0.25
    figure, axis = plt.subplots(figsize=(13, 5.5))
    axis.bar(positions - width, frame["precision"], width, label="Precision")
    axis.bar(positions, frame["recall"], width, label="Recall")
    axis.bar(positions + width, frame["f1"], width, label="F1")
    axis.set(
        title=f"{run_name} per-class test metrics",
        ylabel="Score",
        ylim=(0, 1.05),
        xticks=positions,
        xticklabels=frame["class"],
    )
    axis.tick_params(axis="x", rotation=40)
    axis.legend(ncol=3)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_support(frame, path, run_name):
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(frame["class"], frame["support"], color="#4c78a8")
    axis.set(title=f"{run_name} test support", ylabel="Images")
    axis.tick_params(axis="x", rotation=40)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_confusion_matrix(matrix, class_names, path, title, normalized):
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title=title,
        xlabel="Predicted class",
        ylabel="Actual class",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axis.tick_params(axis="x", rotation=45)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            label = f"{value:.2f}" if normalized else str(int(value))
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    _save_figure(figure, path)


@torch.inference_mode()
def evaluate_and_save_reports(
    model,
    dataloader,
    loss_function,
    device,
    class_names,
    output_dir,
    run_name,
    history=None,
):
    """Evaluate one final model and save reusable CSV, JSON, and PNG reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    number_of_classes = len(class_names)
    model.eval()

    confusion = torch.zeros(number_of_classes, number_of_classes, dtype=torch.int64)
    all_actual = []
    all_predicted = []
    all_confidence = []
    total_loss = 0.0
    total_examples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)

        total_loss += loss_function(logits, labels).item() * labels.size(0)
        total_examples += labels.size(0)
        actual_cpu = labels.cpu()
        predicted_cpu = predictions.cpu()
        indices = actual_cpu * number_of_classes + predicted_cpu
        confusion += torch.bincount(
            indices, minlength=number_of_classes * number_of_classes
        ).reshape(number_of_classes, number_of_classes)
        all_actual.extend(actual_cpu.tolist())
        all_predicted.extend(predicted_cpu.tolist())
        all_confidence.extend(confidence.cpu().tolist())

    true_positive = confusion.diag().float()
    predicted_count = confusion.sum(dim=0).float()
    actual_count = confusion.sum(dim=1).float()
    precision = true_positive / predicted_count.clamp_min(1)
    recall = true_positive / actual_count.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(0.000000000001)
    accuracy = true_positive.sum().item() / total_examples

    class_metrics = pd.DataFrame(
        {
            "class": class_names,
            "precision": precision.numpy(),
            "recall": recall.numpy(),
            "f1": f1.numpy(),
            "support": actual_count.numpy().astype(int),
        }
    )
    class_metrics.to_csv(output_dir / f"{run_name}_class_metrics.csv", index=False)

    confusion_frame = pd.DataFrame(
        confusion.numpy(), index=class_names, columns=class_names
    )
    confusion_frame.index.name = "actual_class"
    confusion_frame.to_csv(output_dir / f"{run_name}_confusion_matrix.csv")
    normalized = confusion.numpy() / np.maximum(
        confusion.sum(dim=1).numpy()[:, None], 1
    )
    normalized_frame = pd.DataFrame(normalized, index=class_names, columns=class_names)
    normalized_frame.index.name = "actual_class"
    normalized_frame.to_csv(output_dir / f"{run_name}_normalized_confusion_matrix.csv")

    image_paths = dataloader.dataset.data["image_path"].tolist()
    predictions_frame = pd.DataFrame(
        {
            "image_path": image_paths,
            "actual_label_id": all_actual,
            "actual_class": [class_names[index] for index in all_actual],
            "predicted_label_id": all_predicted,
            "predicted_class": [class_names[index] for index in all_predicted],
            "confidence": all_confidence,
            "correct": np.equal(all_actual, all_predicted),
        }
    )
    predictions_frame.to_csv(
        output_dir / f"{run_name}_test_predictions.csv", index=False
    )

    summary = {
        "run_name": run_name,
        "test_loss": total_loss / total_examples,
        "test_accuracy": accuracy,
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "test_images": total_examples,
    }
    with (output_dir / f"{run_name}_test_summary.json").open(
        "w", encoding="utf-8"
    ) as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")

    _plot_training_curves(
        history, output_dir / f"{run_name}_training_curves.png", run_name
    )
    _plot_class_metrics(
        class_metrics, output_dir / f"{run_name}_class_metrics.png", run_name
    )
    _plot_support(class_metrics, output_dir / f"{run_name}_class_support.png", run_name)
    _plot_confusion_matrix(
        confusion.numpy(),
        class_names,
        output_dir / f"{run_name}_confusion_matrix.png",
        f"{run_name} confusion matrix",
        normalized=False,
    )
    _plot_confusion_matrix(
        normalized,
        class_names,
        output_dir / f"{run_name}_normalized_confusion_matrix.png",
        f"{run_name} normalized confusion matrix",
        normalized=True,
    )
    return summary


def save_experiment_comparison(rows, output_dir):
    """Save overall validation/test comparison data and chart."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "experiment_comparison.csv", index=False)

    positions = np.arange(len(frame))
    width = 0.38
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(
        positions - width / 2,
        frame["best_validation_accuracy"],
        width,
        label="Best validation",
    )
    axis.bar(
        positions + width / 2,
        frame["test_accuracy"],
        width,
        label="Test",
    )
    axis.set(
        title="Experiment accuracy comparison",
        ylabel="Accuracy",
        ylim=(0.9, 1.0),
        xticks=positions,
        xticklabels=frame["run_name"],
    )
    axis.tick_params(axis="x", rotation=40)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, output_dir / "experiment_accuracy_comparison.png")
