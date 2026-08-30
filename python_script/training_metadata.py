"""Shared helpers for saving reproducible training metadata."""

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def dataset_summary(split_dir, project_root):
    summary = {
        "split_directory": str(split_dir.relative_to(project_root)),
        "splits": {},
    }
    class_mapping = None

    for split_name in ("train", "validation", "test"):
        csv_path = split_dir / f"{split_name}.csv"
        frame = pd.read_csv(csv_path)
        counts = frame["label"].value_counts().sort_index()
        summary["splits"][split_name] = {
            "csv_path": str(csv_path.relative_to(project_root)),
            "image_count": int(len(frame)),
            "class_counts": {name: int(count) for name, count in counts.items()},
        }
        mapping = frame[["label_id", "label"]].drop_duplicates().sort_values("label_id")
        current_mapping = {
            str(int(row.label_id)): row.label for row in mapping.itertuples()
        }
        if class_mapping is None:
            class_mapping = current_mapping
        elif current_mapping != class_mapping:
            raise ValueError("Class mappings differ between split CSV files")

    summary["class_mapping"] = class_mapping
    summary["number_of_classes"] = len(class_mapping)
    summary["split_strategy"] = "stratified 70% train, 15% validation, 15% test"
    return summary


def model_summary(model, model_name, input_shape, pretrained_weights=None):
    return {
        "name": model_name,
        "class_name": model.__class__.__name__,
        "architecture": str(model),
        "input_shape": list(input_shape),
        "total_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "pretrained_weights": pretrained_weights,
        "output_type": "raw logits",
    }


def optimizer_summary(optimizer):
    defaults = optimizer.defaults
    return {
        "name": optimizer.__class__.__name__,
        "learning_rate": float(defaults["lr"]),
        "betas": [float(value) for value in defaults.get("betas", ())],
        "epsilon": float(defaults.get("eps", 0.0)),
        "weight_decay": float(defaults.get("weight_decay", 0.0)),
        "amsgrad": bool(defaults.get("amsgrad", False)),
        "maximize": bool(defaults.get("maximize", False)),
        "learning_rate_scheduler": None,
    }


def runtime_summary(device):
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "operating_system": platform.platform(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
    }


def save_metadata(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")
