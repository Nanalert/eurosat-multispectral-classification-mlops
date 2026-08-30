"""Shared production inference for EuroSAT RGB classification."""

import hashlib
import json
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "production" / "resnet18-production.pt"
DEFAULT_METADATA_PATH = (
    PROJECT_ROOT / "models" / "production" / "resnet18-production_metadata.json"
)


@lru_cache(maxsize=8)
def checkpoint_sha256(model_path=DEFAULT_MODEL_PATH):
    """Return a stable fingerprint for a model checkpoint."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. Run 'dvc pull' first."
        )

    digest = hashlib.sha256()
    with model_path.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_metadata(metadata_path):
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}. Run 'dvc pull' first."
        )

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    try:
        model_name = metadata["model"]["name"]
        class_mapping = metadata["dataset"]["class_mapping"]
        expected_class_ids = {str(index) for index in range(len(class_mapping))}
        if set(class_mapping) != expected_class_ids:
            raise ValueError("class_mapping IDs must be consecutive and start at 0")
        class_names = [class_mapping[str(index)] for index in range(len(class_mapping))]
        image_size = tuple(metadata["preprocessing"]["image_size"])
        mean = tuple(metadata["preprocessing"]["normalization_mean"])
        standard_deviation = tuple(
            metadata["preprocessing"]["normalization_standard_deviation"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid model metadata: {metadata_path}") from error

    if model_name != "ResNet18":
        raise ValueError("The production inference module requires ResNet18 metadata")
    if len(image_size) != 2 or len(mean) != 3 or len(standard_deviation) != 3:
        raise ValueError(f"Invalid preprocessing metadata: {metadata_path}")
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"Class names must be unique: {metadata_path}")
    return metadata, class_names, image_size, mean, standard_deviation


@lru_cache(maxsize=8)
def get_model_details(
    model_path=DEFAULT_MODEL_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
):
    """Return deployment details without loading model weights into PyTorch."""
    metadata, class_names, image_size, _, _ = _read_metadata(metadata_path)
    fingerprint = checkpoint_sha256(model_path)
    run_name = metadata.get("run_name", metadata["model"]["name"])
    return {
        "model": metadata["model"]["name"],
        "run_name": run_name,
        "model_version": f"{run_name}-{fingerprint[:12]}",
        "checkpoint_sha256": fingerprint,
        "number_of_classes": len(class_names),
        "input_size": list(image_size),
        "training_finished_at_utc": metadata.get("training", {}).get("finished_at_utc"),
        "best_epoch": metadata.get("results", {}).get("best_epoch"),
        "best_validation_accuracy": metadata.get("results", {}).get(
            "best_validation_accuracy"
        ),
        "test_accuracy": metadata.get("results", {}).get("test_accuracy"),
    }


def _open_image(image_source):
    try:
        if isinstance(image_source, Image.Image):
            return image_source.copy().convert("RGB")

        if isinstance(image_source, (str, Path)):
            image_path = Path(image_source)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            with Image.open(image_path) as image:
                return image.convert("RGB")

        if isinstance(image_source, (bytes, bytearray)):
            with Image.open(BytesIO(image_source)) as image:
                return image.convert("RGB")

        if hasattr(image_source, "read"):
            original_position = None
            if hasattr(image_source, "tell") and hasattr(image_source, "seek"):
                try:
                    original_position = image_source.tell()
                    image_source.seek(0)
                except (OSError, ValueError):
                    pass
            image_bytes = image_source.read()
            if original_position is not None and hasattr(image_source, "seek"):
                image_source.seek(original_position)
            if not isinstance(image_bytes, (bytes, bytearray)):
                raise ValueError("The readable file object must provide image bytes")
            with Image.open(BytesIO(image_bytes)) as image:
                return image.convert("RGB")
    except FileNotFoundError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("The supplied file is not a readable image") from error

    raise TypeError(
        "image_source must be a path, PIL image, bytes, or readable file object"
    )


class EuroSATPredictor:
    """Load the production model once and use it for repeated predictions."""

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        metadata_path=DEFAULT_METADATA_PATH,
        device=None,
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}. Run 'dvc pull' first."
            )

        (
            self.metadata,
            self.class_names,
            self.image_size,
            mean,
            standard_deviation,
        ) = _read_metadata(self.metadata_path)
        self.details = get_model_details(self.model_path, self.metadata_path)
        self.model_version = self.details["model_version"]
        configured_device = device or os.getenv("MODEL_DEVICE")
        self.device = torch.device(
            configured_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, standard_deviation),
            ]
        )
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        state = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image_source):
        """Return one normalized NCHW tensor on the predictor's device."""
        image = _open_image(image_source)
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict(self, image_source, top_k=3):
        """Return the predicted class, confidence, and all class probabilities."""
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")
        if not 1 <= top_k <= len(self.class_names):
            raise ValueError(f"top_k must be between 1 and {len(self.class_names)}")

        tensor = self.preprocess(image_source)
        probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu()
        top_probabilities, top_indices = probabilities.topk(top_k)
        label_id = int(top_indices[0].item())

        return {
            "label_id": label_id,
            "label": self.class_names[label_id],
            "confidence": float(top_probabilities[0].item()),
            "top_predictions": [
                {
                    "label_id": int(index.item()),
                    "label": self.class_names[int(index.item())],
                    "confidence": float(probability.item()),
                }
                for probability, index in zip(
                    top_probabilities, top_indices, strict=True
                )
            ],
            "probabilities": {
                class_name: float(probabilities[index].item())
                for index, class_name in enumerate(self.class_names)
            },
        }


@lru_cache(maxsize=4)
def load_model(
    model_path=DEFAULT_MODEL_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
    device=None,
):
    """Return a cached predictor so repeated calls do not reload model weights."""
    return EuroSATPredictor(model_path, metadata_path, device)


def preprocess_image(
    image_source,
    model_path=DEFAULT_MODEL_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
    device=None,
):
    """Preprocess an image using the production model's saved configuration."""
    return load_model(model_path, metadata_path, device).preprocess(image_source)


def predict(
    image_source,
    top_k=3,
    model_path=DEFAULT_MODEL_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
    device=None,
):
    """Make a prediction using the cached production model."""
    return load_model(model_path, metadata_path, device).predict(image_source, top_k)
