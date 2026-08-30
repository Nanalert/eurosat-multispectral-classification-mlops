import io
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from python_script.inference import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    EuroSATPredictor,
    checkpoint_sha256,
    get_model_details,
    load_model,
    predict,
    preprocess_image,
)


def test_load_model_is_cached(predictor):
    assert predictor is load_model(device="cpu")
    assert predictor.model.training is False
    assert len(predictor.class_names) == 10
    assert predictor.model_version.startswith("resnet18-production-")


def test_model_details_match_checkpoint_and_metadata(predictor):
    details = get_model_details()

    assert details["checkpoint_sha256"] == checkpoint_sha256()
    assert len(details["checkpoint_sha256"]) == 64
    assert details["model_version"] == predictor.model_version
    assert details["number_of_classes"] == len(predictor.class_names)


def test_preprocess_image(sample_image_path):
    tensor = preprocess_image(sample_image_path, device="cpu")

    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"


def test_prediction_is_json_safe(predictor, sample_image_path):
    result = predictor.predict(sample_image_path, top_k=3)

    assert result["label"] == "River"
    assert result["label_id"] == 8
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_predictions"]) == 3
    assert len(result["probabilities"]) == 10
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    json.dumps(result)


def test_supported_image_inputs(predictor, sample_image_path, sample_image_bytes):
    path_result = predictor.predict(sample_image_path)
    bytes_result = predictor.predict(sample_image_bytes)
    stream = io.BytesIO(sample_image_bytes)
    stream.seek(12)

    with Image.open(sample_image_path) as image:
        pil_result = predictor.predict(image)
    stream_result = predictor.predict(stream)

    assert stream.tell() == 12
    assert {
        path_result["label"],
        bytes_result["label"],
        pil_result["label"],
        stream_result["label"],
    } == {"River"}


@pytest.mark.parametrize(
    ("pillow_format", "mode"),
    [
        ("JPEG", "RGB"),
        ("PNG", "RGBA"),
        ("WEBP", "RGB"),
        ("TIFF", "L"),
    ],
)
def test_supported_image_formats(predictor, pillow_format, mode):
    image = Image.new(mode, (81, 53), color=128)
    buffer = io.BytesIO()
    image.save(buffer, format=pillow_format)

    tensor = predictor.preprocess(buffer.getvalue())

    assert tensor.shape == (1, 3, 64, 64)
    assert torch.isfinite(tensor).all()


def test_cpu_is_selected_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cpu_predictor = EuroSATPredictor(device=None)

    assert cpu_predictor.device.type == "cpu"


def test_model_device_environment_overrides_auto_selection(monkeypatch):
    monkeypatch.setenv("MODEL_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    cpu_predictor = EuroSATPredictor(device=None)

    assert cpu_predictor.device.type == "cpu"


def test_unavailable_requested_cuda_is_reported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        EuroSATPredictor(device="cuda")


def test_missing_checkpoint_and_metadata_are_reported(tmp_path):
    missing_model = tmp_path / "missing.pt"
    missing_metadata = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        EuroSATPredictor(missing_model, DEFAULT_METADATA_PATH, device="cpu")
    with pytest.raises(FileNotFoundError, match="Model metadata not found"):
        EuroSATPredictor(DEFAULT_MODEL_PATH, missing_metadata, device="cpu")


def test_invalid_class_mapping_is_rejected(tmp_path):
    metadata = json.loads(Path(DEFAULT_METADATA_PATH).read_text(encoding="utf-8"))
    metadata["dataset"]["class_mapping"]["1"] = "AnnualCrop"
    metadata_path = tmp_path / "invalid-metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Class names must be unique"):
        EuroSATPredictor(DEFAULT_MODEL_PATH, metadata_path, device="cpu")


@pytest.mark.parametrize("top_k", [0, 11])
def test_invalid_top_k(predictor, sample_image_path, top_k):
    with pytest.raises(ValueError):
        predictor.predict(sample_image_path, top_k=top_k)


def test_invalid_inputs(predictor):
    with pytest.raises(ValueError, match="not a readable image"):
        predictor.predict(b"not an image")
    with pytest.raises(FileNotFoundError, match="Image not found"):
        predictor.predict("missing-image.jpg")
    with pytest.raises(TypeError, match="image_source"):
        predictor.predict(object())

    class InvalidReadable:
        def read(self):
            return "not bytes"

    with pytest.raises(ValueError, match="not a readable image"):
        predictor.predict(InvalidReadable())


@pytest.mark.parametrize("top_k", [True, 1.5, "3"])
def test_top_k_requires_an_integer(predictor, sample_image_path, top_k):
    with pytest.raises(TypeError, match="top_k must be an integer"):
        predictor.predict(sample_image_path, top_k=top_k)


def test_top_level_predict(sample_image_path):
    assert predict(sample_image_path, device="cpu")["label"] == "River"
