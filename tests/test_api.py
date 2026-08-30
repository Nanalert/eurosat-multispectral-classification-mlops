import asyncio
import logging
from contextlib import asynccontextmanager

import httpx2
import pytest

import app.api as api_module
from app.api import create_app
from python_script.inference import load_model

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


class FakePredictor:
    metadata = {"model": {"name": "ResNet18"}}
    device = "cpu"
    class_names = CLASS_NAMES
    image_size = (64, 64)
    model_version = "resnet18-test-0123456789ab"
    details = {
        "model": "ResNet18",
        "run_name": "resnet18-test",
        "model_version": model_version,
        "checkpoint_sha256": "0" * 64,
        "number_of_classes": 10,
        "input_size": [64, 64],
        "training_finished_at_utc": None,
        "best_epoch": 1,
        "best_validation_accuracy": 0.9,
        "test_accuracy": 0.89,
    }

    def predict(self, image_bytes, top_k=3):
        if image_bytes == b"bad image":
            raise ValueError("The supplied file is not a readable image")
        ranked = [
            {"label_id": 8, "label": "River", "confidence": 0.90},
            {"label_id": 5, "label": "Pasture", "confidence": 0.06},
            {"label_id": 0, "label": "AnnualCrop", "confidence": 0.04},
        ][:top_k]
        probabilities = {name: 0.0 for name in CLASS_NAMES}
        probabilities.update({"River": 0.90, "Pasture": 0.06, "AnnualCrop": 0.04})
        return {
            "label_id": 8,
            "label": "River",
            "confidence": 0.90,
            "top_predictions": ranked,
            "probabilities": probabilities,
        }


class ExplodingPredictor(FakePredictor):
    def predict(self, image_bytes, top_k=3):
        raise RuntimeError("unexpected inference failure")


@asynccontextmanager
async def client_for(application):
    async with application.router.lifespan_context(application):
        transport = httpx2.ASGITransport(app=application)
        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            yield client


@pytest.mark.anyio
async def test_root_and_health():
    application = create_app(predictor_loader=FakePredictor)
    async with client_for(application) as client:
        assert (await client.get("/")).json()["docs"] == "/docs"
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": "ResNet18",
        "model_version": FakePredictor.model_version,
        "device": "cpu",
        "number_of_classes": 10,
        "input_size": [64, 64],
    }


@pytest.mark.anyio
async def test_version_endpoint():
    application = create_app(predictor_loader=FakePredictor)
    async with client_for(application) as client:
        response = await client.get("/version")

    assert response.status_code == 200
    assert response.json() == FakePredictor.details


@pytest.mark.anyio
async def test_prediction_endpoint():
    application = create_app(predictor_loader=FakePredictor)
    async with client_for(application) as client:
        response = await client.post(
            "/predict?top_k=2",
            files={"image": ("river.jpg", b"valid image", "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["label"] == "River"
    assert len(response.json()["top_predictions"]) == 2
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


@pytest.mark.anyio
async def test_concurrent_prediction_requests():
    application = create_app(predictor_loader=FakePredictor)
    async with client_for(application) as client:
        responses = await asyncio.gather(
            *[
                client.post(
                    "/predict",
                    files={
                        "image": (f"river-{index}.jpg", b"valid image", "image/jpeg")
                    },
                )
                for index in range(8)
            ]
        )

    assert all(response.status_code == 200 for response in responses)
    assert {response.json()["label"] for response in responses} == {"River"}


@pytest.mark.anyio
async def test_request_is_logged(caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    application = create_app(predictor_loader=FakePredictor)
    async with client_for(application) as client:
        await client.get("/health")

    assert "Request method=GET path=/health status=200" in caplog.text


@pytest.mark.anyio
async def test_unexpected_error_is_logged_without_leaking_details(caplog):
    caplog.set_level(logging.ERROR, logger="uvicorn.error")
    application = create_app(predictor_loader=ExplodingPredictor)
    async with application.router.lifespan_context(application):
        transport = httpx2.ASGITransport(
            app=application,
            raise_app_exceptions=False,
        )
        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/predict",
                files={"image": ("river.jpg", b"valid image", "image/jpeg")},
            )

    assert response.status_code == 500
    assert "unexpected inference failure" not in response.text
    assert "Request failed method=POST path=/predict" in caplog.text


@pytest.mark.anyio
async def test_rejects_invalid_content_type():
    application = create_app(predictor_loader=FakePredictor)
    async with client_for(application) as client:
        response = await client.post(
            "/predict",
            files={"image": ("notes.txt", b"text", "text/plain")},
        )

    assert response.status_code == 415


@pytest.mark.anyio
async def test_rejects_empty_and_unreadable_images():
    application = create_app(predictor_loader=FakePredictor)
    async with client_for(application) as client:
        empty_response = await client.post(
            "/predict",
            files={"image": ("empty.jpg", b"", "image/jpeg")},
        )
        invalid_response = await client.post(
            "/predict",
            files={"image": ("bad.jpg", b"bad image", "image/jpeg")},
        )

    assert empty_response.status_code == 400
    assert invalid_response.status_code == 400


@pytest.mark.anyio
async def test_rejects_large_upload(monkeypatch):
    monkeypatch.setattr(api_module, "MAX_UPLOAD_BYTES", 4)
    application = create_app(predictor_loader=FakePredictor)
    async with client_for(application) as client:
        response = await client.post(
            "/predict",
            files={"image": ("large.jpg", b"12345", "image/jpeg")},
        )

    assert response.status_code == 413


@pytest.mark.anyio
async def test_validates_top_k():
    application = create_app(predictor_loader=FakePredictor)
    async with client_for(application) as client:
        response = await client.post(
            "/predict?top_k=0",
            files={"image": ("river.jpg", b"valid image", "image/jpeg")},
        )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_real_model_prediction(sample_image_bytes):
    application = create_app(predictor_loader=lambda: load_model(device="cpu"))
    async with client_for(application) as client:
        response = await client.post(
            "/predict",
            files={"image": ("river.jpg", sample_image_bytes, "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["label"] == "River"
