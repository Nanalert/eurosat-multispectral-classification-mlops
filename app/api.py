"""FastAPI service for EuroSAT RGB predictions."""

import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field

from python_script.inference import load_model

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LOGGER = logging.getLogger("uvicorn.error")


class RankedPrediction(BaseModel):
    label_id: int
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    label_id: int
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_predictions: list[RankedPrediction]
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model: str
    model_version: str
    device: str
    number_of_classes: int
    input_size: list[int]


class VersionResponse(BaseModel):
    model: str
    run_name: str
    model_version: str
    checkpoint_sha256: str
    number_of_classes: int
    input_size: list[int]
    training_finished_at_utc: str | None
    best_epoch: int | None
    best_validation_accuracy: float | None
    test_accuracy: float | None


def create_app(predictor_loader=load_model):
    @asynccontextmanager
    async def lifespan(application):
        application.state.predictor = predictor_loader()
        LOGGER.info(
            "Loaded model=%s device=%s",
            application.state.predictor.model_version,
            application.state.predictor.device,
        )
        yield

    application = FastAPI(
        title="EuroSAT RGB Classification API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def log_request(request: Request, call_next):
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception(
                "Request failed method=%s path=%s",
                request.method,
                request.url.path,
            )
            raise
        duration_ms = (perf_counter() - started) * 1000
        LOGGER.info(
            "Request method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        return response

    @application.get("/", include_in_schema=False)
    async def service_information():
        return {"service": application.title, "docs": "/docs", "health": "/health"}

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request):
        predictor = request.app.state.predictor
        return {
            "status": "ready",
            "model": predictor.metadata["model"]["name"],
            "model_version": predictor.model_version,
            "device": str(predictor.device),
            "number_of_classes": len(predictor.class_names),
            "input_size": list(predictor.image_size),
        }

    @application.get("/version", response_model=VersionResponse)
    async def version(request: Request):
        return request.app.state.predictor.details

    @application.post("/predict", response_model=PredictionResponse)
    async def predict_image(
        request: Request,
        image: UploadFile = File(...),
        top_k: int = Query(default=3, ge=1, le=10),
    ):
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Upload must have an image content type",
            )

        image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
        await image.close()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image is empty",
            )
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Uploaded image exceeds the 10 MB limit",
            )

        try:
            return request.app.state.predictor.predict(image_bytes, top_k)
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error

    return application


app = create_app()
