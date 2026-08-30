from pathlib import Path

import pytest

from python_script.inference import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "River_100.jpg"


@pytest.fixture(scope="session")
def sample_image_path():
    return SAMPLE_IMAGE


@pytest.fixture(scope="session")
def sample_image_bytes(sample_image_path):
    return sample_image_path.read_bytes()


@pytest.fixture(scope="session")
def predictor():
    return load_model(device="cpu")


@pytest.fixture
def anyio_backend():
    return "asyncio"
