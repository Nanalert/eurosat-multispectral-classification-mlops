import io
import subprocess
import sys
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PROJECT_ROOT = APP_PATH.parents[1]


def test_streamlit_app_starts():
    application = AppTest.from_file(APP_PATH, default_timeout=15).run()

    assert not application.exception
    assert application.title[0].value == "EuroSAT RGB Classifier"
    assert application.slider[0].value == 3
    assert application.get("file_uploader")
    assert any(
        "resnet18-production-" in caption.value for caption in application.caption
    )


def test_streamlit_command_starts_from_project_root():
    result = subprocess.run(
        [sys.executable, "app/streamlit_app.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_streamlit_real_image_prediction(sample_image_path, sample_image_bytes):
    application = AppTest.from_file(APP_PATH, default_timeout=30).run()
    uploader = application.get("file_uploader")[0]

    uploader.upload(
        sample_image_path.name,
        sample_image_bytes,
        "image/jpeg",
    ).run()
    application.button[0].click().run(timeout=30)

    assert not application.exception
    assert application.subheader[0].value == "River"
    assert application.metric[0].label == "Confidence"


def test_streamlit_rejects_corrupt_image():
    application = AppTest.from_file(APP_PATH, default_timeout=15).run()

    application.get("file_uploader")[0].upload(
        "broken.png",
        b"not an image",
        "image/png",
    ).run()

    assert not application.exception
    assert application.error[0].value == "The supplied file is not a readable image"
    assert application.button[0].disabled


def test_streamlit_rejects_oversized_image():
    application = AppTest.from_file(APP_PATH, default_timeout=15).run()

    application.get("file_uploader")[0].upload(
        "large.png",
        b"0" * (10 * 1024 * 1024 + 1),
        "image/png",
    ).run()

    assert not application.exception
    assert application.error[0].value == "The uploaded image exceeds the 10 MB limit"
    assert application.button[0].disabled


def test_streamlit_accepts_dark_grayscale_image():
    image_bytes = io.BytesIO()
    Image.new("L", (64, 64), color=1).save(image_bytes, format="TIFF")
    application = AppTest.from_file(APP_PATH, default_timeout=30).run()

    application.get("file_uploader")[0].upload(
        "dark-image.tiff",
        image_bytes.getvalue(),
        "image/tiff",
    ).run()
    application.button[0].click().run(timeout=30)

    assert not application.exception
    assert application.metric[0].label == "Confidence"
