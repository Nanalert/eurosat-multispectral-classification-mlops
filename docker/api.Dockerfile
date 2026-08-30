FROM python:3.10.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DEVICE=cpu

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home-dir /home/app app

COPY requirements/runtime-common.txt requirements/runtime-api.txt /tmp/requirements/
RUN pip install --upgrade pip==26.2.1 && \
    pip install torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r /tmp/requirements/runtime-common.txt \
        -r /tmp/requirements/runtime-api.txt

COPY python_script/inference.py python_script/inference.py
COPY app/__init__.py app/api.py app/
COPY models/production/resnet18-production.pt models/production/resnet18-production.pt
COPY models/production/resnet18-production_metadata.json models/production/resnet18-production_metadata.json

RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
