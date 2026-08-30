# Docker usage

Build and start both services:

```bash
docker compose up --build -d
```

Open:

- Streamlit: <http://127.0.0.1:8501>
- FastAPI docs: <http://127.0.0.1:8000/docs>
- FastAPI health: <http://127.0.0.1:8000/health>

Change host ports without editing Compose:

```bash
API_PORT=8020 STREAMLIT_PORT=8520 docker compose up -d
```

Inspect and stop the services:

```bash
docker compose ps
docker compose logs --tail=100
docker compose down
```

The images run inference on CPU, include only the production checkpoint, and run
as a non-root user with read-only root filesystems. Rebuild the images whenever
the inference code, application code, dependencies, or production model changes.
