from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = PROJECT_ROOT / "compose.yaml"
DOCKERFILES = [
    PROJECT_ROOT / "docker" / "api.Dockerfile",
    PROJECT_ROOT / "docker" / "streamlit.Dockerfile",
]


def test_compose_services_are_hardened_and_healthy():
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))

    assert set(compose["services"]) == {"api", "streamlit"}
    for service in compose["services"].values():
        assert service["read_only"] is True
        assert service["init"] is True
        assert service["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in service["security_opt"]
        assert service["environment"]["MODEL_DEVICE"] == "cpu"
        assert service["healthcheck"]["test"][0] == "CMD"
        assert service["ports"][0].startswith("127.0.0.1:")


def test_dockerfiles_use_pinned_base_and_non_root_user():
    for dockerfile in DOCKERFILES:
        content = dockerfile.read_text(encoding="utf-8")

        assert content.startswith("FROM python:3.10.12-slim\n")
        assert "USER app" in content
        assert "HEALTHCHECK" in content
        assert "COPY data" not in content
        assert content.index("USER app") < content.index("CMD [")


def test_docker_context_keeps_only_production_model_artifacts():
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "data\n" in dockerignore
    assert "models/production/*" in dockerignore
    assert "!models/production/resnet18-production.pt" in dockerignore
    assert "!models/production/resnet18-production_metadata.json" in dockerignore
