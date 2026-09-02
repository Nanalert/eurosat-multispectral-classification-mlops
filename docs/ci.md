# Continuous Integration and Delivery

The repository uses two GitHub Actions workflows. `ci.yml` runs Ruff, Pytest
with at least 90% coverage, and DVC pipeline validation. `containers.yml`
builds both service images and publishes them to GitHub Container Registry
after a successful push to `main`.

Both workflows download `resnet18-production.pt` and
`resnet18-production_metadata.json` from the `production-model-v1` GitHub
Release and verify their SHA-256 hashes before use.

The protected `main` branch requires these checks:

- `Quality / Ruff`
- `Tests / Pytest`
- `Data / DVC`
- `Docker / api`
- `Docker / streamlit`

Dependency updates are proposed by Dependabot and must pass the same checks
before they are merged. Container images are published only for successful
pushes to `main`; pull requests build the images without publishing them.
