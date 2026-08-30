"""Verify production model artifacts downloaded for CI and deployment."""

import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARTIFACTS = {
    PROJECT_ROOT / "models" / "production" / "resnet18-production.pt": (
        "300e4e9475ff5e3786eb4f419ee6b24671edd0a71089b310fa99bce87deb4d4a"
    ),
    PROJECT_ROOT / "models" / "production" / "resnet18-production_metadata.json": (
        "94a24dc9ab74e3e7d1cd07d1777dd23ddd2d64d3c0804ac42290e6407b4fee40"
    ),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact_file:
        for block in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(expected_artifacts=EXPECTED_ARTIFACTS):
    verified = []
    for path, expected_hash in expected_artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(f"Production artifact not found: {path}")
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Checksum mismatch for {path}: expected {expected_hash}, "
                f"received {actual_hash}"
            )
        verified.append(path)
    return verified


def main():
    for path in verify_artifacts():
        print(f"Verified {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
