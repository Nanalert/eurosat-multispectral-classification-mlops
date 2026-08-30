import pytest

from python_script.verify_production_artifacts import file_sha256, verify_artifacts


def test_verify_production_artifacts():
    verified = verify_artifacts()

    assert len(verified) == 2


def test_missing_and_modified_artifacts_are_rejected(tmp_path):
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="Production artifact not found"):
        verify_artifacts({missing: "0" * 64})

    modified = tmp_path / "model.pt"
    modified.write_bytes(b"wrong model")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_artifacts({modified: "0" * 64})
    assert file_sha256(modified) != "0" * 64
