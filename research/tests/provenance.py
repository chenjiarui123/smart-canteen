from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_registered_snapshot(
    manifest_path: Path, snapshot_root: Path
) -> dict[str, Any]:
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in record["protected"].items():
        path = snapshot_root / name
        assert path.is_file(), f"missing frozen snapshot: {name}"
        assert sha256_file(path) == expected, f"frozen snapshot mismatch: {name}"
    return record
