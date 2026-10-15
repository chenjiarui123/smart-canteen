#!/usr/bin/env python3
"""Build a UTF-8, cross-platform deployment archive for this project."""

from __future__ import annotations

import argparse
import hashlib
import stat
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIR_NAMES = {
    ".venv",
    "node_modules",
    ".pytest_cache",
    "__pycache__",
}

EXCLUDED_RUNTIME_DIRS = {
    Path("data/backups"),
    Path("data/staging"),
    Path("data/.locks"),
}


def should_include(relative_path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in relative_path.parts):
        return False
    if relative_path.name == ".DS_Store" or relative_path.suffix == ".pyc":
        return False
    return not any(
        relative_path == runtime_dir or runtime_dir in relative_path.parents
        for runtime_dir in EXCLUDED_RUNTIME_DIRS
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_info(archive_name: str, source: Path | None = None) -> zipfile.ZipInfo:
    if source is None:
        now = datetime.now()
        date_time = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        mode = stat.S_IFREG | 0o644
    else:
        modified = datetime.fromtimestamp(source.stat().st_mtime)
        date_time = (
            max(modified.year, 1980),
            modified.month,
            modified.day,
            modified.hour,
            modified.minute,
            modified.second,
        )
        mode = source.stat().st_mode
    info = zipfile.ZipInfo(archive_name, date_time=date_time)
    info.create_system = 3
    info.external_attr = (mode & 0xFFFF) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def build(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and should_include(path.relative_to(PROJECT_ROOT))
    )
    checksums = []
    prefix = PROJECT_ROOT.name

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in files:
            relative = source.relative_to(PROJECT_ROOT)
            archive_name = f"{prefix}/{relative.as_posix()}"
            archive.writestr(zip_info(archive_name, source), source.read_bytes())
            checksums.append(f"{sha256(source)}  ./{relative.as_posix()}")

        for runtime_dir in sorted(EXCLUDED_RUNTIME_DIRS):
            archive_name = f"{prefix}/{runtime_dir.as_posix()}/.gitkeep"
            archive.writestr(zip_info(archive_name), b"")

        manifest = "\n".join(checksums) + "\n"
        archive.writestr(
            zip_info(f"{prefix}/PACKAGE_SHA256SUMS.txt"),
            manifest.encode("utf-8"),
        )

    digest_path = output.with_suffix(output.suffix + ".sha256")
    digest_path.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output ZIP path")
    args = parser.parse_args()
    build(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
