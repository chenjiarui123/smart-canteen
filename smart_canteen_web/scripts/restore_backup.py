#!/usr/bin/env python3
"""Restore one CSV/JSON backup using only cross-platform Python APIs."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from filelock import FileLock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (PROJECT_ROOT / "data").resolve()
LOCK_PATH = DATA_ROOT / ".locks" / "business-data.lock"


def inside_data(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == DATA_ROOT or DATA_ROOT in resolved.parents


def restore(backup: Path, target: Path) -> None:
    backup = backup.resolve()
    target = target.resolve()
    if not inside_data(backup) or not inside_data(target):
        raise SystemExit("备份和目标必须位于项目 data 目录内")
    if not backup.is_file():
        raise SystemExit("备份文件不存在")
    if target.name == "sales.csv":
        raise SystemExit("初始化 sales.csv 禁止通过恢复脚本修改")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(LOCK_PATH), timeout=30):
        if target.exists():
            safety = (
                DATA_ROOT
                / "backups"
                / "restore_safety"
                / f"{target.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}{target.suffix}"
            )
            safety.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, safety)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.stem}_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        try:
            shutil.copy2(backup, temp_path)
            os.replace(temp_path, target)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    print(f"已恢复：{target.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()
    restore(
        args.backup if args.backup.is_absolute() else PROJECT_ROOT / args.backup,
        args.target if args.target.is_absolute() else PROJECT_ROOT / args.target,
    )
