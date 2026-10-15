from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from filelock import FileLock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "smart_canteen_web" / "data"

RESTORES = {
    DATA_ROOT / "datasets" / "a_hupan" / "runtime_actual_sales.csv": (
        DATA_ROOT
        / "backups"
        / "a_hupan"
        / "runtime_actual_sales_20260824_205808_766201.csv"
    ),
    DATA_ROOT / "features" / "weather.csv": (
        DATA_ROOT
        / "backups"
        / "features"
        / "weather_20260824_205747_717005.csv"
    ),
    DATA_ROOT / "features" / "calendar.csv": (
        DATA_ROOT
        / "backups"
        / "features"
        / "calendar_20260824_205731_068313.csv"
    ),
    DATA_ROOT / "manifest.json": (
        DATA_ROOT
        / "backups"
        / "data"
        / "manifest_20260824_205731_076416.json"
    ),
}

UNTOUCHED = [
    DATA_ROOT / "datasets" / "a_hupan" / "sales.csv",
    DATA_ROOT / "datasets" / "a_hupan" / "dish_library.csv",
    DATA_ROOT / "datasets" / "a_hupan" / "recommendation_history.csv",
    DATA_ROOT / "datasets" / "f_jichuyuan" / "sales.csv",
    DATA_ROOT / "datasets" / "f_jichuyuan" / "runtime_actual_sales.csv",
    DATA_ROOT / "datasets" / "f_jichuyuan" / "dish_library.csv",
    DATA_ROOT / "datasets" / "f_jichuyuan" / "recommendation_history.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_restore(source: Path, target: Path) -> None:
    with source.open("rb") as input_handle:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.stem}_rollback_",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as output_handle:
            temp_path = Path(output_handle.name)
            shutil.copyfileobj(input_handle, output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
    try:
        os.replace(temp_path, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def csv_summary(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    result: dict[str, object] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": sha256(path),
    }
    if "日期" in frame.columns and len(frame):
        dates = pd.to_datetime(frame["日期"], errors="raise")
        result["date_min"] = dates.min().strftime("%Y-%m-%d")
        result["date_max"] = dates.max().strftime("%Y-%m-%d")
    if {"年", "月", "日"}.issubset(frame.columns) and len(frame):
        dates = pd.to_datetime(
            {"year": frame["年"], "month": frame["月"], "day": frame["日"]},
            errors="raise",
        )
        result["date_min"] = dates.min().strftime("%Y-%m-%d")
        result["date_max"] = dates.max().strftime("%Y-%m-%d")
    return result


def main() -> None:
    for target, source in RESTORES.items():
        if not target.exists() or not source.exists():
            raise FileNotFoundError(f"缺少回滚文件：{target} 或 {source}")

    untouched_before = {str(path): sha256(path) for path in UNTOUCHED}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snapshot_root = DATA_ROOT / "backups" / f"manual_rollback_{stamp}"
    snapshot_root.mkdir(parents=True, exist_ok=False)

    lock = FileLock(str(DATA_ROOT / ".locks" / "business-data.lock"), timeout=30)
    with lock:
        for target in RESTORES:
            relative = target.relative_to(DATA_ROOT)
            snapshot_path = snapshot_root / relative
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, snapshot_path)

        for target, source in RESTORES.items():
            atomic_restore(source, target)

    untouched_after = {str(path): sha256(path) for path in UNTOUCHED}
    if untouched_before != untouched_after:
        raise RuntimeError("不在回滚范围内的原始文件发生变化")

    for target, source in RESTORES.items():
        if sha256(target) != sha256(source):
            raise RuntimeError(f"恢复后文件与更新前备份不一致：{target}")

    report = {
        "status": "success",
        "snapshot_root": str(snapshot_root),
        "restored": {
            "a_hupan_runtime_sales": csv_summary(
                DATA_ROOT / "datasets" / "a_hupan" / "runtime_actual_sales.csv"
            ),
            "weather": csv_summary(DATA_ROOT / "features" / "weather.csv"),
            "calendar": csv_summary(DATA_ROOT / "features" / "calendar.csv"),
        },
        "manifest_sha256": sha256(DATA_ROOT / "manifest.json"),
        "manifest_updates": json.loads(
            (DATA_ROOT / "manifest.json").read_text(encoding="utf-8-sig")
        ).get("updates", []),
        "untouched_files_verified": len(UNTOUCHED),
    }
    (snapshot_root / "rollback_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
