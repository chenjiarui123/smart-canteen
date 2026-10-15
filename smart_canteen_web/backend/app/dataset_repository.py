"""Centralised CSV/JSON access with cross-platform locks and atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from filelock import FileLock

from .model_loader import frozen_policy
from .paths import DATA_ROOT, ensure_runtime_directories


SALES_COLUMNS = [
    "年",
    "月",
    "日",
    "餐次",
    "星期",
    "是否校历寒暑假",
    "校历假期类型",
    "是否法定节假日假期",
    "法定节假日名称",
    "是否调休工作日",
    "气温（℃）",
    "相对湿度（%）",
    "降水量（毫米）",
    "10米风速（公里/小时）",
    "菜品名称",
    "菜品总销量",
    "菜品售卖单位",
    "订单数量",
    "菜品单价",
    "销售额",
    "平均取菜量",
    "食堂名称",
]

RECOMMENDATION_HISTORY_COLUMNS = [
    "request_id",
    "created_at",
    "date",
    "食堂名称",
    "餐次",
    "菜品名称",
    "pred_qty",
    "pred_orders",
    "pred_rev",
    "score",
    "recommendation_source",
    "reason",
]


def normalize_dish_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.translate(
        str.maketrans({"（": "(", "）": ")", "【": "[", "】": "]"})
    )
    return re.sub(r"\s+", " ", text)


def read_csv_utf8(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    last_error: Exception | None = None
    for encoding in ["utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"CSV 文件不是 UTF-8 或 UTF-8-SIG 编码：{path.name}") from last_error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_filename(value: str) -> str:
    """Remove traversal and Windows-invalid filename characters."""

    name = Path(value).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.rstrip(" .")
    return name[:120] or "upload"


class DatasetRepository:
    def __init__(self, data_root: Path = DATA_ROOT):
        ensure_runtime_directories()
        self.data_root = data_root.resolve()
        self.backup_root = self.data_root / "backups"
        self.staging_root = self.data_root / "staging"
        self.lock_root = self.data_root / ".locks"
        for path in [self.backup_root, self.staging_root, self.lock_root]:
            path.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.data_root / "dataset_registry.json"
        self.manifest_path = self.data_root / "manifest.json"
        self.ui_options_path = self.data_root / "ui_options.json"
        self.weather_path = self.data_root / "features" / "weather.csv"
        self.calendar_path = self.data_root / "features" / "calendar.csv"
        self._business_lock = FileLock(
            str((self.lock_root / "business-data.lock").resolve()), timeout=30
        )
        self._base_model_cache: dict[str, tuple[int, pd.DataFrame]] = {}

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def registry(self) -> dict[str, Any]:
        payload = self._read_json(self.registry_path)
        return payload["datasets"]

    def ui_options(self) -> dict[str, Any]:
        return self._read_json(self.ui_options_path)

    def manifest(self) -> dict[str, Any]:
        return self._read_json(self.manifest_path)

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        datasets = self.registry()
        if dataset_id not in datasets:
            raise ValueError("未找到所选逻辑食堂")
        item = dict(datasets[dataset_id])
        base = (self.data_root / item["directory"]).resolve()
        if self.data_root not in base.parents:
            raise ValueError("数据集目录配置不安全")
        item["id"] = dataset_id
        item["base_path"] = base
        return item

    def dataset_path(self, dataset_id: str, filename: str) -> Path:
        if filename not in {
            "sales.csv",
            "runtime_actual_sales.csv",
            "dish_library.csv",
            "recommendation_history.csv",
        }:
            raise ValueError("不允许访问该数据文件")
        return self.dataset(dataset_id)["base_path"] / filename

    @contextmanager
    def write_lock(self) -> Iterator[None]:
        with self._business_lock:
            yield

    def _backup_path(self, path: Path) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder = self.backup_root / path.parent.name
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{path.stem}_{stamp}{path.suffix}"

    def _atomic_write_csv_unlocked(
        self, frame: pd.DataFrame, path: Path, create_backup: bool = True
    ) -> Path | None:
        path.parent.mkdir(parents=True, exist_ok=True)
        backup: Path | None = None
        if create_backup and path.exists():
            backup = self._backup_path(path)
            shutil.copy2(path, backup)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.stem}_",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
            frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
            verified = read_csv_utf8(temp_path)
            if list(verified.columns) != list(frame.columns):
                raise RuntimeError("临时 CSV 字段验证失败")
            os.replace(temp_path, path)
            temp_path = None
            return backup
        except Exception:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            if backup is not None and backup.exists():
                shutil.copy2(backup, path)
            raise

    def _atomic_write_json_unlocked(
        self, payload: dict[str, Any], path: Path, create_backup: bool = True
    ) -> Path | None:
        path.parent.mkdir(parents=True, exist_ok=True)
        backup: Path | None = None
        if create_backup and path.exists():
            backup = self._backup_path(path)
            shutil.copy2(path, backup)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{path.stem}_",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                temp_path = Path(handle.name)
            json.loads(temp_path.read_text(encoding="utf-8"))
            os.replace(temp_path, path)
            temp_path = None
            return backup
        except Exception:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            if backup is not None and backup.exists():
                shutil.copy2(backup, path)
            raise

    def atomic_write_csv(
        self, frame: pd.DataFrame, path: Path, create_backup: bool = True
    ) -> Path | None:
        with self.write_lock():
            return self._atomic_write_csv_unlocked(frame, path, create_backup)

    def atomic_write_json(
        self, payload: dict[str, Any], path: Path, create_backup: bool = True
    ) -> Path | None:
        with self.write_lock():
            return self._atomic_write_json_unlocked(payload, path, create_backup)

    def read_base_sales_for_model(self, dataset_id: str) -> pd.DataFrame:
        path = self.dataset_path(dataset_id, "sales.csv")
        stamp = path.stat().st_mtime_ns
        cached = self._base_model_cache.get(dataset_id)
        if cached is not None and cached[0] == stamp:
            return cached[1].copy()
        frame = frozen_policy.read_sales(path)
        self._base_model_cache[dataset_id] = (stamp, frame)
        return frame.copy()

    def read_runtime_sales_for_model(self, dataset_id: str) -> pd.DataFrame:
        path = self.dataset_path(dataset_id, "runtime_actual_sales.csv")
        raw = read_csv_utf8(path)
        if raw.empty:
            return pd.DataFrame(columns=self.read_base_sales_for_model(dataset_id).columns)
        return frozen_policy.read_sales(path)

    def combined_sales_for_model(self, dataset_id: str) -> pd.DataFrame:
        base = self.read_base_sales_for_model(dataset_id)
        runtime = self.read_runtime_sales_for_model(dataset_id)
        if runtime.empty:
            return base
        base = base.assign(_storage_priority=0)
        runtime = runtime.assign(_storage_priority=1)
        combined = pd.concat([base, runtime], ignore_index=True)
        combined = (
            combined.sort_values("_storage_priority")
            .drop_duplicates(
                ["date", "餐次", "_dish_norm"],
                keep="last",
            )
            .drop(columns="_storage_priority")
            .sort_values(["date", "餐次", "菜品名称"])
            .reset_index(drop=True)
        )
        return combined

    def combined_sales_raw(self, dataset_id: str) -> pd.DataFrame:
        base = read_csv_utf8(self.dataset_path(dataset_id, "sales.csv"))
        runtime = read_csv_utf8(
            self.dataset_path(dataset_id, "runtime_actual_sales.csv")
        )
        if runtime.empty:
            result = base
        else:
            base = base.assign(_storage_priority=0)
            runtime = runtime.assign(_storage_priority=1)
            result = pd.concat([base, runtime], ignore_index=True)
            date_values = pd.to_datetime(
                {
                    "year": result["年"],
                    "month": result["月"],
                    "day": result["日"],
                }
            )
            result["_business_date"] = date_values
            result["_dish_norm"] = result["菜品名称"].map(normalize_dish_name)
            result = (
                result.sort_values("_storage_priority")
                .drop_duplicates(
                    ["_business_date", "餐次", "_dish_norm"], keep="last"
                )
                .drop(
                    columns=["_storage_priority", "_business_date", "_dish_norm"]
                )
            )
        return result[SALES_COLUMNS].reset_index(drop=True)

    def read_dish_library(self, dataset_id: str) -> pd.DataFrame:
        return read_csv_utf8(self.dataset_path(dataset_id, "dish_library.csv"))

    def read_recommendation_history(self, dataset_id: str) -> pd.DataFrame:
        frame = read_csv_utf8(
            self.dataset_path(dataset_id, "recommendation_history.csv")
        )
        if frame.empty:
            return pd.DataFrame(columns=RECOMMENDATION_HISTORY_COLUMNS)
        # Rows created before the opportunity module are frozen high-frequency
        # recommendations.  Fill the new audit column in memory; do not rewrite
        # or mutate existing history files during reads.
        if "recommendation_source" not in frame.columns:
            frame["recommendation_source"] = "high_frequency"
        return frame.reindex(columns=RECOMMENDATION_HISTORY_COLUMNS)

    def read_weather(self) -> pd.DataFrame:
        return read_csv_utf8(self.weather_path)

    def read_calendar(self) -> pd.DataFrame:
        return read_csv_utf8(self.calendar_path)

    def append_runtime_sales(
        self,
        dataset_id: str,
        new_rows: pd.DataFrame,
        source_file: str,
    ) -> tuple[int, int, bool]:
        path = self.dataset_path(dataset_id, "runtime_actual_sales.csv")
        with self.write_lock():
            current = read_csv_utf8(path)
            if current.empty:
                current = pd.DataFrame(columns=SALES_COLUMNS)
            base = read_csv_utf8(self.dataset_path(dataset_id, "sales.csv"))
            all_existing = (
                base.copy()
                if current.empty
                else pd.concat([base, current], ignore_index=True)
            )
            existing_dates = pd.to_datetime(
                {
                    "year": all_existing["年"],
                    "month": all_existing["月"],
                    "day": all_existing["日"],
                }
            )
            existing_keys = set(
                zip(
                    existing_dates.dt.strftime("%Y-%m-%d"),
                    all_existing["餐次"].astype(str),
                    all_existing["菜品名称"].map(normalize_dish_name),
                )
            )
            incoming_dates = pd.to_datetime(
                {
                    "year": new_rows["年"],
                    "month": new_rows["月"],
                    "day": new_rows["日"],
                }
            )
            incoming_keys = list(
                zip(
                    incoming_dates.dt.strftime("%Y-%m-%d"),
                    new_rows["餐次"].astype(str),
                    new_rows["菜品名称"].map(normalize_dish_name),
                )
            )
            keep = [key not in existing_keys for key in incoming_keys]
            ignored = int(len(keep) - sum(keep))
            accepted = new_rows.loc[keep, SALES_COLUMNS].copy()
            if accepted.empty:
                return 0, ignored, False
            updated = (
                accepted.copy()
                if current.empty
                else pd.concat([current, accepted], ignore_index=True)
            )
            updated["_date"] = pd.to_datetime(
                {
                    "year": updated["年"],
                    "month": updated["月"],
                    "day": updated["日"],
                }
            )
            updated["_dish_norm"] = updated["菜品名称"].map(normalize_dish_name)
            updated = (
                updated.drop_duplicates(
                    ["_date", "餐次", "_dish_norm"], keep="first"
                )
                .sort_values(["_date", "餐次", "菜品名称"])
                .drop(columns=["_date", "_dish_norm"])
            )
            backup = self._atomic_write_csv_unlocked(updated, path, True)
            try:
                self._update_manifest_unlocked(
                    dataset_id=dataset_id,
                    kind="runtime_actual_sales",
                    source_file=source_file,
                    added_rows=len(accepted),
                )
            except Exception:
                if backup is not None and backup.exists():
                    shutil.copy2(backup, path)
                raise
            return int(len(accepted)), ignored, backup is not None

    def append_recommendation_history_unlocked(
        self, dataset_id: str, rows: pd.DataFrame
    ) -> Path | None:
        path = self.dataset_path(dataset_id, "recommendation_history.csv")
        current = read_csv_utf8(path)
        if current.empty:
            current = pd.DataFrame(columns=RECOMMENDATION_HISTORY_COLUMNS)
        elif "recommendation_source" not in current.columns:
            current["recommendation_source"] = "high_frequency"
        current = current.reindex(columns=RECOMMENDATION_HISTORY_COLUMNS)
        new_rows = rows[RECOMMENDATION_HISTORY_COLUMNS]
        updated = (
            new_rows.copy()
            if current.empty
            else pd.concat([current, new_rows], ignore_index=True)
        )
        updated = updated.drop_duplicates(
            ["request_id", "date", "餐次", "菜品名称"], keep="first"
        )
        return self._atomic_write_csv_unlocked(updated, path, True)

    def _update_manifest_unlocked(
        self,
        dataset_id: str | None,
        kind: str,
        source_file: str,
        added_rows: int,
    ) -> None:
        payload = self.manifest()
        now = datetime.now().astimezone().isoformat()
        payload["updated_at"] = now
        payload.setdefault("updates", []).append(
            {
                "updated_at": now,
                "dataset_id": dataset_id,
                "kind": kind,
                "source_file": safe_filename(source_file),
                "added_rows": int(added_rows),
            }
        )
        if dataset_id:
            dataset_meta = payload["datasets"][dataset_id]
            dataset_meta["updated_at"] = now
            counter = f"{kind}_added_rows"
            dataset_meta[counter] = int(dataset_meta.get(counter, 0)) + int(
                added_rows
            )
        self._atomic_write_json_unlocked(payload, self.manifest_path, True)

    def commit_feature_rows(
        self, kind: str, new_rows: pd.DataFrame, source_file: str
    ) -> tuple[int, bool]:
        if kind == "weather":
            path = self.weather_path
        elif kind == "calendar":
            path = self.calendar_path
        else:
            raise ValueError("未知公共特征类型")
        with self.write_lock():
            current = read_csv_utf8(path)
            existing_dates = set(current["日期"].astype(str)) if not current.empty else set()
            accepted = new_rows.loc[
                ~new_rows["日期"].astype(str).isin(existing_dates)
            ].copy()
            if accepted.empty:
                return 0, False
            updated = (
                pd.concat([current, accepted], ignore_index=True)
                .drop_duplicates(["日期"], keep="first")
                .sort_values("日期")
            )
            backup = self._atomic_write_csv_unlocked(updated, path, True)
            try:
                self._update_manifest_unlocked(
                    dataset_id=None,
                    kind=kind,
                    source_file=source_file,
                    added_rows=len(accepted),
                )
            except Exception:
                if backup is not None and backup.exists():
                    shutil.copy2(backup, path)
                raise
            return int(len(accepted)), backup is not None

    def stage_paths(self, token: str) -> tuple[Path, Path]:
        if not re.fullmatch(r"[0-9a-f]{32}", token):
            raise ValueError("预览令牌无效")
        return (
            self.staging_root / f"{token}.json",
            self.staging_root / f"{token}.csv",
        )
