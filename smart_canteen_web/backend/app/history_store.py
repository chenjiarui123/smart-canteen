from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset_repository import (
    RECOMMENDATION_HISTORY_COLUMNS,
    DatasetRepository,
    normalize_dish_name,
    read_csv_utf8,
)
from .paths import RECOMMENDATION_ROOT


MENU_EXPORT_COLUMNS = [
    "日期",
    "食堂名称",
    "餐次",
    "菜品id",
    "菜品名",
    "菜品类别",
    "菜品销量",
    "菜品销售额",
]


class HistoryStore:
    def __init__(
        self,
        repository: DatasetRepository,
        recommendation_root: Path = RECOMMENDATION_ROOT,
    ):
        self.repository = repository
        self.recommendation_root = recommendation_root
        self.index_path = recommendation_root / "index.json"

    def initial_menu_history(
        self,
        dataset_id: str,
        canteen: str,
        meal: str,
        prediction_date: pd.Timestamp,
    ) -> dict[tuple[str, str], list[tuple[pd.Timestamp, set[str]]]]:
        frame = self.repository.read_recommendation_history(dataset_id)
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.loc[
            frame["date"].lt(pd.Timestamp(prediction_date))
            & frame["餐次"].astype(str).eq(str(meal))
            & frame["食堂名称"].astype(str).eq(str(canteen))
        ]
        if frame.empty:
            return {}
        values: list[tuple[pd.Timestamp, set[str]]] = []
        for old_date, group in frame.groupby("date", sort=True):
            # A revised confirmation replaces the comparison menu; do not
            # union different confirmed versions from the same day.
            if "created_at" in group and "request_id" in group:
                latest = group.sort_values("created_at", kind="stable").iloc[-1]["request_id"]
                group = group.loc[group["request_id"].eq(latest)]
            values.append(
                (
                    pd.Timestamp(old_date),
                    set(group["菜品名称"].map(normalize_dish_name)),
                )
            )
        return {(str(canteen), str(meal)): values}

    def previous_day_reference(self, dataset_id, canteen, meal, prediction_date):
        target = pd.Timestamp(prediction_date).normalize() - pd.Timedelta(days=1)
        history = self.initial_menu_history(dataset_id, canteen, meal, prediction_date)
        for old_date, dishes in history.get((str(canteen), str(meal)), []):
            if old_date.normalize() == target and dishes:
                return dishes, "confirmed"
        path = self._automatic_path(dataset_id, meal, target)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["canteen"] == canteen and payload["dishes"]:
                return set(payload["dishes"]), "saved_high_frequency"
        return None, "missing_previous"

    def _automatic_path(self, dataset_id, meal, target):
        self.repository.dataset(dataset_id)
        if meal not in {"早餐", "午餐", "晚餐"}:
            raise ValueError("餐次无效")
        return self.recommendation_root / "automatic_high_frequency" / dataset_id / meal / (pd.Timestamp(target).strftime("%Y-%m-%d") + ".json")

    def save_automatic_high_frequency(self, dataset_id, canteen, meal, target, recommendations):
        path = self._automatic_path(dataset_id, meal, target)
        payload = {"canteen": canteen, "date": pd.Timestamp(target).strftime("%Y-%m-%d"),
                   "meal": meal, "dishes": sorted(set(recommendations["菜品名称"].map(normalize_dish_name))),
                   "created_at": datetime.now().astimezone().isoformat()}
        with self.repository.write_lock():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous.get("canteen") == canteen and previous.get("dishes") == payload["dishes"]:
                    return
            self.repository._atomic_write_json_unlocked(payload, path, True)

    def repeat_counts(
        self,
        initial_history: dict[
            tuple[str, str], list[tuple[pd.Timestamp, set[str]]]
        ],
        canteen: str,
        meal: str,
        prediction_date: pd.Timestamp,
    ) -> tuple[set[str], dict[str, int], dict[str, int]]:
        history = initial_history.get((str(canteen), str(meal)), [])
        prior = history[-1][1] if history else set()
        recent7: dict[str, int] = {}
        recent14: dict[str, int] = {}
        target = pd.Timestamp(prediction_date)
        for old_date, dishes in history:
            for dish in dishes:
                if old_date >= target - pd.Timedelta(days=7):
                    recent7[dish] = recent7.get(dish, 0) + 1
                if old_date >= target - pd.Timedelta(days=14):
                    recent14[dish] = recent14.get(dish, 0) + 1
        return prior, recent7, recent14

    def _next_paths(
        self, dataset_id: str, meal: str, prediction_date: pd.Timestamp
    ) -> tuple[Path, Path]:
        folder = self.recommendation_root / dataset_id / meal
        folder.mkdir(parents=True, exist_ok=True)
        stem = pd.Timestamp(prediction_date).strftime("%Y-%m-%d")
        version = 1
        while True:
            suffix = "" if version == 1 else f"_v{version}"
            csv_path = folder / f"{stem}{suffix}.csv"
            md_path = folder / f"{stem}{suffix}.md"
            if not csv_path.exists() and not md_path.exists():
                return csv_path, md_path
            version += 1

    @staticmethod
    def _atomic_text_unlocked(text: str, path: Path) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.stem}_",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(text)
                temp_path = Path(handle.name)
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"recommendations": {}}
        return json.loads(self.index_path.read_text(encoding="utf-8-sig"))

    def _dish_id_lookup(self, dataset_id: str) -> dict[str, str]:
        library = read_csv_utf8(
            self.repository.dataset_path(dataset_id, "dish_library.csv")
        )
        id_column = next(
            (
                column
                for column in ("菜品id", "菜品ID", "菜品编号", "id")
                if column in library.columns
            ),
            None,
        )
        if id_column is None or "菜品名称" not in library.columns:
            return {}

        def clean_id(value: Any) -> str:
            if pd.isna(value):
                return ""
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value).strip()

        lookup = library[["菜品名称", id_column]].copy()
        lookup["_dish_norm"] = lookup["菜品名称"].map(normalize_dish_name)
        lookup["_dish_id"] = lookup[id_column].map(clean_id)
        return (
            lookup.drop_duplicates("_dish_norm", keep="first")
            .set_index("_dish_norm")["_dish_id"]
            .to_dict()
        )

    def _menu_export(
        self, dataset_id: str, recommendations: pd.DataFrame
    ) -> pd.DataFrame:
        """Build the compact user download without altering model quantities."""

        required = {
            "date",
            "食堂名称",
            "餐次",
            "菜品名称",
            "菜品类别",
            "pred_qty",
            "pred_rev",
        }
        missing = sorted(required - set(recommendations.columns))
        if missing:
            raise ValueError(f"推荐菜单缺少导出字段：{','.join(missing)}")
        source = recommendations.copy()
        units = source.get(
            "菜品售卖单位", pd.Series("份", index=source.index)
        ).fillna("份").astype(str).str.strip()
        quantities = pd.to_numeric(source["pred_qty"], errors="raise")
        # The model quantity for gram-priced dishes counts 50 g portions.  The
        # downloaded business value is total grams, so multiply by 50.  Other
        # sale units already represent their final displayed quantity.
        export_quantities = quantities.where(~units.eq("克"), quantities * 50)
        dish_ids = source["菜品名称"].map(normalize_dish_name).map(
            self._dish_id_lookup(dataset_id)
        ).fillna("")
        export = pd.DataFrame(
            {
                "日期": pd.to_datetime(source["date"]).dt.strftime("%Y-%m-%d"),
                "食堂名称": source["食堂名称"].astype(str),
                "餐次": source["餐次"].astype(str),
                "菜品id": dish_ids.astype(str),
                "菜品名": source["菜品名称"].astype(str),
                "菜品类别": source["菜品类别"].fillna("其他").astype(str),
                "菜品销量": export_quantities,
                "菜品销售额": pd.to_numeric(
                    source["pred_rev"], errors="raise"
                ),
            }
        )
        return export[MENU_EXPORT_COLUMNS]

    def save(
        self,
        *,
        dataset_id: str,
        canteen: str,
        meal: str,
        prediction_date: pd.Timestamp,
        recommendations: pd.DataFrame,
        markdown: str,
    ) -> tuple[str, Path, Path]:
        request_id = uuid.uuid4().hex
        created_at = datetime.now().astimezone().isoformat()
        history_rows = recommendations[
            [
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
        ].copy()
        history_rows.insert(0, "created_at", created_at)
        history_rows.insert(0, "request_id", request_id)
        history_rows["date"] = pd.to_datetime(history_rows["date"]).dt.strftime(
            "%Y-%m-%d"
        )
        history_rows = history_rows[RECOMMENDATION_HISTORY_COLUMNS]

        with self.repository.write_lock():
            csv_path, md_path = self._next_paths(
                dataset_id, meal, prediction_date
            )
            index = self._read_index()
            created: list[Path] = []
            history_backup: Path | None = None
            try:
                export = self._menu_export(dataset_id, recommendations)
                self.repository._atomic_write_csv_unlocked(
                    export, csv_path, create_backup=False
                )
                created.append(csv_path)
                self._atomic_text_unlocked(markdown, md_path)
                created.append(md_path)
                history_backup = self.repository.append_recommendation_history_unlocked(
                    dataset_id, history_rows
                )
                index.setdefault("recommendations", {})[request_id] = {
                    "dataset_id": dataset_id,
                    "canteen": canteen,
                    "meal": meal,
                    "date": pd.Timestamp(prediction_date).strftime("%Y-%m-%d"),
                    "csv": str(csv_path.relative_to(self.recommendation_root)),
                    "markdown": str(md_path.relative_to(self.recommendation_root)),
                    "created_at": created_at,
                }
                self.repository._atomic_write_json_unlocked(
                    index, self.index_path, True
                )
            except Exception:
                history_path = self.repository.dataset_path(
                    dataset_id, "recommendation_history.csv"
                )
                if history_backup is not None and history_backup.exists():
                    import shutil
                    shutil.copy2(history_backup, history_path)
                for path in created:
                    if path.exists():
                        path.unlink()
                raise
        return request_id, csv_path, md_path

    def resolve_download(self, recommendation_id: str, kind: str) -> Path:
        if not (
            len(recommendation_id) == 32
            and all(char in "0123456789abcdef" for char in recommendation_id)
        ):
            raise ValueError("推荐编号无效")
        index = self._read_index()
        item = index.get("recommendations", {}).get(recommendation_id)
        if item is None:
            raise ValueError("未找到推荐文件")
        key = "csv" if kind == "csv" else "markdown"
        path = (self.recommendation_root / item[key]).resolve()
        if self.recommendation_root.resolve() not in path.parents:
            raise ValueError("推荐文件路径不安全")
        if not path.exists():
            raise ValueError("推荐文件不存在")
        return path
