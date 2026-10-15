"""Independent low-frequency, high-volume opportunity recommender.

This module intentionally does not import, edit, or wrap the frozen Trial 18
selection function.  It receives leakage-safe history and static attributes,
then builds a separate opportunity list for explicit user review.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset_repository import normalize_dish_name
from .recommendation_copy import low_frequency_reason


CONFIG_PATH = Path(__file__).with_name("low_frequency_config.json")
SELECTABLE_CATEGORIES = {"荤菜", "素菜", "主食", "甜品", "水果"}


def load_low_frequency_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def low_frequency_config_sha256(path: Path = CONFIG_PATH) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def canonical_holiday_name(value: Any) -> str | None:
    """Map compound/source labels to a single comparable festival name."""

    text = str(value or "").strip().lower()
    if text in {"", "nan", "none", "非节假日", "非假期", "无"}:
        return None
    # Order matters for compound labels such as “国庆节、中秋节”.
    aliases = [
        ("元宵", "元宵节"),
        ("端午", "端午节"),
        ("中秋", "中秋节"),
        ("春节", "春节"),
        ("元旦", "元旦"),
        ("清明", "清明节"),
        ("劳动", "劳动节"),
        ("国庆", "国庆节"),
    ]
    for token, canonical in aliases:
        if token in text:
            return canonical
    return str(value).strip()


def associated_holiday(
    dish_name: str, config: dict[str, Any]
) -> str | None:
    normalized = normalize_dish_name(dish_name)
    for holiday, patterns in config["festival_dish_patterns"].items():
        if any(normalize_dish_name(pattern) in normalized for pattern in patterns):
            return str(holiday)
    return None


def _season(month: int) -> str:
    if month in {3, 4, 5}:
        return "春"
    if month in {6, 7, 8}:
        return "夏"
    if month in {9, 10, 11}:
        return "秋"
    return "冬"


def _temperature_band(value: float) -> str:
    if value <= 0:
        return "严寒"
    if value <= 10:
        return "寒冷"
    if value <= 25:
        return "温和"
    return "炎热"


def _rain_band(value: float) -> str:
    if value <= 0:
        return "无雨"
    if value <= 5:
        return "小雨"
    return "明显降水"


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _performance(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "sample_count": 0,
            "avg_qty": 0.0,
            "avg_orders": 0.0,
            "avg_rev": 0.0,
            "qty_stability": 0.0,
            "orders_per_qty": 0.0,
            "rev_per_qty": 0.0,
        }
    daily = (
        frame.groupby("date", as_index=False)
        .agg(
            qty=("菜品总销量", "sum"),
            orders=("订单数量", "sum"),
            rev=("销售额", "sum"),
        )
        .sort_values("date")
    )
    avg_qty = _safe_number(daily["qty"].mean())
    std_qty = _safe_number(daily["qty"].std(ddof=0))
    stability = 1.0 / (1.0 + std_qty / avg_qty) if avg_qty > 0 else 0.0
    total_qty = _safe_number(daily["qty"].sum())
    return {
        "sample_count": int(len(daily)),
        "avg_qty": avg_qty,
        "avg_orders": _safe_number(daily["orders"].mean()),
        "avg_rev": _safe_number(daily["rev"].mean()),
        "qty_stability": stability,
        "orders_per_qty": (
            _safe_number(daily["orders"].sum()) / total_qty
            if total_qty > 0
            else 0.0
        ),
        "rev_per_qty": (
            _safe_number(daily["rev"].sum()) / total_qty
            if total_qty > 0
            else 0.0
        ),
    }


def _shrunk_ratio(
    matched: pd.DataFrame,
    baseline: float,
    tau: float,
    clip: tuple[float, float],
) -> float:
    if matched.empty or baseline <= 0:
        return 1.0
    perf = _performance(matched)
    count = float(perf["sample_count"])
    reliability = count / (count + tau) if count > 0 else 0.0
    raw = perf["avg_qty"] / baseline if baseline > 0 else 1.0
    ratio = 1.0 + (raw - 1.0) * reliability
    return float(np.clip(ratio, clip[0], clip[1]))


def _attribute_lookup(attributes: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "_dish_norm",
        "菜品名称_库",
        "_library_dish_category",
        "菜品类别",
        "主要食材",
        "食材大类",
        "制作方法",
        "口味",
        "相似菜分组",
        "主食标记",
        "inferred_main_ingredient",
        "ingredient_family",
        "attribute_source",
        "inference_confidence",
        "菜品库属性",
        "推断属性",
        "library_matched",
        "_main_hard",
        "_family_hard",
        "_method_hard",
        "_flavor_hard",
        "_similarity_hard",
    ]
    result = attributes.copy()
    if "_library_dish_category" not in result.columns:
        result["_library_dish_category"] = result.get(
            "菜品类别", pd.Series("其他", index=result.index)
        )
    result["_library_dish_category"] = (
        result["_library_dish_category"]
        .fillna("其他")
        .astype(str)
        .str.strip()
        .where(
            lambda values: values.isin(SELECTABLE_CATEGORIES),
            "其他",
        )
    )
    for column in keep:
        if column not in result.columns:
            if column.startswith("_") or column in {"主食标记", "library_matched"}:
                result[column] = False
            else:
                result[column] = "unknown"
    return result[keep].drop_duplicates("_dish_norm", keep="first")


class LowFrequencyRecommender:
    """Build a deterministic semi-random opportunity list."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_low_frequency_config()

    def _formal_selection_counts(
        self,
        recommendation_history: pd.DataFrame,
        target_date: pd.Timestamp,
        canteen: str,
        meal: str,
    ) -> dict[str, int]:
        if recommendation_history.empty:
            return {}
        frame = recommendation_history.copy()
        if "recommendation_source" not in frame.columns:
            return {}
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.loc[
            frame["date"].lt(target_date)
            & frame["食堂名称"].astype(str).eq(str(canteen))
            & frame["餐次"].astype(str).eq(str(meal))
            & frame["recommendation_source"].astype(str).eq(
                "low_frequency_opportunity"
            )
        ].copy()
        if frame.empty:
            return {}
        frame["_dish_norm"] = frame["菜品名称"].map(normalize_dish_name)
        return frame["_dish_norm"].value_counts().astype(int).to_dict()

    def generate(
        self,
        *,
        history: pd.DataFrame,
        attributes: pd.DataFrame,
        recommendation_history: pd.DataFrame,
        target_date: pd.Timestamp,
        canteen: str,
        meal: str,
        target_context: dict[str, Any],
        high_frequency_dishes: set[str],
        refresh_nonce: int = 0,
        category: str = "全部",
        candidate_count: int = 12,
        excluded_dishes: set[str] | None = None,
    ) -> pd.DataFrame:
        if category not in {"全部", *SELECTABLE_CATEGORIES}:
            raise ValueError("潜力菜类别不在允许范围内")
        max_limit = int(self.config["max_candidate_limit"])
        if not isinstance(candidate_count, int) or isinstance(candidate_count, bool):
            raise ValueError("潜力菜推荐数量必须是整数")
        if candidate_count < 1 or candidate_count > max_limit:
            raise ValueError(f"潜力菜推荐数量必须在1至{max_limit}之间")
        target_date = pd.Timestamp(target_date).normalize()
        prior = history.loc[
            history["date"].lt(target_date)
            & history["食堂名称"].astype(str).eq(str(canteen))
            & history["餐次"].astype(str).eq(str(meal))
        ].copy()
        if prior.empty:
            return pd.DataFrame()
        if prior["date"].max() >= target_date:
            raise AssertionError("低频潜力模块读取了目标日期或未来销售数据")
        prior["_dish_norm"] = prior["菜品名称"].map(normalize_dish_name)
        prior["_holiday_canonical"] = prior["法定节假日名称"].map(
            canonical_holiday_name
        )
        flag = pd.to_numeric(
            prior.get("是否法定节假日假期", 0), errors="coerce"
        ).fillna(0).astype(bool)
        prior["_is_holiday"] = flag | prior["_holiday_canonical"].notna()
        prior["_season"] = prior["date"].dt.month.map(_season)
        prior["_temperature_band_lf"] = pd.to_numeric(
            prior["气温（℃）"], errors="coerce"
        ).fillna(0.0).map(_temperature_band)
        prior["_rain_band_lf"] = pd.to_numeric(
            prior["降水量（毫米）"], errors="coerce"
        ).fillna(0.0).map(_rain_band)

        session_count = max(int(prior["date"].nunique()), 1)
        grouped = prior.groupby("_dish_norm", sort=True)
        frequency = grouped["date"].nunique().rename("offer_count").to_frame()
        frequency["offer_rate"] = frequency["offer_count"] / session_count
        frequency["last_date"] = grouped["date"].max()
        frequency["dish_name"] = grouped["菜品名称"].last()
        attr = _attribute_lookup(attributes).set_index("_dish_norm")
        category_by_dish = attr["_library_dish_category"].to_dict()
        frequency["_library_dish_category"] = [
            category_by_dish.get(str(dish), "其他")
            for dish in frequency.index
        ]
        all_excluded = set(high_frequency_dishes) | set(excluded_dishes or set())
        frequency = frequency.loc[
            frequency["offer_count"].ge(int(self.config["min_offer_count"]))
            & (
                frequency["offer_rate"].le(float(self.config["max_offer_rate"]))
                | frequency["offer_count"].le(int(self.config["max_offer_count"]))
            )
            & (
                (target_date - frequency["last_date"]).dt.days
                <= int(self.config["max_inactive_days"])
            )
            & ~frequency.index.astype(str).isin(all_excluded)
        ].copy()
        if category != "全部":
            # Category filtering deliberately happens before all performance
            # calculations and ranking.  The category comes only from the
            # dish library mapping prepared by the service.
            frequency = frequency.loc[
                frequency["_library_dish_category"].eq(category)
            ].copy()
        if frequency.empty:
            return pd.DataFrame()

        eligible_dishes = set(frequency.index.astype(str))
        ordinary_all = prior.loc[
            ~prior["_is_holiday"]
            & prior["_dish_norm"].astype(str).isin(eligible_dishes)
        ]
        ordinary_global = _performance(ordinary_all)
        global_baseline = max(ordinary_global["avg_qty"], 1.0)
        ordinary_by_dish = {
            dish: _performance(group)
            for dish, group in ordinary_all.groupby("_dish_norm", sort=False)
        }
        known_avgs = [
            value["avg_qty"]
            for value in ordinary_by_dish.values()
            if value["sample_count"] > 0 and value["avg_qty"] > 0
        ]
        if known_avgs:
            high_qty_threshold = float(
                np.quantile(known_avgs, float(self.config["high_qty_quantile"]))
            )
            conservative_baseline = float(np.quantile(known_avgs, 0.25))
        else:
            high_qty_threshold = global_baseline
            conservative_baseline = max(global_baseline * 0.5, 1.0)

        target_holiday = canonical_holiday_name(
            target_context.get("法定节假日名称")
        )
        target_season = _season(target_date.month)
        target_weekday = target_date.weekday()
        target_temp_band = _temperature_band(
            _safe_number(target_context.get("气温（℃）"))
        )
        target_rain_band = _rain_band(
            _safe_number(target_context.get("降水量（毫米）"))
        )
        counts = self._formal_selection_counts(
            recommendation_history, target_date, canteen, meal
        )
        rows: list[dict[str, Any]] = []
        context_tau = float(self.config["context_sample_shrinkage_tau"])
        ordinary_clip = tuple(map(float, self.config["ordinary_context_clip"]))
        weather_clip = tuple(map(float, self.config["weather_context_clip"]))
        holiday_tau = float(self.config["holiday_sample_shrinkage_tau"])

        for dish, freq in frequency.iterrows():
            dish_rows = grouped.get_group(dish)
            ordinary_rows = dish_rows.loc[~dish_rows["_is_holiday"]]
            holiday_rows = dish_rows.loc[dish_rows["_is_holiday"]]
            ordinary = ordinary_by_dish.get(dish, _performance(pd.DataFrame()))
            holiday = _performance(holiday_rows)
            dish_name = str(freq["dish_name"])
            dish_holiday = associated_holiday(dish_name, self.config)
            holiday_matches = (
                dish_holiday is not None and target_holiday == dish_holiday
            )
            matching_rows = holiday_rows.loc[
                holiday_rows["_holiday_canonical"].eq(dish_holiday)
            ] if dish_holiday else holiday_rows.iloc[0:0]
            matching = _performance(matching_rows)

            has_ordinary = ordinary["sample_count"] > 0
            baseline_qty = (
                ordinary["avg_qty"] if has_ordinary else conservative_baseline
            )
            stability = (
                ordinary["qty_stability"]
                if has_ordinary
                else float(self.config["cold_start_score_weight"])
            )
            weekday_rows = ordinary_rows.loc[
                ordinary_rows["date"].dt.weekday.eq(target_weekday)
            ]
            season_rows = ordinary_rows.loc[
                ordinary_rows["_season"].eq(target_season)
            ]
            weekday_factor = _shrunk_ratio(
                weekday_rows, baseline_qty, context_tau, ordinary_clip
            )
            season_factor = _shrunk_ratio(
                season_rows, baseline_qty, context_tau, ordinary_clip
            )
            weather_rows = ordinary_rows.loc[
                ordinary_rows["_temperature_band_lf"].eq(target_temp_band)
                & ordinary_rows["_rain_band_lf"].eq(target_rain_band)
            ]
            weather_factor = _shrunk_ratio(
                weather_rows, baseline_qty, context_tau, weather_clip
            )
            context_match_score = float(
                np.clip(
                    weekday_factor * season_factor * weather_factor,
                    ordinary_clip[0],
                    ordinary_clip[1],
                )
            )
            selected_count = int(counts.get(str(dish), 0))
            selection_penalty = 1.0 / (
                1.0
                + float(self.config["selection_penalty_rate"])
                * selected_count
            )
            ordinary_score = (
                baseline_qty
                * stability
                * context_match_score
                * selection_penalty
            )
            holiday_gain = 0.0
            holiday_score_used = False
            blended_qty = baseline_qty * context_match_score
            orders_ratio = ordinary["orders_per_qty"] if has_ordinary else 0.0
            rev_ratio = ordinary["rev_per_qty"] if has_ordinary else 0.0
            if holiday_matches and matching["sample_count"] > 0:
                reliability = matching["sample_count"] / (
                    matching["sample_count"] + holiday_tau
                )
                shrunk_matching_qty = (
                    baseline_qty
                    + (matching["avg_qty"] - baseline_qty) * reliability
                )
                holiday_gain = max(
                    0.0,
                    shrunk_matching_qty - baseline_qty,
                ) * matching["qty_stability"] * selection_penalty
                blended_qty = baseline_qty + (
                    matching["avg_qty"] - baseline_qty
                ) * reliability
                orders_ratio = (
                    (1.0 - reliability) * orders_ratio
                    + reliability * matching["orders_per_qty"]
                )
                rev_ratio = (
                    (1.0 - reliability) * rev_ratio
                    + reliability * matching["rev_per_qty"]
                )
                holiday_score_used = True
            score = ordinary_score + holiday_gain

            festival_low_sample = (
                dish_holiday is not None
                and ordinary["sample_count"]
                < int(self.config["ordinary_min_samples"])
            )
            qualifies = (
                has_ordinary and ordinary["avg_qty"] >= high_qty_threshold
            ) or festival_low_sample or (holiday_score_used and score > 0)
            if not qualifies or score <= 0:
                continue

            if not has_ordinary:
                trial_fraction = float(self.config["cold_start_trial_fraction"])
            elif holiday_score_used:
                trial_fraction = float(self.config["holiday_trial_fraction"])
            else:
                trial_fraction = float(self.config["ordinary_trial_fraction"])
            suggested_qty = float(
                np.clip(
                    blended_qty * context_match_score * trial_fraction,
                    float(self.config["trial_qty_floor"]),
                    float(self.config["trial_qty_cap"]),
                )
            )
            reason = low_frequency_reason(
                offer_count=int(freq["offer_count"]),
                ordinary_count=int(ordinary["sample_count"]),
                dish_holiday=dish_holiday,
                holiday_matches=holiday_matches,
                holiday_score_used=holiday_score_used,
                matching_count=int(matching["sample_count"]),
                low_sample=ordinary["sample_count"] < int(self.config["ordinary_min_samples"]),
                selected_count=selected_count,
            )
            attribute = attr.loc[dish].to_dict() if dish in attr.index else {}
            library_category = str(
                attribute.get("_library_dish_category", "其他")
            )
            rows.append(
                {
                    "date": target_date,
                    "食堂名称": str(canteen),
                    "餐次": str(meal),
                    "菜品名称": dish_name,
                    "_dish_norm": str(dish),
                    "pred_qty": suggested_qty,
                    "suggested_trial_qty": suggested_qty,
                    "pred_orders": suggested_qty * orders_ratio,
                    "pred_rev": suggested_qty * rev_ratio,
                    "expected_orders_per_qty": orders_ratio,
                    "expected_rev_per_qty": rev_ratio,
                    "score": float(score),
                    "ordinary_performance": float(
                        baseline_qty * stability
                    ),
                    "holiday_performance": float(
                        holiday["avg_qty"] * holiday["qty_stability"]
                    ),
                    "matching_holiday_performance": float(
                        matching["avg_qty"] * matching["qty_stability"]
                    ),
                    "ordinary_sample_count": int(ordinary["sample_count"]),
                    "ordinary_avg_qty": float(ordinary["avg_qty"]),
                    "matching_holiday_sample_count": int(
                        matching["sample_count"]
                    ),
                    "matching_holiday_avg_qty": float(matching["avg_qty"]),
                    "holiday_score_used": bool(holiday_score_used),
                    "matched_holiday_name": (
                        dish_holiday if holiday_matches else None
                    ),
                    "associated_holiday_name": dish_holiday,
                    "context_match_score": context_match_score,
                    "selection_penalty": float(selection_penalty),
                    "formal_selection_count": selected_count,
                    "recommendation_source": "low_frequency_opportunity",
                    "recommendation_reason": reason,
                    "reason": reason,
                    "菜品售卖单位": str(
                        dish_rows.sort_values("date")["菜品售卖单位"].iloc[-1]
                    ),
                    **attribute,
                    # This display/filter category is always the direct dish
                    # library value normalized into the supported UI groups.
                    "菜品类别": library_category,
                }
            )
        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        result["_base_rank"] = result["score"].rank(
            method="first", ascending=False
        )
        # Deterministic pseudo-randomness: refresh_nonce changes only the
        # ordering inside the already qualified pool and is reproducible on all OSes.
        def jitter(row: pd.Series) -> float:
            seed = "|".join(
                [
                    str(canteen),
                    str(meal),
                    target_date.strftime("%Y-%m-%d"),
                    str(int(refresh_nonce)),
                    str(row["_dish_norm"]),
                ]
            )
            number = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)
            unit = number / float(0xFFFFFFFFFFFFFFFF)
            amplitude = float(self.config["semirandom_jitter"])
            return 1.0 - amplitude + 2.0 * amplitude * unit

        result["_refresh_factor"] = result.apply(jitter, axis=1)
        anchor_count = min(int(self.config["anchor_count"]), len(result))
        ordered = result.sort_values(
            ["score", "菜品名称"], ascending=[False, True]
        )
        anchors = ordered.head(anchor_count)
        remainder = ordered.iloc[anchor_count:].copy()
        remainder["_refresh_score"] = (
            remainder["score"] * remainder["_refresh_factor"]
        )
        remainder = remainder.sort_values(
            ["_refresh_score", "菜品名称"], ascending=[False, True]
        )
        result = pd.concat([anchors, remainder], ignore_index=True).head(
            candidate_count
        )
        return result.drop(
            columns=["_base_rank", "_refresh_factor", "_refresh_score"],
            errors="ignore",
        ).reset_index(drop=True)
