"""Read-only comparison against real deployed source and both canteens."""
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smart_canteen_sdk import CanteenEngine
from smart_canteen_sdk.adapters import from_legacy_frames

PROJECT = Path(__file__).resolve().parents[2]
WEB = PROJECT / "smart_canteen_web"
pytestmark = pytest.mark.skipif(not (WEB / "backend" / "app").exists(), reason="original project is unavailable")


@pytest.mark.parametrize("dataset", ["a_hupan", "f_jichuyuan"])
@pytest.mark.parametrize("day,meal", [("2026-05-01", "午餐"), ("2026-06-11", "早餐")])
def test_real_deployed_prediction_parity(dataset, day, meal):
    sys.path.insert(0, str(WEB / "backend"))
    from app.model_loader import frozen_policy
    from app.context_builder import build_target_context, build_inference_context
    from app.high_frequency_recommender import HighFrequencyRecommender
    from app.low_frequency_recommender import LowFrequencyRecommender

    folder = WEB / "data" / "datasets" / dataset
    raw = pd.read_csv(folder / "sales.csv")
    library = pd.read_csv(folder / "dish_library.csv")
    date = pd.Timestamp(day)
    raw_dates = pd.to_datetime(dict(year=raw["年"], month=raw["月"], day=raw["日"]))
    raw = raw.loc[raw_dates.lt(date)].copy()
    snapshot = from_legacy_frames(raw, library, canteen_id=dataset, snapshot_id="parity",
                                  allow_legacy_unit_changes=True)
    request = {"canteen_id": dataset, "prediction_date": day, "meal": meal,
               "weather": {"temperature_c": 22, "humidity_pct": 65, "precipitation_mm": 0, "wind_speed_kmh": 10},
               "calendar": {"school_break": False, "statutory_holiday": day == "2026-05-01",
                            "statutory_holiday_name": "劳动节" if day == "2026-05-01" else "非节假日", "adjusted_workday": False},
               "opportunities": {"enabled": True, "count": 12, "refresh_seed": 2}}
    engine = CanteenEngine()
    proposal = engine.recommend(request, snapshot)

    history = frozen_policy.read_sales(StringIO(raw.to_csv(index=False)))
    attributes, lib = frozen_policy.build_library_attributes(folder / "dish_library.csv")
    categories = lib.set_index("_dish_norm")["类别"].fillna("其他").astype(str).str.strip()
    attributes["_library_dish_category"] = attributes["_dish_norm"].map(categories).fillna("其他")
    calendar = {**request["calendar"], "school_break_type": "非寒暑假"}
    target = build_target_context(date, meal, **calendar, **request["weather"])
    canteen = snapshot.canteen_name
    context = build_inference_context(history=history, attributes=attributes,
                                      prediction_date=date, canteen=canteen, meal=meal, target_context=target)
    units = history.sort_values("date").drop_duplicates("_dish_norm", keep="last").set_index("_dish_norm")["菜品售卖单位"].to_dict()
    high = HighFrequencyRecommender().generate(context=context, history=history, target_date=date,
           canteen=canteen, meal=meal, initial_menu_history={}, prior_norms=set(), recent7={}, recent14={}, sale_units=units)
    low = LowFrequencyRecommender().generate(history=history, attributes=attributes,
          recommendation_history=pd.DataFrame(), target_date=date, canteen=canteen, meal=meal,
          target_context=target, high_frequency_dishes=set(high.recommendations["_dish_norm"]), refresh_nonce=2,
          candidate_count=12, category="全部")
    assert proposal.audit["low_frequency_status"] == "ok"
    for actual, expected in [(proposal.high_frequency_menu, high.recommendations), (proposal.low_frequency_opportunities, low)]:
        assert [i.dish_name for i in actual] == expected.get("菜品名称", pd.Series(dtype=str)).tolist()
        if actual:
            for attr, col in [("model_quantity", "pred_qty"), ("reference_orders", "pred_orders"), ("reference_revenue", "pred_rev"), ("score", "score")]:
                np.testing.assert_allclose([getattr(i, attr) for i in actual], expected[col], rtol=1e-12, atol=1e-8)
    selected = {"high_frequency_dish_ids": [i.dish_id for i in proposal.high_frequency_menu]}
    result = engine.finalize(proposal, selected)
    np.testing.assert_allclose([i.model_quantity for i in result.menu], high.recommendations.pred_qty, rtol=1e-12, atol=1e-8)
