"""Synthetic, non-business demo data; also usable as an HTTP request fixture."""
import argparse
import json
from datetime import date, timedelta

from smart_canteen_sdk import CanteenEngine


def demo_payload():
    canteen = "demo_canteen"
    dishes = [{"dish_id": f"d{i:02d}", "dish_name": f"示例菜品{i}",
               "category": "主食" if i == 0 else "荤菜", "flavor": "清淡" if i % 2 else "咸鲜"}
              for i in range(16)]
    weather = {"temperature_c": 22, "humidity_pct": 60, "precipitation_mm": 0, "wind_speed_kmh": 5}
    calendar = {"school_break": False, "statutory_holiday": False, "adjusted_workday": False}
    start = date(2026, 8, 1)
    history, contexts = [], []
    for offset in range(40):
        day = (start + timedelta(days=offset)).isoformat()
        contexts.append({"date": day, "canteen_id": canteen, "weather": weather, "calendar": calendar})
        for i, dish in enumerate(dishes):
            if i >= 12 and offset % 8 != i % 8:
                continue
            qty = 35 + i + offset % 4 if i < 12 else 150 + i
            history.append({"date": day, "canteen_id": canteen, "meal": "午餐", "dish_id": dish["dish_id"],
                            "quantity": qty, "unit": "份", "order_count": qty, "revenue": qty * 5})
    return {"request": {"canteen_id": canteen, "prediction_date": (start + timedelta(days=40)).isoformat(),
                        "meal": "午餐", "weather": weather, "calendar": calendar,
                        "opportunities": {"enabled": True, "count": 4}},
            "data": {"snapshot_id": "demo_v1", "canteen_id": canteen, "sales_history": history,
                     "dish_catalog": dishes, "historical_context": contexts, "menu_history": []}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", action="store_true")
    args = parser.parse_args()
    payload = demo_payload()
    if args.request_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        engine = CanteenEngine()
        proposal = engine.recommend(**payload)
        result = engine.finalize(proposal, {"high_frequency_dish_ids": [d.dish_id for d in proposal.high_frequency_menu]})
        print(json.dumps({"model": engine.get_model_info(), "proposal_id": proposal.proposal_id,
                          "summary": result.summary, "audit": result.audit}, ensure_ascii=False, indent=2))
