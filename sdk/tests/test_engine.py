from copy import deepcopy
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from smart_canteen_sdk import CanteenEngine, CanteenError, DataSnapshot, Proposal, RecommendationRequest
from smart_canteen_sdk.data import menu_context


def selection(proposal):
    return {"high_frequency_dish_ids": [i.dish_id for i in proposal.high_frequency_menu]}


def test_roundtrip_and_no_files(engine, payload, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = deepcopy(payload)
    proposal = engine.recommend(**payload)
    assert payload == original
    assert proposal.high_frequency_menu
    assert "state" not in proposal.public_dict()
    restored = Proposal.model_validate_json(proposal.model_dump_json())
    result = CanteenEngine().finalize(restored, selection(proposal))
    assert result.audit["constraint_violation_count"] == 0
    assert result.summary["quantities_by_unit"] == pytest.approx(proposal.summary["quantities_by_unit"])
    assert [i.model_dump() for i in result.menu] == [i.model_dump() for i in proposal.high_frequency_menu]
    assert list(tmp_path.iterdir()) == []
    assert result.history_events[0].source == "finalized"
    assert proposal.history_events[0].source == "automatic"


def test_excludes_current_and_future_outcomes(engine, payload):
    baseline = engine.recommend(**payload)
    for day in [payload["request"]["prediction_date"], "2099-01-01"]:
        row = deepcopy(payload["data"]["sales_history"][0])
        row.update(date=day, quantity=1e12, revenue=1e15)
        payload["data"]["sales_history"].append(row)
        context = deepcopy(payload["data"]["historical_context"][0]); context["date"] = day
        payload["data"]["historical_context"].append(context)
    result = engine.recommend(**payload)
    assert result.high_frequency_menu == baseline.high_frequency_menu
    assert result.low_frequency_opportunities == baseline.low_frequency_opportunities
    assert result.audit["excluded_nonpast_sales_rows"] == 2


@pytest.mark.parametrize("kind,code", [("canteen", "CANTEEN_MISMATCH"), ("duplicate", "DUPLICATE_SALE"),
                                     ("unit", "UNIT_CHANGED"), ("context", "MISSING_CONTEXT"),
                                     ("dish", "UNKNOWN_DISH")])
def test_invalid_data(engine, payload, kind, code):
    data = payload["data"]
    if kind == "canteen": data["sales_history"][0]["canteen_id"] = "other"
    if kind == "duplicate": data["sales_history"].append(deepcopy(data["sales_history"][0]))
    if kind == "unit": data["sales_history"][0]["unit"] = "克"
    if kind == "context": data["historical_context"].pop(0)
    if kind == "dish": data["sales_history"][0]["dish_id"] = "missing"
    with pytest.raises(CanteenError) as error:
        engine.recommend(**payload)
    assert error.value.code == code


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_bad_quantities(engine, payload, value):
    payload["data"]["sales_history"][0]["quantity"] = value
    with pytest.raises(ValidationError): engine.recommend(**payload)


def test_grams_and_per_unit_redistribution(engine, payload):
    for row in payload["data"]["sales_history"]:
        if int(row["dish_id"][1:]) % 2:
            row["unit"] = "克"; row["quantity"] *= 50
    proposal = engine.recommend(**payload)
    gram_items = [i for i in proposal.high_frequency_menu if i.unit == "克"]
    assert gram_items
    assert all(i.quantity == i.model_quantity * 50 for i in gram_items)
    selected = selection(proposal)
    selected["high_frequency_dish_ids"].remove(gram_items[0].dish_id)
    selected["redistribute_to_total_demand"] = True
    result = engine.finalize(proposal, selected)
    assert result.summary["quantities_by_unit"] == pytest.approx(proposal.summary["quantities_by_unit"])
    assert result.audit["redistributed_to_total_demand"]
    selected["high_frequency_dish_ids"] = [i.dish_id for i in proposal.high_frequency_menu if i.unit != "克"]
    with pytest.raises(CanteenError, match="某单位"):
        engine.finalize(proposal, selected)


def test_low_frequency_and_disabled(engine, payload):
    proposal = engine.recommend(**payload)
    assert proposal.low_frequency_opportunities
    assert all(i.expected_demand_qty is None and i.suggested_trial_qty is not None for i in proposal.low_frequency_opportunities)
    chosen = selection(proposal)
    chosen["low_frequency_dish_ids"] = [proposal.low_frequency_opportunities[0].dish_id]
    result = engine.finalize(proposal, chosen)
    assert len(result.menu) == len(proposal.high_frequency_menu) + 1
    payload["request"]["opportunities"]["enabled"] = False
    disabled = engine.recommend(**payload)
    assert disabled.high_frequency_menu == proposal.high_frequency_menu
    assert disabled.low_frequency_opportunities == []


def test_invalid_selections_and_corruption(engine, payload):
    proposal = engine.recommend(**payload)
    first = proposal.high_frequency_menu[0].dish_id
    for selected in [[], ["unknown"], [first, first]]:
        with pytest.raises(CanteenError):
            engine.finalize(proposal, {"high_frequency_dish_ids": selected})
    changed = Proposal.model_validate_json(proposal.model_dump_json())
    changed.state["summary"]["pred_total_qty"] += 100
    with pytest.raises(CanteenError) as error:
        engine.finalize(changed, selection(proposal))
    assert error.value.code == "PROPOSAL_CORRUPTED"


def test_hard_similarity_conflict(engine, payload):
    # Select a known low/high pair, then constrain their shared static group.
    proposal = engine.recommend(**payload)
    high_id = proposal.high_frequency_menu[0].dish_id
    low_id = proposal.low_frequency_opportunities[0].dish_id
    for d in payload["data"]["dish_catalog"]:
        if d["dish_id"] in {high_id, low_id}: d["similarity_group"] = "same-group"
    proposal = engine.recommend(**payload)
    if high_id not in {i.dish_id for i in proposal.high_frequency_menu}:
        pytest.fail("fixture must retain high frequency dish")
    with pytest.raises(CanteenError, match="硬冲突"):
        engine.finalize(proposal, {"high_frequency_dish_ids": [high_id], "low_frequency_dish_ids": [low_id]})


def test_menu_version_and_automatic_fallback(payload):
    data = payload["data"]
    day = data["historical_context"][-1]["date"]
    base = {"date": day, "canteen_id": data["canteen_id"], "meal": "午餐", "source": "automatic",
            "menu_id": "auto", "created_at": "2026-09-09T08:00:00+08:00", "dish_ids": ["d00"]}
    data["menu_history"] = [base]
    req = RecommendationRequest.model_validate(payload["request"])
    context = menu_context(DataSnapshot.model_validate(data), req)
    assert context[-1] == "automatic_high_frequency"
    data["menu_history"].extend([
        {**base, "source": "finalized", "menu_id": "old", "created_at": "2026-09-09T09:00:00+08:00", "dish_ids": ["d01"]},
        {**base, "source": "finalized", "menu_id": "new", "created_at": "2026-09-09T10:00:00+08:00", "dish_ids": ["d02"]},
    ])
    context = menu_context(DataSnapshot.model_validate(data), req)
    assert context[-1] == "confirmed"
    assert context[-2] == {"示例菜品2"}
