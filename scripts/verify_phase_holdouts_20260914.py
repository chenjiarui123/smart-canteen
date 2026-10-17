"""Independent holdout rescoring and incident analysis; no training or selection."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import policy
from diagnostics import hit_quantity_diagnostics
from targeted_search import date_type, dump
from phase_release import verify_freeze
from phase_calendar import PhaseCalendar

RUN = ROOT / "runs/phase_20260913"
OUT = ROOT / "outputs/phase_holdout_verification_20260914"
KEYS = policy.GROUP_KEYS


def score(menu, actual):
    m = policy.evaluate_policy_metrics(menu, actual, policy.BASE_CONSTRAINTS)
    hm, hits = hit_quantity_diagnostics(menu, actual)
    m.update(hm)
    a = actual.groupby(KEYS, as_index=False).agg(actual_total=("菜品总销量", "sum"), actual_count=("_dish_norm", "nunique"))
    p = menu.groupby(KEYS, as_index=False).agg(pred_total=("pred_qty", "sum"), pred_count=("菜品名称", "nunique"))
    s = a.merge(p, on=KEYS, how="left", validate="one_to_one").fillna({"pred_total": 0, "pred_count": 0})
    s["bias"] = s.pred_total - s.actual_total
    s["ape_pct"] = 100 * s.bias.abs() / s.actual_total.replace(0, np.nan)
    m["total_wape"] = float(100 * s.bias.abs().sum() / s.actual_total.sum())
    m["session_count"] = len(s)
    assert np.isclose(m["total_mape"], s.ape_pct.mean(), atol=1e-8)
    return m, s, hits


def main():
    frozen = verify_freeze()
    OUT.mkdir(parents=True, exist_ok=False)
    summary, groups, bad_sessions, bad_hits = [], [], [], []
    verification = {"selected": frozen["selected"], "source_freeze_verified": True,
        "method": "Independent rescoring of checksum-verified completed frozen daily replays; no second model selection or tuning", "datasets": {}}
    calendar = PhaseCalendar()
    for ds in ["a_hupan", "f_jichuyuan"]:
        folder = RUN / "retest" / ds
        record = json.loads((folder / "record.json").read_text())
        assert record["status"] == "complete" and frozen["selected"][ds] == "baseline"
        for name, sha in record["sha256"].items():
            assert policy.sha256_file(folder / name) == sha, name
        menu = pd.read_pickle(folder / "menu.pkl.gz")
        actual = pd.read_pickle(folder / "actual.pkl.gz")
        history = policy.read_sales(ROOT / "smart_canteen_web/data/datasets" / ds / "sales.csv")
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), history.loc[history.date.between("2026-01-01", "2026-06-10")].reset_index(drop=True))
        assert not menu.duplicated(KEYS + ["菜品名称"]).any()
        contexts = actual.drop_duplicates(KEYS)[KEYS + ["是否校历寒暑假", "校历假期类型", "是否法定节假日假期", "法定节假日名称", "是否调休工作日"]].copy()
        contexts["date_type"] = contexts.apply(date_type, axis=1)
        contexts["phase_family"] = contexts.date.map(lambda d: calendar.features(d)["phase_family"])
        intervals = [("full_existing_interval", "2026-01-01", "2026-06-10"),
                     ("old_validation" if ds == "a_hupan" else "q1_diagnostic", "2026-01-01", "2026-03-31"),
                     ("old_test" if ds == "a_hupan" else "q2_diagnostic", "2026-04-01", "2026-06-10")]
        for interval, start, end in intervals:
            p = menu.loc[menu.date.between(start, end)]
            a = actual.loc[actual.date.between(start, end)]
            metrics, sessions, hits = score(p, a)
            old = record["results"][interval]
            checked = ["total_mape", "hit_qty_mape", "hit_qty_wmape", "net_benefit", "total_wape"]
            for k in checked:
                assert np.isclose(metrics[k], old[k], rtol=1e-12, atol=1e-8), (ds, interval, k)
            summary.append(dict(dataset=ds, interval=interval, start=start, end=end, **metrics))
            sessions = sessions.merge(contexts[KEYS + ["date_type", "phase_family"]], on=KEYS, validate="one_to_one")
            hits = hits.merge(contexts[KEYS + ["date_type", "phase_family"]], on=KEYS, validate="many_to_one")
            for dimension in ["date_type", "phase_family"]:
                for name, g in sessions.groupby(dimension):
                    keys = g[KEYS]
                    gm, _, _ = score(p.merge(keys, on=KEYS), a.merge(keys, on=KEYS))
                    groups.append(dict(dataset=ds, interval=interval, dimension=dimension, segment=name, **gm))
            for r in sessions.nlargest(8, "ape_pct").to_dict("records"):
                bad_sessions.append(dict(dataset=ds, interval=interval, **r))
            for r in hits.nlargest(8, "ape").to_dict("records"):
                past = history.loc[(history.date < r["date"]) & history["餐次"].eq(r["餐次"]) & history._dish_norm.eq(r["_dish_norm"])]
                meal = sessions.loc[sessions.date.eq(r["date"]) & sessions["餐次"].eq(r["餐次"])].iloc[0]
                bad_hits.append(dict(dataset=ds, interval=interval, **r,
                    ape_pct=100*r["ape"], past_samples=len(past), past_median=float(past["菜品总销量"].median()) if len(past) else None,
                    past_last=float(past.sort_values("date")["菜品总销量"].iloc[-1]) if len(past) else None,
                    meal_actual_total=float(meal.actual_total), meal_pred_total=float(meal.pred_total), meal_ape_pct=float(meal.ape_pct)))
            dump(OUT / (ds + "_" + interval + "_metrics.json"), metrics)
        verification["datasets"][ds] = {"saved_replay_hashes_verified": True, "actual_matches_current_data": True,
            "rescored_metrics_match_saved_metrics": True, "evaluation_sessions": len(actual[KEYS].drop_duplicates())}
    verify_freeze()
    dump(OUT / "verification.json", verification)
    dump(OUT / "summary.json", summary)
    dump(OUT / "date_and_phase_metrics.json", groups)
    dump(OUT / "worst_sessions.json", bad_sessions)
    dump(OUT / "worst_hit_dishes.json", bad_hits)
    cols = ["dataset", "interval", "session_count", "total_mape", "hit_qty_mape", "hit_qty_wmape", "net_benefit"]
    print(pd.DataFrame(summary)[cols].round(4).to_string(index=False), flush=True)
    print("Independent metric verification complete", flush=True)


if __name__ == "__main__":
    main()
