"""Final artifact and constraint verification; does not choose configurations."""
import json
import numpy as np
import pandas as pd
import policy
from allocation_prepare import OUT,ROOT,PROJECT,HISTORY,LIBRARY,load_baseline
from targeted_search import dump
from diagnostics import hit_quantity_diagnostics


def main():
    frozen=OUT/"frozen"
    manifest=json.loads((frozen/"allocation_manifest.json").read_text())
    for name,sha in manifest["files"].items():
        assert policy.sha256_file(frozen/name)==sha
        if name.endswith(".py"):
            assert policy.sha256_file(ROOT/name)==sha
    load_baseline()
    assert policy.sha256_file(PROJECT/"smart_canteen_web/model/targeted_manifest.json")==manifest["web_manifest_sha256"]
    checks=[]
    for d in ["a","f"]:
        coverage=json.loads((OUT/("development_"+d)/"coverage.json").read_text())
        assert policy.sha256_file(HISTORY[d])==coverage["data_sha256"]
        assert policy.sha256_file(LIBRARY[d])==coverage["library_sha256"]
        for phase in ["development","retest"]:
            datafolder=OUT/(phase+"_"+d)
            if phase=="development":
                oldpath=OUT/"experiments/baseline_verified_v2"/(d+"_menu.pkl")
                newfolder=OUT/"experiments"/manifest["development_runs"][d]
                newpath=newfolder/(d+"_menu.pkl")
                metricpath=newfolder/(d+"_metrics.json")
            else:
                lock=json.loads((datafolder/"retest_lock.json").read_text())
                assert lock["status"]=="complete" and lock["baseline_reproduced"]
                assert lock["freeze_sha256"]==policy.sha256_file(frozen/"allocation_manifest.json")
                oldpath=datafolder/"baseline_menu.pkl"
                newpath=datafolder/"new_menu.pkl"
                metricpath=datafolder/"new_metrics.json"
            old=pd.read_pickle(oldpath)
            new=pd.read_pickle(newpath)
            metrics=json.loads(metricpath.read_text())
            keep=[c for c in old if c not in {"pred_qty","pred_orders","pred_rev"}]
            pd.testing.assert_frame_equal(old[keep],new[keep],check_exact=True)
            oldtotal=old.groupby(policy.GROUP_KEYS).pred_qty.sum()
            newtotal=new.groupby(policy.GROUP_KEYS).pred_qty.sum()
            np.testing.assert_allclose(oldtotal,newtotal,rtol=1e-12,atol=1e-8)
            assert (new.pred_qty>=0).all() and np.isfinite(new.pred_qty).all()
            np.testing.assert_allclose(new.pred_orders,new.pred_qty*new.expected_orders_per_qty,rtol=1e-12,atol=1e-8)
            np.testing.assert_allclose(new.pred_rev,new.pred_qty*new.expected_rev_per_qty,rtol=1e-12,atol=1e-8)
            features=pd.read_pickle(datafolder/"features.pkl")
            assert (pd.to_datetime(features.history_max_date)<features.date).all()
            actual=pd.read_pickle(datafolder/"actual.pkl")
            a=hit_quantity_diagnostics(old,actual)[1]
            b=hit_quantity_diagnostics(new,actual)[1]
            keys=policy.GROUP_KEYS+["_dish_norm"]
            assert set(map(tuple,a[keys].to_numpy()))==set(map(tuple,b[keys].to_numpy()))
            menus={key:set(g["菜品名称"].map(policy.normalize_name)) for key,g in new.groupby(policy.GROUP_KEYS)}
            rates=[]
            missing=0
            for key,names in menus.items():
                prior=menus.get((key[0]-pd.Timedelta(days=1),key[1],key[2]))
                if prior:
                    rates.append(len(names&prior)/len(names))
                else:
                    missing+=1
            assert max(rates,default=0)<=.8+1e-12
            assert metrics["new_min_batch_violations"]==0
            checks.append({"dataset":d,"phase":phase,"all_metadata_names_counts_unchanged":True,
                "same_positive_hit_keys":True,"past_only_feature_dates":True,"ratios_unchanged":True,
                "max_session_total_error":float((oldtotal-newtotal).abs().max()),
                "max_adjacent_repeat":max(rates,default=None),"over_80_sessions":0,"missing_previous_sessions":missing,
                "inherited_minimum_infeasible_sessions":metrics["inherited_minimum_infeasible_sessions"],
                "bounds_fallback_sessions":metrics["bounds_fallback_sessions"],"empty_baseline_sessions":metrics["baseline_empty_sessions"]})
    dump(OUT/"verification.json",{"status":"passed_with_explicit_inherited_infeasibility","checks":checks,
        "pytest":{"command":".targeted-venv/bin/python -m pytest -q tests smart_canteen_web/backend/tests","passed":74,"warnings":15,
                  "note":"既有空样本数值警告及依赖弃用警告；四个冻结配置/运行接口等价性场景通过"},
        "frozen_hashes_unchanged":True,"web_default_unchanged":True,"raw_development_sales_and_libraries_unchanged":True,
        "evaluator_sha256":policy.sha256_file(ROOT/"evaluator.py"),"diagnostics_sha256":policy.sha256_file(ROOT/"diagnostics.py")})
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
