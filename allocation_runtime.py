"""Opt-in daily runtime for the frozen allocator, without changing web defaults."""
from __future__ import annotations

import json
import types
import pandas as pd
import policy
from allocation_prepare import OUT,load_baseline
from allocation_features import OfferedHistory
from allocation_policy import allocate_session,HistoricalBias,demand_estimate


class AllocationRuntime:
    """One logical canteen per instance; recommend whole day, then observe.

    `contexts` use the existing baseline interface, with past-only sales statistics
    and declared target-date calendar/weather inputs. The wrapper obtains hidden
    raw baseline weights itself; callers need not fabricate allocation features.
    Returned audit feasibility must be checked before operational publication.
    """
    def __init__(self, history, dataset, config=None, initial_menu_history=None):
        if dataset not in {"a","f"}:
            raise ValueError("Unknown logical canteen")
        self.module,self.baseline_config,_=load_baseline()
        if config is None:
            config=json.loads((OUT/"frozen/allocation_config.json").read_text())[dataset]
        self.config=dict(config)
        self.offered=OfferedHistory()
        self.offered.append(history)
        self.bias=HistoricalBias(config.get("bias_tau",10),config.get("bias","dish"),config.get("bias_strength",1))
        self.menu_history={} if initial_menu_history is None else {k:list(v) for k,v in initial_menu_history.items()}
        self.pending=None

    def recommend(self,contexts):
        if self.pending is not None:
            raise ValueError("Observe the completed prediction day before advancing")
        dates={pd.Timestamp(c["date"]) for c in contexts}
        if len(dates)!=1:
            raise ValueError("Provide every requested meal for one target date")
        date=next(iter(dates))
        contexts=[dict(c,day_group_count=len(contexts)) for c in contexts]
        if self.offered.max_date is not None and self.offered.max_date>=date:
            raise ValueError("History must precede target date")
        raw=[]
        original=self.module.enforce_repeat
        def capture(*args):
            result=original(*args)
            c=args[-1]
            for r in result.to_dict("records"):
                raw.append({"date":pd.Timestamp(c["date"]),"食堂名称":c["食堂名称"],"餐次":c["餐次"],"_dish_norm":r["_dish_norm"],
                    "uncalibrated_qty":float(r["_allocation_weight"]),"baseline_qty_stability":float(r["_qty_stability"]),
                    "baseline_recent_weight":float(r["_adaptive_recent_weight"])})
            return result
        core=self.module._integrated_function()
        function=types.FunctionType(core.__code__,dict(core.__globals__,enforce_repeat=capture),core.__name__,core.__defaults__)
        function.__kwdefaults__=core.__kwdefaults__
        audits=[]
        baseline=function(contexts,self.baseline_config,policy.BASE_CONSTRAINTS,selection_audit=audits,initial_menu_history=self.menu_history)
        results,allocation_audits,observations=[],[],[]
        rawframe=pd.DataFrame(raw)
        for c in contexts:
            menu=baseline.loc[baseline["餐次"].eq(c["餐次"]) & baseline["食堂名称"].eq(c["食堂名称"])].copy()
            self.menu_history.setdefault((c["食堂名称"],c["餐次"]),[]).append((date,set(menu["菜品名称"].map(policy.normalize_name))))
            if menu.empty:
                continue
            f=self.offered.features(menu,c["target_context"]).merge(rawframe,on=policy.GROUP_KEYS+["_dish_norm"],validate="one_to_one")
            factors=self.bias.factors(f) if self.config.get("bias") else None
            result,audit=allocate_session(menu,f,self.config,factors)
            results.append(result)
            allocation_audits.append(audit)
            observations.append((f,demand_estimate(f,self.config)))
        self.pending=(date,observations)
        new=pd.concat(results).sort_index() if results else baseline.copy()
        return new,{"baseline":audits,"allocation":allocation_audits}

    def observe(self,actual):
        if self.pending is None:
            raise ValueError("No pending prediction day")
        date,observations=self.pending
        if not actual.date.eq(date).all() or actual.empty:
            raise ValueError("Observe exactly the completed prediction date")
        if self.config.get("bias"):
            from evaluator import prepare_actual
            truth=prepare_actual(actual)
            for features,estimate in observations:
                self.bias.observe(features,estimate,truth)
        self.offered.append(actual)
        self.pending=None
