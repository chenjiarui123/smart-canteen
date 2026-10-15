"""Read-only instrumentation of the exact deployed baseline's six stages."""
from __future__ import annotations
import inspect
import types
import numpy as np
import pandas as pd
import policy

STAGES=["initial_history","date_conditioned","stability_weight","total_allocated","minimum_processed","repeat_final"]


class BaselineTrace:
    def __init__(self,module):
        self.module=module
        self.rows={}
        core=module._integrated_function()
        original_attach=core.__globals__["_attach_min_batch_estimates"]
        original_allocate=core.__globals__["_allocate_and_prune_min_batch"]
        original_repeat=core.__globals__["enforce_repeat"]

        def attach(frame,config,local_count):
            caller=inspect.currentframe().f_back
            try:
                values=caller.f_locals
                context=values["context"]
                initial=(values["adaptive_recent_weight"]*values["recent_robust"]+(1-values["adaptive_recent_weight"])*values["long_robust"])
                result=original_attach(frame,config,local_count)
                for idx,r in result.iterrows():
                    key=(pd.Timestamp(context["date"]),context["食堂名称"],context["餐次"],r["_dish_norm"])
                    self.rows[key]={"date":key[0],"食堂名称":key[1],"餐次":key[2],"_dish_norm":key[3],
                        "菜品名称":r["菜品名称_库"],"菜品类别":r["菜品类别"],"initial_history":float(initial.loc[idx]),
                        "date_conditioned":float(r["_allocation_weight"]),
                        "stability_weight":float(r["_allocation_weight"]*r["_qty_stability"]**float(config.get("allocation_uncertainty_power",0))),
                        "qty_stability":float(r["_qty_stability"]),"min_batch_qty":float(r["_min_batch_qty"]),
                        "history_count":float(local_count.loc[idx]),"adaptive_recent_weight":float(r["_adaptive_recent_weight"]),
                        "history_max_date":context["max_history_date"]}
                return result
            finally:
                del caller

        def allocate(frame,total,config):
            caller=inspect.currentframe().f_back
            try:
                context=caller.f_locals["context"]
                plain,_=original_allocate(frame,total,dict(config,min_batch_mode="none"))
                result,removed=original_allocate(frame,total,config)
                self.record(context,plain,"total_allocated")
                self.record(context,result,"minimum_processed")
                return result,removed
            finally:
                del caller

        def repeat(*args):
            result=original_repeat(*args)
            self.record(args[-1],result,"repeat_final")
            return result

        self.function=types.FunctionType(core.__code__,dict(core.__globals__,_attach_min_batch_estimates=attach,
            _allocate_and_prune_min_batch=allocate,enforce_repeat=repeat),core.__name__,core.__defaults__)
        self.function.__kwdefaults__=core.__kwdefaults__

    def record(self,context,frame,stage):
        for r in frame.to_dict("records"):
            key=(pd.Timestamp(context["date"]),context["食堂名称"],context["餐次"],r["_dish_norm"])
            self.rows[key][stage]=float(r["pred_qty"])

    def generate(self,contexts,config,menu_history):
        self.rows={}
        contexts=[dict(c) for c in contexts]
        audit=[]
        menu=self.function(contexts,config,policy.BASE_CONSTRAINTS,selection_audit=audit,initial_menu_history=menu_history)
        indexed={(pd.Timestamp(c["date"]),c["食堂名称"],c["餐次"]):c for c in contexts}
        for a in audit:
            c=indexed[(pd.Timestamp(a["date"]),a["食堂名称"],a["餐次"])]
            a.update(c.get("_targeted_audit",{}))
        trace=pd.DataFrame(self.rows.values())
        for stage in STAGES:
            if stage not in trace:
                trace[stage]=np.nan
        trace["removed_at_minimum"]=trace.total_allocated.notna() & trace.minimum_processed.isna()
        trace["added_at_repeat"]=trace.minimum_processed.isna() & trace.repeat_final.notna()
        trace["removed_at_repeat"]=trace.minimum_processed.notna() & trace.repeat_final.isna()
        return menu,audit,trace
