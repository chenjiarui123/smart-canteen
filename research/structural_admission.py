"""Past-only expert admission by meal/calendar state; no target-outcome access."""
from collections import defaultdict
import numpy as np
import pandas as pd


class HistoricalAdmission:
    def __init__(self,thresholds):
        self.thresholds=thresholds
        self.records=defaultdict(list)
        self.max_date=None

    @staticmethod
    def key(features):
        return (str(features["食堂名称"].iloc[0]),str(features["餐次"].iloc[0]),str(features.calendar_state.iloc[0]))

    @staticmethod
    def aggregate(records,label):
        s={k:sum(r[label][k] for r in records) for k in ["hits","ape_sum","abs_error","actual_qty","qty_error","net","nonhit_qty","pred_qty"]}
        return {"hits":s["hits"],"mape":s["ape_sum"]/max(s["hits"],1)*100,
            "wmape":s["abs_error"]/max(s["actual_qty"],1e-12)*100,"bias":s["qty_error"]/max(s["actual_qty"],1e-12)*100,
            "net":s["net"],"nonhit_share":s["nonhit_qty"]/max(s["pred_qty"],1e-12)}

    def decide(self,features):
        date=pd.Timestamp(features.date.iloc[0])
        if self.max_date is not None and self.max_date>=date:
            raise ValueError("Admission cannot use target-day outcomes")
        records=self.records[self.key(features)][-30:]
        b=self.aggregate(records,"baseline");n=self.aggregate(records,"expert")
        reasons=[]
        t=self.thresholds
        if len(records)<3 or b["hits"]<t["subgroup_min_hits_for_gate"]:reasons.append("insufficient_previous_prediction_evidence")
        if n["mape"]>b["mape"]-t["minimum_hit_mape_improvement_pp"]:reasons.append("past_hit_mape_not_better")
        if n["wmape"]>b["wmape"]*(1+t["hit_wmape_relative_increase_max"]):reasons.append("past_wmape_risk")
        if n["net"]<b["net"]-abs(b["net"])*t["net_relative_loss_max"]:reasons.append("past_net_risk")
        if abs(n["bias"])>abs(b["bias"])+t["absolute_weighted_bias_increase_pp_max"]:reasons.append("past_bias_risk")
        if n["nonhit_share"]>b["nonhit_share"]+t["nonhit_share_absolute_increase_max"]:reasons.append("past_nonhit_allocation_risk")
        return not reasons,{"admitted":not reasons,"reasons":reasons,"previous_sessions":len(records),
            "previous_max_date":max((r["date"] for r in records),default=None),"baseline_history":b,"expert_history":n}

    @staticmethod
    def sufficient_statistics(menu,actual):
        from evaluator import prepare_actual,prepare_recommendations
        p=prepare_recommendations(menu)
        a=prepare_actual(actual)
        g=p.merge(a[["date","食堂名称","餐次","_dish_norm","actual_qty","actual_rev"]],on=["date","食堂名称","餐次","_dish_norm"],how="left")
        hit=g.actual_qty.gt(0)
        h=g.loc[hit]
        return {"hits":len(h),"ape_sum":float(((h.pred_qty-h.actual_qty).abs()/h.actual_qty).sum()),
            "abs_error":float((h.pred_qty-h.actual_qty).abs().sum()),"actual_qty":float(h.actual_qty.sum()),
            "qty_error":float((h.pred_qty-h.actual_qty).sum()),
            "net":float(g.actual_rev.fillna(0).sum()-.5*(g.pred_rev-g.actual_rev.fillna(0)).clip(lower=0).sum()),
            "nonhit_qty":float(g.loc[~hit,"pred_qty"].sum()),"pred_qty":float(g.pred_qty.sum())}

    def observe_day(self,observations,actual):
        date=pd.Timestamp(actual.date.iloc[0])
        if not actual.date.eq(date).all() or (self.max_date is not None and self.max_date>=date):
            raise ValueError("Admission observations must be a new completed day")
        for features,baseline,expert in observations:
            if not features.date.eq(date).all():raise ValueError("Admission observation date mismatch")
            self.records[self.key(features)].append({"date":date,"baseline":self.sufficient_statistics(baseline,actual),"expert":self.sufficient_statistics(expert,actual)})
        self.max_date=date
