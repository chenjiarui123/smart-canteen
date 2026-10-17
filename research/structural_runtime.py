"""Optional structural forecast runtime, without changing web registration."""
import json
import pandas as pd
import policy
from allocation_prepare import load_baseline
from allocation_features import OfferedHistory
from structural_trace import BaselineTrace
from structural_features import StructuralHistory
from structural_policy import allocate,ForecastReliability,initial_estimate
from structural_borrowing import PeerResidualModel
from structural_admission import HistoricalAdmission
from structural_data import OUT
from targeted_search import date_type


class StructuralRuntime:
    def __init__(self,history,attributes,dataset,config=None,initial_menu_history=None):
        if dataset not in {"a_hupan","f_jichuyuan"}:
            raise ValueError("Unknown logical canteen")
        module,base_config,_=load_baseline()
        self.baseline_config=dict(base_config,runtime_history_end_inactivity=True)
        self.config=dict(config) if config is not None else json.loads((OUT/"frozen/structural_config.json").read_text())[dataset]
        self.trace=BaselineTrace(module)
        self.offered=OfferedHistory();self.offered.append(history)
        self.structural=StructuralHistory(attributes);self.structural.append(history)
        self.reliability=ForecastReliability()
        self.peer=PeerResidualModel()
        self.admission=HistoricalAdmission(json.loads((OUT/"preregistered_protocol.json").read_text())["selection"])
        self.menu_history={} if initial_menu_history is None else {k:list(v) for k,v in initial_menu_history.items()}
        self.pending=None

    def recommend(self,contexts):
        if self.pending is not None:raise ValueError("Observe the completed day before advancing")
        dates={pd.Timestamp(c["date"]) for c in contexts}
        if len(dates)!=1:raise ValueError("Supply one complete target day")
        date=next(iter(dates))
        if self.offered.max_date is not None and self.offered.max_date>=date:
            raise ValueError("History must precede prediction day")
        if any(c["max_history_date"] is not None and pd.Timestamp(c["max_history_date"])>=date for c in contexts):
            raise ValueError("Baseline contexts contain nonhistorical sales")
        baseline,base_audit,trace=self.trace.generate(contexts,self.baseline_config,self.menu_history)
        new_menus,audits,features,estimates,peer_features,design=[],[],[],[],[],[]
        admission_observations=[]
        for c in contexts:
            menu=baseline.loc[baseline["餐次"].eq(c["餐次"]) & baseline["食堂名称"].eq(c["食堂名称"])].copy()
            if menu.empty:continue
            f=self.offered.features(menu,c["target_context"])
            f["date_type"]=date_type(dict(c["target_context"],date=date))
            f=f.merge(trace[policy.GROUP_KEYS+["_dish_norm","initial_history","date_conditioned","stability_weight","qty_stability","adaptive_recent_weight"]],on=policy.GROUP_KEYS+["_dish_norm"],validate="one_to_one")
            f["uncalibrated_qty"]=f.date_conditioned
            f["baseline_qty_stability"]=f.qty_stability
            f["baseline_recent_weight"]=f.adaptive_recent_weight
            f=self.structural.augment(menu,f)
            override=None
            if self.config.get("peer_borrowing"):
                pf=f.copy();pf["baseline_qty"]=initial_estimate(f,self.config)
                override,x=self.peer.predict(menu,pf)
                peer_features.append(pf);design.append(pd.DataFrame(x))
            new,audit,estimate=allocate(menu,f,self.config,self.reliability,override)
            if self.config.get("admission_guard"):
                expert=new.copy()
                admitted,decision=self.admission.decide(f)
                if not admitted:
                    new,audit,_=allocate(menu,f,{"identity":True,"separate_preparation":self.config.get("separate_preparation",False)})
                audit["admission"]=decision
                admission_observations.append((f,menu,expert))
            new_menus.append(new);audits.append(audit);features.append(f);estimates.extend(estimate)
            self.menu_history.setdefault((c["食堂名称"],c["餐次"]),[]).append((date,set(menu["菜品名称"].map(policy.normalize_name))))
        self.pending=(date,features,estimates,peer_features,design,admission_observations)
        new=pd.concat(new_menus).sort_index() if new_menus else baseline.copy()
        return new,{"baseline":base_audit,"coordination":audits,"baseline_stage_trace":trace,
            "all_requested_sessions_generated":len(new_menus)==len(contexts)}

    def observe(self,actual):
        if self.pending is None:raise ValueError("No pending prediction")
        date,features,estimates,peer_features,design,admission_observations=self.pending
        if actual.empty or not actual.date.eq(date).all():raise ValueError("Observe the just-predicted day only")
        if features and self.config.get("reliability"):
            self.reliability.observe_day(pd.concat(features,ignore_index=True),estimates,actual)
        if peer_features and self.config.get("peer_borrowing"):
            self.peer.observe_day(pd.concat(peer_features,ignore_index=True),pd.concat(design,ignore_index=True).to_numpy(),estimates,actual)
        if admission_observations:
            self.admission.observe_day(admission_observations,actual)
        self.offered.append(actual);self.structural.append(actual)
        self.pending=None
