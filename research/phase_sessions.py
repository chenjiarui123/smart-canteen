"""Joint calendar-stage session levels and independently estimated dish counts."""
import numpy as np
import pandas as pd
from phase_calendar import PhaseCalendar
from targeted_search import date_type
import policy


class SessionModel:
    def __init__(self,calendar,window=7,trend=False):self.calendar=calendar;self.window=window;self.trend=trend;self.rows=[];self.max_date=None

    def observe(self,day):
        date=pd.Timestamp(day.date.iloc[0]);assert day.date.eq(date).all()
        if self.max_date is not None:assert self.max_date<date
        phase=self.calendar.features(date,self.window)
        for (canteen,meal),g in day.groupby(["食堂名称","餐次"]):
            self.rows.append(dict(date=date,canteen=canteen,meal=meal,qty=float(g["菜品总销量"].sum()),count=g._dish_norm.nunique(),
                calendar_state=str(g._calendar_state.iloc[0]),date_type=date_type(g.iloc[0]),**phase))
        self.max_date=date

    def predict(self,date,canteen,meal,target,baseline_total,baseline_count):
        date=pd.Timestamp(date)
        assert self.max_date is None or self.max_date<date
        phase=self.calendar.features(date,self.window)
        result=dict(date=date,食堂名称=canteen,餐次=meal,baseline_total=float(baseline_total),baseline_count=int(baseline_count),
            predicted_total=float(baseline_total),predicted_count=int(baseline_count),history_max_date=self.max_date,window=self.window,**phase)
        h=[r for r in self.rows if r["canteen"]==canteen and r["meal"]==meal]
        if not phase["phase_special"]:result["source"]="ordinary_final_total_lock";return result
        state=str(target.get("_calendar_state",""))
        levels=[("compatible_calendar",[r for r in h if r["calendar_state"]==state]),
                ("phase_family",[r for r in h if r["phase_family"]==phase["phase_family"]]),
                ("named_joint_phase",[r for r in h if r["phase"]==phase["phase"]])]
        source=[];seen=set()
        for metric,parent in [("qty",float(baseline_total)),("count",float(baseline_count))]:
            value=parent;used=[]
            for level,rows in levels:
                identity=tuple(r["date"] for r in rows)
                if not rows or identity in seen and metric=="qty":continue
                if identity in used:continue
                used.append(identity)
                n=len(rows);estimate=float(np.median([r[metric] for r in rows]));w=n/(n+3)
                value=float(np.expm1(w*np.log1p(estimate)+(1-w)*np.log1p(max(value,0))))
                if metric=="qty":source.append(dict(level=level,n=n,median=estimate));seen.add(identity)
            recent=[r for r in h if r["phase_family"]==phase["phase_family"]][-3:]
            if self.trend and recent:
                w=len(recent)/(len(recent)+3);value=(1-w)*value+w*float(np.median([r[metric] for r in recent]))
            result["predicted_total" if metric=="qty" else "predicted_count"]=max(value,0) if metric=="qty" else max(1,int(round(value)))
        result["source"]=source or "no_phase_evidence_baseline_fallback";return result


def forecasts(a,window=7,trend=False):
    model=SessionModel(PhaseCalendar(),window,trend)
    for _,day in a["warmup"].groupby("date",sort=True):model.observe(day)
    baseline=a["baseline_menu"].groupby(policy.GROUP_KEYS).agg(qty=("pred_qty","sum"),count=("菜品名称","nunique")).to_dict("index")
    rows=[]
    for date,day in a["actual"].groupby("date",sort=True):
        for (canteen,meal),g in day.groupby(["食堂名称","餐次"]):
            b=baseline.get((date,canteen,meal),dict(qty=0,count=0))
            # Only target calendar metadata and known meal identity passed.
            target={"_calendar_state":g._calendar_state.iloc[0]}
            rows.append(model.predict(date,canteen,meal,target,b["qty"],b["count"]))
        model.observe(day)
    return pd.DataFrame(rows)
