"""Calendar-only holiday boundaries; missing dates never establish a start."""
from functools import lru_cache
import numpy as np
import pandas as pd
import policy
from phase_data import OUT, verify
from targeted_search import dump


class PhaseCalendar:
    def __init__(self, frame=None):
        if frame is None:
            verify()
            frame=pd.read_csv(OUT/"snapshot/smart_canteen_web/data/features/calendar.csv")
        frame=frame.copy();frame["date"]=pd.to_datetime(frame["日期"])
        self.rows=frame.set_index("date").to_dict("index");self.events=[]
        for kind,flag,name in [("school","是否校历寒暑假","校历假期类型"),("legal","是否法定节假日假期","法定节假日名称")]:
            flagged=frame.loc[frame[flag].map(policy._flag)].sort_values("date")
            groups=[]
            for r in flagged.to_dict("records"):
                if groups and r["date"]-groups[-1][-1]["date"]==pd.Timedelta(days=1) and r[name]==groups[-1][-1][name]:groups[-1].append(r)
                else:groups.append([r])
            for group in groups:
                first,last=group[0]["date"],group[-1]["date"]
                before=self.rows.get(first-pd.Timedelta(days=1));after=self.rows.get(last+pd.Timedelta(days=1))
                start_known=before is not None and not policy._flag(before[flag])
                end_known=after is not None and not policy._flag(after[flag])
                self.events.append(dict(kind=kind,name=group[0][name],observed_start=first,observed_end=last,
                    start=first if start_known else None,end=last if end_known else None,
                    start_known=start_known,end_known=end_known))

    @lru_cache(maxsize=None)
    def features(self, day, window=7):
        day=pd.Timestamp(day);row=self.rows.get(day)
        result=dict(calendar_known=row is not None,phase="ordinary",phase_family="ordinary",phase_event="none",phase_special=False)
        active={};states={}
        for kind in ["school","legal"]:
            events=[e for e in self.events if e["kind"]==kind]
            current=next((e for e in events if e["observed_start"]<=day<=e["observed_end"]),None)
            active[kind]=current
            # Nearest known boundary may lie in future: calendar is known input.
            starts=[e for e in events if e["start"] is not None]
            ends=[e for e in events if e["end"] is not None]
            near_start=min(starts,key=lambda e:abs((day-e["start"]).days)) if starts else None
            near_end=min(ends,key=lambda e:abs((day-e["end"]).days)) if ends else None
            ds=(day-near_start["start"]).days if near_start else np.nan
            de=(day-near_end["end"]).days if near_end else np.nan
            if current:
                ds=(day-current["start"]).days if current["start_known"] else np.nan
                de=(day-current["end"]).days if current["end_known"] else np.nan
            prior=[e for e in ends if e["end"]<day]
            since=(day-max(e["end"] for e in prior)).days if prior else np.nan
            result.update({kind+"_start_distance":ds,kind+"_end_distance":de,
                kind+"_elapsed":ds+1 if current and np.isfinite(ds) else np.nan,
                kind+"_since_end":since,kind+"_active":current is not None,
                kind+"_start_unknown":bool(current and not current["start_known"]),
                kind+"_end_unknown":bool(current and not current["end_known"])})
            if current:
                state="end_return" if kind=="school" and np.isfinite(de) and -window<de<=0 else "ongoing"
                states[kind]=(state,current["name"])
            elif near_start and -window<=ds<0:states[kind]=("pre",near_start["name"])
            elif near_end and 0<de<=window:states[kind]=("recovery",near_end["name"])
        result["joint_overlap"]=bool(active["school"] and active["legal"])
        if active["legal"]:
            family="spring_core" if "春节" in active["legal"]["name"] else "legal_core"
            event=(str(active["school"]["name"])+"+" if active["school"] else "")+str(active["legal"]["name"])
        elif states:
            # A named legal transition inside a school vacation stays joint.
            kind="legal" if "legal" in states else "school"
            state,event=states[kind];family=kind+"_"+state
            if active["school"] and kind=="legal":event=str(active["school"]["name"])+"+"+str(event)
        else:family="ordinary";event="none"
        if row is None:family="unknown_calendar";event="unknown"
        result.update(phase_family=family,phase_event=event,phase=family+":"+event,
                      phase_special=family not in {"ordinary","unknown_calendar"})
        return result

    def audit(self):
        dump(OUT/"calendar_events.json",self.events)
        rows=[dict(date=d,window=w,**self.features(d,w)) for d in pd.date_range(min(self.rows),max(self.rows)) for w in [3,7,14]]
        pd.DataFrame(rows).to_csv(OUT/"calendar_phase_audit.csv",index=False)


if __name__=="__main__":PhaseCalendar().audit()
