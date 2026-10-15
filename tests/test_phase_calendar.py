import pandas as pd
import numpy as np
from phase_calendar import PhaseCalendar


def test_missing_calendar_does_not_invent_start():
    rows=[]
    for date,school in [("2025-07-01",0),("2025-07-21",1),("2025-07-22",1),("2025-07-23",0)]:
        rows.append({"日期":date,"是否校历寒暑假":school,"校历假期类型":"暑假" if school else "非寒暑假",
                     "是否法定节假日假期":0,"法定节假日名称":"非节假日"})
    c=PhaseCalendar(pd.DataFrame(rows));f=c.features(pd.Timestamp("2025-07-21"))
    assert np.isnan(f["school_start_distance"]) and f["school_start_unknown"]
    assert f["school_end_distance"]==-1
    assert c.features(pd.Timestamp("2025-07-10"))["phase_family"]=="unknown_calendar"
    assert c.features(pd.Timestamp("2025-07-23"))["phase_family"]=="school_recovery"


def test_overlap_is_one_joint_state():
    rows=[]
    for day in range(1,6):
        rows.append({"日期":f"2025-02-0{day}","是否校历寒暑假":int(1<day<5),"校历假期类型":"寒假",
            "是否法定节假日假期":int(day==3),"法定节假日名称":"春节"})
    c=PhaseCalendar(pd.DataFrame(rows));f=c.features(pd.Timestamp("2025-02-03"))
    assert f["phase_family"]=="spring_core" and f["joint_overlap"]
    assert f["phase_event"]=="寒假+春节" and f["school_elapsed"]==2
