"""Execute one preregistered structural batch; not a fixed trial-count limit."""
import argparse
from phase_item_experiments import run as item
from phase_session_experiments import run as session


def run(dataset,batch):
    suffix="a" if dataset=="a_hupan" else "f"
    if batch=="session":
        for n,c in [("phase_direct3",{"demand":True,"window":3}),("phase_direct14",{"demand":True,"window":14}),
                    ("phase_count7",{"count":True,"window":7}),("phase_joint7",{"demand":True,"count":True,"window":7})]:session(n+"_"+suffix,c,[dataset])
    elif batch=="items":
        for n,c in [("own_typical",{"mode":"typical"}),("own_hierarchy",{"mode":"typical","hierarchy":True}),
            ("own_drop_hierarchy",{"source":"own_drop_"+suffix,"hierarchy":True}),
            ("own_drop_mixed",{"source":"own_drop_"+suffix,"objective":"mixed"}),
            ("own_drop_relative_loss",{"source":"own_drop_"+suffix,"objective":"relative"}),
            ("own_drop_mean",{"source":"own_drop_"+suffix,"objective":"mean"}),
            ("own_drop_no_phase",{"mode":"drop","phase":False}),
            ("own_drop_hits",{"mode":"drop","hits_only":True}),
            ("own_drop_risk",{"source":"own_drop_"+suffix,"coordination":"risk"})]:item(n+"_"+suffix,c,[dataset])


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("dataset");p.add_argument("batch",choices=["session","items"]);a=p.parse_args();run(a.dataset,a.batch)
