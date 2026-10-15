"""Add complete demand/preparation views without altering experiment outputs."""
import json
import numpy as np
import pandas as pd
import policy
from phase_data import OUT,DATASETS,verify
from targeted_search import dump


def run():
    verify();rows=[]
    paths=list((OUT/"experiments").glob("*/*/menu.pkl.gz"))+list(OUT.glob("development_*/baseline_menu.pkl"))+list((OUT/"retest").glob("*/menu.pkl.gz"))
    for p in paths:
        menu=pd.read_pickle(p);out=menu.copy()
        out["expected_demand_qty"]=out.pred_qty
        out["suggested_preparation_qty"]=np.maximum(out.pred_qty,out.min_batch_qty)
        out["minimum_batch_surplus_risk"]=np.maximum(out.min_batch_qty-out.pred_qty,0)
        out["preparation_orders"]=out.suggested_preparation_qty*out.expected_orders_per_qty
        out["preparation_revenue"]=out.suggested_preparation_qty*out.expected_rev_per_qty
        np.testing.assert_array_equal(menu.pred_qty,out.pred_qty)
        dest=p.parent/"complete_demand_preparation.csv.gz";out.to_csv(dest,index=False)
        rows.append(dict(source=str(p.relative_to(OUT)),source_sha256=policy.sha256_file(p),export=str(dest.relative_to(OUT)),export_sha256=policy.sha256_file(dest),
            rows=len(out),below_minimum_demand=int((out.pred_qty<out.min_batch_qty-1e-8).sum()),suggested_extra_qty=float(out.minimum_batch_surplus_risk.sum())))
    dump(OUT/"demand_preparation_export_manifest.json",dict(source_unchanged=True,reason="Complete view fills missing advice on unmodified ordinary rows in early session experiments; no forecast/evaluation changes",exports=rows))
    print("Complete demand/preparation exports",len(rows),flush=True)


if __name__=="__main__":run()
