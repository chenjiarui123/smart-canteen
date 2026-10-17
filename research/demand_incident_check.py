"""Reproduce metadata invariant failure without editing evidence or predictors."""
import numpy as np
import pandas as pd
import policy
from demand_data import OUT
from demand_experiments import load
from demand_models import decision
from demand_evaluate import KEYS
from demand_coordination import coordinate


def main():
    a,f,t=load("a_hupan");b=a["baseline_menu"]
    old=a["features"].set_index(KEYS).loc[pd.MultiIndex.from_frame(f[KEYS])].reset_index()
    cache=np.load(OUT/"experiments/conditional_all_log/a_hupan/distributions.npz")
    samples=cache["samples"];point=decision(samples,"absolute",f.historical_scale.to_numpy())
    keep=[c for c in b if c not in {"pred_qty","pred_orders","pred_rev"}]
    rows=[]
    for key,index in b.groupby(policy.GROUP_KEYS,sort=False).indices.items():
        menu=b.iloc[index]
        new,_=coordinate(menu,old.iloc[index].reset_index(drop=True),f.iloc[index],point[index],samples[index])
        pd.testing.assert_frame_equal(new[keep],menu[keep],check_exact=True)
        rows.append(new)
    combined=pd.concat(rows).sort_index()
    for phase in ["before_evaluation","after_evaluation"]:
        different=combined.weather_state.ne(b.weather_state)
        print(phase,"weather mismatches",int(different.sum()),flush=True)
        if different.any():print(pd.DataFrame({"date":b.date,"old":b.weather_state,"new":combined.weather_state}).loc[different].head(5).to_string(index=False),flush=True)
        pd.testing.assert_frame_equal(combined[keep],b[keep],check_exact=True)
        if phase=="before_evaluation":policy.evaluate_policy_metrics(combined,a["actual"],policy.BASE_CONSTRAINTS)
    print("metadata failure not reproduced; all stage copies and evaluator input remain exact",flush=True)


if __name__=="__main__":main()
