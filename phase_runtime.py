"""Opt-in, one-day API. Does not register or replace the web strategy.

Returns the retained baseline with separate preparation advice. Rejected
research models remain executable through the full chronological runner,
never promoted or refitted with a different one-day fitting schedule.
"""
import numpy as np
import pandas as pd
import policy
from phase_features import KEYS


def predict_day(baseline_menu,prediction_features,offered_features,past_training,config=None):
    menu=baseline_menu.copy()
    if menu.empty:return menu,dict(status="empty",web_default_changed=False)
    if menu.date.nunique()!=1 or menu["食堂名称"].nunique()!=1:raise ValueError("One canteen, one complete prediction day")
    day=pd.Timestamp(menu.date.iloc[0]);config=dict(config or {})
    if not past_training.empty and past_training.date.max()>=day:raise ValueError("Only strictly past supplied records allowed")
    if not prediction_features.date.eq(day).all() or pd.to_datetime(prediction_features.history_max_date).ge(day).any():raise ValueError("Prediction features are not pre-day")
    index=pd.MultiIndex.from_frame(menu[policy.GROUP_KEYS].assign(_dish_norm=menu["菜品名称"].map(policy.normalize_name)))
    f=prediction_features.set_index(KEYS).loc[index].reset_index();old=offered_features.set_index(KEYS).loc[index].reset_index()
    np.testing.assert_allclose(old.baseline_qty,menu.pred_qty,rtol=1e-12,atol=1e-8)
    if not config:
        menu["expected_demand_qty"]=menu.pred_qty
        menu["suggested_preparation_qty"]=np.maximum(menu.pred_qty,menu.min_batch_qty)
        menu["minimum_batch_surplus_risk"]=np.maximum(menu.min_batch_qty-menu.pred_qty,0)
        return menu,dict(status="retained_baseline",web_default_changed=False,reason="No research model passed release gates")
    raise ValueError("No research configuration passed release gates. Use the full chronological experiment runner for research; this retained runtime cannot deploy rejected configurations.")
