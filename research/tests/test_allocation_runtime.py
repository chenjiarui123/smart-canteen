import pandas as pd
import pytest

import policy
from allocation_runtime import AllocationRuntime
from allocation_prepare import HISTORY,LIBRARY,OUT


@pytest.mark.parametrize("trial,dataset,config",[
    ("stable_relative_2","a",{"statistic":"uncalibrated_qty","stability_power":2,"projection_power":2}),
    ("baseline_residual","a",{"statistic":"baseline_qty","bias":"dish","bias_tau":10}),
    ("final_verified_a","a",None),
    ("final_verified_f","f",None),
])
def test_optional_runtime_matches_cached_development_and_has_date_barrier(trial,dataset,config):
    path=OUT/"experiments"/trial/(dataset+"_menu.pkl")
    if not path.exists():
        pytest.skip("Development fixture not prepared")
    sales=policy.read_sales(HISTORY[dataset])
    history,actual,_=policy.training_backtest_frames(sales)
    attributes,_=policy.build_library_attributes(LIBRARY[dataset])
    runtime=AllocationRuntime(history,dataset,config)
    expected=pd.read_pickle(path)
    for date in sorted(actual.date.unique())[:2]:
        today=actual.loc[actual.date.eq(date)]
        contexts=policy.build_contexts(history,today,attributes)
        new,audit=runtime.recommend(contexts)
        old=expected.loc[expected.date.eq(date)].reset_index(drop=True)
        pd.testing.assert_frame_equal(new.reset_index(drop=True),old,check_exact=False,rtol=1e-12,atol=1e-8)
        assert len(audit["allocation"])==3
        with pytest.raises(ValueError,match="Observe"):
            runtime.recommend(contexts)
        runtime.observe(today)
        history=pd.concat([history,today],ignore_index=True)
