import json
import pandas as pd
import pytest
import policy
from structural_data import OUT
from structural_runtime import StructuralRuntime


@pytest.mark.parametrize("trial",["A_B","peer_only","guarded_A_C_F"])
def test_runtime_matches_full_development_first_two_days(trial):
    dataset="a_hupan"
    folder=OUT/"experiments"/trial
    if not (folder/dataset/"menu.pkl.gz").exists():pytest.skip("Development fixture not yet prepared")
    config=json.loads((folder/"record.json").read_text())["config"]
    data=OUT/("development_"+dataset)
    history=pd.read_pickle(data/"warmup.pkl")
    actual=pd.read_pickle(data/"actual.pkl")
    attributes=pd.read_pickle(data/"attributes.pkl")
    runtime=StructuralRuntime(history,attributes,dataset,config)
    expected=pd.read_pickle(folder/dataset/"menu.pkl.gz")
    for date in sorted(actual.date.unique())[:2]:
        today=actual.loc[actual.date.eq(date)]
        contexts=policy.build_contexts(history,today,attributes)
        menu,audit=runtime.recommend(contexts)
        pd.testing.assert_frame_equal(menu.reset_index(drop=True),expected.loc[expected.date.eq(date)].reset_index(drop=True),check_exact=False,rtol=1e-12,atol=1e-8)
        assert audit["all_requested_sessions_generated"]
        assert not audit["baseline_stage_trace"].empty
        with pytest.raises(ValueError,match="Observe"):
            runtime.recommend(contexts)
        runtime.observe(today)
        history=pd.concat([history,today],ignore_index=True)
