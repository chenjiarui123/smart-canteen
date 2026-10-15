"""Post-freeze reporting-only fix for two diagnostic name collisions.

phase_evaluate augmented targeted metrics with hit_stats, overwriting generic
overprediction_qty/underprediction_qty with HIT amounts. Preserve all original
records; derive explicitly named total and hit amounts from saved predictions.
No primary metric, gate, forecast, data, evaluator or frozen source is changed.
"""
import json
import numpy as np
import pandas as pd
from phase_data import OUT,DATASETS
from phase_release import verify_freeze
from targeted_search import dump


def run():
    verify_freeze();rows=[]
    def clarify(dataset,name,interval,metrics,sessions,source):
        q=sessions.pred_total-sessions.actual_total
        wape=100*q.abs().sum()/sessions.actual_total.sum()
        np.testing.assert_allclose(wape,metrics["total_wape"],rtol=1e-12,atol=1e-8)
        rows.append(dict(dataset=dataset,trial=name,interval=interval,total_overprediction_qty=float(q.clip(lower=0).sum()),
            total_underprediction_qty=float((-q).clip(lower=0).sum()),total_weighted_qty_bias=float(100*q.sum()/sessions.actual_total.sum()),
            total_qty_bias=float(q.sum()),hit_overprediction_qty=metrics["overprediction_qty"],hit_underprediction_qty=metrics["underprediction_qty"],
            total_mape=metrics["total_mape"],hit_qty_mape=metrics["hit_qty_mape"],net_benefit=metrics["net_benefit"],total_wape=metrics["total_wape"],source=source))
    for ds in DATASETS:
        base=OUT/("development_"+ds)
        clarify(ds,"baseline","development",json.loads((base/"full_metrics.json").read_text()),pd.read_csv(base/"full_sessions.csv"),str(base.relative_to(OUT)))
        for p in (OUT/"experiments").glob("*/"+ds+"/metrics.json"):
            clarify(ds,p.parent.parent.name,"development",json.loads(p.read_text()),pd.read_csv(p.parent/"sessions.csv"),str(p.relative_to(OUT)))
        retest=OUT/"retest"/ds
        for interval,m in json.loads((retest/"metrics.json").read_text()).items():
            clarify(ds,"retained_baseline",interval,m,pd.read_csv(retest/(interval+"_sessions.csv")),str(retest.relative_to(OUT)))
    values=pd.DataFrame(rows);values.to_csv(OUT/"clarified_over_under_metrics.csv",index=False)
    original=pd.read_csv(OUT/"all_experiment_metrics.csv")
    columns=["dataset","trial","total_overprediction_qty","total_underprediction_qty","hit_overprediction_qty","hit_underprediction_qty","total_weighted_qty_bias"]
    merged=original.drop(columns=["overprediction_qty","underprediction_qty"],errors="ignore").merge(values.loc[values.interval.eq("development"),columns],on=["dataset","trial"],how="left",validate="one_to_one")
    merged.to_csv(OUT/"clarified_experiment_metrics.csv",index=False)
    dump(OUT/"diagnostic_field_clarification.json",dict(discovered_after_freeze=True,kind="reporting_only_name_collision",forecast_or_gate_changed=False,original_records_preserved=True,
        cause="phase_evaluate hit_stats update overwrote generic overprediction_qty/underprediction_qty from session diagnostics; primary metrics and risk gates do not use these collided keys",
        canonical_total="sum of positive/negative (saved session pred_total - actual_total)",canonical_hit="original generic values from hit_stats, now explicitly prefixed hit_",
        total_wape_crosscheck="all rows match original saved total_wape within rtol1e-12 atol1e-8",tables=["clarified_over_under_metrics.csv","clarified_experiment_metrics.csv"]))
    p=OUT/"阶段与单菜结构实验报告.md";text=p.read_text();marker="\n## 辅助高低估字段核验说明\n"
    if marker in text:text=text.split(marker)[0]
    text+=marker+"\n最终核对发现辅助统计的同名字段覆盖：原metrics中overprediction_qty/underprediction_qty被命中单菜高低估量覆盖。原日志与冻结源码保留，不静默重写；已根据保存的逐餐结果补充total_overprediction_qty/total_underprediction_qty，并把命中项量明确命名为hit_overprediction_qty/hit_underprediction_qty。请使用clarified_experiment_metrics.csv和clarified_over_under_metrics.csv读取这四项。所有整餐WAPE与保存逐餐结果重新核对一致；三项主要指标、验收决定、预测与2026复测次数均不改变。这是报告口径修正，不是测试后调参。\n"
    p.write_text(text,encoding="utf-8");print("Diagnostic fields clarified without changing frozen outcomes",len(values),flush=True)


if __name__=="__main__":run()
