"""Development-only stopping/selection audit before any new locked retest."""
import json
import pandas as pd
from allocation_prepare import OUT,ROOT,load_baseline
from targeted_search import dump
import policy


def main():
    assert not (OUT/"frozen").exists(), "Selection must precede freeze"
    baseline=json.loads((OUT/"experiments/baseline_verified_v2/record.json").read_text())
    originals={"a":"relative_joint_final","f":"residual_tight_bounds"}
    final={"a":"final_verified_a","f":"final_verified_f"}
    tail=["joint_median_audit","joint_weather_audit","joint_reliability_audit"]
    records=[json.loads(p.read_text()) for p in (OUT/"experiments").glob("*/record.json")]
    assert not any(r["status"]=="running" for r in records)
    reviews=[]
    for r in records:
        item={"name":r["name"],"status":r["status"],"hypothesis":r["hypothesis"],"failure":r.get("failure"),"source_hashes":r["source_hashes"]}
        for d in ["a","f"]:
            if d not in r:
                continue
            m,b=r[d],baseline[d]
            reasons=[]
            if r["status"]!="complete": reasons.append("未完成完整实验，不参与选择")
            if m.get("new_min_batch_violations",0) or m["max_session_sum_error"]>1e-7: reasons.append("新增约束违规")
            if m["net_benefit"]<b["net_benefit"]*.99: reasons.append("净收益下降超过1%风险门槛")
            if m["hit_qty_wmape"]>b["hit_qty_wmape"]*1.05: reasons.append("命中WMAPE增加超过5%相对风险门槛")
            if m["nonhit_pred_qty_share"]>b["nonhit_pred_qty_share"]+.005: reasons.append("未命中分配份额增加超过0.5个百分点")
            item[d]={"eligible":not reasons,"risk_reasons":reasons,"hit_qty_mape":m["hit_qty_mape"],"metrics":m}
        reviews.append(item)
    for d in ["a","f"]:
        a=pd.read_pickle(OUT/"experiments"/originals[d]/(d+"_menu.pkl"))
        b=pd.read_pickle(OUT/"experiments"/final[d]/(d+"_menu.pkl"))
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        selected=json.loads((OUT/"experiments"/final[d]/"record.json").read_text())[d]
        eligible=[r[d]["hit_qty_mape"] for r in reviews if d in r and r[d]["eligible"]]
        assert selected["hit_qty_mape"]<=min(eligible)+1e-10
        for name in tail:
            r=next(r for r in reviews if r["name"]==name)
            assert not r[d]["eligible"] or r[d]["hit_qty_mape"]>=selected["hit_qty_mape"]-.25
    _,_,web=load_baseline()
    audit={"status":"development_search_stopped_and_verified","created_at":pd.Timestamp.now().isoformat(),
        "locked_2026_sales_used_for_selection":False,"selected_originals":originals,"selected_final":final,
        "config_simplification_exact_menu_match":True,"completed_runs":sum(r["status"]=="complete" for r in records),
        "failed_runs":sum(r["status"]=="failed" for r in records),"no_fixed_trial_count_limit":True,
        "selection_rule":"通过约束与量化风险门槛后，分别最小化命中MAPE；总量MAPE不变。门槛于开发选择审计固定，非预注册临床阈值，且不表示日期类型无退化。",
        "risk_gates":{"net_relative_loss_max":.01,"hit_wmape_relative_increase_max":.05,"nonhit_share_absolute_increase_max":.005},
        "stop_reason":"已探索供应条件统计、近期收缩、联合日期/天气、稳健分位、不同投影、可靠性和滚动残差；最佳刷新后最后三种不同机制均无合格的至少0.25个百分点额外改善。30%未达到，收益与日期退化已披露，不继续重复等价实验。",
        "final_distinct_directions":tail,"material_improvement_pp":.25,
        "unresolved":["A开发缺少寒假与重叠样本","F四个固定总量与最低批量数学冲突","F开发8月13日午餐实际总量0.26，固定预测12664.72不能由分配修复","A命中WMAPE上升","F净收益与普通日期/重叠状态部分退化","两食堂开发均未达到30%"],
        "web_default_switched":False,"web_model_version":web["model_version"],"reviews":reviews}
    dump(OUT/"selection_stop_audit.json",audit)
    print("selection/stop audit verified",final)


if __name__=="__main__": main()
