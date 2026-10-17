"""Evidence-only allocation report. Never selects parameters from retest data."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import policy
from allocation_prepare import ROOT,OUT
from targeted_search import dump
from targeted_report import table
from evaluator import prepare_actual


def constraints_evidence(folder,baseline,features,actual):
    m=baseline.copy()
    m["_dish_norm"]=m["菜品名称"].map(policy.normalize_name)
    m=m.merge(features[[*policy.GROUP_KEYS,"_dish_norm","q99"]],on=policy.GROUP_KEYS+["_dish_norm"],validate="one_to_one")
    m=m.merge(prepare_actual(actual)[[*policy.GROUP_KEYS,"_dish_norm","actual_qty"]],on=policy.GROUP_KEYS+["_dish_norm"],how="left")
    evidence=[]
    for key,g in m.groupby(policy.GROUP_KEYS):
        hit=g.actual_qty.gt(0)
        h=g.loc[hit]
        if h.empty:
            continue
        b=g.pred_qty.to_numpy(float)
        minimum=g.min_batch_qty.to_numpy(float)
        values=h.actual_qty.to_numpy(float)
        lower=h.min_batch_qty.to_numpy(float)
        upper=np.maximum(h.pred_qty.to_numpy(float),np.maximum(lower,np.minimum(h.q99.to_numpy(float)*2,h.pred_qty.to_numpy(float)*2)))
        floor=float(100*(np.maximum(lower-values,0)/values).mean())
        entry=dict(zip(policy.GROUP_KEYS,key))
        entry.update(baseline_total=float(b.sum()),actual_selected_total=float(h.actual_qty.sum()),hit_count=len(h),
                     recommended_count=len(g),minimum_only_mape_lower_bound=floor,
                     minimum_total_shortfall=max(0.,float(minimum.sum()-b.sum())))
        if entry["minimum_total_shortfall"]>1e-8:
            entry.update(oracle_status="infeasible_minimum",oracle_mape=None)
        else:
            # Diagnostic only: nonhits cannot absorb any extra baseline quantity.
            # This oracle knows actuals, ignores concentration, and is NEVER used
            # by prediction or search. Its bound is conditional on this envelope.
            q=np.clip(values,lower,upper)
            hit_total_min=float(b.sum()-g.loc[~hit,"pred_qty"].sum())
            hit_total_max=float(b.sum()-g.loc[~hit,"min_batch_qty"].sum())
            target=np.clip(q.sum(),max(hit_total_min,lower.sum()),min(hit_total_max,upper.sum()))
            if hit_total_min>upper.sum()+1e-8 or hit_total_max<lower.sum()-1e-8:
                entry.update(oracle_status="infeasible_envelope",oracle_mape=None)
            else:
                delta=target-q.sum()
                for j in np.argsort(-values):
                    step=min(abs(delta),(upper[j]-q[j]) if delta>0 else (q[j]-lower[j]))
                    q[j]+=np.sign(delta)*step
                    delta-=np.sign(delta)*step
                    if abs(delta)<1e-8:
                        break
                entry.update(oracle_status="diagnostic_actual_known_not_predictor",oracle_mape=float(100*(np.abs(q-values)/values).mean()))
        evidence.append(entry)
    result=pd.DataFrame(evidence)
    result.to_csv(folder/"fixed_total_constraint_evidence.csv",index=False)
    return result


def main():
    manifest=json.loads((OUT/"frozen/allocation_manifest.json").read_text())
    selected=manifest["development_runs"]
    comparisons=[]
    records=[]
    for p in sorted((OUT/"experiments").glob("*/record.json")):
        r=json.loads(p.read_text())
        failure_notes={"baseline":"F历史准备尚未完成，读取保护中止；A部分结果保留", "baseline_verified":"F基线4餐总量低于最低批量；原检查中止，后续显式保留不可行状态", "offered_mean":"F基线最低批量冲突导致中止；A部分结果保留"}
        row={"实验":r["name"],"状态":r["status"],"假设":r["hypothesis"],"失败原因":failure_notes.get(r["name"],r.get("failure",""))}
        for d in ["a","f"]:
            if d in r:
                row.update({d+"_MAPE":r[d]["hit_qty_mape"],d+"_net":r[d]["net_benefit"],d+"_非命中分配":r[d]["nonhit_pred_qty_share"]})
        records.append(row)
    pd.DataFrame(records).to_csv(OUT/"experiment_comparison.csv",index=False)
    achieved_groups=[]
    for d in ["a","f"]:
        oldfolder=OUT/"experiments/baseline_verified_v2"
        newfolder=OUT/"experiments"/selected[d]
        for label,folder,prefix in [("旧",oldfolder,d),("新",newfolder,d)]:
            m=json.loads((folder/(prefix+"_metrics.json")).read_text())
            comparisons.append(dict(食堂=d,区间="2025开发",策略=label,**m))
        retest=OUT/("retest_"+d)
        lock=json.loads((retest/"retest_lock.json").read_text())
        assert lock["status"]=="complete"
        intervals=[("2026全段复测","")]
        if d=="a":
            intervals += [("2026一季度验证复测","_validation_retest"),("2026四月至六月测试复测","_test_retest")]
        for interval,suffix in intervals:
            for label,prefix in [("旧","baseline"),("新","new")]:
                m=json.loads((retest/(prefix+suffix+"_metrics.json")).read_text())
                comparisons.append(dict(食堂=d,区间=interval,策略=label,**m))
    comparison=pd.DataFrame(comparisons)
    comparison.to_csv(OUT/"new_old_comparison.csv",index=False)
    report=["# 固定菜单与整餐总量的单菜分配优化结果","", "30%目标未达到。新分配模块独立交付，改善只发生在部分指标和区间，未切换网页默认策略。", "",
        "## 主要结果", "",table(comparison,["食堂","区间","策略","hit_qty_mape","total_mape","net_benefit","hit_qty_wmape","hit_qty_median_ape"]),"",
        "A对应湖畔，F对应基础园。MAPE/WMAPE以百分数表示，net_benefit保持原项目公式及金额口径。各区间分别判断是否低于30%，不合并食堂掩盖失败。", "",
        "## 复现与误差来源", "",
        "A开发区间2025-08-04—12-31，450餐次；F开发区间2025-01-15—12-31，915餐次。均沿用14天预热和逐日滚动，三餐使用同一日初历史。A开发缺少寒假/重叠，F开发覆盖全部七类日期。2026实际数据在冻结后复测，不参与参数选择；原始数据、菜品库、评价代码和网页默认模型未修改。",
        "命中MAPE严格沿用diagnostics.hit_quantity_diagnostics：按日期、食堂、餐次、规范菜名匹配，仅保留actual_qty>0的命中记录，APE对记录等权平均。每次名单相同，命中集合不变；极小销量和高误差记录没有删除。原策略本身已经使用供应条件几何统计和稳定性加权，不是简单的缺失补零预测。",
        "新历史特征按已观测供应记录统计；缺失日期不补零，显式零单独记录，正销量条件估计只使用正观测。最新单位标签不一致的历史不混用；乱码单位菜品锁回基线，不猜测单位换算。销量/订单/销售额仍沿用项目模型计量以及基线expected_*_per_qty比例。近期与上下文估计采用样本量收缩，日期状态联合估计是绝对量替换而不是连续乘多个因子。",
        "整餐统一校准并非唯一误差来源：极端低销量、少样本、长期间隔及固定总量均有限制。校准前后误差与详细分组见各实验*_metrics.json和*_hit_segments.csv。", "",
        "F开发2025-08-13午餐实际总量只有0.26，冻结预测总量12664.72；单餐总量MAPE约487万%，保留该餐使整体原口径总量MAPE达到5360.79%。此值不是单菜分配改变造成。可能存在停供或销售记录不完整，但未核实前不能擅自删改数据；对应F开发总量WAPE为10.79%，MAE为1234.02。", "",
        "## 绝对误差与极端误差诊断", "",table(comparison,["食堂","区间","策略","total_wape","total_mae","hit_qty_mae","ape_p95","ape_p99","ape_max"]),"",
        "## 策略与约束", "",
        "所选配置见frozen/allocation_config.json。仅修改pred_qty、pred_orders、pred_rev，名单、菜数、最低批量字段、总量字段及其余元数据逐字段一致。分配是固定和的有界二次投影，历史上界、相对变化上界、HHI和最大菜品份额共同防止集中转移；上界不可行时显式回退而非静默扩上限。预测函数不接收目标日实际销量或命中标记。历史残差若启用，仅在当天所有餐次预测结束后更新。",
        "默认安全界：单菜不超过基线2倍与历史q99两倍的较小值，不低于基线25%及适用最低批量；HHI不超过基线1.15倍，最大单菜份额增量不超过3个百分点。历史上界与浓度限制冲突时回退基线并报告。回退不代表合理历史范围已经满足。", "",
        table(comparison,["食堂","区间","策略","max_session_sum_error","new_min_batch_violations","inherited_minimum_infeasible_sessions","bounds_fallback_sessions","nonhit_pred_qty_share","max_hhi_ratio","max_top_share_increase"]),"",
        "F开发存在4个基线固有最低批量冲突：这些餐次固定总量甚至低于唯一菜品的最低批量。新模块返回明确不可行状态并保留基线值以保证评估完整；这些结果不能作为满足最低批量的自动备餐方案发布，且不计作新模块合规。名单与总量不变使相邻日重复率、缺少前日对照和原策略的菜单不可行状态完全不变。", ""]
    for d in ["a","f"]:
        for phase in ["development","retest"]:
            folder=OUT/(phase+"_"+d)
            base=pd.read_csv(folder/"baseline_menu.csv",parse_dates=["date"])
            features=pd.read_pickle(folder/"features.pkl")
            actual=pd.read_pickle(folder/"actual.pkl")
            evidence=constraints_evidence(folder,base,features,actual)
            report += [f"## {d} {phase}：固定总量限制的具体证据", "",
                "以下oracle仅为事后诊断：知道真实销量、不许向未命中菜增加基线份额、保留最低批量和放宽的历史范围，且忽略浓度限制。它不是预测器，也不是30%整体目标不可能的证明。minimum_only界只使用最低批量与真实销量，亦不参与选择。", "",
                table(evidence.sort_values("minimum_only_mape_lower_bound",ascending=False).head(5),["date","餐次","baseline_total","actual_selected_total","hit_count","minimum_only_mape_lower_bound","oracle_mape","minimum_total_shortfall"]),""]
        old=OUT/"experiments/baseline_verified_v2"
        new=OUT/"experiments"/selected[d]
        for phase,of,nf,op,nprefix in [("开发",old,new,d,d),("冻结复测",OUT/("retest_"+d),OUT/("retest_"+d),"baseline","new")]:
            a=pd.read_csv(of/(op+"_session_segments.csv"))
            b=pd.read_csv(nf/(nprefix+"_session_segments.csv"))
            merged=a.merge(b,on=["dimension","segment"],suffixes=("_old","_new"))
            merged.to_csv(OUT/(d+"_"+phase+"_session_comparison.csv"),index=False)
            for r in merged.loc[merged.hit_qty_mape_new.lt(30) & merged.hit_item_count_new.gt(0) & merged.dimension.isin(["month","餐次","date_type"])].to_dict("records"):
                achieved_groups.append({"食堂":d,"阶段":phase,"维度":r["dimension"],"分组":r["segment"],"命中记录数":r["hit_item_count_new"],"新MAPE":r["hit_qty_mape_new"]})
            report += [f"## {d} {phase}日期类型与普通/假期", "",table(merged.loc[merged.dimension.isin(["date_type","regular_holiday"])],["dimension","segment","hit_qty_mape_old","hit_qty_mape_new","total_mape_old","total_mape_new","net_benefit_old","net_benefit_new"]),""]
            hs1=pd.read_csv(of/(op+"_hit_segments.csv"))
            hs2=pd.read_csv(nf/(nprefix+"_hit_segments.csv"))
            hs1.merge(hs2,on=["dimension","segment"],suffixes=("_old","_new")).to_csv(OUT/(d+"_"+phase+"_hit_comparison.csv"),index=False)
            hit_rows=pd.read_pickle(nf/(nprefix+"_hits.pkl"))
            calibration=[]
            for dimension in ["month","餐次","date_type","historical_volume","frequency","samples","recent_samples"]:
                for name,g in hit_rows.groupby(dimension,observed=True):
                    calibration.append({"dimension":dimension,"segment":str(name),"hit_count":len(g),
                        "raw_mape":float(100*((g.uncalibrated_qty-g.actual_qty).abs()/g.actual_qty).mean()),
                        "baseline_calibrated_mape":float(100*((g.baseline_qty-g.actual_qty).abs()/g.actual_qty).mean()),
                        "new_allocated_mape":float(100*((g.pred_qty-g.actual_qty).abs()/g.actual_qty).mean())})
            pd.DataFrame(calibration).to_csv(OUT/(d+"_"+phase+"_calibration_diagnostics.csv"),index=False)
    report += ["## 30%达标范围", "", "两个食堂及整体区间必须分别看主要结果表，不能用下面的局部分组替代整段目标。低样本分组只作描述，分组互有重叠。", ""]
    if achieved_groups:
        report += [table(pd.DataFrame(achieved_groups),["食堂","阶段","维度","分组","命中记录数","新MAPE"]),""]
        pd.DataFrame(achieved_groups).to_csv(OUT/"below_30_groups.csv",index=False)
    else:
        report += ["所报告的月份、餐次与日期类型分组中，没有新策略MAPE低于30%的有效命中分组。",""]
    report += ["## 所有实验（含失败）", "",table(pd.DataFrame(records),["实验","状态","a_MAPE","f_MAPE","a_net","f_net","失败原因"]), "",
        "停止审计、候选选择与已知权衡见selection_stop_audit.json；逐轮假设、参数、完整指标、源码快照及哈希见experiments/*/record.json。验证见verification.json。凡未达30%或假期/净收益退化的区间，均不宣称全面改善。", "",
        "## 使用与回退", "",
        "推荐使用allocation_runtime.AllocationRuntime：以原历史与a/f食堂配置初始化，recommend(contexts)一次预测当天各餐，结束后observe(actual)才更新历史；封装器自动捕获冻结基线的校准前权重和稳定性。它与缓存滚动实验有等价性测试。也可调用allocation_policy.allocate_session，但须完整提供历史特征及隐藏基线权重。返回审计allocation_feasible=false时，不应作为新的合规自动备餐计划直接发布。无需修改菜单生成代码，停止调用分配模块即可回退。网页当前仍使用原冻结targeted策略。"]
    (OUT/"单菜分配优化对比报告.md").write_text("\n".join(report),encoding="utf-8")
    print(OUT/"单菜分配优化对比报告.md")


if __name__=="__main__":
    main()
