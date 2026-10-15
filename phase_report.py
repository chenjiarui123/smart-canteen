"""Build reproducible full evidence tables and Chinese delivery report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from phase_data import ROOT,OUT,DATASETS,verify
from phase_evaluate import fold_names
from demand_evaluate import auc
from targeted_search import dump


def table(frame,columns):
    f=frame.reindex(columns=columns)
    def fmt(x):
        if isinstance(x,(float,np.floating)):return "无样本" if not np.isfinite(x) else f"{x:,.4f}"
        return str(x).replace("|","／").replace("\n"," ")
    return "| "+" | ".join(columns)+" |\n|"+"|".join(["---"]*len(columns))+"|\n"+"\n".join("| "+" | ".join(fmt(v) for v in row)+" |" for row in f.itertuples(index=False,name=None))+"\n"


def collect():
    rows=[];segments=[];raw=[];risk=[];baselines=[];comparisons=[]
    for ds in DATASETS:
        folder=OUT/("development_"+ds);m=json.loads((folder/"full_metrics.json").read_text());baselines.append(dict(dataset=ds,**m))
        frame=pd.read_csv(folder/"full_segments.csv");frame["dataset"]=ds;frame["trial"]="baseline";segments.append(frame)
    for record_path in sorted((OUT/"experiments").glob("*/record.json")):
        r=json.loads(record_path.read_text())
        for ds,v in r["datasets"].items():
            sub=record_path.parent/ds;m=v.get("metrics",{});rm=v.get("raw",{})
            rows.append(dict(dataset=ds,trial=r["name"],kind=r["kind"],status=v.get("status",r["status"]),seconds=r.get("seconds"),
                eligible=v.get("gate",{}).get("eligible",False),reasons="; ".join(v.get("gate",{}).get("reasons",rm.get("precoordination_gate",{}).get("reasons",[]))),
                **m,**{k:value for k,value in rm.items() if not isinstance(value,dict)}))
            if (sub/"segments.csv").exists():
                f=pd.read_csv(sub/"segments.csv");f["dataset"]=ds;f["trial"]=r["name"];segments.append(f)
            if (sub/"raw_segments.csv").exists():
                f=pd.read_csv(sub/"raw_segments.csv");f["dataset"]=ds;f["trial"]=r["name"];raw.append(f)
            if (sub/"pre_post_detail.pkl.gz").exists():
                detail=pd.read_pickle(sub/"pre_post_detail.pkl.gz");obs=detail.loc[detail.actual_qty.notna() & detail.sample_count.ge(3) & detail.own_drop_threshold.notna()]
                for dim in ["overall","month","餐次","date_type","sample_group","frequency_group"]:
                    for name,g in ([("all",obs)] if dim=="overall" else obs.groupby(dim)):
                        y=g.own_drop_observed.to_numpy(bool);p=g.drop_risk.to_numpy(float);width=g.upper80-g.lower80
                        risk.append(dict(dataset=ds,trial=r["name"],dimension=dim,segment=str(name),samples=len(g),own_drop_auc=auc(y,p),
                            prior_auc=auc(y,g.prior_own_drop_rate),brier=float(np.mean((p-y)**2)),prior_brier=float(np.mean((g.prior_own_drop_rate-y)**2)),
                            event_rate=float(y.mean()),coverage=float(((g.actual_qty>=g.lower80)&(g.actual_qty<=g.upper80)).mean()),
                            mean_width=float(width.mean()),interval_score=float((width+10*(g.lower80-g.actual_qty).clip(lower=0)+10*(g.actual_qty-g.upper80).clip(lower=0)).mean())))
    all_metrics=pd.DataFrame(rows);all_segments=pd.concat(segments,ignore_index=True);all_raw=pd.concat(raw,ignore_index=True)
    for name,f in [("all_experiment_metrics",all_metrics),("all_segments",all_segments),("all_raw_segments",all_raw),("risk_interval_segments",pd.DataFrame(risk))]:f.to_csv(OUT/(name+".csv"),index=False)
    for ds in DATASETS:
        suffix="a" if ds=="a_hupan" else "f"
        for full,reduced,meaning in [("phase_joint7","phase_direct7","remove count"),("phase_joint7","phase_count7","remove volume"),("own_drop","own_relative","remove own-drop states"),
            ("own_drop","own_drop_no_phase","remove phase distances"),("own_drop","own_drop_hits","all supplied versus historical hits"),("own_drop_hierarchy","own_drop","remove category layer"),("own_drop_hierarchy","own_hierarchy","remove own-drop predictor")]:
            p=all_metrics.loc[all_metrics.dataset.eq(ds)].set_index("trial")
            if full+"_"+suffix in p.index and reduced+"_"+suffix in p.index:
                a,b=p.loc[full+"_"+suffix],p.loc[reduced+"_"+suffix]
                comparisons.append(dict(dataset=ds,full=full,reduced=reduced,meaning=meaning,full_total=a.total_mape,reduced_total=b.total_mape,full_hit=a.hit_qty_mape,reduced_hit=b.hit_qty_mape,full_net=a.net_benefit,reduced_net=b.net_benefit))
    pd.DataFrame(comparisons).to_csv(OUT/"combinations_ablations.csv",index=False)
    return pd.DataFrame(baselines),all_metrics,all_segments,all_raw,pd.DataFrame(risk),pd.DataFrame(comparisons)


def report():
    verify();baseline,metrics,segments,raw,risk,ablations=collect()
    frozen=json.loads((OUT/"frozen/selection.json").read_text()) if (OUT/"frozen/selection.json").exists() else None
    lines=["# 米饭合并后：阶段需求、单菜自身突降与类别分配实验报告", "",
        "## 当前结论", "",
        "两食堂均保留基线，网页默认策略未改变。没有候选通过事前门槛；不宣称实现30%目标。" if frozen else "开发实验进行中；本报告是阶段性汇总，不能视为最终冻结或复测交付。", "",
        "本轮使用当前米饭合并数据重新生成基线，不复用合并前预测。独立模块包含日历阶段、绝对整餐需求与菜数、相对本菜典型销量的突降状态、类别需求及风险协调入口。实现与效果分开判断。", "",
        "## 基线、数据与时间纪律", "",
        "网页版本 targeted_repeat_holiday_20260911，runtime_history_end_inactivity=True。A为2025-08-05—12-31，F为2025-01-15—12-31，各自14个日历日预热。每个预测日只使用此前销售，同一天所有餐次完成预测后再观察。没有补未供应零值、恢复已删除记录或改原evaluator。数量沿用原混合计量口径，克已在原读取流程换为50克份数，不再次换算，不跨原单位调量。", "",
        table(baseline,["dataset","evaluated_group_count","total_mape","hit_qty_mape","net_benefit","hit_qty_wmape","total_wape","repeat_over_80_sessions","missing_previous_sessions"]),
        "数据与代码快照、哈希在 registration.json、snapshot/；新基线全量在 development_*/baseline_menu.csv、baseline_stages.csv.gz。第一日追踪器输出与网页直接调用逐字段精确相等。", "",
        "开发划分：A早期8月、中期9—10月、后期11—12月；F早期1—6月、中期7—9月、后期10—12月。三个区间均为开发，不是新的独立验证集。完整月份、餐次、日期类型和fold三项指标见 all_segments.csv；不以2026结果选择或停止。", "",
        "## 日历证据与缺口", "",
        "仅用显式日历标记及已知相邻非假期日期确认边界。2025-07-06—07-20日历缺失，因此暑假开始距离为空，7月21日是首个已观测假期标记，不是已确认开假日。已向用户询问校历，确认前不编造。已知寒假、暑假结束及法定假期仍可建模；重叠为联合命名状态，不重复乘下降因子。3、7、14日阶段窗口在实验前登记。详见 calendar_events.json、calendar_phase_audit.csv。", "",
        "## 验收门槛与探索机制", "",
        "phase_protocol.json与phase_trial_design.json在结果前登记。单菜模式固定原最终名单、总量及原单位总量，命中MAPE需至少降0.25个百分点；特殊阶段允许需求/菜数独立改变，总量MAPE需相对改善至少0.05%。net损失≤1%，命中WMAPE与整餐WAPE退化≤5%，未命中预测量占比增加≤0.5个百分点，绝对加权偏差增加≤2个百分点，过量预测销售额增加≤1%。≥100命中分组退化≤max(1个百分点,5%)。运行时也对更细日期类型×餐次及阶段分组保守拒绝，未放宽阈值。", "",
        "阶段模型直接收缩绝对需求/真实出菜数量，不把销量降幅照搬到菜数；趋势版对阶段历史与最近3个相同阶段餐次作绝对值融合。没有统一假期下降比例或强制缩放下限。菜数模式沿用原合格候选、选菜约束和80%限制；原单位空间或集中度不容纳时显式回退，回退与前日菜单不兼容时不发布该餐，保留该餐评估。", "",
        "独立单菜模型学习所有已供应记录相对本菜历史典型销量的变化。突降阈值取预测前本菜历史相对量q10，向同餐次同单位历史收缩；没有历史阈值时不伪造标签。分类不使用绝对菜品规模特征，另与本菜先验概率比较，防止把小菜识别当作突降识别。深度2、叶子至少80、每7天重拟合，无神经网络及大规模参数搜索。", "",
        "类别模型先预测同餐次同单位的类别记录销量，学习当前推荐同类菜数相对历史菜数的影响，再在类别内按独立单菜预测分配，不读取当天真实类别总量。历史真实菜单结构与当前推荐结构差异在 menu_structure_shift.csv。", "",
        "单菜总量协调用原有有界投影作共同对照；只有协调前MAPE、WMAPE、正常销量门槛均通过才允许进入新预计风险协调。不通过的风险实验记录为跳过，不能声称已在真实数据验证其收益。最低批量单独输出为max(需求,最低批量)建议，不抬高主指标需求预测。", "",
        "目标对比是在相同条件分布上采用对数、均值、相对误差与相对/绝对混合决策；树分裂仍按对数平方误差，不能说成树分裂也直接优化了MAPE。本轮没有重新包装上一轮IRLS残差实验。", "",
        "## 全部单机制与组合", "",
        table(metrics,["dataset","trial","status","total_mape","raw_mape","hit_qty_mape","hit_qty_wmape","net_benefit","eligible"]),
        "完整风险指标、运行成本和拒绝原因见 all_experiment_metrics.csv，逐轮假设与代码版本见 experiments/*/record.json及source/。缓存复用仅针对同一已完成独立模型的分布、损失决策或消融，不重复拟合包装成新方向。", "",
        "本轮共34份数据集实验记录：32份完成评分，2份预计风险协调因协调前门槛失败被跳过。A自身突降模型最终MAPE65.1372%，F59.3651%，均差于60.6434%/55.4773%基线。相对损失将原始MAPE降至A57.9712%、F49.9707%，但固定总量后为64.5507%/61.8189%，不能当作有效改善。", "",
        "F阶段趋势模型把整餐MAPE37.4223%降至32.6507%，但WAPE10.7019%升至11.6785%，未达风险门槛。春节核心期MAPE772.7775%→657.9261%（仍很差）；节前44.8819%→33.5104%；校历结束前恢复56.5425%→32.9670%；法定假期结束后16.9591%→26.7346%、校历结束后13.6108%→16.9151%，恢复并不稳定。普通稳定阶段总量完全不变。A阶段直接模型整餐MAPE11.6315%→13.6656%，同样淘汰。", "",
        "A菜数单改动命中MAPE60.1186%看似下降，但10餐因约束不可行未发布，总量MAPE升至13.3973%，覆盖与填充退化；F销量/菜数组合命中MAPE54.1598%也低于基线，却以总量MAPE41.3141%、覆盖与填充损失为代价。困难餐次仍在评估，不能把这类命中集合变化视为单菜预测改善。固定原菜单的配对结果另存fixed_menu_paired_metrics.json。", "",
        "## 组合与逐项移除", "",table(ablations,["dataset","full","reduced","meaning","full_total","reduced_total","full_hit","reduced_hit"]),
        "## 同菜同样本定位、极低记录与突降区分", "",
        "development_*/same_item_stage_diagnosis.csv在每一相邻阶段只比较两者均存在的同菜同真实记录；不同阶段对照行的名单不完全相同，不能跨行当作同一总体。初始候选池大于最终推荐池，最终固定名单的完整初始/日期修正/最终/新原始误差见 all_raw_segments.csv。总量协调含稳定性权重，不单独归因于最低批量或稳定性。", "",
        table(raw.loc[raw.dimension.eq("overall") & raw.trial.str.startswith("own_drop_") & ~raw.trial.str.contains("hierarchy|mixed|relative_loss|mean|no_phase|hits|risk")],["dataset","trial","stage","hit_item_count","hit_qty_mape","hit_qty_wmape","weighted_qty_bias"]),
        "显式零销量单独绝对误差评价；本轮开发未发现显式零记录，不能声称验证了零销量能力。极低真实量全部保留，原MAPE不截断，同时报告WMAPE、绝对误差与p95/p99。", "",
        table(metrics.loc[metrics.trial.isin(["own_drop_a","own_drop_f"])],["dataset","trial","own_drop_auc","prior_own_drop_auc","drop_brier","prior_drop_brier","interval_coverage","interval_width","interval_score"]),
        "概率、80%区间覆盖/宽度/区间分数按月份、餐次、日期类型、频率/样本组见 risk_interval_segments.csv。区间在协调前记录销量尺度评价；协调后只报告点误差，不把原始区间误称为协调后覆盖。高误差真实样本在 experiments/*/*/failure_samples.csv。仅有记录销量无法识别少做、售罄或潜在真实需求。", "",
        "F自身突降AUC0.7256、Brier0.0958，比本菜历史先验AUC0.4943/Brier0.1021好，说明有一定可排序信号，不能说完全不可识别；A AUC0.6476但Brier0.1264比先验0.1233差，概率校准仍不足。信号尚未带来合格点预测。区间相对无条件参考宽度/覆盖见interval_reference_comparison.csv，不能只凭宽区间接近80%覆盖声称准确。类别组合中的风险概率来自原单菜状态分类，区间则随类别分配缩放，二者用途分开。", "",
        "## 各类日期与普通日期稳定性", ""]
    dates=["普通工作日","普通周末","调休工作日","法定节假日","寒假","暑假","寒暑假与法定节假日重叠"]
    selected=segments.loc[segments.trial.eq("baseline") & segments.dimension.eq("date_type")]
    for ds in DATASETS:
        values=selected.loc[selected.dataset.eq(ds)].set_index("segment").reindex(dates).reset_index();values["dataset"]=ds
        lines.extend([f"### {ds}：最终保留基线，以下也是冻结开发结果", "",table(values,["segment","session_count","total_mape","hit_qty_mape","net_benefit","pred_count_mean","actual_count_mean","pred_qty_sum","actual_qty_sum"])])
    lines.extend(["普通稳定日期按每个实验窗口固定原最终整餐及单位总量；阶段定义不同，‘普通日期’含靠近节前/恢复的日期，不能与‘普通稳定阶段’混为一谈。菜数实验如因不可行未发布导致普通总量变化，会额外拒绝，不能算降低预测有效。最终保留基线，因此所有日期类型最终菜数、总量和三个主指标相对新基线变化均为0；探索候选的变化见 all_segments.csv及各餐session_forecasts.csv。", "",
        "holiday_count_volume_changes.csv逐日期类型/阶段列出每个销量/菜数候选的菜数与总量变动百分比、真实量和高低估。chronological_selection_stability.csv列出早中期选择区间与后期稳定区间；fixed_menu_integrity_audit.json逐实验验证全部元数据、逐餐原单位总量和训练截止日。", "",
        "## 冻结、既有区间复测与停止依据", ""])
    retest=[]
    for ds in DATASETS:
        p=OUT/"retest"/ds/"record.json"
        if p.exists():
            r=json.loads(p.read_text())
            for interval,m in r.get("results",{}).items():retest.append(dict(dataset=ds,interval=interval,status=r["status"],**m))
    if retest:
        rt=pd.DataFrame(retest);rt.to_csv(OUT/"frozen_retest_summary.csv",index=False)
        lines.append(table(rt,["dataset","interval","total_mape","hit_qty_mape","net_benefit","hit_qty_wmape","total_wape","repeat_over_80_sessions"]))
    else:lines.append("既有2026区间复测尚未全部完成，不提供未执行的测试结论。")
    lines.extend(["A：1—3月为原验证区间，4月1日—6月10日为原测试区间；F：1月1日—6月10日整体是原测试区间，Q1/Q2只作诊断。上述区间已被查看过，明确称既有区间复测，不声称独立新测试。冻结后每个数据集完整滚动一次，不能再根据结果修改策略。", "",
        "停止依据：阶段需求、单菜自身突降、类别分配三个实质不同方向以及销量/菜数组合、类别/突降组合和逐项移除均未通过事前门槛。没有有效的协调前候选时不堆叠复杂分配，不无限重复等价调参。不是因为达到某个固定实验次数。没有新策略刷新最佳，基线就是最终简化方案。", "",
        "## 需要补充的业务信息", "",
        "需要可核实学校校历（尤其2025暑假开始）、各餐实际开餐安排/就餐人数与返校比例、逐菜备餐量与补菜时间、售罄时刻、缺货/停供、剩余量及退货、菜品销售时段、促销和菜单替换记录。否则无法区分需求突然降低与临时少做、早售罄等供应过程；不能给没有观测依据的解释或承诺30%必达。", "",
        "## 代码、测试及复现", "",
        "独立入口：phase_baseline.py、phase_features.py、phase_item_experiments.py、phase_session_experiments.py；策略冻结与复测：phase_release.py。命令与可重复执行边界见 REPRODUCE.md。逐菜明细及分布在每轮pre_post_detail.csv.gz、distribution.npz；逐餐及不发布/回退原因在sessions.csv、allocation_audit.csv、menu_audit.json；原evaluator及网页默认文件保持注册哈希不变。", "",
        "phase_runtime.predict_day是最终保留版本的独立入口，直接接受已锁定基线输出，逐字段保持原预测并单列备餐建议；明确拒绝未通过门槛的研究配置，避免将无状态逐日重拟合误当作每7日重拟合的实验策略。研究模型仍可通过完整滚动入口复现。各轮complete_demand_preparation.csv.gz补齐所有菜的需求/备餐字段，原始实验文件不覆盖，销量及评价不改变；导出清单记录源哈希。", "",
        "旧数据快照绑定的4个demand_runtime测试因米饭合并触发原指纹拒绝，记录在test_legacy_fixture_incident.json，未绕过或伪装全绿。当前独立模块与米饭身份迁移的测试单独记录。", ""])
    (OUT/"阶段与单菜结构实验报告.md").write_text("\n".join(lines),encoding="utf-8")
    print(OUT/"阶段与单菜结构实验报告.md",flush=True)


if __name__=="__main__":report()
