"""Generate a development-only comparison report from frozen evidence."""
import json
import shlex
import numpy as np
import pandas as pd
from demand_data import ROOT,OUT,DATASETS,verify
from structural_report import table
from targeted_search import dump


def main():
    verify()
    selection=json.loads((OUT/"selection_and_stop.json").read_text())
    audit=json.loads((OUT/"final_verification.json").read_text())
    metrics=pd.read_csv(OUT/"all_experiment_metrics.csv")
    segments=pd.read_csv(OUT/"all_session_segments.csv")
    records={p.parent.name:json.loads(p.read_text()) for p in (OUT/"experiments").glob("*/record.json")}
    sections=["# 独立单菜记录销量模型：开发回测与冻结报告\n",
        "## 结论\n",
        "没有新预测通过事前固定的协调前及最终风险门槛。A／湖畔、F／基础园均保留原部署销量预测，三项主要指标变化为0，未达到命中单菜MAPE低于30%的目标。新模块实现完成不等于预测效果改善，未切换网页默认策略。\n",
        "本轮确实完成了独立模型、全部供应记录训练、条件销量分布、低销量状态分类、不同相对/绝对目标的探索，而不是继续调整旧份额或小幅残差分配。直接相对误差拟合降低了协调前MAPE，却引入明显普遍低估；固定总量后的MAPE反而更差。不能为了发布模型而放宽门槛。\n",
        "独立输出保留预计记录销量，同时增加 `max(需求,最低批量)` 的备餐建议和额外批量风险，销量/订单/销售额不随备餐量抬高。它没有进入网页默认流程，也不计作精度改善。\n",
        "## 基线复现、范围和评价纪律\n",
        "实际部署版本为 `targeted_repeat_holiday_20260911`，包括现有假期修正及80%重复限制。A、F各自从相同14个日历日预热状态完整重跑，与上一轮保存的最终菜单和特征逐字段一致。使用网页实际的 `runtime_history_end_inactivity=True`，没有拿旧allocation实验的不同配置充当基线。\n",
        "当前清理后的 `smart_canteen_web/data/datasets/` 两套销量和菜品库哈希均未改变；没有恢复已删除记录、增删低销量样本、修改标签/单位/菜名或evaluator。开发期A为2025-08-05—12-31（447餐），F为2025-01-15—12-31（914餐）。A没有寒假/重叠假期开发样本，F七类日期均有数据。2026销量和实际菜单不参与本轮选择、调参、停止或验证。\n",
        "第一阶段直接锁定旧策略最终名单及每餐最终总量，并额外锁定每个原售卖单位的总量，阻断影子菜单的间接影响。不跨单位转移、不重复50克换算；原指标采用原项目混合计量口径，不能把合计误解为总重量。未开餐和缺少记录不补零。\n"]
    base=[]
    for ds in DATASETS:
        m=selection["baseline_metrics"][ds]
        fm=json.loads((OUT/ds/"feature_manifest.json").read_text())
        base.append(dict(食堂=ds,评估餐次=m["evaluated_group_count"],全部供应训练记录=fm["training_rows"],显式零销量记录=fm["zero_record_count"],
            总量MAPE=m["total_mape"],命中MAPE=m["hit_qty_mape"],net_benefit=m["net_benefit"],命中WMAPE=m["hit_qty_wmape"],命中率=m["dish_hit_rate"]))
    sections.append(table(pd.DataFrame(base)))
    sections += ["当前开发数据没有显式零销量记录；代码仍保留零值处理与单独绝对误差评价，并用合成测试验证，不能声称已获得零销量事件的真实验证效果。缺少记录绝不自动转为零。所有预测最多使用前一日销售统计；当天所有餐次预测结束后，才构造当天已供应记录的训练样本并加入下一日可用历史。\n",
        "## 预注册门槛与实验设计\n",
        "沿用上一轮：最终命中MAPE至少改善0.25个百分点；net损失≤1%；命中WMAPE相对增加≤5%；未命中预测量占比增加≤0.5个百分点；加权偏差绝对值增加≤2个百分点。分月、餐次、日期类型中≥100个正销量命中的组，MAPE退化≤max(1个百分点,基线5%)；额外检查历史频率、样本量和正常记录量组。\n",
        "新增事前进入条件：协调前同一命中集合的MAPE须比旧日期修正估计低至少0.25个百分点，WMAPE不能相对变差超过5%，正常记录量组不能明显退化。无合格协调前模型，不使用新复杂协调掩盖失败。80%区间覆盖率至少70%，宽度不超过同模型无条件残差参考的120%；同时报告区间分数而非只报覆盖率。\n",
        "模型结构固定为浅层（最大深度2，叶子至少40记录），每7个日历日从此前允许样本重拟合；同餐次×单位少于80训练行时回退独立历史典型值。没有大规模超参搜索。后续条件分布及直接损失模型的设计原因在出结果前分别记入 `conditional_design.json`、`direct_loss_design.json`，没有改变验收门槛。\n",
        "## 同菜、同样本的误差定位\n"]
    stage=pd.read_csv(OUT/"baseline_paired_stage_diagnostics.csv")
    sections.append(table(stage.query("dimension=='overall'").drop(columns=["dimension","segment"])))
    sections += ["每行只比较两个阶段都存在的同一批正销量命中菜；不能把不同阶段名单变动当作单步因果效应。`date_conditioned→total_allocated` 包含稳定性权重及总量协调，不能单独归因于稳定性折扣。上一轮同总量、同菜的稳定性移除对照已显示原折扣有一定收益，因此本轮未预设它是主要问题。最低批量与重复替换的共同样本平均影响也不足以解释全部误差。\n",
        "按极低、低、正常记录量及历史频率/样本量/日期类型/餐次的完整同样本诊断见 `baseline_paired_stage_diagnostics.csv`。分组低阈值为预测前同餐次×单位正销量历史q10，极低阈值q01；目标实际销量只用于事后分组评价，不参与模型分支选择。\n",
        "## 新机制与训练对照\n",
        "1. 独立对数线性模型直接拟合所有已供应记录的销量，不限于旧策略命中项。浅层分段版本拟合独立模型的条件误差，不是围绕旧预测做受限乘子修正。旧预测仅在专门的withold对照中作为可选特征。\n",
        "2. 条件经验分布直接学习‘记录销量／本菜此前历史典型量’的分布。这个单位保持的目标变换不以旧策略预测为中心，也没有旧值附近的变化限制；用于减少新日历状态下的线性外推。\n",
        "3. 低销量两状态模型先学习低记录量概率，再组合低状态与正常状态的销量分布。历史q10阈值逐日形成，训练与预测都不读目标实际销量。\n",
        "4. 对相同历史分布比较log典型值、算术均值、绝对误差中位数、相对误差及相对/历史尺度绝对误差各半的决策。树的分割仍以对数平方误差学习；不能把叶子风险决策说成所有树分裂都优化MAPE。进一步用8步固定IRLS直接拟合相对/混合损失，残差平滑0.01仅用于求解，未截断评估误差。正量的相对权重不裁剪，零量用单独绝对项，决不代入1/0。\n",
        "5. 条件分布风险协调实现了各菜预计误差的分段线性边际成本排序，分单位守恒；集中度不满足时按事前方式与基线混合并再次检查边界，不能声称是带全部集中度条件的精确全局最优。它通过小规模穷举对照测试，但没有协调前模型通过进入门槛，因此真实开发期的新风险协调被明确跳过。\n",
        "### 全部供应与只学历史命中、旧预测特征\n"]
    control_names=["linear_all_log","linear_hits_log","linear_withold_log","conditional_all_log","conditional_hits_log","conditional_withold_log"]
    sections.append(table(metrics[metrics.trial.isin(control_names)][["dataset","trial","raw_mape","raw_wmape","hit_qty_mape","net_benefit"]]))
    sections += ["仅命中对照使用相同全部供应14日预热种子，之后只加入旧策略此前选中且实际供应的记录。全部供应通常优于这种选择性样本，但尚不足以优于原策略。加入旧单菜预测并未稳定挽救独立模型；这轮不能归结为‘新模型只是被旧预测束缚’。\n",
        "### 所有单机制、组合及目标比较\n"]
    sections.append(table(metrics[["dataset","trial","raw_mape","raw_wmape","hit_qty_mape","hit_qty_wmape","net_benefit"]]))
    sections += [f"共{len(records)}份运行记录：包括1份被断言拦截的技术失败、同配置复核，以及1份因协调前门槛失败而跳过新风险分配的记录。失败不删，复核不计作新机制。所有完成实验的总量MAPE与基线相同；表中的最终单菜指标来自原投影方法的对照，不代表通过发布门槛。\n",
        "每轮完整风险指标、假设配置、父分布及源码哈希分别在 `all_experiment_metrics.csv` 与 `experiments/*/record.json`；逐组拒绝原因在 `gate_review.json`。没有把技术实现成功算成预测改善。\n",
        "## 为什么相对目标的MAPE下降仍被淘汰\n"]
    evidence=[]
    for ds in DATASETS:
        raw=pd.read_csv(OUT/"experiments/direct_relative"/ds/"raw_segments.csv")
        evidence.append(raw.query("dimension in ['overall','volume_segment']").assign(dataset=ds)[["dataset","dimension","segment","stage","hit_qty_mape","hit_qty_wmape","weighted_qty_bias"]])
    sections.append(table(pd.concat(evidence)))
    sections += ["直接相对拟合在协调前将A的MAPE从64.7870%降至59.8102%，F从56.1607%降至51.5283%。但A的WMAPE从23.3173%升至38.1702%，加权偏差约−28.07%；F的WMAPE从18.6957%升至27.9872%，偏差约−18.53%。正常记录量MAPE也分别从30.80%升至38.17%、25.65%升至32.00%。\n",
        "这些下降主要来自压低预测后减轻极低实际量的相对误差，而不是对典型销量的更准确识别。固定整餐总量再投影，最终命中MAPE为A68.3380%、F58.6617%，都比基线差。F净收益虽升至约4,349,989.90，也不能越过更高优先级的误差目标。混合绝对目标、低状态组合和消融均未解除这一取舍。\n",
        "## 消融与结构贡献\n"]
    sections.append(table(pd.read_csv(OUT/"ablation_comparisons.csv")))
    sections += ["每个对照明确移除或加入的结构含义；没有把多个配置一起变化冒充单组件消融。没有形成可保留的有效组合。条件分布+低状态+相对/混合决策已做对应去除低状态、去除条件分区和去除损失分量的对照，未靠重复参数微调增加实验次数。\n",
        "## 低销量风险、区间及菜单结构偏移\n"]
    risk=pd.read_csv(OUT/"risk_discrimination_controls.csv")
    selected_names=["linear_all_log","conditional_all_log","conditional_hurdle_log","direct_relative"]
    sections.append(table(risk[risk.trial.isin(selected_names)]))
    sections += ["看似较高的低销量AUC大部分可由‘本来就卖得少的菜’解释：仅按历史典型销量排序，A/F的AUC已约0.8485/0.8707。F的两状态分类器AUC约0.8309，甚至低于该简单对照；没有证据说明它能够可靠识别突发极低销量。风险指标仅在固定菜单中确实供应的记录上评价，未供应项不当作低销量或零。\n"]
    sections.append(table(metrics[metrics.trial.isin(selected_names)][["dataset","trial","interval_coverage","interval_width_mean","interval_width_relative_to_actual","interval_width_ratio","interval_score_mean"]]))
    sections += ["区间为协调前记录销量的80%经验分布区间，报告覆盖、宽度和区间分数；没有通过增大区间来选择方案。41点分位数是分布近似，不是将评估尾部截断。经验残差来自训练样本，覆盖率由之后的逐日输出检验，未声称已做独立校准。协调后的点预测未重新构造联合条件区间，不能把上表当成整餐约束后的校准保证。最终保留原预测，不发布未经通过的新区间。\n"]
    sections.append(table(pd.read_csv(OUT/"training_prediction_menu_shift.csv")))
    sections += ["历史训练用实际供应菜单、预测用策略菜单，类别内数量确有分布差异，例如A午餐对应菜的类别同伴数平均约50.82对46.11；这限制竞争特征的可迁移性。没有用目标真实菜单填补差异。目标日天气和日期允许作为已知输入；假期结束距离没有可靠的事前完整校历接口，因此未从未来销售记录反推，仅使用此前观察到的假期状态持续/结束时间。\n",
        "## 分日期类型与分月份\n",
        "以下以直接相对模型为代表展示失败取舍，完整所有实验在 `all_session_segments.csv`。总量MAPE固定不变；最终采用结果各组均等于原基线，不把表中的候选当作新线上版本。\n"]
    for dimension in ["regular_holiday","date_type","month"]:
        grouped=[]
        for ds in DATASETS:
            b=pd.read_csv(OUT/"final_selected"/ds/"session_segments.csv").query("dimension==@dimension")
            c=segments[(segments.dataset==ds)&(segments.trial=="direct_relative")&(segments.dimension==dimension)]
            j=b.merge(c,on=["dimension","segment"],suffixes=("_baseline","_candidate"))
            j["dataset"]=ds
            grouped.append(j[["dataset","segment","evaluated_group_count_baseline","total_mape_baseline","hit_qty_mape_baseline","hit_qty_mape_candidate","net_benefit_baseline","net_benefit_candidate"]])
        sections.append(table(pd.concat(grouped)))
    sections += ["## 典型失败记录与现有信息不足\n"]
    cases=[]
    for ds in DATASETS:
        d=pd.read_pickle(OUT/"experiments/direct_relative"/ds/"pre_post_detail.pkl.gz")
        h=d.loc[d.actual_qty>0].copy();h["raw_ape"]=(h.raw_prediction-h.actual_qty).abs()/h.actual_qty*100
        cases.append(h.nlargest(4,"raw_ape").assign(dataset=ds)[["dataset","date","餐次","菜品名称","actual_qty","historical_scale","sample_count","old_date_conditioned","raw_prediction","final_prediction","raw_ape"]])
    sections.append(table(pd.concat(cases)))
    sections += ["极低记录量可能来自少量备货、临时停供、较晚出菜、提前售罄、顾客变化或交易记录口径；仅凭销量不能区分这些原因。本轮没有把它们直接认定为错误，没有删样本，也不声称识别了真实潜在需求。线性模型在新日历状态下曾出现历史典型量35—62倍的外推；直接条件分布减轻了这类外推，却仍不能稳定处理低量尾部和稀疏假期状态。\n",
        "最需要补充的业务信息：逐菜计划/实际备货量、补菜与售罄时间、实际售卖时长及停供原因；餐次就餐人数/订单到达变化；活动、调课、停餐与事前校历；临时替代菜和菜单调整记录；库存约束和团餐预订。只有把‘未供应、供应少、售罄’与‘需求少’分开，才能进一步判断预测目标是否应从记录销量转向潜在需求。\n",
        "## 约束、冻结与停止\n"]
    sections.append(table(pd.DataFrame([dict(dataset=d,**v) for d,v in audit["repeat"].items()])))
    sections += ["所有已完成实验保存相同菜单和正销量命中集合，逐餐/单位总量守恒；有前日对照的自动菜单均≤80%，缺少对照独立标注。没有减少菜、删除困难餐次或转移到未命中菜来人为降低指标，未命中预测量份额仍作为拒绝门槛。\n",
        "新独立输出的需求不强制最低制作批量，备餐建议单列。最终仍用原需求，其中F原有4处需求低于最低批量，独立备餐建议的额外量合计约310.65原项目模型计量单位；这不是可跨单位解释的重量。原基线另有1餐超出本轮新历史上界的继承例外，仍明确报告，不能把保留旧预测说成已满足所有新分配边界。\n",
        "已完成至少三类不同结构方向及合理组合/消融：独立全部供应回归、条件分布/低状态、相对/混合学习目标；均未通过事前门槛，因此按规则停止。新风险协调未获进入资格，有明确跳过记录而非偷偷运行后挑选结果。没有固定实验次数上限，也没有继续重复等价调参。\n",
        "冻结配置在 `frozen/config.json`：两个食堂均为原销量预测+独立备餐建议。网页默认和回滚版本保持不变。未执行可选2026既有测试复测，因为没有新的合格预测；2026从未参与选择或停止，不能把旧测试重新称为独立测试。\n",
        "## 验证、技术失败和交付索引\n",
        "`conditional_absolute` 曾被weather_state元数据一致性断言拦截，作为技术失败排除并保留。逐餐拷贝、拼接前后及原evaluator前后重新检查未复现原因；同配置 `conditional_absolute_verified` 通过了增强的完整元数据检查，但预测效果仍不合格。没有删除失败，也没有把未确定根因包装成已修复的模型问题。详见 `metadata_incident.json` 和独立复查脚本。\n",
        "本轮完整验证保存于 `final_verification.json`：每个完成实验重新计算原三指标及风险指标，检查名单/元数据、命中集合、订单/销售额比例、逐餐和单位总量、拟合截止日期与历史特征日期。数据、菜品库、协议和原评价器哈希均未改变。\n",
        "运行与接口说明：根目录 `INDEPENDENT_DEMAND.md`；模块 `demand_runtime.py`；完整命令 `reproduction_commands.txt`；全部实验 `experiments/`；最终保留逐菜输出 `final_selected/`；停止/配置 `selection_and_stop.json`。所有复测和研究范围限制都随交付保留。\n"]
    sections.append("逐餐协调前后明细另见各实验 `pre_post_sessions.csv`；分月、餐次、日期类型、频率、样本组及低量组的区间覆盖/宽度/Brier/AUC见 `interval_risk_segments.csv`，其中还报告最终点预测落在原协调前区间之外的比例，不把原始区间误作协调后联合区间。\n")
    test=OUT/"test_verification.json"
    if test.exists():sections.append("测试结果："+json.loads(test.read_text())["summary"]+"。\n")
    (OUT/"独立单菜需求模型优化报告.md").write_text("\n".join(sections),encoding="utf-8")
    commands=["# Executed recipes; run only in a separately registered reproduction root. Preserve original failures."]
    for name,r in sorted(records.items(),key=lambda x:x[1].get("started_at","")):
        command=".targeted-venv/bin/python -u demand_experiments.py "+shlex.quote(name)+" --config "+shlex.quote(json.dumps(r["config"],ensure_ascii=False))
        if r.get("source_model"):command+=" --source "+shlex.quote(r["source_model"])
        commands.append(command)
    (OUT/"reproduction_commands.txt").write_text("\n".join(commands)+"\n",encoding="utf-8")
    verify();print(OUT/"独立单菜需求模型优化报告.md",flush=True)


if __name__=="__main__":main()
