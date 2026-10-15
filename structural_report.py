"""Read-only analysis of completed development outputs; generate delivery docs."""
import json
import numpy as np
import pandas as pd
from structural_data import ROOT, OUT, DATASETS, verify_protected
from targeted_search import dump, evaluate as extended_evaluate


def table(frame):
    def fmt(value):
        if pd.isna(value): return "缺少样本/不适用"
        if isinstance(value, (float, np.floating)): return f"{value:,.4f}"
        return str(value).replace("|", "／").replace("\n", " ")
    return "| " + " | ".join(frame.columns) + " |\n|" + "|".join(["---"]*len(frame.columns)) + "|\n" + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)) + "\n"


def main():
    verify_protected()
    decision = json.loads((OUT/"selection_and_stop_audit.json").read_text())
    audit = json.loads((OUT/"final_requirement_audit.json").read_text())
    allm = pd.read_csv(OUT/"all_experiment_metrics.csv")
    segments = pd.read_csv(OUT/"all_session_segments.csv")
    records = {p.parent.name:json.loads(p.read_text()) for p in (OUT/"experiments").glob("*/record.json")}
    p = ["# 智慧食堂单菜结构性预测优化报告\n",
         "## 最终结论\n",
         "本轮没有取得满足预先固定门槛的稳定改进，两个食堂均保留现有部署策略。没有达到命中单菜MAPE低于30%的目标，不发布失败组合，不放宽门槛，也不删除低销量或困难日期。独立新模块、完整实验及逐餐逐菜结果已经保存，网页默认未改变。\n",
         "当前网页基线已经包含上一轮的80%相邻日重复硬限制和假期校准，版本为 `targeted_repeat_holiday_20260911`。此前定向优化的完整新旧对比见 `runs/targeted_20260911/新旧策略对比报告.md`；本报告是后续单菜模块研究，不能把两个不同数据版本/区间的指标混为一次比较。\n",
         "## 数据、复现和锁定范围\n",
         "只使用 `smart_canteen_web/data/datasets/{a_hupan,f_jichuyuan}/sales.csv` 及对应当前菜品库。保留网页已清理的记录状态，不从 `数据/` 原始文件恢复，不新增删除、不修改标签、单位或evaluator。2026只允许读取日期列以隔离行，销量和真实菜单没有加载进本轮选择、停止或审计。\n",
         "A从2025-07-22开始，14个日历日预热后评估2025-08-05—12-31，共447餐；F从2025-01-01开始，评估2025-01-15—12-31，共914餐。缺少销售记录不补零，未开餐不制造餐次。A无寒假/重叠假期开发样本；F覆盖七种类型，但重叠状态只有31条正销量命中，仍完整报告。\n",
         "实际网页使用 `runtime_history_end_inactivity=True`，本轮复现也启用它；旧allocation实验没有同样配置，不能复用其基线数值。基线与既有清理后输出的2025部分一致，逐阶段追踪与直接调用首日逐字段一致。最终当前源码对照再次重跑，两食堂全部原预测字段与基线完全相同。\n",
         "第一阶段直接锁定最终名单、每餐最终总量，额外锁定原售卖单位组总量；因此影子菜单的初始估计不会悄悄改变总量。不把‘克、份、个’互相转移，不做新的50克换算。原评价器仍采用原项目数量口径；混合单位数量合计只作该口径诊断，不能解释为物理重量。\n",
         "## 选择门槛（实验前固定）\n",
         "主目标顺序为总销量MAPE→命中单菜MAPE→net_benefit。固定总量误差容差为rtol=1e-12、atol=1e-8；命中MAPE至少下降0.25个百分点，net下降不超过1%，命中WMAPE相对上升不超过5%，未命中预测量占比上升不超过0.5个百分点，命中加权偏差绝对值增加不超过2个百分点。\n",
         "分月、餐次、日期类型中正销量命中≥100项的组，命中MAPE允许退化不超过max(1个百分点, 该组基线的5%)；小样本组不隐藏。单菜上界为min(历史q99×2, 基线×2)，HHI不超过基线1.15倍、最大单菜份额不增加超过3个百分点。冲突回退整餐基线并记录，不将需求移给其他单位或少数菜。完整协议见 `preregistered_protocol.json`。\n",
         "## 基线与最终保留结果\n"]
    base = allm[allm.trial.eq("final_verified_control")]
    p.append(table(base[["dataset","evaluated_group_count","total_mape","hit_qty_mape","net_benefit","hit_qty_wmape","total_wape","total_mae","dish_hit_rate","nonhit_pred_qty_share"]]))
    p += ["最终保留结果与上表基线相同：三项主指标变化均为0。MAPE/WAPE/WMAPE为百分数，命中率、份额为0—1，net_benefit沿用原项目口径。\n", "## 误差从哪里产生\n",
          "下表每一行比较的是同一批菜品，且限定在最终菜单集合中；不是把不同阶段不同命中样本数直接相减。稳定性阶段保存的原始数值是未归一化权重，不能当成独立销量预测；这里在同总量下对比去掉/保留稳定性权重的分配。\n"]
    paired = pd.read_csv(OUT/"paired_stage_changes.csv")
    p.append(table(paired.query("scope=='final_menu_subset'").drop(columns="scope")))
    p += ["A的日期修正只小幅降低误差，F反而增加约1.01个百分点；但两食堂原稳定性分配在同菜同总量对照中均有收益，因此不能简单删掉波动折扣就断言会更好。最低批量处理对保留菜的平均误差变化很小，重复率换菜对A保留菜略差、对F略好，不能把全部高误差归因于某一个后处理步骤。\n",
          "最低批量阶段删除A172项、F211项；重复率控制各增加并移除A4476项、F8132项。逐项的新增、删除及阶段缺失均保留，缺失不是零预测。需求/备餐分离可以解释经营风险，但只增加字段、不改变需求的对照没有销量指标改善。\n",
          "以下大误差案例只用于事后诊断，不据目标日真实值制定规则。所有低销量记录保留：\n"]
    examples = []
    for ds in DATASETS:
        f = pd.read_csv(OUT/("development_"+ds)/"worst_final_dish_stage_cases.csv").head(3)
        examples.append(f.assign(dataset=ds)[["dataset","date","餐次","菜品名称","actual_qty","initial_history","repeat_final","min_batch_qty","sample_count","final_ape"]])
    p.append(table(pd.concat(examples)))
    p += ["例如F的酸菜鸡柳历史均值约108.51，目标日实际只有0.28，最终预测约139.18。这解释了为何MAPE很高，而WMAPE和中位误差较低；但不能据此自动删该记录或把它在预测时识别成异常。最低批量对理论最优需求造成的事后单项误差下界均值仅A5.66、F4.26个百分点，这不是当前总误差的充分解释，更不能据此声称30%目标不可能。\n",
          "全部分餐次、日期类型、历史销量、供给频率、历史样本数见各实验 `hit_segments.csv`；阶段明细见 `development_*/stage_error_segments.csv`。历史量/频率分组阈值仅由开发期历史特征分布计算，用于诊断，不参与预测或目标日实际量分组。\n",
          "## 单机制、组合与消融\n",
          "A=需求与备餐分离；B=条件份额（包含状态/天气联合回退）；C=按餐次/单位分层共享残差；E=历史菜单竞争关系；F=历史预测误差可靠性协调。D没有包装成独立单试验，日期/天气已作为B和C内部条件。竞争只学习关联，不宣称因果弹性；测试目标菜单来自基线，不使用当天真实出菜结构。\n",
          "平面/分层份额、竞争、可靠性、共享残差四个不同结构方向先各自验证，再验证解除备餐下界等互补组合和历史表现准入。历史准入只依据此前同餐次/状态的预测表现，证据不足整餐用基线，所有困难日期仍纳入评价。没有固定实验次数上限，也未把旧参数搜索换名字当新结构。\n"]
    compact = []
    for name,r in sorted(records.items(), key=lambda item:item[1].get("started_at", "")):
        row = {"实验":name}
        for ds,label in [("a_hupan","A"),("f_jichuyuan","F")]:
            x = r["datasets"].get(ds)
            row[label+"命中MAPE"] = x["metrics"]["hit_qty_mape"] if x else np.nan
            row[label+"net_benefit"] = x["metrics"]["net_benefit"] if x else np.nan
        row["结论"] = "基线对照" if name in {"baseline","baseline_f","final_verified_control"} else "两食堂均未通过"
        compact.append(row)
    p.append(table(pd.DataFrame(compact)))
    p += [f"共{len(records)}个完整运行记录（含A/F分开基线及最终对照），失败实验均保留。总量指标在所有实验中保持基线，不因少推荐菜或换评价字段得到改善。\n",
          "全部指标、假设和失败原因：`all_experiment_metrics.csv`。逐组的原始门槛与退化证据：各实验 `gate_review.json`；每轮源码SHA-256、配置和结果在 `record.json`，源码快照在 `source/`。准备和交付审计中的运行问题也保留于 `preparation_incidents.json`、`release_incidents.json`，未重算预测掩盖问题。\n",
          "严格逐项消融映射如下；已验证配置恰好只移除指定组件，不把多个变化的实验误称为单组件消融：\n"]
    p.append(table(pd.DataFrame(json.loads((OUT/"ablation_audit.json").read_text()))))
    p += ["`A_F` 同时移除了旧波动折扣，不能作为 `A_C_F` 的严格−C对照；对应对照是 `A_F_baseline_anchor`。保留基线初始估计时F协调几乎是恒等操作，历史准入对 `guarded_A_F_baseline` 两食堂均0餐采纳，不能宣称它有独立改善。\n",
          "### 有希望但被淘汰的方案\n",
          "F的 `A_C_F` 命中MAPE从55.4773%降至54.5302%，net从4,104,653.67升至4,161,912.12，整餐总量不变。但7月命中MAPE由63.2156%升至72.1447%，退化8.9292个百分点，大于允许的3.1608，故淘汰。\n",
          "加历史准入后，`guarded_A_C_F` 在F采纳402餐，整体命中MAPE约54.6340%；7月仍升到67.2802%，退化4.0646个百分点，仍不通过。A该方案采纳80餐且整体误差变差。去掉F后的受保护A+C仅改善F约0.1637个百分点，不足0.25门槛。不能为了保留这一趋势而放宽稳定性标准。\n",
          "## 普通日期、假期和各日期类型\n",
          "下表的‘最终保留’就是现有基线；两个实验列用于显示淘汰方案的取舍，不是线上新策略。A不存在的寒假/重叠组不填0。\n"]
    for ds in DATASETS:
        p.append("### "+ds+"\n")
        for dimension in ["regular_holiday","date_type"]:
            data = segments[(segments.dataset==ds)&(segments.dimension==dimension)&segments.trial.isin(["final_verified_control","A_C_F","guarded_A_C_F"])]
            data = data[["segment","trial","evaluated_group_count","total_mape","hit_qty_mape","net_benefit","hit_qty_wmape","weighted_qty_bias","dish_hit_rate"]].copy()
            data["trial"] = data.trial.replace({"final_verified_control":"基线=最终保留"})
            p.append(table(data))
    extended_rows = []
    for ds in DATASETS:
        folder = OUT/("development_"+ds)
        actual = pd.read_pickle(folder/"actual.pkl")
        ba = json.loads((folder/"baseline_audit.json").read_text())
        for a in ba: a["date"] = pd.Timestamp(a["date"])
        menu = pd.read_pickle(OUT/"experiments/final_verified_control"/ds/"menu.pkl.gz")
        _, sessions, extended = extended_evaluate(menu, actual, ba)
        sessions.to_csv(folder/"final_menu_demand_diagnostics.csv", index=False)
        extended_rows.append(pd.DataFrame(extended).assign(dataset=ds))
    extended = pd.concat(extended_rows, ignore_index=True)
    extended.to_csv(OUT/"final_count_demand_segments.csv", index=False)
    p += ["### 数量、超量、低估及小销量诊断\n",
          "第一阶段和最终选择均不改变名单与总量，所以相对本轮基线各类假期的菜数、总销量变化均为0；不声称这一轮继续降低了假期高估。以下列出当前仍存在的偏差，包括低估，而不是仅看MAPE。F暑假和寒假的命中需求总体偏低，重叠假期仍存在很大的高估风险，不能对所有假期统一乘一个下降因子。\n"]
    p.append(table(extended.query("dimension in ['overall','date_type']")[["dataset","segment","session_count","actual_count_mean","pred_count_mean","count_bias_mean","actual_qty_sum","pred_qty_sum","overprediction_qty","underprediction_qty","total_wape","total_mae","menu_fill_rate"]]))
    p += ["这里保留原MAPE，同时报告WAPE和绝对误差；极小真实量不排除，未命中量不转嫁。`final_count_demand_segments.csv` 还包含同食堂×餐次×日期类型明细；各实验的 `sessions.csv` 与固定菜单可逐餐核对。\n",
          "## 80%硬约束复核与替换菜\n"]
    p.append(table(pd.DataFrame([dict(dataset=d, **r) for d,r in audit["repeat"].items()])))
    p += ["重新按标准化菜名去重、所有类别、同食堂同餐次的前一自然日菜单计算：有对照餐次全部≤80%；缺少前日对照保留缺失值，不记作0。所有本轮实验名单与基线严格一致，所以也继承这一结果；没有加入潜力菜或缩减菜单来改变分母。\n",
          "原evaluator的 `repeat_rate` 是旧口径，在F整体可超过0.8，不是本任务的相邻自然日指标；不得据此误判硬限制失败，也不得修改原指标定义。正确逐餐指标见 `development_*/adjacent_final_audit.csv`。\n",
          "替换来自部署策略原同餐次合格候选池，要求至少3次可靠历史，冷启动有同餐次历史回退。候选池、替换前后菜名、历史计数、缺口和来源见 `development_*/baseline_audit.json`；分阶段变化见压缩CSV。原网页仍优先使用前日已确认菜单，没有时使用保存的高频推荐；人工超限只提醒、自动策略硬限制。\n",
          "## 备餐、边界冲突和剩余问题\n"]
    p.append(table(allm[allm.trial.isin(["final_verified_control","preparation_only","A_C_F","guarded_A_C_F"])][["dataset","trial","below_minimum_demand_items","below_minimum_preparation_items","surplus_risk_qty","conflict_sessions"]]))
    p += ["每次实验的 `preparation_metrics` 另以备餐量评价，但主指标仍按预计需求。分离时备餐=max(需求,最低批量)，差额是最低批量的额外风险量，不是实际剩余或浪费率。未分离的identity严格保留原输出：F仍有4餐原始需求低于最低批量，不能声称原备餐规则已全部解决。\n",
          "交付审计另发现：F 2025-08-26早餐有1道基线菜超过本轮新增实验q99上界，原identity审计状态仅写allocated。它属于完全未变的旧预测，不是新分配造成的突破；已逐餐证明与基线相同，并在 `inherited_baseline_box_exceptions.json` 明列。不能把该控制组当成满足所有新箱约束的改进策略。涉及identity回退的实验仍是未发布候选。\n",
          "## 冻结、停止和复测纪律\n",
          "至少四种不同结构方向及其合理组合、严格移除组件消融均没有带来达到预设门槛的稳定改善；历史准入也未解决分段退化。最终恢复最简identity并完成当前源码复跑，因此按事前停止规则结束，不再重复等价实验。第二阶段的菜单更改未获进入资格，没有启动。\n",
          "已冻结 `frozen/structural_config.json`、源码及哈希清单，两食堂均选择identity。2026是已多次分析的旧区间，本轮没有新增可接受预测，故不再执行可选复测；不能将旧结果重新称为独立测试或用于调整门槛。保留有希望的失败候选供未来新增训练证据时研究，但不据此上线。\n",
          "## 交付与验收索引\n",
          "运行及接入指南：项目根目录 `STRUCTURAL_OPTIMIZATION.md`。独立接口：`structural_runtime.py`。全部实验主指标：`all_experiment_metrics.csv`；全部分段：`all_session_segments.csv`；逐项消融：`ablation_audit.json`；选择/停止：`selection_and_stop_audit.json`；最终逐餐验证：`final_requirement_audit.json`。最终对照和代表性失败方案提供 `menu.csv.gz`、`hits.csv.gz`，各轮pickle明细也完整保留。\n",
          "验证包括每轮原evaluator指标重算、菜单元数据/正销量命中集合相等、逐餐及单位组总量守恒、订单/销售额比例一致、历史最大日期严格早于目标日、保护层前日信息检查、源码快照哈希、清理后数据/库/网页/协议未改变。边界与冷启动限制如上明列，不以‘通过审计’掩盖未达成30%或未做上线验证。\n"]
    tests = OUT/"test_verification.json"
    if tests.exists():
        p.append("回归测试："+json.loads(tests.read_text())["summary"]+"。\n")
    (OUT/"结构性预测优化报告.md").write_text("\n".join(p), encoding="utf-8")
    verify_protected()
    print(OUT/"结构性预测优化报告.md")


if __name__ == "__main__": main()
