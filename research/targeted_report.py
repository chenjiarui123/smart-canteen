"""Render the auditable comparison from completed run artifacts only."""
import json
from pathlib import Path

import pandas as pd

import policy
from targeted_search import OUT, ROOT, dump, evaluate


def table(frame, columns):
    frame = frame[columns]
    header = "| " + " | ".join(columns) + " |\n|" + "|".join(["---"]*len(columns)) + "|\n"
    def fmt(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "缺少样本/对照"
        if isinstance(value, float):
            return f"{value:,.4f}"
        return str(value).replace("|", "/")
    return header + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))


def main():
    manifest = json.loads((OUT / "frozen/targeted_manifest.json").read_text())
    best = manifest["development_run"]
    train = policy.read_sales(policy.TRAIN_PATH)
    actual = policy.training_backtest_frames(train)[1]
    development = {}
    for name in ["baseline", best]:
        folder = OUT / name
        recs = pd.read_csv(folder / "recommended_menu.csv", parse_dates=["date"])
        audit = json.loads((folder / "audit.json").read_text())
        for a in audit:
            a["date"] = pd.Timestamp(a["date"])
        metrics, sessions, segments = evaluate(recs, actual, audit)
        dump(folder / "complete_metrics.json", metrics)
        sessions.to_csv(folder / "complete_sessions.csv", index=False)
        pd.DataFrame(segments).to_csv(folder / "complete_segments.csv", index=False)
        development[name] = (metrics, sessions, pd.DataFrame(segments))
    b, n = development["baseline"][0], development[best][0]
    rows = []
    for metric in ["total_mape", "hit_qty_mape", "net_benefit", "total_wape", "total_mae", "adjacent_repeat_rate",
                   "adjacent_repeat_max", "repeat_over_80_sessions", "missing_previous_sessions", "infeasible_sessions",
                   "menu_count_mae", "count_bias_mean", "overprediction_qty", "underprediction_qty", "dish_hit_rate", "menu_fill_rate"]:
        rows.append({"指标": metric, "旧策略": b[metric], "新策略": n[metric], "变化": n[metric]-b[metric] if b[metric] is not None and n[metric] is not None else None})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "development_comparison.csv", index=False)
    report = ["# 智慧食堂定向优化与冻结复测报告", "", f"冻结版本：{manifest['model_version']}；最终开发复核：{best}。", "",
              "## 结论与适用范围", "",
              f"开发回测450餐次，自动菜单超80%餐次：{b['repeat_over_80_sessions']} → {n['repeat_over_80_sessions']}；缺少前日对照{n['missing_previous_sessions']}餐次，不计作0%。不可行{n['infeasible_sessions']}餐次。",
              f"总销量MAPE {b['total_mape']:.4f}% → {n['total_mape']:.4f}%；命中MAPE {b['hit_qty_mape']:.4f}% → {n['hit_qty_mape']:.4f}%；net_benefit {b['net_benefit']:.2f} → {n['net_benefit']:.2f}。",
              "该结果必须按三项目标分别判断，不能表述为所有指标同时改善。测试区间是既有数据的冻结后复测，不是新独立盲测。", "",
              "## 复现、范围与原因", "",
              "沿用根目录policy.py、config.json、固定evaluator.py与训练文件。开发区间2025-08-04—2025-12-31，14天历史预热；每餐销售统计只使用预测日前历史，每天三餐共享同一个日初快照。固定开发集只有长春A食堂，包含普通工作日、普通周末、暑假、法定节假日与调休；寒假和校历法定重叠没有开发样本，不伪造其开发指标。",
              "完整复现基线与旧README存在细小数值差异，因此所有新旧比较均使用当前环境重跑的基线。快速等价适配层在450餐次全部输出字段上与该基线一致；初版错误引入网页活跃度参照日的实验已保留并排除。",
              "原重复处理只有轻度软惩罚，且使用上次菜单而非严格前一天。原假期需求仅启用法定节假日与重叠状态，组合因子下限0.50；菜数只向状态/需求估计修正25%。但开发集暑假已低估，因此最终方案不对暑假需求额外下调。", "",
              "## 策略规则", "",
              "高频菜单按标准化菜名去重、所有类别纳入。最低批量筛选后锁定当餐N，要求重复菜数≤floor(4N/5)，由同餐次原合格候选池选取有至少3次历史依据的替换菜；冷启动允许同餐次跨食堂历史回退。使用预测得分、历史稳定性及软搭配排序，同时保持相似菜硬约束。重分配后再次检查最低批量、菜数与重复配额；不会靠潜力菜扩大分母，也不会为达标缩小N。",
              "若找不到同时满足条件的菜单，保留完整需求、目标菜数、替换缺口和原因，不发布超限或缩水自动菜单。此类餐次仍保留在评估范围；缺口是当前确定性组合搜索的诊断，不宣称求解了所有可能组合的数学不可行性证明。",
              "假期需求与菜数各自使用相近状态历史。法定假期按同餐次法定类别→具体假期；重叠状态按对应寒/暑假父状态→联合状态。优先同食堂，缺少时回退同餐次可用历史，按n/(n+tau)收缩，使用log1p几何混合。收缩父值为当前预测，避免长期普通高需求主导。使用绝对假期预测后不再叠加天气、星期、趋势下降因子；允许预测低于原0.50因子下限。无相近历史首日回退旧预测，无法保证识别全新停餐断点。",
              "整餐需求保持独立，单菜按权重及稳定性守恒分配；订单与销售额继续使用原历史单位比例。没有直接压低销售额、改评价公式或删除低销量日期。", "",
              "## 开发集整体比较", "", table(comparison, ["指标", "旧策略", "新策略", "变化"]), "",
              "MAPE/WAPE列单位为%；重复率、命中率、填充率为0—1。数量和销量保留原项目单位（克制菜的模型计量为50克份），没有在评估中混入网页总克数转换。", ""]
    for label, dim in [("分日期类型", "date_type"), ("普通日期与假期", "regular_holiday")]:
        report += ["## " + label, ""]
        before = development["baseline"][2]
        after = development[best][2]
        compare = before.loc[before.dimension.eq(dim)].merge(after.loc[after.dimension.eq(dim)], on="segment", suffixes=("_old", "_new"))
        compare.to_csv(OUT / (dim + "_comparison.csv"), index=False)
        report += [table(compare, ["segment", "session_count_old", "total_mape_old", "total_mape_new", "hit_qty_mape_old", "hit_qty_mape_new", "net_benefit_old", "net_benefit_new"]), "",
                   table(compare, ["segment", "actual_count_mean_old", "pred_count_mean_old", "pred_count_mean_new", "actual_qty_sum_old", "pred_qty_sum_old", "pred_qty_sum_new", "overprediction_qty_old", "overprediction_qty_new", "underprediction_qty_old", "underprediction_qty_new"]), ""]
    replacements = []
    for a in json.loads((OUT / best / "audit.json").read_text()):
        for r in a.get("replacements", []):
            replacements.append({"date": pd.Timestamp(a["date"]), "食堂名称": a["食堂名称"], "餐次": a["餐次"], **r})
    if replacements:
        from evaluator import prepare_actual
        replacement_frame = pd.DataFrame(replacements)
        truth = prepare_actual(actual)
        replacement_frame = replacement_frame.merge(truth, left_on=policy.GROUP_KEYS+["dish"], right_on=policy.GROUP_KEYS+["_dish_norm"], how="left")
        replacement_frame["hit"] = replacement_frame.actual_qty.notna()
        positive = replacement_frame.loc[replacement_frame.actual_qty.gt(0)]
        replacement_mape = float(((positive.pred_qty-positive.actual_qty).abs()/positive.actual_qty).mean()*100)
        replacement_frame.to_csv(OUT / "replacement_outcomes.csv", index=False)
        report += ["## 替换菜来源与表现", "",
                   f"开发期共{len(replacement_frame)}次菜品替换（同一道菜在不同餐次分别计数），来自同餐次原候选池；本地历史出菜次数中位数{replacement_frame.local_history_count.median():.0f}，最少{replacement_frame.local_history_count.min():.0f}。替换菜实际命中率{replacement_frame['hit'].mean():.2%}，其中正销量命中项原口径MAPE {replacement_mape:.4f}%。该分组不是额外选择目标，不改变整体指标。逐项见replacement_outcomes.csv。", ""]
    # Every experiment retains its original artifact plus a separate judgement.
    records = [json.loads(p.read_text()) for p in OUT.glob("*/record.json")]
    records.sort(key=lambda r:r["started_at"])
    experiment_rows, reviews = [], []
    for r in records:
        m = r.get("metrics", {})
        name = r["name"]
        if name == best:
            reason = "最终源代码复核及冻结结果"
        elif name == "baseline":
            reason = "复现基线，不满足80%硬限制"
        elif name in {"adapter_equivalence", "repeat_only"}:
            reason = "初版网页活跃度参照混入回测，已由v2修复；排除选择"
        elif name == "adapter_equivalence_v2":
            reason = "全字段等价验证通过，非改进候选"
        elif not r.get("patch", {}).get("adjacent_repeat_hard"):
            reason = "单模块验证；本轮未执行80%限制，不单独发布"
        elif m.get("infeasible_sessions", 0):
            reason = "开发期存在未发布菜单，不接受以缺菜单改善指标"
        elif m.get("total_mape", 1e9) > n["total_mape"] * 1.0005:
            reason = "首要目标总销量MAPE较最终方案差"
        elif m.get("hit_qty_mape", 1e9) > n["hit_qty_mape"] * 1.0005:
            reason = "总量接近，但第二目标命中MAPE较差"
        elif m.get("net_benefit", 0) < n["net_benefit"]:
            reason = "前两项目标接近，净收益较低"
        else:
            reason = "保留的中间最佳/行为等价配置，最终以复核版本交付"
        reviews.append({"name":name, "hypothesis":r["hypothesis"], "decision":reason, "source_hashes":r["source_hashes"], "metrics":m})
        experiment_rows.append({"实验":name, "总量MAPE":m.get("total_mape"), "命中MAPE":m.get("hit_qty_mape"), "net_benefit":m.get("net_benefit"), "超限":m.get("repeat_over_80_sessions"), "结论":reason})
    dump(OUT / "experiment_reviews.json", reviews)
    pd.DataFrame(experiment_rows).to_csv(OUT / "experiment_comparison.csv", index=False)
    report += ["## 实验与停止依据", "", table(pd.DataFrame(experiment_rows), ["实验", "总量MAPE", "命中MAPE", "net_benefit", "超限", "结论"]), "",
               "先分别验证重复限制、假期需求、假期菜数，再组合。比较首先要求自动菜单可行且无80%违规；随后按总量MAPE、命中MAPE、net_benefit顺序，沿用0.05%相对变化作为实质改进参考，忽略小于该阈值的数值抖动。没有固定试验次数上限。停止原因及实际连续无改进方向见stop_audit.json。若相对无约束基线仍存在收益或命中误差退化，应明确作为未解决权衡，不称为三目标Pareto改进。", ""]
    for dataset in ["hupan", "jichuyuan"]:
        folder = OUT / ("retest_" + dataset)
        if not (folder / "retest_lock.json").exists():
            continue
        lock = json.loads((folder / "retest_lock.json").read_text())
        if lock["status"] != "complete":
            report += [f"## {dataset} 冻结复测", "", "复测尚未完成。", ""]
            continue
        report += [f"## {dataset} 冻结后复测（2026-01-01—2026-06-10）", ""]
        for dim in ["overall", "date_type", "regular_holiday"]:
            a = pd.read_csv(folder / "baseline_segments.csv")
            z = pd.read_csv(folder / "new_segments.csv")
            merged = a.loc[a.dimension.eq(dim)].merge(z.loc[z.dimension.eq(dim)], on="segment", suffixes=("_old", "_new"))
            merged.to_csv(folder / (dim + "_comparison.csv"), index=False)
            report += [table(merged, ["segment", "session_count_old", "total_mape_old", "total_mape_new", "hit_qty_mape_old", "hit_qty_mape_new", "net_benefit_old", "net_benefit_new"]), "",
                       table(merged, ["segment", "pred_count_mean_old", "pred_count_mean_new", "pred_qty_sum_old", "pred_qty_sum_new", "total_wape_new", "total_mae_new", "repeat_over_80_sessions_new", "infeasible_sessions_new"]), ""]
        if dataset == "hupan":
            # Rebuild split diagnostics from the saved frozen predictions;
            # retain March 31 as April 1's comparator via the full-run audit.
            external_actual = policy.read_sales(ROOT / "数据/菜品销量_吉大湖畔智慧餐厅_菜名修正后.csv")
            for label in ["baseline", "new"]:
                saved = pd.read_csv(folder / (label + "_recommended_menu.csv"), parse_dates=["date"])
                audits = json.loads((folder / (label + "_audit.json")).read_text())
                for a in audits:
                    a["date"] = pd.Timestamp(a["date"])
                for split, start, end in [("validation_retest", "2026-01-01", "2026-03-31"), ("test_retest", "2026-04-01", "2026-06-10")]:
                    m, s, seg = evaluate(saved.loc[saved.date.between(start,end)], external_actual.loc[external_actual.date.between(start,end)],
                                         [a for a in audits if pd.Timestamp(start) <= a["date"] <= pd.Timestamp(end)])
                    dump(folder / f"{label}_{split}_metrics.json", m)
                    s.to_csv(folder / f"{label}_{split}_sessions.csv", index=False)
                    pd.DataFrame(seg).to_csv(folder / f"{label}_{split}_segments.csv", index=False)
            splitrows = []
            for split in ["validation_retest", "test_retest"]:
                for label in ["baseline", "new"]:
                    m = json.loads((folder / f"{label}_{split}_metrics.json").read_text())
                    splitrows.append(dict(阶段=split, 策略=label, **{k:m[k] for k in ["total_mape", "hit_qty_mape", "net_benefit", "total_wape", "total_mae", "repeat_over_80_sessions", "missing_previous_sessions", "infeasible_sessions"]}))
            report += [table(pd.DataFrame(splitrows), list(splitrows[0])), ""]
        a = pd.read_csv(folder / "baseline_segments.csv")
        z = pd.read_csv(folder / "new_segments.csv")
        states = a.loc[a.dimension.eq("date_type")].merge(z.loc[z.dimension.eq("date_type")], on="segment", suffixes=("_old", "_new"))
        report += ["### 剩余风险与未改善分组", ""]
        for r in states.to_dict("records"):
            bias_pct = 100*r["qty_bias_new"]/r["actual_qty_sum_new"] if r["actual_qty_sum_new"] else 0
            if r["total_mape_new"] > r["total_mape_old"]*1.01:
                report += [f"- {r['segment']}：总销量MAPE {r['total_mape_old']:.2f}% → {r['total_mape_new']:.2f}%，未改善；新策略累计销量偏差{bias_pct:+.2f}%。"]
            elif abs(bias_pct) > 10:
                report += [f"- {r['segment']}：即使平均误差改善，新策略累计销量仍{'高估' if bias_pct > 0 else '低估'}{abs(bias_pct):.2f}%，不能认定问题已经解决。"]
        report += ["", "以上问题来自冻结后的复测诊断，没有据此回调参数。寒假/重叠没有开发样本；长期同假期中位数可能无法跟随假期内的快速需求变化，需要后续独立开发数据验证。", ""]
        failures = pd.read_csv(folder / "new_sessions.csv")
        failures = failures.loc[failures.repeat_status.eq("infeasible")]
        failures.to_csv(folder / "infeasible_sessions.csv", index=False)
        report += [f"不可行餐次{len(failures)}个，详细日期、餐次、缺口与原因见该复测目录infeasible_sessions.csv。未发布餐次仍计入总评估；对应原计划需求见new_audit.json。", ""]
    report += ["## 验证", "", "后端与回测58项测试通过（含跨区间前日对照诊断），前端16项测试通过，生产构建成功；当前/API文档/健康检查均返回200，健康检查确认新策略版本及两个食堂数据完整性。见verification.json。", "", "## 网页行为与运行", "",
               "网页优先使用前一天同餐次最近一次已确认菜单；同日多次确认不求并集。无确认记录时使用automatic_high_frequency目录保存的前日高频结果；两者都没有则显示缺少对照。低频草稿不写入该高频对照。最终人工菜单超80%显示提醒，仍允许保存。",
               "新策略独立保存于smart_canteen_web/model/targeted_policy.py、targeted_config.json、targeted_manifest.json；原Trial18文件与历史评估锁保留。启动仍使用smart_canteen_web/run_app.py。命令、复现实验和回退步骤见TARGETED_OPTIMIZATION.md。",
               "逐食堂、餐次和日期类型的完整指标在各实验/复测目录segments.csv（dimension=date_type_canteen_meal）；逐餐误差、数量、对照状态在sessions.csv；替换来源、稳定性、历史次数、配额与缺口在audit.json。", ""]
    path = OUT / "新旧策略对比报告.md"
    path.write_text("\n".join(report), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
