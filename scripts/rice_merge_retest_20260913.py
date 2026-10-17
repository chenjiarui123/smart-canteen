"""User-confirmed rice identity migration and fixed-policy rolling retest.

Does not change any forecast parameters, original source files, or old experiments.
"""
from pathlib import Path
from decimal import Decimal
import argparse
import importlib.util
import json
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/rice_merge_retest_20260913"
DATA = ROOT / "smart_canteen_web/data"
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "smart_canteen_web/backend"))
import policy
from targeted_search import dump, evaluate
from diagnostics import hit_quantity_diagnostics


def decimal_sum(values):
    return sum((Decimal(str(v)) for v in values), Decimal(0))


def merge_rice(frame):
    """Merge only the two explicitly confirmed names; preserve every other row."""
    mask = frame["菜品名称"].isin(["米饭", "米饭🍚"])
    rice = frame.loc[mask].copy()
    keys = ["年", "月", "日", "食堂名称", "餐次"]
    sums = ["菜品总销量", "订单数量", "销售额"]
    mutable = {"菜品名称", "平均取菜量", *sums}
    rows = []
    merged_groups = 0
    for _, group in rice.groupby(keys, sort=False):
        for col in frame.columns:
            if col not in mutable:
                assert group[col].nunique(dropna=False) == 1, (col, group.to_dict("records"))
        assert group["菜品售卖单位"].iloc[0] == "克"
        row = group.iloc[0].copy()
        row["菜品名称"] = "米饭"
        if len(group) > 1:
            merged_groups += 1
            for col in sums:
                row[col] = str(decimal_sum(group[col]))
            orders = Decimal(row["订单数量"])
            assert orders > 0
            # Raw model quantity for weighed dishes is in 50-gram portions.
            row["平均取菜量"] = str((Decimal(row["菜品总销量"]) * 50 / orders).quantize(Decimal("0.000001")))
        rows.append(row)
    merged = pd.DataFrame(rows, columns=frame.columns)
    result = pd.concat([frame.loc[~mask], merged]).sort_index().reset_index(drop=True)
    pd.testing.assert_frame_equal(frame.loc[~mask].reset_index(drop=True),
                                 result.loc[~result["菜品名称"].eq("米饭")].reset_index(drop=True))
    for col in sums:
        assert decimal_sum(frame[col]) == decimal_sum(result[col]), col
    before_totals = frame.groupby(keys)[sums].agg(decimal_sum)
    after_totals = result.groupby(keys)[sums].agg(decimal_sum)
    pd.testing.assert_frame_equal(before_totals, after_totals)
    assert not result["菜品名称"].eq("米饭🍚").any()
    assert not result.loc[result["菜品名称"].eq("米饭")].duplicated(keys).any()
    return result, {"before_rows": len(frame), "after_rows": len(result),
                    "renamed_alias_rows": int(frame["菜品名称"].eq("米饭🍚").sum()),
                    "merged_sessions": merged_groups,
                    "before_rice_rows": len(rice), "after_rice_rows": len(merged),
                    "all_other_rows_unchanged": True, "session_totals_preserved": True,
                    "totals": {col: str(decimal_sum(result[col])) for col in sums}}


def migrate():
    from app.dataset_repository import DatasetRepository
    assert not OUT.exists(), "Do not overwrite an existing migration/retest"
    OUT.mkdir(parents=True)
    repository = DatasetRepository(DATA)
    paths = [DATA / "datasets/a_hupan/sales.csv", DATA / "datasets/a_hupan/dish_library.csv", DATA / "manifest.json"]
    # Preserve raw strings, untouched metadata and a byte-for-byte rollback copy.
    with repository.write_lock():
        sales = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
        library = pd.read_csv(paths[1], dtype=str, keep_default_na=False)
        after, audit = merge_rice(sales)
        rice_library = library.loc[library["菜品名称"].isin(["米饭", "米饭🍚"])]
        assert len(rice_library) == 2
        canonical = library.loc[library["菜品名称"].eq("米饭")]
        assert len(canonical) == 1
        new_library = library.loc[~library["菜品名称"].eq("米饭🍚")].reset_index(drop=True)
        audit.update(library_before=len(library), library_after=len(new_library),
                     retained_library_record=canonical.to_dict("records")[0],
                     removed_duplicate_library_record=rice_library.loc[rice_library["菜品名称"].eq("米饭🍚")].to_dict("records")[0])
        protected = [p for p in (ROOT / "smart_canteen_web/model").iterdir() if p.is_file()]
        protected += [DATA / "datasets/f_jichuyuan" / n for n in ["sales.csv", "dish_library.csv"]]
        protected += [ROOT / "research" / n for n in ["policy.py", "evaluator.py", "diagnostics.py"]]
        audit["protected_hashes"] = {str(p.relative_to(ROOT)): policy.sha256_file(p) for p in protected}
        for path in paths:
            backup = OUT / "backup" / path.relative_to(DATA)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        sales.loc[sales["菜品名称"].isin(["米饭", "米饭🍚"])].to_csv(OUT / "rice_before.csv", index=False, encoding="utf-8-sig")
        after.loc[after["菜品名称"].eq("米饭")].to_csv(OUT / "rice_after.csv", index=False, encoding="utf-8-sig")
        try:
            repository._atomic_write_csv_unlocked(after, paths[0])
            repository._atomic_write_csv_unlocked(new_library, paths[1])
            for path, expected in [(paths[0], after), (paths[1], new_library)]:
                pd.testing.assert_frame_equal(pd.read_csv(path, dtype=str, keep_default_na=False), expected)
            manifest = repository.manifest()
            now = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
            item = manifest["datasets"]["a_hupan"]
            item.update(sales_rows=len(after), sales_sha256=policy.sha256_file(paths[0]),
                        dish_library_sha256=policy.sha256_file(paths[1]), updated_at=now)
            # Keep original-source row counts unchanged; record current active count separately.
            item["dish_library_rows"] = len(new_library)
            manifest["updated_at"] = now
            manifest.setdefault("updates", []).append({"type": "user_confirmed_dish_alias_merge", "dataset": "a_hupan",
                "alias": "米饭🍚", "canonical": "米饭", "updated_at": now,
                "audit_directory": str(OUT.relative_to(ROOT)), "forecast_parameters_changed": False})
            repository._atomic_write_json_unlocked(manifest, paths[2])
            for path in paths:
                path.chmod((OUT / "backup" / path.relative_to(DATA)).stat().st_mode & 0o777)
        except Exception:
            for path in paths:
                shutil.copyfile(OUT / "backup" / path.relative_to(DATA), path)
            raise
        audit["after_hashes"] = {str(p.relative_to(ROOT)): policy.sha256_file(p) for p in paths}
        audit["archived_predictions_and_original_sources_unchanged"] = True
        for name, digest in audit["protected_hashes"].items():
            assert policy.sha256_file(ROOT / name) == digest
        dump(OUT / "merge_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def run(dataset):
    # Reuse the verified full-history daily-walk implementation in a NEW output
    # root. No old cache, model, protocol or experiment output is overwritten.
    spec = importlib.util.spec_from_file_location("rice_daily_walk", ROOT / "outputs/cleaned_retest_20260912/retest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = OUT
    module.run(dataset)
    summarize(dataset)


def summarize(dataset):
    folder = OUT / dataset
    recs = pd.read_csv(folder / "recommended_menu.csv", parse_dates=["date"])
    actual = policy.read_sales(DATA / "datasets" / dataset / "sales.csv")
    actual = actual.loc[actual.date.ge(recs.date.min())]
    audits = json.loads((folder / "audit.json").read_text())
    for row in audits:
        row["date"] = pd.Timestamp(row["date"])
    splits = [("development", str(recs.date.min().date()), "2025-12-31"),
              ("validation" if dataset == "a_hupan" else "q1_diagnostic_not_independent_validation", "2026-01-01", "2026-03-31"),
              ("test" if dataset == "a_hupan" else "q2_test_subperiod", "2026-04-01", "2026-06-10")]
    if dataset == "f_jichuyuan":
        splits.append(("original_test", "2026-01-01", "2026-06-10"))
    summary, all_segments = [], []
    for label, start, end in splits:
        a = actual.loc[actual.date.between(start, end)]
        r = recs.loc[recs.date.between(start, end)]
        au = [v for v in audits if pd.Timestamp(start) <= v["date"] <= pd.Timestamp(end)]
        m, sessions, segments = evaluate(r, a, au)
        _, hits = hit_quantity_diagnostics(r, a)
        row = dict(dataset=dataset, split=label, start=start, end=end, **m)
        summary.append(row)
        dump(folder / (label + "_metrics.json"), row)
        sessions.to_csv(folder / (label + "_sessions.csv"), index=False, encoding="utf-8-sig")
        hits.to_csv(folder / (label + "_hit_errors.csv.gz"), index=False, compression="gzip")
        all_segments.extend(dict(dataset=dataset, split=label, **segment) for segment in segments)
    pd.DataFrame(summary).to_csv(folder / "split_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_segments).to_csv(folder / "split_segments.csv", index=False, encoding="utf-8-sig")
    dump(folder / "split_summary.json", summary)
    print(dataset, "SPLIT EVALUATION COMPLETE", flush=True)


def compare_and_report():
    """Re-score prior predictions on their original data; never overwrite them."""
    comparison, date_rows = [], []
    selected_keys = ["total_mape", "hit_qty_mape", "hit_qty_wmape", "net_benefit",
                     "evaluated_group_count", "dish_hit_rate"]
    sample_rows = []
    for dataset in ["a_hupan", "f_jichuyuan"]:
        folder = OUT / dataset
        coverage = json.loads((folder / "coverage.json").read_text())
        assert coverage["status"] == "complete"
        previous = ROOT / "outputs/cleaned_retest_20260912" / dataset
        previous_coverage = json.loads((previous / "coverage.json").read_text())
        old_path = OUT / "backup/datasets/a_hupan/sales.csv" if dataset == "a_hupan" else DATA / "datasets/f_jichuyuan/sales.csv"
        assert policy.sha256_file(old_path) == previous_coverage["sales_sha256"]
        assert json.loads((previous / "effective_config.json").read_text()) == json.loads((folder / "effective_config.json").read_text())
        for name in (folder / "model_snapshot").iterdir():
            assert policy.sha256_file(name) == policy.sha256_file(previous / "model_snapshot" / name.name)
        old_actual = policy.read_sales(old_path)
        old_recs = pd.read_csv(previous / "recommended_menu.csv", parse_dates=["date"])
        old_audit = json.loads((previous / "audit.json").read_text())
        for row in old_audit:
            row["date"] = pd.Timestamp(row["date"])
        for new in json.loads((folder / "split_summary.json").read_text()):
            start, end, split = new["start"], new["end"], new["split"]
            a = old_actual.loc[old_actual.date.between(start, end)]
            r = old_recs.loc[old_recs.date.between(start, end)]
            au = [v for v in old_audit if pd.Timestamp(start) <= v["date"] <= pd.Timestamp(end)]
            before, _, _ = evaluate(r, a, au)
            comparison.append(dict(dataset=dataset, split=split, start=start, end=end,
                **{"before_" + k: before[k] for k in selected_keys},
                **{"after_" + k: new[k] for k in selected_keys}))
            hits = pd.read_csv(folder / (split + "_hit_errors.csv.gz"))
            # Save enough source detail to inspect either relative or absolute errors.
            hits["dataset"] = dataset
            hits["split"] = split
            sample_rows.append(hits.head(10))
        segments = pd.read_csv(folder / "split_segments.csv")
        date_rows.append(segments.loc[segments["dimension"].eq("date_type")])
    comp = pd.DataFrame(comparison)
    comp.to_csv(OUT / "合并前后各区间指标.csv", index=False, encoding="utf-8-sig")
    dates = pd.concat(date_rows, ignore_index=True)
    dates.to_csv(OUT / "各区间日期类型指标.csv", index=False, encoding="utf-8-sig")
    pd.concat(sample_rows, ignore_index=True).to_csv(OUT / "各区间高误差命中菜品.csv", index=False, encoding="utf-8-sig")
    audit = json.loads((OUT / "merge_audit.json").read_text())
    for name, digest in {**audit["protected_hashes"], **audit["after_hashes"]}.items():
        assert policy.sha256_file(ROOT / name) == digest, name
    dump(OUT / "verification.json", {"protected_files_unchanged": True,
         "new_data_unchanged_during_retest": True, "same_model_and_parameters_as_previous_evaluation": True,
         "test_results_used_for_tuning": False, "all_splits_complete": True})
    def table(frame):
        return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                           "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                          ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)])
    split_names = {"development": "训练/开发", "validation": "原验证区间复测", "test": "原测试区间复测",
                   "q1_diagnostic_not_independent_validation": "1—3月诊断（不是独立验证集）",
                   "q2_test_subperiod": "4—6月测试子区间", "original_test": "原测试区间整体复测"}
    dataset_names = {"a_hupan": "A／湖畔", "f_jichuyuan": "F／基础园"}
    display = []
    for r in comp.to_dict("records"):
        display.append({"食堂": dataset_names[r["dataset"]], "区间": split_names[r["split"]],
            "日期": r["start"] + "—" + r["end"], "餐次": r["after_evaluated_group_count"],
            "总量MAPE（前→后）": f'{r["before_total_mape"]:.2f}% → {r["after_total_mape"]:.2f}%',
            "命中单菜MAPE（前→后）": f'{r["before_hit_qty_mape"]:.2f}% → {r["after_hit_qty_mape"]:.2f}%',
            "net_benefit（前→后，元）": f'{r["before_net_benefit"]:,.2f} → {r["after_net_benefit"]:,.2f}'})
    report = ["# 米饭名称统一与训练、验证、测试区间复评", "",
        "## 修改内容", "",
        "湖畔314条米饭🍚记录统一为米饭，41个同餐重复组合并相加，销售表91969行变为91928行；菜品库2148道变为2147道，保留原米饭（ID 726）及其属性。合并行的平均取菜量按总销量×50÷订单数重新计算。其他销售记录和菜品属性不变。",
        "每餐和全库的销量、订单、销售额均守恒；基础园数据、两食堂策略参数与评估公式均未改变。原始来源文件、旧实验和旧推荐导出不覆盖，备份与迁移审计在本目录。", "",
        "## 评估方法与划分", "",
        "从同样的14个日历日预热期开始，逐日重新生成菜单。每日各餐预测完成后，才把当天实销量加入后续历史。2026区间也采用此前已设定的这一滚动更新方式，不是只凭2025历史一次性预测半年。期间没有根据验证/测试结果调参或选择模型。",
        "上一轮协议把2025年作为开发区间，2026年全部锁定，仅允许冻结后可选复测；由于没有合格新模型，上一轮未执行这项可选复测。本次按用户要求补齐。",
        "湖畔原验证区间为2026-01-01—03-31，原测试区间为04-01—06-10。基础园原测试区间为2026-01-01—06-10，并无独立验证集，本报告仅额外拆分1—3月和4—6月作诊断，不将其重新命名为独立验证。两者2026区间已在历次工作中查看过，属于既有区间复测，不是未见过的独立泛化证明。",
        "总量MAPE是逐餐总销量的绝对百分比误差均值。命中单菜MAPE只在推荐与实际共同存在且实际销量为正的记录上计算等权均值；WMAPE是这些记录的绝对误差和除以真实销量和。没有删除低销量或假期记录。net_benefit为历史命中菜销售额扣除过量预测销售额的0.5倍，不是实际经营利润。", "",
        "## 合并前后各区间", "", table(pd.DataFrame(display)), "",
        "菜品身份统一同时改变历史统计、候选菜选择和指标匹配集合，因此这里的变化是数据口径修正及重放效果，不是模型训练能力的提升。", ""]
    for (ds, split), group in dates.groupby(["dataset", "split"], sort=False):
        show = group[["segment", "session_count", "total_mape", "hit_qty_mape", "hit_qty_wmape"]].copy()
        show.columns = ["日期类型", "餐次数", "总量MAPE", "命中单菜MAPE", "命中单菜WMAPE"]
        for c in show.columns[2:]:
            show[c] = show[c].map(lambda x: f"{x:.2f}%")
        report += ["## " + dataset_names[ds] + "：" + split_names[split], "", table(show), ""]
    report += ["## 两个米饭样本", "",
        "湖畔2025-12-20午餐合并后的实际米饭销量为1535.52，重新预测1506.17，APE为1.91%；2025-12-25晚餐实际1204.80，重新预测1118.94，APE为7.13%。单位沿用模型记录口径，称重菜数量为50克份数，并非直接克数。", "",
        "## 春节重叠期是否下降不足", "",
        "这些极端餐次的确降幅不足，但并非简单的统一惩罚参数太小。当前假期分支使用同类假期历史的绝对销量估计：2025-01-28午餐没有此前春节重叠样本，回退到15个寒假同餐样本，预测6407.86，约为常规参考9242.04的69.33%；实际315.14仅为参考的3.41%。初一午餐已有1个春节重叠样本，预测降到576.07（参考的6.23%），但实际7.08（参考的约0.077%）。这些比例仅用于事后解释，不能据此设置测试日期专用参数。",
        "该分支将趋势、短期变化和天气乘数设为1，避免重复缩放，但也缺少跟踪假期内部急降的机制。holiday_shrinkage_tau=0.25是历史估计收缩参数，不是直接的处罚倍数。仅把所有假期统一压低，会增加假期结束与返校恢复时的低估。", "",
        "## 是否增加假期前后阶段", "",
        "建议增加由已知校历和法定假日表自动生成的阶段信息：距放假开始天数、假期进行到第几天、距假期结束天数及结束后经过天数。由历史学习节前下降、节中极低供餐和返校恢复；阶段长度及强度只能在开发数据中确定。样本不足时向普通日期/同类假期估计收缩，不能多个下降因子盲目相乘。缺失校历边界不能通过未来销量或无销售记录反推。",
        "本次仅实施米饭合并及复评，没有修改或训练假期机制。", "",
        "## 校验与文件", "",
        "本次合并与数据更新相关测试12项通过，网页数据完整性检查正常。全量测试107项通过、4项失败：4项均为旧独立模型实验的旧数据指纹检查，因授权修改湖畔数据而触发；旧指纹未改写、检查未绕过。复现旧实验时须使用本次保留的合并前备份及其原环境。",
        "汇总：合并前后各区间指标.csv；日期分类：各区间日期类型指标.csv；高误差命中菜品：各区间高误差命中菜品.csv。每个食堂目录保留全部推荐菜单、逐餐指标、分区间命中菜品明细及模型快照。merge_audit.json和verification.json提供数据守恒及策略未变化校验。", ""]
    (OUT / "米饭合并与分区间复评报告.md").write_text("\n".join(report), encoding="utf-8")
    print(comp.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["migrate", "run", "summarize", "compare"])
    parser.add_argument("dataset", nargs="?", choices=["a_hupan", "f_jichuyuan"])
    args = parser.parse_args()
    if args.action == "migrate":
        migrate()
    elif args.action == "run":
        run(args.dataset)
    elif args.action == "compare":
        compare_and_report()
    else:
        summarize(args.dataset)
