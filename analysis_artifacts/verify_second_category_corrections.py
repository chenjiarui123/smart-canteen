from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = ROOT / "research" / "数据" / "backups"
REPORT_DIR = ROOT / "reports"

CURRENT = {
    "canonical_A湖畔": ROOT / "research" / "数据" / "长春A食堂_菜品库.csv",
    "runtime_A湖畔": ROOT / "smart_canteen_web" / "data" / "datasets" / "a_hupan" / "dish_library.csv",
    "canonical_F基础园": ROOT / "research" / "数据" / "长春F食堂_菜品库.csv",
    "runtime_F基础园": ROOT / "smart_canteen_web" / "data" / "datasets" / "f_jichuyuan" / "dish_library.csv",
}

EXPECTED_CHANGES = {
    "canonical_A湖畔": 110,
    "runtime_A湖畔": 110,
    "canonical_F基础园": 84,
    "runtime_F基础园": 84,
}

ALLOWED = {"荤菜", "素菜", "主食", "水果", "甜品"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    backups = sorted(BACKUP_ROOT.glob("菜品类别第二轮修正_*"))
    assert backups, "未找到第二轮备份目录"
    backup = backups[-1]

    reports = sorted(REPORT_DIR.glob("菜品类别第二轮修正报告_*.csv"))
    assert reports, "未找到第二轮修正报告"
    report = reports[-1]

    for label, current_path in CURRENT.items():
        backup_path = backup / f"{label}_{current_path.name}"
        before = read_rows(backup_path)
        after = read_rows(current_path)
        assert len(before) == len(after), f"{label}: 行数发生变化"
        assert before and after, f"{label}: 文件为空"
        assert before[0].keys() == after[0].keys(), f"{label}: 列结构发生变化"

        changed = 0
        for index, (old, new) in enumerate(zip(before, after), start=2):
            for column in old:
                if column == "类别":
                    if old[column] != new[column]:
                        changed += 1
                else:
                    assert old[column] == new[column], (
                        f"{label}: 第{index}行非类别字段 {column} 被修改"
                    )
            assert new["类别"] in ALLOWED, (
                f"{label}: 第{index}行存在非法类别 {new['类别']}"
            )
        assert changed == EXPECTED_CHANGES[label], (
            f"{label}: 类别改动数 {changed} != {EXPECTED_CHANGES[label]}"
        )
        print(label, "rows=", len(after), "category_changes=", changed,
              "distribution=", dict(Counter(row["类别"] for row in after)))

    for canonical_label, runtime_label in (
        ("canonical_A湖畔", "runtime_A湖畔"),
        ("canonical_F基础园", "runtime_F基础园"),
    ):
        canonical = read_rows(CURRENT[canonical_label])
        runtime = read_rows(CURRENT[runtime_label])
        # Runtime libraries intentionally keep a reduced column set. Verify every
        # field used by the web app rather than requiring identical schemas.
        runtime_fields = list(runtime[0])
        assert len(canonical) == len(runtime), f"{canonical_label} 与网页运行副本行数不一致"
        for index, (full_row, runtime_row) in enumerate(zip(canonical, runtime), start=2):
            for field in runtime_fields:
                assert full_row[field] == runtime_row[field], (
                    f"{canonical_label}: 第{index}行字段 {field} 与网页运行副本不一致"
                )

    report_rows = read_rows(report)
    assert len(report_rows) == 194, f"报告行数 {len(report_rows)} != 194"
    report_keys = {
        (row["菜品库"], row["菜品id"], row["菜品名称"], row["原类别"], row["修改后类别"])
        for row in report_rows
    }
    assert len(report_keys) == len(report_rows), "报告存在重复修正记录"
    assert any(
        row.get("菜品名称") == "红茶瑞士卷" and row.get("修改后类别") == "甜品"
        for row in report_rows
    ), "报告未记录红茶瑞士卷修正"

    red_tea_labels = []
    for label in CURRENT:
        matches = [row for row in read_rows(CURRENT[label]) if row.get("菜品名称") == "红茶瑞士卷"]
        if matches:
            red_tea_labels.append(label)
            assert all(row["类别"] == "甜品" for row in matches), f"{label}: 红茶瑞士卷仍未修正"
    assert red_tea_labels, "两个菜品库均未找到红茶瑞士卷"

    print("runtime_sync=ok")
    print("report_rows=", len(report_rows))
    print("red_tea_swiss_roll=甜品")
    print("backup=", backup)
    print("report=", report)


if __name__ == "__main__":
    main()
