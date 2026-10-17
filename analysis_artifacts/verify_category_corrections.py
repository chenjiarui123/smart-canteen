from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKUP = sorted((ROOT / "research/数据/backups").glob("菜品类别修正_*"))[-1]
FILES = {
    "canonical_A": ROOT / "research/数据/长春A食堂_菜品库.csv",
    "runtime_A": ROOT / "smart_canteen_web/data/datasets/a_hupan/dish_library.csv",
    "canonical_F": ROOT / "research/数据/长春F食堂_菜品库.csv",
    "runtime_F": ROOT / "smart_canteen_web/data/datasets/f_jichuyuan/dish_library.csv",
}
ALLOWED = {"荤菜", "素菜", "主食", "水果", "甜品"}


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


changed_counts: dict[str, int] = {}
for key, current_path in FILES.items():
    backup_path = BACKUP / f"{key}_{current_path.name}"
    fields_before, before = read(backup_path)
    fields_after, after = read(current_path)
    assert fields_before == fields_after, (key, "列结构变化")
    assert len(before) == len(after), (key, "行数变化")
    changed = 0
    for old, new in zip(before, after, strict=True):
        assert old["菜品id"] == new["菜品id"]
        for field in fields_before:
            if field == "类别":
                if old[field] != new[field]:
                    changed += 1
            else:
                assert old[field] == new[field], (key, old["菜品名称"], field)
        assert new["类别"] in ALLOWED, (key, new["菜品名称"], new["类别"])
    changed_counts[key] = changed

for dataset, canonical_key, runtime_key in (
    ("A", "canonical_A", "runtime_A"),
    ("F", "canonical_F", "runtime_F"),
):
    canonical = pd.read_csv(FILES[canonical_key], dtype=str)
    runtime = pd.read_csv(FILES[runtime_key], dtype=str)
    merged = canonical[["菜品id", "菜品名称", "类别"]].merge(
        runtime[["菜品id", "菜品名称", "类别"]],
        on=["菜品id", "菜品名称"],
        suffixes=("_canonical", "_runtime"),
        validate="one_to_one",
    )
    assert len(merged) == len(canonical) == len(runtime), dataset
    assert (merged["类别_canonical"] == merged["类别_runtime"]).all(), dataset

report = pd.read_csv(ROOT / "reports/菜品类别修正报告_20260803.csv", dtype=str)
assert len(report) == changed_counts["canonical_A"] + changed_counts["canonical_F"]
assert not report.duplicated(["菜品库", "菜品id"]).any()
assert set(report["修改后类别"]).issubset(ALLOWED)

print("备份目录", BACKUP)
print("仅类别列修改计数", changed_counts)
print("报告明细数", len(report))
for dataset, path in (("A", FILES["canonical_A"]), ("F", FILES["canonical_F"])):
    frame = pd.read_csv(path)
    print(dataset, "行数", len(frame), "类别分布", frame["类别"].value_counts().to_dict())
print("验证通过")
