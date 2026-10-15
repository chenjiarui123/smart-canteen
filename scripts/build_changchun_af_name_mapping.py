#!/usr/bin/env python3
"""Build a reproducible old-name/price to corrected-name mapping for canteen A or F.

This is a one-time audit helper. It uses the existing corrected dataset as the
reconciliation target, then solves the ambiguous garbled-name assignments under
exact quantity, order-count, and sales-amount constraints.
"""

from __future__ import annotations

import argparse
import difflib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


CANTEENS = {
    "A": {
        "name": "长春A食堂",
        "source": Path("1-长春A食堂/长春A食堂菜品销量.xlsx"),
        "reference": Path("数据/长春A食堂_菜品销量_菜名修正后.csv"),
        "library": Path("数据/长春A食堂_菜品库.csv"),
    },
    "F": {
        "name": "长春F食堂",
        "source": Path("6-长春F食堂/长春F食堂菜品销量.xlsx"),
        "reference": Path("数据/长春F食堂_菜品销量_菜名修正后.csv"),
        "library": Path("数据/长春F食堂_菜品库.csv"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canteen", choices=sorted(CANTEENS), required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="项目根目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出 CSV；默认写入数据/长春X食堂_历史菜品名称转换映射.csv",
    )
    return parser.parse_args()


def processed_quantity(raw: pd.DataFrame) -> pd.Series:
    price_unit = raw["菜品单价"].astype(str).str.extract(r"元/(\d+)(.*)$")
    denominator = pd.to_numeric(price_unit[0], errors="coerce")
    unit = price_unit[1].fillna("")
    gram_like = denominator.gt(1) & unit.map(lambda value: value == "克" or "�" in value)
    result = raw["菜品总销量"].astype(float).copy()
    result.loc[gram_like] = result.loc[gram_like] / denominator.loc[gram_like]
    return result


def prepare_inputs(root: Path, code: str):
    config = CANTEENS[code]
    raw = pd.read_excel(root / config["source"])
    reference = pd.read_csv(root / config["reference"])
    library = pd.read_csv(root / config["library"])

    raw["日期"] = pd.to_datetime(raw["日期"]).dt.strftime("%Y-%m-%d")
    if "日期" not in reference.columns and {"年", "月", "日"}.issubset(reference.columns):
        reference["日期"] = pd.to_datetime(
            {
                "year": reference["年"],
                "month": reference["月"],
                "day": reference["日"],
            }
        ).dt.strftime("%Y-%m-%d")
    raw["销量分_处理"] = np.rint(processed_quantity(raw) * 100).astype("int64")
    raw["销售额分_处理"] = np.rint(raw["销售额"].fillna(0) * 100).astype("int64")
    reference["销量分_处理"] = np.rint(reference["菜品总销量"] * 100).astype("int64")
    reference["销售额分_处理"] = np.rint(reference["销售额"].fillna(0) * 100).astype("int64")
    return raw, reference, library


def build_mapping(root: Path, code: str) -> pd.DataFrame:
    config = CANTEENS[code]
    raw, reference, library = prepare_inputs(root, code)
    is_garbled = raw["菜品名称"].astype(str).str.contains("�", regex=False)
    clean_names = set(raw.loc[~is_garbled, "菜品名称"].astype(str))

    residual = {
        (row.日期, row.餐次, row.菜品名称): [
            int(row.销量分_处理),
            int(row.订单数量),
            int(row.销售额分_处理),
        ]
        for row in reference.itertuples()
    }
    for row in raw.loc[~is_garbled].itertuples():
        values = residual[(row.日期, row.餐次, str(row.菜品名称))]
        values[0] -= int(row.销量分_处理)
        values[1] -= int(row.订单数量)
        values[2] -= int(row.销售额分_处理)

    if any(any(value < 0 for value in values) for values in residual.values()):
        raise RuntimeError("原名保留记录已超过参考结果，无法建立一致映射")

    active_by_meal = defaultdict(list)
    for (date, meal, target), values in residual.items():
        if any(value > 0 for value in values):
            active_by_meal[(date, meal)].append((target, values))

    key_labels: list[tuple[str, str]] = []
    key_contributions: list[dict[tuple[str, str], tuple[int, int, int]]] = []
    candidates: list[list[str]] = []
    garbled = raw.loc[is_garbled]

    for key, rows in garbled.groupby(["菜品名称", "菜品单价"], sort=False):
        grouped = rows.groupby(["日期", "餐次"], sort=False).agg(
            quantity=("销量分_处理", "sum"),
            orders=("订单数量", "sum"),
            sales=("销售额分_处理", "sum"),
        )
        contribution = {
            (date, meal): (int(row.quantity), int(row.orders), int(row.sales))
            for (date, meal), row in grouped.iterrows()
        }
        possible = None
        for date_meal, values in contribution.items():
            fitting = {
                target
                for target, capacity in active_by_meal[date_meal]
                if all(capacity[index] >= values[index] for index in range(3))
            }
            possible = fitting if possible is None else possible & fitting

        if not possible:
            raise RuntimeError(f"映射键没有可行目标：{key!r}")
        key_labels.append((str(key[0]), str(key[1])))
        key_contributions.append(contribution)
        candidates.append(sorted(possible))

    variable_keys: list[int] = []
    variable_targets: list[str] = []
    objective: list[float] = []
    for key_index, (key, possible) in enumerate(zip(key_labels, candidates)):
        old_name, old_price = key
        skeleton = re.sub("�+", "", old_name)
        for target in possible:
            variable_keys.append(key_index)
            variable_targets.append(target)
            similarity = difflib.SequenceMatcher(None, skeleton, target).ratio()
            exact_price_seen = not raw.loc[
                raw["菜品名称"].eq(target) & raw["菜品单价"].astype(str).eq(old_price)
            ].empty
            objective.append(1 - similarity - (0.15 if exact_price_seen else 0))

    variables_by_key = defaultdict(list)
    for variable_index, key_index in enumerate(variable_keys):
        variables_by_key[key_index].append(variable_index)

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[int] = []
    right_hand_side: list[int] = []
    constraint_index = 0

    for key_index in range(len(key_labels)):
        for variable_index in variables_by_key[key_index]:
            matrix_rows.append(constraint_index)
            matrix_columns.append(variable_index)
            matrix_values.append(1)
        right_hand_side.append(1)
        constraint_index += 1

    residual_rows = {}
    for group_key, values in residual.items():
        if any(value > 0 for value in values):
            for metric_index, value in enumerate(values):
                residual_rows[(group_key, metric_index)] = constraint_index
                right_hand_side.append(value)
                constraint_index += 1

    for variable_index, (key_index, target) in enumerate(
        zip(variable_keys, variable_targets)
    ):
        for (date, meal), values in key_contributions[key_index].items():
            group_key = (date, meal, target)
            for metric_index, value in enumerate(values):
                row_index = residual_rows.get((group_key, metric_index))
                if row_index is not None and value:
                    matrix_rows.append(row_index)
                    matrix_columns.append(variable_index)
                    matrix_values.append(value)

    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(constraint_index, len(variable_keys)),
    ).tocsr()
    bounds = np.asarray(right_hand_side, dtype=float)
    result = milp(
        c=np.asarray(objective),
        integrality=np.ones(len(variable_keys)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, bounds, bounds),
        options={"time_limit": 120, "mip_rel_gap": 0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"映射约束求解失败：{result.message}")

    selected = np.flatnonzero(np.asarray(result.x) > 0.5)
    if len(selected) != len(key_labels):
        raise RuntimeError("映射解不完整")
    bad_mapping = {
        key_labels[variable_keys[index]]: variable_targets[index] for index in selected
    }

    distinct = (
        raw.groupby(["菜品名称", "菜品单价"], sort=False)
        .agg(原始记录数=("菜品名称", "size"))
        .reset_index()
        .rename(columns={"菜品名称": "旧菜品名称", "菜品单价": "原菜品单价"})
    )
    distinct["新菜品名称"] = [
        bad_mapping.get((str(old_name), str(old_price)), str(old_name))
        for old_name, old_price in distinct[["旧菜品名称", "原菜品单价"]].itertuples(
            index=False, name=None
        )
    ]
    id_by_name = library.set_index("菜品名称")["菜品id"]
    distinct["菜品id"] = distinct["新菜品名称"].map(id_by_name)
    if distinct["菜品id"].isna().any():
        missing = distinct.loc[distinct["菜品id"].isna(), "新菜品名称"].unique()
        raise RuntimeError(f"菜品库缺少菜品 ID：{missing[:10].tolist()}")

    distinct.insert(0, "食堂名称", config["name"])
    distinct["菜品id"] = distinct["菜品id"].astype("int64")
    distinct["是否菜名修正"] = distinct["旧菜品名称"] != distinct["新菜品名称"]
    return distinct[
        [
            "食堂名称",
            "旧菜品名称",
            "原菜品单价",
            "新菜品名称",
            "菜品id",
            "是否菜名修正",
            "原始记录数",
        ]
    ]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (
        args.output
        or root / "数据" / f"长春{args.canteen}食堂_历史菜品名称转换映射.csv"
    )
    mapping = build_mapping(root, args.canteen)
    output.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output, index=False, encoding="utf-8-sig")
    changed = int(mapping["是否菜名修正"].sum())
    print(f"已生成 {output}：{len(mapping)} 个菜名+单价映射，其中 {changed} 个需要修正")


if __name__ == "__main__":
    main()
