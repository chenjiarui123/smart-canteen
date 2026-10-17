#!/usr/bin/env python3
"""Build mappings for Jilin University canteens and extend A/F dish libraries."""

from __future__ import annotations

import argparse
import difflib
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {
    "hupan": {
        "name": "吉大湖畔智慧餐厅",
        "source": Path("research/数据/菜品销量_吉大湖畔智慧餐厅.csv"),
        "base_canteen": "长春A食堂",
        "library": Path("research/数据/长春A食堂_菜品库.csv"),
        "extension_source": "湖畔食堂扩展",
    },
    "jichuyuan": {
        "name": "吉大南校基础园智慧餐厅",
        "source": Path("research/数据/菜品销量_吉大南校基础园智慧餐厅.csv"),
        "base_canteen": "长春F食堂",
        "library": Path("research/数据/长春F食堂_菜品库.csv"),
        "extension_source": "基础园食堂扩展",
    },
}

LIBRARY_COLUMNS = [
    "菜品id",
    "菜品名称",
    "平均单价",
    "类别",
    "颜色",
    "营养",
    "口味",
    "历史平均销量",
    "历史平均销售额",
    "售卖天数",
    "数据天数",
    "总销量_处理后",
    "总销量_原始",
    "总销售额",
    "订单数量",
    "记录数",
    "原始菜名数",
    "特征来源",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1], help="项目根目录"
    )
    parser.add_argument(
        "--output-a-mapping",
        type=Path,
        help="A组映射；默认写入数据/长春A食堂_菜品名称转换映射.csv",
    )
    parser.add_argument(
        "--output-f-mapping",
        type=Path,
        help="F组映射；默认写入数据/长春F食堂_菜品名称转换映射.csv",
    )
    return parser.parse_args()


def normalized_name(value: str, drop_prefix: bool = False) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = "".join(
        char
        for char in text
        if char.isalnum() or "\u4e00" <= char <= "\u9fff" or char == "�"
    )
    if drop_prefix:
        text = re.sub(r"^[adjkmqxyz]{1,2}(?=[\u4e00-\u9fff])", "", text)
    return text


def price_number(value: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)元/", str(value))
    return float(match.group(1)) if match else math.nan


def processed_quantity(frame: pd.DataFrame) -> pd.Series:
    price_unit = frame["菜品单价"].astype(str).str.extract(r"元/(\d+)(.*)$")
    denominator = pd.to_numeric(price_unit[0], errors="coerce")
    unit = price_unit[1].fillna("")
    gram_like = denominator.gt(1) & unit.map(lambda value: value == "克" or "�" in value)
    result = frame["菜品总销量"].astype(float).copy()
    result.loc[gram_like] = result.loc[gram_like] / denominator.loc[gram_like]
    return result


def unique_lookup(names, *, drop_prefix: bool = False) -> dict[str, str]:
    grouped = defaultdict(list)
    for name in names:
        grouped[normalized_name(name, drop_prefix)].append(str(name))
    return {key: values[0] for key, values in grouped.items() if key and len(set(values)) == 1}


def choose_historical_name(options: pd.DataFrame, raw_price: str) -> tuple[str, float]:
    price = price_number(raw_price)
    scored = []
    for target, rows in options.groupby("新菜品名称", sort=False):
        exact = rows["原菜品单价"].astype(str).eq(str(raw_price)).any()
        candidate_prices = rows["原菜品单价"].map(price_number).dropna().to_numpy()
        difference = (
            float(np.min(np.abs(candidate_prices - price)))
            if candidate_prices.size and not math.isnan(price)
            else 999.0
        )
        scored.append((0 if exact else 1, difference, -len(rows), str(target)))
    best = min(scored)
    confidence = 0.98 if best[0] == 0 else max(0.75, 0.95 - best[1] * 0.2)
    return best[3], confidence


def ordered_segments_match(garbled: str, candidate: str) -> bool:
    segments = [normalized_name(part) for part in re.split("�+", garbled)]
    segments = [segment for segment in segments if segment]
    target = normalized_name(candidate)
    position = 0
    for segment in segments:
        found = target.find(segment, position)
        if found < 0:
            return False
        position = found + len(segment)
    return bool(segments)


def one_character_mask_match(garbled: str, candidate: str) -> bool:
    mask = re.sub("�+", "?", normalized_name(garbled))
    target = normalized_name(candidate)
    return len(mask) == len(target) and all(
        left == "?" or left == right for left, right in zip(mask, target)
    )


def garbled_score(
    old_name: str,
    raw_price: str,
    meal: str,
    old_dates: set[str],
    candidate: str,
    price_context: dict[str, set[str]],
    meal_context: dict[str, set[str]],
    date_meal_context: dict[str, set[tuple[str, str]]],
    date_meal_price_context: dict[str, set[tuple[str, str, str]]],
    target_clean_names: set[str],
) -> float:
    skeleton = normalized_name(old_name).replace("�", "")
    candidate_key = normalized_name(candidate)
    ratio = difflib.SequenceMatcher(None, skeleton, candidate_key).ratio()
    score = ratio * 0.58
    if ordered_segments_match(old_name, candidate):
        score += 0.18
    if one_character_mask_match(old_name, candidate):
        score += 0.22
    if skeleton and candidate_key and skeleton[0] == candidate_key[0]:
        score += 0.05
    if skeleton and candidate_key and skeleton[-1] == candidate_key[-1]:
        score += 0.04
    if str(raw_price) in price_context.get(candidate, set()):
        score += 0.22
    else:
        raw_value = price_number(raw_price)
        values = [price_number(value) for value in price_context.get(candidate, set())]
        values = [value for value in values if not math.isnan(value)]
        if values and not math.isnan(raw_value):
            difference = min(abs(value - raw_value) for value in values)
            score += max(0, 0.12 - difference * 0.08)
    if meal in meal_context.get(candidate, set()):
        score += 0.04
    if old_dates:
        appearances = date_meal_context.get(candidate, set())
        overlap = sum((date, meal) in appearances for date in old_dates) / len(old_dates)
        score += overlap * 0.18
        exact_appearances = date_meal_price_context.get(candidate, set())
        exact_overlap = sum(
            (date, meal, str(raw_price)) in exact_appearances for date in old_dates
        ) / len(old_dates)
        score += exact_overlap * 0.28
    if candidate in target_clean_names:
        score += 0.04
    expected_length = len(re.sub("�+", "?", normalized_name(old_name)))
    score -= min(abs(expected_length - len(candidate_key)) * 0.015, 0.12)
    return score


def infer_features(name: str, reference_rows: pd.DataFrame) -> dict[str, str]:
    if not reference_rows.empty:
        row = reference_rows.iloc[0]
        return {
            "类别": str(row["类别"]),
            "颜色": str(row["颜色"]),
            "营养": str(row["营养"]),
            "口味": str(row["口味"]),
            "特征来源": str(row.get("特征来源", "已有菜品库匹配")),
        }

    text = normalized_name(name)
    fruit_words = ["苹果", "香蕉", "西瓜", "橙", "柚", "梨", "葡萄", "草莓", "蓝莓", "火龙果", "哈密瓜", "桃"]
    dessert_words = ["蛋糕", "布丁", "冰粉", "冰淇淋", "慕斯", "奶茶", "酸奶", "果冻", "甜品", "巧克力", "蛋挞", "泡芙"]
    staple_words = ["米饭", "炒饭", "粥", "面", "馒头", "花卷", "包子", "饺", "馄饨", "烧麦", "饼", "面包", "三明治", "汉堡", "发糕", "窝头"]
    meat_words = ["猪", "牛", "羊", "鸡", "鸭", "鹅", "鱼", "虾", "蟹", "贝", "肉", "排骨", "肘", "肠", "肚", "肥肠", "蛋"]
    vegetable_words = ["菜", "瓜", "豆", "笋", "菇", "菌", "藕", "茄", "椒", "萝卜", "土豆", "山药", "木耳", "海带", "玉米", "沙拉"]

    if any(word in text for word in fruit_words) and not any(word in text for word in meat_words):
        category = "水果"
    elif any(word in text for word in dessert_words):
        category = "甜品"
    elif any(word in text for word in staple_words):
        category = "主食"
    elif any(word in text for word in meat_words):
        category = "荤菜"
    elif any(word in text for word in vegetable_words):
        category = "素菜"
    else:
        category = "素菜"

    if any(word in text for word in ["菠菜", "油菜", "西兰花", "生菜", "芹", "韭", "黄瓜", "青椒", "菜心", "豆角", "沙拉"]):
        color = "绿"
    elif any(word in text for word in ["番茄", "西红柿", "红椒", "红烧", "红油", "辣子", "草莓"]):
        color = "红"
    elif any(word in text for word in ["南瓜", "胡萝卜", "橙"]):
        color = "橙"
    elif any(word in text for word in ["紫薯", "紫米", "紫甘蓝"]):
        color = "紫"
    elif any(word in text for word in ["黑米", "黑椒", "木耳", "海带"]):
        color = "黑"
    elif any(word in text for word in ["白", "豆腐", "米饭", "馒头", "山药", "奶"]):
        color = "白"
    elif any(word in text for word in ["炸", "煎", "酥", "锅包", "蛋黄"]):
        color = "金黄"
    elif any(word in text for word in ["玉米", "鸡蛋", "咖喱", "黄"]):
        color = "黄"
    else:
        color = "棕" if category == "荤菜" else "绿"

    if category in {"主食", "甜品"}:
        nutrition = "高碳水"
    elif category in {"素菜", "水果"}:
        nutrition = "高纤维"
    elif any(word in text for word in ["炸", "酥", "五花", "肥肠", "奶油", "芝士", "锅包"]):
        nutrition = "高脂肪"
    else:
        nutrition = "高蛋白"

    if "酸辣" in text:
        taste = "酸辣"
    elif any(word in text for word in ["糖醋", "酸甜", "菠萝", "咕咾"]):
        taste = "酸甜"
    elif any(word in text for word in ["辣", "麻", "椒", "水煮", "香锅"]):
        taste = "辣"
    elif any(word in text for word in ["甜", "糖", "蜜", "奶", "枣", "豆沙", "巧克力"]):
        taste = "甜"
    elif "酸" in text:
        taste = "酸"
    elif any(word in text for word in ["清蒸", "白灼", "低脂", "沙拉", "粥", "原味"]):
        taste = "清淡"
    else:
        taste = "咸"
    return {"类别": category, "颜色": color, "营养": nutrition, "口味": taste, "特征来源": "菜名规则推断"}


def build_first_pass(
    source: pd.DataFrame,
    base_library: pd.DataFrame,
    historical_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    grouped = (
        source.groupby(["菜品名称", "菜品单价", "餐次"], sort=False)
        .agg(
            原始记录数=("菜品名称", "size"),
            日期集合=("_日期", lambda values: set(values.astype(str))),
        )
        .reset_index()
    )
    direct = historical_mapping.set_index(["旧菜品名称", "原菜品单价"])["新菜品名称"].to_dict()
    by_old = {name: rows for name, rows in historical_mapping.groupby("旧菜品名称", sort=False)}
    library_names = set(base_library["菜品名称"].astype(str))
    historical_names = set(historical_mapping["新菜品名称"].astype(str))
    library_norm = unique_lookup(library_names)
    rows = []
    target_clean_names = set()

    for row in grouped.itertuples(index=False):
        old_name, raw_price, meal = str(row.菜品名称), str(row.菜品单价), str(row.餐次)
        key = (old_name, raw_price)
        target = None
        source_label = None
        confidence = None
        if key in direct:
            target = str(direct[key])
            source_label = "现有A/F映射-菜名单价"
            confidence = 1.0
        elif old_name in by_old:
            target, confidence = choose_historical_name(by_old[old_name], raw_price)
            source_label = "现有A/F映射-同旧名近价"
        elif old_name in library_names:
            target = old_name
            source_label = (
                "现有菜品库精确匹配"
                if old_name in historical_names
                else "新增菜品-原名保留"
            )
            confidence = 1.0
        elif "�" not in old_name:
            normalized_target = library_norm.get(normalized_name(old_name))
            if normalized_target:
                target = normalized_target
                source_label = (
                    "现有菜品库清洗后匹配"
                    if normalized_target in historical_names
                    else "新增菜品-清洗后匹配"
                )
                confidence = 0.98
            else:
                target = old_name
                source_label = "新菜品-原名保留"
                confidence = 1.0
        if target is not None and "�" not in target:
            target_clean_names.add(target)
        rows.append(
            {
                "旧菜品名称": old_name,
                "原菜品单价": raw_price,
                "餐次": meal,
                "新菜品名称": target,
                "原始记录数": int(row.原始记录数),
                "日期集合": row.日期集合,
                "映射来源": source_label,
                "匹配置信度": confidence,
            }
        )
    return pd.DataFrame(rows), target_clean_names


def resolve_garbled(
    first_pass: pd.DataFrame,
    base_library: pd.DataFrame,
    historical_mapping: pd.DataFrame,
    target_clean_names: set[str],
) -> pd.DataFrame:
    candidates = sorted(set(base_library["菜品名称"].astype(str)) | target_clean_names)
    price_context = defaultdict(set)
    meal_context = defaultdict(set)
    date_meal_context = defaultdict(set)
    date_meal_price_context = defaultdict(set)
    for row in historical_mapping.itertuples():
        price_context[str(row.新菜品名称)].add(str(row.原菜品单价))
    for row in first_pass.loc[first_pass["新菜品名称"].notna()].itertuples():
        target = str(row.新菜品名称)
        price_context[target].add(str(row.原菜品单价))
        meal_context[target].add(str(row.餐次))
        for date in row.日期集合:
            date_meal_context[target].add((str(date), str(row.餐次)))
            date_meal_price_context[target].add(
                (str(date), str(row.餐次), str(row.原菜品单价))
            )
    for row in base_library.itertuples():
        price_context[str(row.菜品名称)].add(f"{float(row.平均单价):.2f}元/50克")

    unresolved = first_pass["新菜品名称"].isna()
    for index, row in first_pass.loc[unresolved].iterrows():
        old_name = str(row["旧菜品名称"])
        skeleton = normalized_name(old_name).replace("�", "")
        ordered = [candidate for candidate in candidates if ordered_segments_match(old_name, candidate)]
        fuzzy = sorted(
            candidates,
            key=lambda candidate: difflib.SequenceMatcher(
                None, skeleton, normalized_name(candidate)
            ).ratio(),
            reverse=True,
        )[:30]
        shortlist = list(dict.fromkeys(ordered + fuzzy))
        scored = [
            (
                garbled_score(
                    old_name,
                    str(row["原菜品单价"]),
                    str(row["餐次"]),
                    set(row["日期集合"]),
                    candidate,
                    price_context,
                    meal_context,
                    date_meal_context,
                    date_meal_price_context,
                    target_clean_names,
                ),
                candidate,
            )
            for candidate in shortlist
        ]
        scored.sort(reverse=True)
        best_score, best_name = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -1.0
        margin = best_score - second_score
        if (one_character_mask_match(old_name, best_name) and margin >= 0.05) or (
            best_score >= 1.05 and margin >= 0.10
        ):
            confidence_label = "高"
            numeric_confidence = min(0.99, 0.90 + max(margin, 0) * 0.2)
        elif best_score >= 0.82 and margin >= 0.04:
            confidence_label = "中"
            numeric_confidence = min(0.89, 0.72 + max(margin, 0) * 0.25)
        else:
            confidence_label = "低"
            numeric_confidence = max(0.50, min(0.71, 0.55 + max(margin, 0) * 0.2))
        first_pass.at[index, "新菜品名称"] = best_name
        first_pass.at[index, "映射来源"] = f"乱码名称+单价+餐次推断-{confidence_label}"
        first_pass.at[index, "匹配置信度"] = round(numeric_confidence, 4)

    signatures = first_pass["旧菜品名称"].astype(str).map(
        lambda value: re.sub("�+", "�", value)
    )
    low_rows = first_pass["映射来源"].astype(str).str.endswith("-低")
    for index in first_pass.index[low_rows]:
        peers = first_pass.loc[
            signatures.eq(signatures.at[index])
            & first_pass["原菜品单价"].astype(str).eq(
                str(first_pass.at[index, "原菜品单价"])
            )
            & ~first_pass["映射来源"].astype(str).str.endswith("-低")
        ]
        if peers.empty:
            continue
        weighted = peers.groupby("新菜品名称")["原始记录数"].sum().sort_values(ascending=False)
        if len(weighted) == 1 or weighted.iloc[0] > weighted.iloc[1]:
            first_pass.at[index, "新菜品名称"] = str(weighted.index[0])
            first_pass.at[index, "映射来源"] = "乱码同形变体继承-高"
            first_pass.at[index, "匹配置信度"] = 0.95
    return first_pass


def attach_mapping(source: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    lookup = mapping.set_index(["旧菜品名称", "原菜品单价", "餐次"])["新菜品名称"]
    keys = pd.MultiIndex.from_frame(
        source[["菜品名称", "菜品单价", "餐次"]].rename(
            columns={"菜品名称": "旧菜品名称", "菜品单价": "原菜品单价"}
        )
    )
    result = source.copy()
    result["新菜品名称"] = lookup.reindex(keys).to_numpy()
    if result["新菜品名称"].isna().any():
        raise RuntimeError("目标食堂映射未覆盖全部记录")
    return result


def build_new_library_rows(
    mapped_source: pd.DataFrame,
    base_library: pd.DataFrame,
    feature_references: pd.DataFrame,
    name_to_id: dict[str, int],
    extension_source: str,
) -> pd.DataFrame:
    existing_names = set(base_library["菜品名称"].astype(str))
    new_names = sorted(set(mapped_source["新菜品名称"].astype(str)) - existing_names)
    if not new_names:
        return pd.DataFrame(columns=LIBRARY_COLUMNS)

    mapped_source = mapped_source.copy()
    mapped_source["_处理后销量"] = processed_quantity(mapped_source)
    mapped_source["_价格数值"] = mapped_source["菜品单价"].map(price_number)
    data_days = mapped_source["_日期"].nunique()
    feature_exact = {name: rows for name, rows in feature_references.groupby("菜品名称", sort=False)}
    feature_norm = unique_lookup(feature_references["菜品名称"].astype(str))
    feature_no_prefix = unique_lookup(
        feature_references["菜品名称"].astype(str), drop_prefix=True
    )
    rows = []

    for name in new_names:
        group = mapped_source.loc[mapped_source["新菜品名称"].eq(name)]
        weights = group["订单数量"].fillna(0).astype(float)
        valid_price = group["_价格数值"].notna()
        if valid_price.any() and weights.loc[valid_price].sum() > 0:
            average_price = np.average(
                group.loc[valid_price, "_价格数值"], weights=weights.loc[valid_price]
            )
        else:
            average_price = group["_价格数值"].mean()

        feature_rows = feature_exact.get(name, pd.DataFrame())
        feature_source = None
        if feature_rows.empty:
            normalized_match = feature_norm.get(normalized_name(name))
            if normalized_match:
                feature_rows = feature_exact[normalized_match]
                feature_source = "特征库清洗后匹配"
        if feature_rows.empty:
            no_prefix_match = feature_no_prefix.get(normalized_name(name, drop_prefix=True))
            if no_prefix_match:
                feature_rows = feature_exact[no_prefix_match]
                feature_source = "特征库去前缀匹配"
        features = infer_features(name, feature_rows)
        if feature_source:
            features["特征来源"] = feature_source
        elif not feature_rows.empty and "特征来源" not in feature_rows.columns:
            features["特征来源"] = "dishes.csv精确匹配"
        elif not feature_rows.empty:
            features["特征来源"] = str(feature_rows.iloc[0].get("特征来源", "已有菜品库匹配"))
        if features["特征来源"] == "菜名规则推断":
            features["特征来源"] = f"{extension_source}-菜名规则推断"

        total_processed = float(group["_处理后销量"].sum())
        total_raw = float(group["菜品总销量"].sum())
        total_sales = float(group["销售额"].sum())
        rows.append(
            {
                "菜品id": int(name_to_id[name]),
                "菜品名称": name,
                "平均单价": round(float(average_price), 4),
                "类别": features["类别"],
                "颜色": features["颜色"],
                "营养": features["营养"],
                "口味": features["口味"],
                "历史平均销量": round(total_processed / data_days, 6),
                "历史平均销售额": round(total_sales / data_days, 6),
                "售卖天数": int(group["_日期"].nunique()),
                "数据天数": int(data_days),
                "总销量_处理后": round(total_processed, 6),
                "总销量_原始": round(total_raw, 6),
                "总销售额": round(total_sales, 2),
                "订单数量": int(group["订单数量"].sum()),
                "记录数": int(len(group)),
                "原始菜名数": int(group["菜品名称"].nunique()),
                "特征来源": features["特征来源"],
            }
        )
    return pd.DataFrame(rows, columns=LIBRARY_COLUMNS)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_mappings = {
        "hupan": args.output_a_mapping
        or root / "research" / "数据" / "长春A食堂_菜品名称转换映射.csv",
        "jichuyuan": args.output_f_mapping
        or root / "research" / "数据" / "长春F食堂_菜品名称转换映射.csv",
    }
    historical_paths = [
        root / "research" / "数据" / "长春A食堂_历史菜品名称转换映射.csv",
        root / "research" / "数据" / "长春F食堂_历史菜品名称转换映射.csv",
    ]
    if all(path.exists() for path in historical_paths):
        historical_mapping = pd.concat(
            [pd.read_csv(path) for path in historical_paths], ignore_index=True
        )
    else:
        historical_mapping = pd.read_csv(
            root / "research" / "数据" / "长春A_F食堂_菜品名称转换映射.csv"
        )
    dishes = pd.read_csv(root / "research" / "dishes.csv")
    libraries = {
        code: pd.read_csv(root / config["library"]) for code, config in TARGETS.items()
    }

    all_library_rows = pd.concat(list(libraries.values()), ignore_index=True)
    id_name_pairs = pd.concat(
        [all_library_rows[["菜品id", "菜品名称"]], dishes[["菜品id", "菜品名称"]]],
        ignore_index=True,
    ).drop_duplicates()
    id_conflicts = id_name_pairs.groupby("菜品id")["菜品名称"].nunique()
    if (id_conflicts > 1).any():
        raise RuntimeError("现有菜品 ID 对应多个不同名称")
    name_to_id = (
        id_name_pairs.drop_duplicates("菜品名称").set_index("菜品名称")["菜品id"].astype(int).to_dict()
    )
    next_id = int(id_name_pairs["菜品id"].max()) + 1

    target_assets = {}
    all_target_names = set()
    for code, config in TARGETS.items():
        source = pd.read_csv(root / config["source"])
        source["_日期"] = pd.to_datetime(
            {"year": source["年"], "month": source["月"], "day": source["日"]}
        ).dt.strftime("%Y-%m-%d")
        base_library = libraries[code]
        base_mapping = historical_mapping.loc[
            historical_mapping["食堂名称"].eq(config["base_canteen"])
        ].copy()
        first_pass, clean_names = build_first_pass(source, base_library, base_mapping)
        mapping = resolve_garbled(first_pass, base_library, base_mapping, clean_names)
        if mapping["新菜品名称"].astype(str).str.contains("�", regex=False).any():
            raise RuntimeError(f"{config['name']} 映射结果仍包含乱码")
        mapped_source = attach_mapping(source, mapping)
        target_assets[code] = (source, mapping, mapped_source)
        all_target_names.update(mapped_source["新菜品名称"].astype(str))

    for name in sorted(all_target_names):
        if name not in name_to_id:
            name_to_id[name] = next_id
            next_id += 1

    feature_references = pd.concat(
        [
            all_library_rows[["菜品名称", "类别", "颜色", "营养", "口味", "特征来源"]],
            dishes[["菜品名称", "类别", "颜色", "营养", "口味"]].assign(
                特征来源="dishes.csv精确匹配"
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("菜品名称")

    target_mappings = {}
    summary_rows = []
    for code, config in TARGETS.items():
        source, mapping, mapped_source = target_assets[code]
        mapping = mapping.copy()
        mapping.insert(0, "食堂名称", config["name"])
        mapping["菜品id"] = mapping["新菜品名称"].map(name_to_id).astype("int64")
        mapping["是否菜名修正"] = mapping["旧菜品名称"] != mapping["新菜品名称"]
        mapping = mapping.drop(columns=["日期集合"])
        target_mappings[code] = mapping

        new_rows = build_new_library_rows(
            mapped_source,
            libraries[code],
            feature_references,
            name_to_id,
            config["extension_source"],
        )
        if not new_rows.empty:
            libraries[code] = pd.concat([libraries[code], new_rows], ignore_index=True)
            libraries[code].to_csv(root / config["library"], index=False, encoding="utf-8-sig")

        confidence_counts = mapping["映射来源"].value_counts()
        historical_names = set(
            historical_mapping.loc[
                historical_mapping["食堂名称"].eq(config["base_canteen"]), "新菜品名称"
            ].astype(str)
        )
        extended_name_count = len(
            set(mapped_source["新菜品名称"].astype(str)) - historical_names
        )
        summary_rows.append(
            {
                "食堂名称": config["name"],
                "原始记录数": len(source),
                "原始菜名数": source["菜品名称"].nunique(),
                "标准菜名数": mapped_source["新菜品名称"].nunique(),
                "新增菜品数": extended_name_count,
                "乱码映射数": int(mapping["旧菜品名称"].astype(str).str.contains("�", regex=False).sum()),
                "低置信度乱码映射数": int(
                    mapping["映射来源"].astype(str).str.endswith("-低").sum()
                ),
                "映射覆盖率": 1.0,
                "现有映射优先命中数": int(
                    confidence_counts.get("现有A/F映射-菜名单价", 0)
                    + confidence_counts.get("现有A/F映射-同旧名近价", 0)
                ),
            }
        )

    historical_output = historical_mapping.copy()
    historical_output["餐次"] = "*"
    historical_output["映射来源"] = "历史A/F映射"
    historical_output["匹配置信度"] = 1.0
    mapping_columns = [
        "食堂名称",
        "旧菜品名称",
        "原菜品单价",
        "餐次",
        "新菜品名称",
        "菜品id",
        "是否菜名修正",
        "原始记录数",
        "映射来源",
        "匹配置信度",
    ]
    for code, config in TARGETS.items():
        historical_group = historical_output.loc[
            historical_output["食堂名称"].eq(config["base_canteen"])
        ]
        group_mapping = pd.concat(
            [historical_group, target_mappings[code]], ignore_index=True, sort=False
        )
        output_path = output_mappings[code]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        group_mapping[mapping_columns].to_csv(
            output_path, index=False, encoding="utf-8-sig"
        )
        print(f"已生成 {output_path}：{len(group_mapping)} 条映射")
    summary = pd.DataFrame(summary_rows)
    summary_path = root / "research" / "数据" / "湖畔_基础园_菜品映射与菜品库扩展汇总.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
