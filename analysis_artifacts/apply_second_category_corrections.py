from __future__ import annotations

import argparse
import csv
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = {
    "A湖畔": ROOT / "research/数据/长春A食堂_菜品库.csv",
    "F基础园": ROOT / "research/数据/长春F食堂_菜品库.csv",
}
RUNTIME = {
    "A湖畔": ROOT / "smart_canteen_web/data/datasets/a_hupan/dish_library.csv",
    "F基础园": ROOT / "smart_canteen_web/data/datasets/f_jichuyuan/dish_library.csv",
}


def norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"^[adjkmqxyz]{1,2}(?=[\u4e00-\u9fff])", "", text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


RAW_TARGETS = {
    "甜品": {
        "红茶瑞士卷", "K抹茶瑞士卷", "K土豆面包", "k手撕面包", "K奶酪面包",
        "K华妃爱喝小冰粉", "K冰粉", "K红豆冰粉", "红酒醉雪梨", "橙汁焗南瓜",
        "金丝香蕉", "金盏蜜瓜香芋菠萝忠", "橙汁贝贝瓜", "Q麦脆香蕉",
        "K广式雷沙奶黄", "K樱花酥", "K天蓬元帅爱喝小奶盖", "K榴莲酥",
        "K虎皮卷", "K红小豆卷", "K一品酥", "K七彩糖果紫薯卷", "K抹茶抱抱卷",
        "K枣泥酥", "K紫薯红豆卷", "K红糖四角糕", "K红糖山楂卷",
        "K红糖蜜豆卷", "K绿豆酥", "南瓜🎃松糕",
        "可可芝士奶酪包", "国潮酥", "奥利奥可可奶盖", "奶香吐司", "开心糕",
        "手工沙琪玛", "抹茶炼乳奶酪包", "杏花糯米糕", "樱花奶豆腐",
        "紫米奶酪糕", "红豆红糖双拼卷", "红丝绒榴莲裂纹包", "K火腿玉米吐司",
        "抱抱卷", "拔丝酥皇菜", "奶油松松小马", "奶酥提子", "绿豆红枣卷",
        "酒酿冰酥酪", "果酱排包", "杂粮排包", "K绿豆沙酥饼",
        "K紫薯大枣糕", "K蜜豆酥饼",
        "冰皮绿豆饼", "果脯地瓜酥", "K豆沙酥饼", "紫薯玫瑰糕",
        "山药紫薯糕", "炸牛奶", "k抹茶奶酪小贝", "k牛奶曲奇", "牛奶棒",
        "红豆烤蛋奶", "k肉松包",
    },
    "主食": {
        "宫保鸡丁炒饭", "黑椒鸡片炒饭", "流沙包", "豆包(红豆馅）",
        "K香甜玉米粒饼", "K酱香卷", "拌饭", "K东北特色香煎豆包",
        "K黑芝麻包", "杂粮包", "东北大饭包", "海带包饭", "J韩式拌饭",
        "K玉米山楂包", "K金三紫菜包饭", "东北油滋包", "奶香花生包",
        "川香麻辣粉", "油润香包（油滋酸菜）", "甜辣年糕", "轻食拌饭",
        "葱香千层卷", "蜂蜜芥末年糕", "麻辣粉耗子", "拇指煎包",
        "香菇小白菜小笼包", "花生包", "大枣包", "南瓜包", "土豆包",
        "抹茶莲子包", "春卷", "甘蓝包", "香菇菜包", "香煎热狗卷",
        "南瓜热狗卷", "酱香肉包", "猪肉长豆包", "K青海包（猪肉酸菜）",
        "吉廷翠辣香肉包", "川香肉沫豆腐包", "鲜虾玲珑包", "M奥尔良鸡肉包",
        "销魂鸡腿饭", "牛肉饭", "猪肉韭菜包", "甏肉干饭", "K荷包蛋堡",
    },
    "荤菜": {
        "木须柿子", "干锅手撕包菜炒肉", "炸藕盒", "肉沫麻婆豆腐",
        "干锅茶树菇炒肉", "红烧脊骨", "八珍半月牙骨", "大油边炒脆骨",
        "尖椒炒月牙骨", "酥焖脆骨", "酥闷脆骨", "蒜香小排", "美味花千骨",
        "锡纸麻辣大排", "王氏酱骨", "川香月牙骨", "重庆小排", "精烧月牙骨",
        "海米角瓜条", "孜然月牙骨", "Y麻辣月牙骨", "八珍月牙骨",
        "J水晶里脊", "Q泡椒无骨凤爪", "关东煮脆骨丸", "干煸脆骨丸",
        "庐山拌大骨", "果味小排", "民间烤脊骨", "江南小排",
        "泡椒柠檬无骨凤爪", "甜辣里脊", "糯米蒸大排", "红烧大排",
        "脊骨酸菜", "酸辣脆骨丸", "锡纸小排", "香㸆大排", "香㸆龙骨",
        "香辣无骨凤爪", "麻辣脆骨丸", "脊骨烩酸菜", "烤大排", "蒜泥脊骨",
        "避风塘烤大排", "话梅小排", "意大利黑醋小排", "绝味状元小排",
        "柠檬仔排", "葱香大排", "干锅小排", "炝拌脆骨", "金桔烧小排",
        "浓烫脊骨汤", "海米丝瓜", "X酱棒骨~特惠菜", "九层塔爆鲜鱿",
        "叉烧里脊", "木须火腿丝瓜", "芝麻里脊", "薄荷秘制小排",
        "豆花里脊", "风沙小排", "香叶焗小排", "香辣杏花仔排",
    },
    "素菜": {
        "橙汁藕片🔆", "葱烧素鸡豆腐", "素鸡豆腐", "鱼香茄条",
        "鱼香豆付泡", "鱼香茄煲",
    },
}

REASONS = {
    "甜品": "名称明确为糕点、烘焙、甜饮或加工甜食",
    "主食": "名称明确为饭、包、卷、粉、年糕或其他主食形态",
    "荤菜": "名称明确含肉、蛋、海米、水产、骨排或内脏",
    "素菜": "名称明确为不含肉类的蔬菜或豆制品菜肴",
}


TARGETS: dict[str, str] = {}
for category, names in RAW_TARGETS.items():
    for name in names:
        key = norm(name)
        if key in TARGETS and TARGETS[key] != category:
            raise AssertionError((name, TARGETS[key], category))
        TARGETS[key] = category


def build_changes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, path in CANONICAL.items():
        frame = pd.read_csv(path)
        for row in frame.itertuples(index=False):
            target = TARGETS.get(norm(row.菜品名称))
            if target and target != str(row.类别):
                rows.append({
                    "菜品库": dataset,
                    "菜品id": row.菜品id,
                    "菜品名称": row.菜品名称,
                    "原类别": row.类别,
                    "修改后类别": target,
                    "修改依据": REASONS[target],
                    "原特征来源": row.特征来源,
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["菜品库", "修改后类别", "菜品名称"]).reset_index(drop=True)
    return result


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def apply_file(path: Path, mapping: dict[str, str]) -> int:
    fields, rows = read_csv(path)
    before = [row.copy() for row in rows]
    count = 0
    for row in rows:
        target = mapping.get(norm(row["菜品名称"]))
        if target and row["类别"] != target:
            row["类别"] = target
            count += 1
    for old, new in zip(before, rows, strict=True):
        for field in fields:
            if field != "类别" and old[field] != new[field]:
                raise AssertionError((path, old["菜品名称"], field))
    write_csv(path, fields, rows)
    return count


def apply(result: pd.DataFrame) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ROOT / "research/数据/backups" / f"菜品类别第二轮修正_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    for dataset, path in CANONICAL.items():
        shutil.copy2(path, backup / f"canonical_{dataset}_{path.name}")
    for dataset, path in RUNTIME.items():
        shutil.copy2(path, backup / f"runtime_{dataset}_{path.name}")

    counts: dict[str, int] = {}
    for dataset in CANONICAL:
        subset = result[result["菜品库"] == dataset]
        mapping = {norm(name): category for name, category in zip(
            subset["菜品名称"], subset["修改后类别"], strict=True
        )}
        counts[f"canonical_{dataset}"] = apply_file(CANONICAL[dataset], mapping)
        counts[f"runtime_{dataset}"] = apply_file(RUNTIME[dataset], mapping)

    report = ROOT / "reports/菜品类别第二轮修正报告_20260803.csv"
    result.to_csv(report, index=False, encoding="utf-8-sig")
    print("备份目录", backup)
    print("修改计数", counts)
    print("报告", report)


parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()
changes = build_changes()
print("候选修改数", len(changes))
if not changes.empty:
    print(changes.groupby(["菜品库", "原类别", "修改后类别"]).size().to_string())
    print(changes.to_csv(index=False))
if args.apply:
    apply(changes)
