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
LIBRARIES = {
    "A": ROOT / "数据/长春A食堂_菜品库.csv",
    "F": ROOT / "数据/长春F食堂_菜品库.csv",
}
RUNTIME_LIBRARIES = {
    "A": ROOT / "smart_canteen_web/data/datasets/a_hupan/dish_library.csv",
    "F": ROOT / "smart_canteen_web/data/datasets/f_jichuyuan/dish_library.csv",
}


def norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"^[adjkmqxyz]{1,2}(?=[\u4e00-\u9fff])", "", text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


MANUAL = {
    "粉蒸鳕鱼": "荤菜",
    "水煎包": "主食",
    "特价荤菜一": "荤菜",
    "大福冰淇淋": "甜品",
    "桃酥炉果": "甜品",
    "糖蒜": "素菜",
    "甜蒜": "素菜",
    "糖醋精排": "荤菜",
    "糖醋胡萝卜丝": "素菜",
    "糖醋荷包蛋": "荤菜",
    "酸奶奥利奥锅包肉": "荤菜",
    "豆腐脑": "主食",
    "豆腐脑卤": "主食",
    "红酒雪梨": "甜品",
    "冰糖雪梨": "甜品",
    "蓝莓山药": "甜品",
    "蓝莓山药": "甜品",
    "贝贝蓝莓山药": "甜品",
    "蜜汁蓝莓山药": "甜品",
    "奶香山药": "甜品",
    "奶香果脯山药": "甜品",
    "蜜汁山药": "甜品",
    "山楂山药": "甜品",
    "蜜枣蒸南瓜": "甜品",
    "蜜汁小南瓜": "甜品",
    "鱼香鱼豆腐": "荤菜",
    "海米冬瓜": "荤菜",
    "腐竹拌火腿": "荤菜",
    "蟹味菇烧丝瓜": "素菜",
    "黄金沙绿贝贝瓜": "素菜",
    "黄金沙蜜贝贝瓜": "甜品",
    "小炒贝贝瓜": "素菜",
    "香炒贝贝瓜": "素菜",
    "果味贝贝瓜": "素菜",
    "口味贝贝瓜": "素菜",
    "绝味贝贝瓜": "素菜",
    "牛舌饼": "主食",
    "小堡贝": "主食",
    "紫甘蓝拌苹果": "素菜",
    "鱼香茄子墩": "素菜",
    "鱼香自制豆腐": "素菜",
    "鱼香日本豆腐": "素菜",
    "鱼香杏鲍菇": "素菜",
    "番茄年糕条": "主食",
    "辣白菜炒年糕": "主食",
    "韩式年糕条": "主食",
    "红豆汉堡": "甜品",
    "奶香椒麻山药": "素菜",
    "蜜汁烧香菇": "素菜",
    "蜜汁茄盒": "荤菜",
    "果味酸奶棕": "主食",
    "金沙玉米虾": "荤菜",
    "红茶瑞士卷": "甜品",
    "木须柿子": "荤菜",
}

AMBIGUOUS = {
    "无油鸡蛋糕",
    "农家一碗香",
    "鸭血粉丝",
    "南京鸭血粉丝",
    "蟹黄干捞粉丝煲",
    "鲜虾干捞粉丝煲",
    "蜜汁三鲜",
    "蜜焖三鲜",
    "橙汁焗南瓜",
    "橙汁贝贝瓜",
    "绝味贝勒爷",
    "面包鸡蛋饼",
    "金盏蜜瓜香芋菠萝忠",
}

FRESH_FRUIT = {
    "水果", "t水果", "冬枣", "哈密瓜", "圣女果", "富士苹果", "红富士苹果",
    "苹果", "果冻橙", "橙子", "白兰瓜", "白心火龙果", "皇冠梨", "香梨",
    "西瓜", "香蕉", "草莓", "葡萄", "蓝莓", "火龙果", "猕猴桃", "梨", "沙糖桔",
}

FRUIT_WORDS = (
    "苹果", "香蕉", "西瓜", "橙子", "果冻橙", "哈密瓜", "白兰瓜", "火龙果",
    "皇冠梨", "香梨", "雪梨", "冬枣", "草莓", "蓝莓", "葡萄", "猕猴桃",
)
FRUIT_PROCESS = (
    "蛋糕", "面包", "吐司", "餐包", "慕斯", "大福", "派", "酥", "饼", "糕",
    "奶昔", "雪媚娘", "泡芙", "麻糍", "糖葫芦", "沙拉", "果酱", "果汁",
    "拌", "焗", "藕", "白菜", "南瓜", "锅包肉", "肉", "鸡", "寿司", "青团",
    "烤", "炸", "脆皮", "芝士", "燕麦", "圈", "排", "汤", "山药",
)

STAPLE_WORDS = (
    "米饭", "炒饭", "盖饭", "饭团", "盒饭", "粥", "面条", "炒面", "拌面",
    "汤面", "冷面", "米线", "馄饨", "饺子", "水饺", "蒸饺", "煎饺", "锅贴",
    "包子", "水煎包", "馒头", "花卷", "烧麦", "油条", "发糕", "粽子", "窝头",
    "豆沙包", "糖包", "核桃包", "流沙包", "蜜豆包", "糖三角", "寿司", "馅饼",
    "肉夹馍", "汉堡", "三明治", "意面", "盖面",
)
STAPLE_EXCLUDE = (
    "蛋糕", "面包", "月饼", "桃酥", "酥饼", "饼干", "泡芙", "麻薯",
)

DESSERT_WORDS = (
    "蛋糕", "慕斯", "布丁", "冰淇淋", "雪媚娘", "麻薯", "麻糍", "蛋挞",
    "泡芙", "甜甜圈", "桃酥", "炉果", "饼干", "奶昔", "奶茶", "酸奶",
    "果冻", "月饼", "青团", "奶贝", "杨枝甘露", "巧克力", "绿豆糕",
    "红豆糕", "凉糕", "杏花糕", "山药糕", "冰糕", "糖葫芦", "水果沙拉",
    "面包", "蛋黄酥", "桂花糕", "瑞士卷",
)

ENTREE_MEAT_WORDS = (
    "锅包肉", "炖粉条", "粉蒸", "粉丝扇贝", "扇贝蒸粉丝", "精排骨",
    "蒸排骨", "蒸肉", "肉片炒", "肉片炖", "肉沫水晶粉", "肉沫玉米",
    "萝卜丸子粉丝汤", "五花肉冻豆腐", "玉米虾",
)

MEAT_WORDS = (
    "猪", "牛", "羊", "鸡", "鸭", "鹅", "鱼", "虾", "蟹", "贝", "鱿鱼",
    "章鱼", "蛤", "肉", "排骨", "精排", "肘", "肠", "肚", "肥肠", "猪肝",
    "牛腩", "牛柳", "牛排", "鸡腿", "鸡翅", "鸡柳", "鸡丁", "鸡片", "鸡排",
    "鸭腿", "鸭翅", "蛋", "茄盒", "藕盒",
)

VEG_WORDS = (
    "白菜", "菠菜", "油菜", "生菜", "芹菜", "韭菜", "圆白菜", "甘蓝", "黄瓜",
    "茄子", "番茄", "西红柿", "土豆", "萝卜", "山药", "莲藕", "藕片", "南瓜",
    "冬瓜", "苦瓜", "丝瓜", "豆角", "豆芽", "豆苗", "豆腐", "豆干", "干豆腐",
    "腐竹", "青椒", "尖椒", "木耳", "蘑菇", "香菇", "金针菇", "杏鲍菇",
    "海带", "西兰花", "菜花", "笋", "秋葵", "茼蒿", "蒜", "莼菜",
)


def proposed(name: str, current: str) -> tuple[str | None, str]:
    text = norm(name)
    if text in AMBIGUOUS:
        return None, "名称含义或菜谱不确定"
    if text in MANUAL:
        return MANUAL[text], "人工确认规则"
    if text in FRESH_FRUIT:
        return "水果", "鲜果精确词"
    if "特价荤菜" in text:
        return "荤菜", "名称明确含荤菜"
    dessert_hit = any(word in text for word in DESSERT_WORDS)
    if dessert_hit:
        return "甜品", "明确甜品形态"
    staple_hit = any(word in text for word in STAPLE_WORDS) and not any(
        word in text for word in STAPLE_EXCLUDE
    )
    if staple_hit:
        return "主食", "明确主食形态"
    animal_text = (
        text.replace("鱼香", "")
        .replace("贝贝瓜", "")
        .replace("贝勒爷", "")
        .replace("蟹味菇", "")
        .replace("鸡枞", "")
        .replace("鸡腿菇", "")
    )
    meat_hit = any(word in animal_text for word in MEAT_WORDS)
    if meat_hit and current != "主食":
        return "荤菜", "明确肉蛋水产"
    if meat_hit and current == "主食" and any(word in text for word in ENTREE_MEAT_WORDS):
        return "荤菜", "肉类菜肴被粉条或粉丝误判为主食"
    if any(word in text for word in FRUIT_WORDS) and not any(
        word in text for word in FRUIT_PROCESS
    ):
        return "水果", "鲜果名称且无加工词"
    if current == "水果":
        if any(word in text for word in VEG_WORDS):
            return "素菜", "含水果调味的蔬菜菜肴"
        return "甜品", "水果加工点心或甜品"
    if current in {"甜品", "荤菜"} and any(word in text for word in VEG_WORDS):
        return "素菜", "明确蔬菜豆制品"
    return None, "无法高置信度判断"


def build_changes(all_sources: bool = False) -> pd.DataFrame:
    rows = []
    for dataset, path in LIBRARIES.items():
        frame = pd.read_csv(path)
        for row in frame.itertuples(index=False):
            # 仅纠正现有名称规则推断形成的标签；菜品库已有人工/参考标签不自动覆盖。
            if not all_sources and "规则推断" not in str(row.特征来源):
                continue
            suggestion, reason = proposed(str(row.菜品名称), str(row.类别))
            if suggestion and suggestion != str(row.类别):
                rows.append(
                    {
                        "菜品库": dataset,
                        "菜品id": row.菜品id,
                        "菜品名称": row.菜品名称,
                        "原类别": row.类别,
                        "修改后类别": suggestion,
                        "修改依据": reason,
                        "原特征来源": row.特征来源,
                    }
                )
    return pd.DataFrame(rows)


def read_csv_strings(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv_strings(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def apply_to_file(path: Path, name_to_category: dict[str, str]) -> int:
    fields, rows = read_csv_strings(path)
    category_col = "类别" if "类别" in fields else "菜品类别"
    if "菜品名称" not in fields or category_col not in fields:
        raise ValueError(f"{path} 缺少菜品名称或类别列")
    before = [row.copy() for row in rows]
    changed = 0
    for row in rows:
        target = name_to_category.get(str(row["菜品名称"]))
        if target and row[category_col] != target:
            row[category_col] = target
            changed += 1
    for old, new in zip(before, rows, strict=True):
        for field in fields:
            if field != category_col and old[field] != new[field]:
                raise AssertionError(f"非类别字段发生变化：{path} / {field}")
    write_csv_strings(path, fields, rows)
    return changed


def apply_changes(result: pd.DataFrame) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ROOT / "数据/backups" / f"菜品类别修正_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    all_paths = {**{f"canonical_{k}": v for k, v in LIBRARIES.items()},
                 **{f"runtime_{k}": v for k, v in RUNTIME_LIBRARIES.items()}}
    for key, path in all_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        shutil.copy2(path, backup_dir / f"{key}_{path.name}")

    counts: dict[str, int] = {}
    for dataset in LIBRARIES:
        subset = result[result["菜品库"] == dataset]
        mapping = dict(zip(subset["菜品名称"], subset["修改后类别"], strict=True))
        counts[f"canonical_{dataset}"] = apply_to_file(LIBRARIES[dataset], mapping)
        counts[f"runtime_{dataset}"] = apply_to_file(RUNTIME_LIBRARIES[dataset], mapping)

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "菜品类别修正报告_20260803.csv"
    result.to_csv(report_path, index=False, encoding="utf-8-sig")
    print("备份目录", backup_dir)
    print("修改计数", counts)
    print("明细报告", report_path)


parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true")
parser.add_argument("--all-sources", action="store_true")
args = parser.parse_args()
result = build_changes(all_sources=args.all_sources)
print("候选修改数", len(result))
if not result.empty:
    print(result.groupby(["原类别", "修改后类别"]).size().to_string())
if args.apply:
    apply_changes(result)
else:
    print(result.to_csv(index=False))
