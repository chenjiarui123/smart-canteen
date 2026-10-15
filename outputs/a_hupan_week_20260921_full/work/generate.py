import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sdk/src'))
from smart_canteen_sdk import CanteenEngine
from smart_canteen_sdk.schemas import MenuHistory
from smart_canteen_sdk.adapters import load_legacy_csv

OUT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'smart_canteen_web/data/datasets/a_hupan'
PRIOR_DIR = ROOT / 'outputs/a_hupan_week_20260914/A湖畔_20260914-20260920_一周三餐/菜单'
MEALS = ['早餐', '午餐', '晚餐']
WEEKDAYS = '一二三四五六日'

def read_common_ids(path: Path):
    rows = list(csv.reader(path.open(encoding='utf-8-sig', newline='')))
    header_index = next(i for i, row in enumerate(rows) if row and row[0] == '行类型')
    header = rows[header_index]
    records = [dict(zip(header, row)) for row in rows[header_index + 1:]]
    return [row['菜品ID'] for row in records if row.get('行类型') == '菜品' and row.get('推荐来源') == '常用菜']

def prior_history():
    history = []
    for meal in MEALS:
        path = PRIOR_DIR / f'A湖畔_2026-09-20_周日_{meal}.csv'
        dish_ids = read_common_ids(path)
        if not dish_ids:
            raise RuntimeError(f'9月20日{meal}历史菜单为空: {path}')
        history.append(MenuHistory.model_validate({
            'date': '2026-09-20',
            'canteen_id': 'a_hupan',
            'meal': meal,
            'menu_id': f'prior_20260920_{meal}',
            'created_at': '2026-09-20T12:00:00+08:00',
            'source': 'automatic',
            'dish_ids': dish_ids,
            'low_frequency_dish_ids': [],
        }))
    return history

def main():
    data = load_legacy_csv(
        DATASET / 'sales.csv', DATASET / 'dish_library.csv',
        canteen_id='a_hupan', snapshot_id='weekly_20260921_history_20260610',
        runtime_sales_path=DATASET / 'runtime_actual_sales.csv',
        recommendation_history_path=DATASET / 'recommendation_history.csv',
        allow_legacy_unit_changes=True,
    )
    data.menu_history = prior_history()
    sales = pd.read_csv(DATASET / 'sales.csv', encoding='utf-8-sig')
    typical = sales.loc[sales['月'].eq(9)].drop_duplicates(['年', '月', '日'])
    mapping = {
        'temperature_c': '气温（℃）', 'humidity_pct': '相对湿度（%）',
        'precipitation_mm': '降水量（毫米）', 'wind_speed_kmh': '10米风速（公里/小时）',
    }
    weather = {key: float(typical[column].median()) for key, column in mapping.items()}
    engine = CanteenEngine()
    entries = []
    for day in pd.date_range('2026-09-21', '2026-09-27'):
        weekday = '周' + WEEKDAYS[day.weekday()]
        for meal in MEALS:
            request = {
                'canteen_id': 'a_hupan', 'prediction_date': str(day.date()), 'meal': meal,
                'weather': weather,
                'calendar': {'school_break': False, 'statutory_holiday': False, 'adjusted_workday': False},
                'opportunities': {'enabled': True, 'category': '全部', 'count': 24, 'refresh_seed': 0},
            }
            proposal = engine.recommend(request, data)
            if len(proposal.low_frequency_opportunities) != 24:
                raise RuntimeError(f'{day.date()} {meal} 低频候选不是24道')
            if proposal.audit['constraint_violation_count'] != 0:
                raise RuntimeError(f'{day.date()} {meal} 存在约束违规')
            if proposal.audit['adjacent_reference_source'] != 'automatic_high_frequency':
                raise RuntimeError(f'{day.date()} {meal} 没有参考前一天同餐次菜单')
            if proposal.audit['adjacent_repeat_rate'] is None:
                raise RuntimeError(f'{day.date()} {meal} 前一天重复率为空')
            entries.append({
                'date': str(day.date()), 'weekday': weekday, 'meal': meal,
                'high': [item.model_dump(mode='json') for item in proposal.high_frequency_menu],
                'low': [item.model_dump(mode='json') for item in proposal.low_frequency_opportunities],
                'audit': proposal.audit, 'weather': weather,
            })
            data.menu_history.extend(proposal.history_events)
    (OUT / 'work/week.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'entries': len(entries), 'low_total': sum(len(x['low']) for x in entries),
        'high_total': sum(len(x['high']) for x in entries),
        'reference_sources': sorted(set(x['audit']['adjacent_reference_source'] for x in entries)),
        'repeat_rates': [round(x['audit']['adjacent_repeat_rate'], 6) for x in entries],
    }, ensure_ascii=False))

if __name__ == '__main__':
    main()
