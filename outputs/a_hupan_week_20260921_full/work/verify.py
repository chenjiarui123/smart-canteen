import csv
import hashlib
import json
import math
import re
import zipfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
entries = json.loads((OUT/'work/week.json').read_text(encoding='utf-8'))
assert len(entries) == 21
assert len({(e['date'], e['meal']) for e in entries}) == 21
assert {e['date'] for e in entries} == {(date(2026,9,21)+timedelta(days=n)).isoformat() for n in range(7)}
assert {e['meal'] for e in entries} == {'早餐','午餐','晚餐'}
with (ROOT/'smart_canteen_web/data/datasets/a_hupan/dish_library.csv').open(encoding='utf-8-sig') as f:
    library = {r['菜品id']:r for r in csv.DictReader(f)}
previous = {}
for meal in ['早餐','午餐','晚餐']:
    path = ROOT/f'outputs/a_hupan_week_20260914/menus/A湖畔_2026-09-20_周日_{meal}.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:
        prior = list(csv.reader(f))
    previous[meal] = {r[3] for r in prior if r[0]=='菜品' and r[1]=='常用菜'}
checks = []
for e in entries:
    path = OUT/f"menus/A湖畔_{e['weekday']}_{e['meal']}.csv"
    assert path.read_bytes().startswith(b'\xef\xbb\xbf')
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == ['推荐来源','菜品种类','菜品ID','菜品名称','预测销售额（元）','预测销量','销量单位']
    assert all(len(r)==7 and any(r) for r in rows)
    assert not re.search(r'\d{4}-\d{2}-\d{2}|情景预测|气温|真实销售历史截至|低频菜为备选',path.read_text(encoding='utf-8-sig'))
    records = [r for r in rows[1:] if r[2]]
    expected = {i['dish_id']:i for i in e['high']+e['low']}
    assert len(records)==len(expected)==len(e['high'])+24
    assert len({r[2] for r in records})==len(records)
    assert sum(r[0]=='低频潜力菜' for r in records)==24
    group = None
    for r in rows[1:]:
        if not r[2]:
            assert not any(r[2:])
            group = r[:2]
            continue
        assert group==r[:2]
        i = expected[r[2]]
        assert r[0] == ('常用菜' if i['recommendation_source']=='high_frequency' else '低频潜力菜')
        assert r[1]==i['category']==library[r[2]]['类别']
        assert r[3]==i['dish_name']==library[r[2]]['菜品名称']
        assert r[6]==i['unit']
        for text,number in [(r[4],i['reference_revenue']),(r[5],i['quantity'])]:
            assert math.isfinite(float(text)) and float(text)>=0
            assert abs(float(text)-number)<0.00501
        assert i['quantity']==i['model_quantity']*i['business_units_per_model_unit']
    current = {i['dish_id'] for i in e['high']}
    overlap = len(current & previous[e['meal']])
    assert math.isclose(e['audit']['adjacent_repeat_rate'],overlap/len(current))
    assert e['audit']['adjacent_reference_source']=='automatic_high_frequency'
    assert e['audit']['constraint_violation_count']==0
    previous[e['meal']] = current
    checks.append({'weekday':e['weekday'],'meal':e['meal'],'high':len(current),'low':24,
                   'shared_with_previous':overlap,'repeat_rate':overlap/len(current),
                   'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
sample_path = ROOT/'outputs/a_hupan_week_20260921_sample/work/sample.json'
sample = json.loads(sample_path.read_text(encoding='utf-8'))['proposal']
entry = next(e for e in entries if e['weekday']=='周一' and e['meal']=='午餐')
assert entry['high']==sample['high_frequency_menu']
assert entry['low']==sample['low_frequency_opportunities']
assert len(list((OUT/'menus').glob('*.csv')))==21
(OUT/'work/verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
archive = OUT/'A湖畔_周一至周日_一周三餐.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for c in checks:
        path=OUT/f"menus/A湖畔_{c['weekday']}_{c['meal']}.csv"
        z.write(path,path.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist())==21
    assert all(n.endswith('.csv') for n in z.namelist())
print(json.dumps({'files':len(checks),'high_total':sum(c['high'] for c in checks),
                  'low_total':sum(c['low'] for c in checks),'monday_sample_matches':True,
                  'all_previous_meal_references_verified':True,'archive':str(archive)},ensure_ascii=False))
