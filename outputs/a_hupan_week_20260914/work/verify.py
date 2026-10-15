import csv
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

OUT=Path(__file__).resolve().parent.parent
ROOT=OUT.parents[1]
entries=json.loads((OUT/'work/week.json').read_text())
assert len(entries)==21
sample=json.loads((OUT/'work/sample.json').read_text())[0]
actual=next(e for e in entries if e['date']=='2026-09-14' and e['meal']=='午餐')
assert sample['high']==actual['high'] and sample['low']==actual['low']
library={}
with (ROOT/'smart_canteen_web/data/datasets/a_hupan/dish_library.csv').open(encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):library[str(row['菜品id'])]=row
checks=[]
for entry in entries:
    name=f"A湖畔_{entry['date']}_{entry['weekday']}_{entry['meal']}.csv"
    path=OUT/'menus'/name
    assert path.read_bytes().startswith(b'\xef\xbb\xbf')
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.reader(f))
    assert all(len(r)==9 for r in rows)
    records=[r for r in rows if r[0]=='菜品']
    assert len(records)==len(entry['high'])+len(entry['low'])
    expected={i['dish_id']:i for i in entry['high']+entry['low']}
    assert len(expected)==len(records)
    for r in records:
        i=expected[r[3]]
        assert r[4]==i['dish_name']==library[r[3]]['菜品名称']
        assert r[2]==i['category']==library[r[3]]['类别']
        assert abs(float(r[5])-i['reference_revenue'])<=.00501
        assert abs(float(r[6])-i['quantity'])<=.00501
        assert r[7]==i['unit']
        assert math.isfinite(float(r[5])) and math.isfinite(float(r[6]))
        assert i['quantity']==i['model_quantity']*i['business_units_per_model_unit']
    previous=None
    for r in rows[6:]:
        if r[0]=='类别分组':previous=(r[1],r[2])
        elif r[0]=='菜品':assert previous==(r[1],r[2])
    assert entry['audit']['constraint_violation_count']==0
    checks.append({'date':entry['date'],'meal':entry['meal'],'high_count':len(entry['high']),
        'low_count':len(entry['low']),'shortage':max(0,24-len(entry['low'])),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'adjacent_repeat_rate':entry['audit']['adjacent_repeat_rate']})
assert len(list((OUT/'menus').glob('*.csv')))==21
(OUT/'work/verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
short=[c for c in checks if c['shortage']]
text='''# A湖畔智慧餐厅 2026年9月14日至20日菜单

每天早餐、午餐、晚餐分别一个CSV，共21个。文件使用UTF-8 BOM，可用Excel打开。常用菜与低频候选菜分别按荤菜、素菜、主食、甜品、水果及其他类别分组。每个类别另起标题行，每道菜独立一行，类别内部按销售额参考值降序。

每道菜提供菜品ID、菜名、菜品种类、预测销售额、预测销量和售卖单位。菜品ID与种类直接对应原菜品库。常用菜为需求预测；低频菜在数量列列示试餐参考量，销售额是按试餐量计算的参考值。低频菜为备选，不表示24道都已经选入正式菜单，也不应将所有候选合计理解为最终整餐需求。

本次采用用户同意的情景假设：非寒暑假、非法定节假日、非调休。天气取现有历史2025年9月30个日期的中位数：22.9℃、相对湿度53%、降水0毫米、风速8.95公里/小时。不是2026年目标周的实测天气或天气预报。

真实销售历史截至2026年6月10日，预测周距历史末日96至102天，属于远期情景预测。使用SDK 0.1.0与当前部署策略targeted_repeat_holiday_20260911，未重训、未修改参数。明确启用旧数据单位兼容模式，保留少数菜品曾发生历史单位变化的既有模型行为；不代表已经完成历史单位清洗。

销售数据固定不变，预测生成的自动常用菜单在本次计算中提供给后续日期，供前一天同餐次重复比较；不写入实际销售，也不冒充用户已确认菜单。原业务数据及正式推荐记录未修改。周一缺少周日菜单时相邻日重复率不可计算。

标准接口已经完成克的换算，CSV中的克是实际克数，请勿再次乘50。不同单位的销量不可直接相加。数值保留两位小数。

先展示的周一午餐实例与批量生成的周一午餐一致。校验涵盖21份CSV、菜品ID和类别匹配、数量和金额与SDK输出一致、各餐无重复菜品及常用菜硬约束违规数为0。

## 每餐数量

| 日期 | 餐次 | 常用菜 | 低频菜 | 低频缺额 |
|---|---|---:|---:|---:|
'''
for c in checks:text+=f"| {c['date']} | {c['meal']} | {c['high_count']} | {c['low_count']} | {c['shortage']} |\n"
if short:text+='\n部分餐次当前策略合格低频候选不足24道，已保留所有合格候选并明确标注缺额，没有重复菜或放宽筛选条件凑数。\n'
else:text+='\n全部21餐均返回24道低频候选，共504条低频候选记录。\n'
(OUT/'使用说明.md').write_text(text)
with zipfile.ZipFile(OUT/'A湖畔_20260914-20260920_一周三餐.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted((OUT/'menus').glob('*.csv')):z.write(p,'菜单/'+p.name)
    z.write(OUT/'使用说明.md','使用说明.md')
print(json.dumps({'files':len(checks),'high_total':sum(c['high_count'] for c in checks),'low_total':sum(c['low_count'] for c in checks),'shortages':short},ensure_ascii=False))
