import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sdk/src'))
from smart_canteen_sdk import CanteenEngine
from smart_canteen_sdk.adapters import load_legacy_csv

OUT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    folder = ROOT / 'smart_canteen_web/data/datasets/a_hupan'
    data = load_legacy_csv(folder/'sales.csv', folder/'dish_library.csv',
        canteen_id='a_hupan', snapshot_id='weekly_20260914_history_20260610',
        runtime_sales_path=folder/'runtime_actual_sales.csv',
        recommendation_history_path=folder/'recommendation_history.csv',
        allow_legacy_unit_changes=True)
    weather_source=pd.read_csv(folder/'sales.csv')
    weather_source=weather_source.loc[weather_source['月'].eq(9)].drop_duplicates(['年','月','日'])
    mapping={'temperature_c':'气温（℃）','humidity_pct':'相对湿度（%）','precipitation_mm':'降水量（毫米）','wind_speed_kmh':'10米风速（公里/小时）'}
    weather={k:float(weather_source[v].median()) for k,v in mapping.items()}
    engine=CanteenEngine()
    entries=[]
    for day in pd.date_range('2026-09-14','2026-09-20'):
        for meal in ['早餐','午餐','晚餐']:
            if args.sample and (str(day.date())!='2026-09-14' or meal!='午餐'):continue
            request={'canteen_id':'a_hupan','prediction_date':str(day.date()),'meal':meal,
                'weather':weather,'calendar':{'school_break':False,'statutory_holiday':False,'adjusted_workday':False},
                'opportunities':{'enabled':True,'category':'全部','count':24,'refresh_seed':0}}
            p=engine.recommend(request,data)
            assert p.audit['low_frequency_status']=='ok'
            assert p.audit['constraint_violation_count']==0
            ids=[i.dish_id for i in p.high_frequency_menu+p.low_frequency_opportunities]
            assert len(ids)==len(set(ids))
            entry={'date':str(day.date()),'weekday':'周'+'一二三四五六日'[day.weekday()], 'meal':meal,
                'high':[i.model_dump() for i in p.high_frequency_menu],
                'low':[i.model_dump() for i in p.low_frequency_opportunities],
                'audit':p.audit,'weather':weather,'request':request}
            entries.append(entry)
            # Plan-only automatic history; never treat recommendations as actual sales or confirmed selections.
            if not args.sample:data.menu_history.extend(p.history_events)
            (OUT/'work'/f'{day.date()}_{meal}.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2))
            print(json.dumps({'date':entry['date'],'meal':meal,'high':len(entry['high']),'low':len(entry['low'])},ensure_ascii=False),flush=True)
    (OUT/'work'/('sample.json' if args.sample else 'week.json')).write_text(json.dumps(entries,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
