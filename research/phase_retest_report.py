"""Report-only post-freeze coverage; cannot change selection or predictions."""
import json
import pandas as pd
from phase_data import OUT,DATASETS
from phase_release import verify_freeze
from phase_report import table


def run():
    frozen=verify_freeze();rows=[];months=[]
    types=["普通工作日","普通周末","调休工作日","法定节假日","寒假","暑假","寒暑假与法定节假日重叠"]
    for ds in DATASETS:
        folder=OUT/"retest"/ds;record=json.loads((folder/"record.json").read_text())
        assert record["status"]=="complete"
        for interval in record["results"]:
            frame=pd.read_csv(folder/(interval+"_segments.csv"))
            groups=frame.loc[frame.dimension.eq("date_type")].set_index("segment").reindex(types).reset_index()
            groups["dataset"]=ds;groups["interval"]=interval
            groups["coverage_status"]=groups.session_count.map(lambda v:"无样本" if pd.isna(v) else "已复测")
            rows.append(groups)
            if interval=="full_existing_interval":
                month=frame.loc[frame.dimension.eq("month")].set_index("segment").reindex([f"2026-{m:02d}" for m in range(1,7)]).reset_index()
                month["dataset"]=ds;months.append(month)
    coverage=pd.concat(rows,ignore_index=True);monthly=pd.concat(months,ignore_index=True)
    coverage.to_csv(OUT/"retest_date_type_coverage.csv",index=False);monthly.to_csv(OUT/"retest_monthly_metrics.csv",index=False)
    text=["\n## 既有区间复测：日期类型覆盖与分月结果\n",
        "以下为冻结后一次完整滚动得到的结果，没有用于任何模型/阈值/停止判断。‘无样本’不代表误差为0，也不代表该场景已解决。最终保留基线，最终与新基线的预测逐餐相同，净收益等变化均为0。\n"]
    for ds in DATASETS:
        text.extend([f"### {ds}：2026年1月1日—6月10日日期类型\n",
            table(coverage.loc[coverage.dataset.eq(ds)&coverage.interval.eq("full_existing_interval")],
                ["segment","coverage_status","session_count","total_mape","hit_qty_mape","net_benefit","total_wape","pred_count_mean","actual_count_mean"])])
    text.extend(["### 分月诊断（非独立验证）\n",table(monthly,["dataset","segment","session_count","total_mape","hit_qty_mape","net_benefit","total_wape"])])
    failures=[]
    for ds,suffix in [("a_hupan","a"),("f_jichuyuan","f")]:
        frame=pd.read_csv(OUT/"experiments"/("own_drop_"+suffix)/ds/"failure_samples.csv").head(3)
        frame["dataset"]=ds;failures.append(frame)
    text.extend(["### 开发期典型失败（保留原低记录量，不判定为脏数据）\n",
        table(pd.concat(failures,ignore_index=True),["dataset","date","餐次","菜品名称","actual_qty","historical_scale","raw_prediction","final_prediction","sample_count","drop_risk"]),
        "这些菜此前有9—184次记录，仍出现模型未能识别的骤低记录量；不能简单归因于历史样本少或最低批量。数值沿用项目计量，原售卖单位‘克’已在原读取流程折成50克份。没有供应过程证据，不能把这些记录当错误删除，也不能断言是真实需求骤降。\n",
        "早期无可靠本菜历史或无历史阈值的记录不纳入突降分类标签训练和风险评价；明细中的布尔比较占位值须结合sample_count及own_drop_threshold有效性读取，不把未知事件解释为正常销量。\n"])
    p=OUT/"阶段与单菜结构实验报告.md";existing=p.read_text();marker="\n## 既有区间复测：日期类型覆盖与分月结果\n"
    if marker in existing:existing=existing.split(marker)[0]
    p.write_text(existing+"\n".join(text),encoding="utf-8")
    print("Post-freeze coverage report complete; strategy unchanged",flush=True)


if __name__=="__main__":run()
