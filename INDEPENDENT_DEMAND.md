# 独立单菜记录销量模型：运行与交付

本轮沿用清理后网页数据和原evaluator，未找到同时通过协调前与最终风险门槛的新预测。最终保留原部署预测；独立接口可以附带最低批量备餐建议，但这不改变销量、订单、预计销售额，也不是预测精度改善。网页默认未切换。

## 独立模块

- `demand_features.py`：全部真实供应记录的逐日特征。预测菜单的类别数量只来自旧策略生成名单；当天真实供应结构仅在当天预测结束后的观测阶段用于未来训练。
- `demand_models.py`：独立对数线性、浅层分段、条件经验分布、低销量两状态模型，以及相对/混合损失直接拟合。不依赖旧单菜预测的小幅乘子。
- `demand_coordination.py`：原投影对照和条件分布预计误差分配。新风险分配只通过合成小问题验证；没有协调前模型通过门槛，因此本轮历史数据中的新风险协调按协议跳过。
- `demand_experiments.py`：训练历史隔离、模型对照、保存分布后的不同决策目标比较、原指标及独立诊断。
- `demand_runtime.py`：按完整日期推进的显式独立接口；不注册到网页。
- `demand_reproduce.py`：当前实际部署基线完整重跑并与原缓存逐字段比较。
- `demand_release.py`：完整性、逐餐/单位总量、原指标重算、停止选择和冻结。
- `demand_report.py`：从已冻结开发结果生成报告，不做新实验。

算法只使用项目现有NumPy、pandas，无新安装依赖。冻结目录是版本档案，仍依赖现有项目和缓存，不是独立安装SDK。

## 查看与验证

在项目根目录运行：

```bash
.targeted-venv/bin/python -m pytest -q tests smart_canteen_web/backend/tests
.targeted-venv/bin/python demand_report.py
```

报告目录：`runs/demand_20260913/`。`all_experiment_metrics.csv` 汇总全部完成实验；`ablation_comparisons.csv` 保存逐项对照；`baseline_paired_stage_diagnostics.csv` 是同菜同正销量样本的阶段变化；`risk_discrimination_controls.csv` 区分真正风险信息与仅按菜品大小排序的效果。

每轮 `experiments/<实验名>/<食堂>/` 包含：

- `pre_post_detail.csv.gz` / `pre_post_detail.pkl.gz`：协调前后销量、历史阈值、低销量风险、原始80%区间、实际命中结果及模型历史截止日。
- `menu.pkl.gz`：全部最终菜单和销量、订单、销售额、建议备餐量；不是只保存命中行。
- `raw_segments.csv`、`hit_segments.csv`、`session_segments.csv`、`sessions.csv`：协调前后、低量/正常量、分月/餐次/日期类型等结果。
- `raw_metrics.json`、`metrics.json`、`gate_review.json`：完整指标和未通过门槛的原因。
- `allocation_audit.csv`：固定总量、单位隔离、范围冲突、集中度及备餐风险。
- `pre_post_sessions.csv`：逐餐旧估计、新协调前估计、固定总量后的结果、备餐量及原始区间诊断；总表 `interval_risk_segments.csv` 单列各日期/月/餐次/样本组的覆盖率和宽度。

模型拟合轮保存41点分布 `distributions.npz`。改变均值/相对误差等输出决策时复用同一已验证分布，并记录父实验及哈希，不重复训练相同模型。区间是协调前分布诊断，不声称它是固定整餐总量条件下已经重新校准的联合预测区间；没有新分布被发布到网页。

## 本轮复现顺序

已完成的目录拒绝覆盖，冻结后实验入口拒绝继续在本轮调参。要完整独立复现，请在单独项目副本设置新的 `demand_data.OUT` 目录，保持原协议和数据，先注册再运行；不得删掉现有冻结或失败记录来覆盖结果。

```bash
.targeted-venv/bin/python demand_data.py
.targeted-venv/bin/python demand_reproduce.py a_hupan
.targeted-venv/bin/python demand_reproduce.py f_jichuyuan
.targeted-venv/bin/python demand_features.py a_hupan
.targeted-venv/bin/python demand_features.py f_jichuyuan
.targeted-venv/bin/python demand_experiments.py linear_all_log --config '{"family":"linear","objective":"log"}'
.targeted-venv/bin/python demand_experiments.py linear_hits_log --config '{"family":"linear","objective":"log","hits_only":true}'
.targeted-venv/bin/python demand_experiments.py conditional_all_log --config '{"family":"conditional","objective":"log"}'
.targeted-venv/bin/python demand_experiments.py conditional_hurdle_log --config '{"family":"conditional_hurdle","objective":"log"}'
.targeted-venv/bin/python demand_experiments.py conditional_relative --source conditional_all_log --config '{"family":"conditional","objective":"relative"}'
.targeted-venv/bin/python demand_experiments.py direct_relative --config '{"family":"relative_linear","objective":"absolute"}'
```

其他单机制、目标和消融的准确配置、父分布、代码版本都在各自 `record.json`；最终报告另导出 `reproduction_commands.txt`，按依赖顺序列出本轮已运行命令。失败的 `conditional_absolute` 原记录保留；`conditional_absolute_verified` 是同配置技术复核，不计为新模型方向。此轮没有重新用2026选择或提前停止，也没有执行可选2026复测。

## 接口约定

```python
from demand_runtime import LockedMenuPredictor

predictor = LockedMenuPredictor("a_hupan")  # 读取冻结选择，当前是原预测
menu, audit = predictor.predict_day(
    baseline_menu=old_final_menu_for_complete_day,
    old_features=old_past_only_features,
    features=new_past_only_features,
    completed_training=all_recorded_training_rows_before_target_day,
)
```

`features` 由 `RecordedHistory.features` 构造；特征帧不能含目标日实际量、命中标记。已完成训练表来自同一历史状态逐日生成的特征及其随后观察到的销量，不是把当前统计量回填到历史行。输入必须同一食堂、同一完整日期，训练和特征历史都严格早于预测日。一个对象按日期递增使用，同一天不能逐餐调用并加入当天结果。

要复现实验模型，显式传其配置。模型内部每7个日历日用此前全部允许训练样本重拟合，少于80个同餐次/单位样本时用独立历史典型值回退。对象包含拟合状态；任意中间日冷启动重训与原滚动过程中“最近一次周度拟合”不等价。需要从相同预热状态按日重放，不应宣称该接口具备已验证的跨重启模型持久化。

当前冻结配置保留原需求，没有接受新风险概率或区间，因此接口返回的区间为缺失而非0。如显式使用实验配置，审计会标记 `experimental=True`；不代表通过发布门槛。

## 需求与备餐

`pred_qty` / `expected_demand_qty` 是预计记录销量；`suggested_preparation_qty=max(需求,最低批量)` 是独立备餐建议；`minimum_batch_surplus_risk` 为两者差额，只代表最低批量额外暴露量，不是已经识别出的真实浪费。

订单、销售额继续按原需求和历史单位比例计算，不能把多备的量计入预计卖出的销售额。备餐评价单独保存为 `preparation_metrics`，不替代主要指标。保留预测的独立备餐建议能显式呈现F的4处既有最低批量问题，但没有改变网页原输出。

如以后有新模型通过门槛，接入点仍是 `smart_canteen_web/backend/app/high_frequency_recommender.py`，字段在 `recommender.py` / `schemas.py` 透传。前端 `src/utils/displayUnits.js` 对单位“克”的原模型数量统一乘50一次；新数量字段必须传原模型单位，不可在预测器中提前换算。不同售卖单位的数量不能互相转移或解释成等价重量。
