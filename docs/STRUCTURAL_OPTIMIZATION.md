# 单菜结构性预测实验：运行与网页接入

本轮没有新模型通过预先固定的开发门槛，A、F均保留网页已部署的 `targeted_repeat_holiday_20260911`。新增代码是独立实验模块，不会自动覆盖网页策略。`runs/structural_20260912/frozen/structural_config.json` 对两个食堂都选择 `identity: true`，表示保持原菜单、需求量和备餐输出，不表示预测误差为零。

## 文件与结果

- `structural_data.py`：当前网页清理后数据的开发期隔离与哈希校验。
- `structural_trace.py`、`structural_prepare.py`：实际部署基线的逐阶段追踪及最终菜单、总量锁定。
- `structural_features.py`：同售卖单位的条件份额、类别份额和菜单竞争特征。
- `structural_policy.py`：需求与备餐分离、固定总量协调、冲突回退。
- `structural_borrowing.py`：按餐次和售卖单位隔离的共享残差模型。
- `structural_admission.py`：仅用历史预测表现的整餐准入保护。
- `structural_runtime.py`：可选的逐日预测接口，尚未注册到网页。
- `structural_experiments.py`、`structural_guard_experiment.py`：单机制、组合和保护层实验。
- `structural_release.py`：重新验证已保存输出、复核门槛和冻结选择，不做调参。
- `runs/structural_20260912/结构性预测优化报告.md`：结果、失败原因和停止依据。

`frozen/` 是代码及选择版本档案，依赖项目现有路径和数据，不是可脱离项目安装的SDK。原部署、旧分配实验和旧数据清理记录均保留。

## 复核现有结果

从项目根目录执行，使用现有依赖环境：

```bash
.targeted-venv/bin/python -m pytest -q tests smart_canteen_web/backend/tests
.targeted-venv/bin/python structural_report.py
```

报告命令只读取开发期缓存和实验结果，不加载2026销量，也不重新选择方案。完整结果在 `experiments/<实验名>/<食堂>/`：`metrics.json`、`sessions.csv`、`session_segments.csv`、`hit_segments.csv`、`allocation_audit.csv`、`menu.pkl.gz`、`hits.pkl.gz`。最终对照、`A_C_F` 和 `guarded_A_C_F` 另有可直接读取的压缩CSV逐菜明细。不得将不同售卖单位的数量加总解释成总重量。

以下是本轮实际实验流程。已完成的准备目录、实验名和冻结目录均拒绝覆盖；不要原样重跑覆盖证据。若需独立复现，请复制项目到单独目录，保留原始运行档案，在副本建立新的运行目录，再按同一协议运行。

```bash
.targeted-venv/bin/python structural_data.py
.targeted-venv/bin/python structural_prepare.py a_hupan
.targeted-venv/bin/python structural_prepare.py f_jichuyuan
.targeted-venv/bin/python structural_features.py a_hupan
.targeted-venv/bin/python structural_features.py f_jichuyuan
.targeted-venv/bin/python structural_experiments.py baseline --datasets a_hupan
.targeted-venv/bin/python structural_experiments.py baseline_f --datasets f_jichuyuan
.targeted-venv/bin/python structural_experiments.py preparation_only shares_flat shares_hierarchical competition_only reliability_only
.targeted-venv/bin/python structural_experiments.py --spec structural_combinations.json
.targeted-venv/bin/python structural_experiments.py --spec structural_peer_trials.json
.targeted-venv/bin/python structural_experiments.py --spec structural_peer_ablations.json
.targeted-venv/bin/python structural_guard_experiment.py A_C_F guarded_A_C_F
.targeted-venv/bin/python structural_guard_experiment.py C_F guarded_C_F
.targeted-venv/bin/python structural_guard_experiment.py A_F_baseline_anchor guarded_A_F_baseline
.targeted-venv/bin/python structural_guard_experiment.py A_C guarded_A_C
.targeted-venv/bin/python structural_experiments.py --spec structural_final_control.json
.targeted-venv/bin/python structural_release.py
.targeted-venv/bin/python structural_report.py
```

`structural_evaluate.stage_diagnostics(dataset)` 与 `structural_diagnosis_report.py` 生成逐阶段诊断；它们不改变预测。无须重跑同一失败实验来“增加次数”。本轮已满足停止条件，任何新研究应先说明新增信息或机制，再注册新的协议，不使用已反复分析的2026数据做选择。

## 逐日接口

接口的真实使用方式如下。这里的 `contexts` 必须由现有网页上下文构建逻辑生成，只含历史销售统计和目标日期已知的日历、天气信息，不能传目标日真实菜单或销量。

```python
from structural_runtime import StructuralRuntime

runtime = StructuralRuntime(
    history=history_before_first_prediction,
    attributes=current_dish_attributes,
    dataset="a_hupan",
    initial_menu_history=previous_confirmed_or_saved_high_frequency_menus,
)  # 不传config时读取本轮冻结配置；当前为identity

menu, audit = runtime.recommend(contexts_for_one_complete_day)
# 当天所有请求餐次都预测完后，才提交当天真实记录：
runtime.observe(completed_day_actual_sales)
```

`history` 和 `attributes` 沿用项目已有清洗与属性生成函数返回的数据结构，不接受任意原始表。`initial_menu_history` 使用现有格式 `{(食堂名称, 餐次): [(Timestamp, 标准化菜名集合), ...]}`。前一天优先最近已确认版本，没有时用保存的高频推荐；不能拿前一次任意日期代替前一自然日。

该接口按完整日期推进，未 `observe` 前不能再次预测；相同日期的餐次不能通过多次调用逐餐学习当天销量。没有真实销售记录不能自行补零；明确未开餐的日期不要伪造销售记录。空菜单或 `all_requested_sessions_generated=False` 必须展示审计原因，不能当作需求为零。

份额历史在初始化时加载；残差、可靠性和准入状态则来自真正的逐日预测误差。若显式试用 `peer_borrowing`、`reliability` 或 `admission_guard`，必须从预热期按时序重放预测—观测，或持久化并恢复服务状态。仅用全部历史创建一个新对象，不等于训练好这些组件。当前状态仅在内存中，尚无经过验证的跨重启持久化实现。保护层接口只验证了前两天冷启动的一致性，未声称完成成熟准入分支的上线验证。

## 需求、备餐和经营风险展示

| 字段 | 含义与口径 |
|---|---|
| `pred_qty` / `expected_demand_qty` | 预计售出需求，继续用于原销量指标、订单和销售额 |
| `suggested_preparation_qty` | 单独启用需求/备餐分离时为 `max(需求, 最低批量)` |
| `minimum_batch_surplus_risk` | 建议备餐减去需求的非负差值，是批量造成的额外暴露量，不是实测剩余 |
| `demand_before_coordination` | 单菜协调前的估计，供诊断使用 |
| `historical_error_adjustment_weight` | 基于过去预测误差的协调权重，不是历史波动惩罚 |

只有 `separate_preparation=True` 的实验明确抬高备餐建议；当前保留的identity输出完全沿用基线，因此F的4处既有最低批量冲突仍须提醒，不能把字段名当作“已保证最低批量”。备餐量另算备餐指标，不替代需求指标。订单=`需求×历史每销量订单比`；销售额=`需求×历史每销量金额比`，不可随备餐量一起抬高预计销售额。

若以后有方案通过门槛，网页接入位置为：

1. `smart_canteen_web/backend/app/high_frequency_recommender.py`：在已有高频选择和重复率审计之后，增加独立、显式的预测版本分支；保留原版本回滚。
2. `smart_canteen_web/backend/app/recommender.py` 与 `schemas.py`：新增字段透传及类型定义，保留原字段兼容性；异常必须带原因，不能自动吞掉。
3. 前端展示“预计售出”“建议备餐”“最低批量额外风险”三列，并分别显示单位。
4. `frontend/src/utils/displayUnits.js` 对售卖单位为“克”的模型数量统一乘50一次。新增数量字段都传原始模型单位交给该格式化函数；预测层不得预乘50，不能重复换算。

本轮不切换网页，不自动启用失败组合。用户人工增删菜品的权利和超80%提醒保持原逻辑。竞争特征能够根据新的生成菜单重新计算，但由于第一阶段未达标，本轮未进入改变选菜的第二阶段，也没有发布“人工菜单实时重新优化”功能。
