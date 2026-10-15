# 智慧食堂启发式菜单推荐：因子与容量阶段

本阶段从上一阶段最佳 Trial 108 继续优化，没有从零开发。旧代码、旧账本、旧 `versions/`、旧菜单和外部评估产物已完整归档到：

`archive/factor_capacity_phase0_trial108_20260727/`

新阶段 Trial 从 1 重新计数，共完成 21 个相同 expanding-window 回测。固定 `evaluator.py` 未修改，SHA-256 为：

`4f5cd9494b40ea021744640bc86551febb4f019eb4a668ddc37f20fcea9fcffc`

所有新实验和参数选择只使用长春 A 食堂训练集。Trial 18 完成冻结后，按用户明确要求读取湖畔餐厅外部数据一次，生成独立的 `trial18_external_evaluation.lock.json`；上一阶段 Trial 108 的 `adaptive_test_evaluation.lock.json` 原样保留，没有覆盖。外部结果只用于最终报告，没有继续调参。

## 数据与严格时间划分

- 训练数据：`数据/长春A食堂_菜品销量_菜名修正后.csv`
- 菜品库：`数据/长春A食堂_菜品库.csv`
- 历史预热区：2025-07-21 至 2025-08-03
- 训练内开发回测：2025-08-04 至 2025-12-31，共 150 天、450 个餐次
- 冻结迁移验证：2026-01-01 至 2026-03-31，共 270 个餐次
- 单次锁定测试：2026-04-01 至 2026-06-10，共 212 个餐次
- 基本任务键：`(date, 食堂名称, 餐次)`

每个预测日的早餐、午餐和晚餐共享当天开始前的历史快照；当天所有餐次预测结束后，才把当天实际销售加入历史。销售特征始终满足 `history_date < prediction_date`。预测当日的星期、校历、法定假期和天气属于已知上下文，可以使用。

## 菜品库匹配与推断

菜名匹配只做 NFKC 全半角统一、空格清理、大小写统一和常见括号统一，不修改菜品库。菜品库属性优先；缺失主要食材时使用静态、保守的菜名规则，仅高置信度属性可形成硬冲突。

本阶段训练数据没有未匹配菜品；`unmatched_dishes.csv` 记录了 503 个 low/unknown 低置信度属性，供人工审计，未把它们强行用于硬约束。

## 最终保留策略：Trial 18

### 1. 销量组合因子控制

餐次总量先计算基础预测，再分别输出：

- `weekday_total_factor`
- `group_trend_factor`
- `group_momentum_factor`
- `calendar_factor`
- `weather_factor`
- `weather_residual_factor`
- `raw_combined_factor`
- `final_combined_factor`

最终保留的组合方式为：

```text
raw_combined_factor
= weekday × trend × momentum × calendar × weather_residual

final_combined_factor
= clip(raw_combined_factor, 0.50, 1.30)

pred_total_qty
= base_total_qty × final_combined_factor
```

训练回测中 57.11% 的餐次同时存在至少两个下降因子，原始最低组合因子为 0.4753；最终最低因子被控制在 0.50。更激进的 0.65 下限使国庆状态 MAPE 从 22.59% 恶化到 30.00%，已失败回滚。

校历假期与法定节假日重叠时只生成一个联合 `_calendar_state`，不会乘两个假期因子。调休工作日按工作日父状态处理。假期已存在日历状态时，天气只保留 25% 弱残差；该规则在当前回测期数值等价，但作为防重复修正的结构约束保留。

### 2. 独立餐次总量与总量守恒

最终使用 `independent_shadow_calibrated`：

- 固定旧基准容量生成一个不可见的“影子菜单”，仅用于复现原有餐次总量校准；
- 实际菜单容量只能改变影子排序取多少道菜，不能改变餐次 `pred_total_qty`；
- 入选菜品按单品权重和销量稳定性重新分配；
- 每个餐次严格满足 `sum(pred_qty) = pred_total_qty`。

完整回测的总量分配 WAPE 为 `7.60e-17`。

### 3. 节假日菜单容量

实际容量不再只用同星期菜品数中位数。容量模型同时计算：

1. 历史同类日期真实菜品数的层级估计；
2. `pred_total_qty / 历史合理每菜份数` 的需求容量。

两者权重各 50%，再以 25% 强度向旧容量修正：

```text
joint_capacity = 0.5 × state_capacity + 0.5 × demand_capacity
planned_capacity = old_capacity + 0.25 × (joint_capacity - old_capacity)
```

层级回退为：食堂+餐次+联合日期状态 → 日期大类 → 普通工作日/周末 → 同餐次全局。状态样本按 `n / (n + 5)` 向父状态收缩。容量天气强度最终为 0；10% 弱天气容量消融使菜单 MAE和假期偏差都变差，因此不保留。

### 4. 最小备餐批量

菜品最小批量使用预测日前正销量的 25 分位，至少需要 3 次菜品历史；样本不足时依次回退到同类别同餐次、同餐次全局。最终阈值尺度为历史 25 分位的 20%。

如果分配量低于阈值，删除最低批量比/最低得分菜品，再把完整餐次总量守恒分配给剩余菜品。完整回测最终 `below_min_batch_count = 0`。

## 训练内三个主要目标与容量诊断

| 指标 | 新阶段基线 Trial 1 | 最终 Trial 18 | 变化 |
|---|---:|---:|---:|
| Total MAPE | 12.877442% | **12.867842%** | -0.009600 个百分点 |
| 命中菜品 MAPE | 61.535969% | **61.083468%** | -0.452502 个百分点 |
| Net benefit | 1,753,482.26 | **1,786,560.41** | +33,078.16 |
| Menu count MAE | 18.822222 | **15.584444** | -17.20% |
| Menu count MAPE | 15.034895% | **12.440449%** | -17.25% |
| Holiday menu count bias | +1.666667 | **+0.793103** | 绝对偏差 -52.41% |
| Over-capacity session rate | 48.6667% | **43.3333%** | -5.3334 个百分点 |
| Excess dish count | 2,541 | **1,943** | -598 |
| Below-min-batch count | 23,218 | **0** | -23,218 |
| High-revenue candidate recall | 94.5197% | **94.5197%** | 无下降 |
| 硬约束违规 | 0 | **0** | 0 |

容量调整没有改变 Total MAPE 的计算口径，也没有通过丢弃餐次销量改善净收益。相对基线，三个主要目标均改善。

## Trial 18 训练、验证和测试结果

| 阶段 | Total MAPE | 命中菜品 MAPE | 命中 WAPE | Net benefit | Revenue coverage | Dish hit rate | 硬约束违规 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 训练内滚动回测 | **12.8678%** | 61.0835% | 28.1550% | **1,786,560.41** | 64.3959% | 61.0062% | 0 |
| 冻结迁移验证 | 258.8766% | 83.4504% | 36.8587% | 565,333.82 | 61.0539% | 50.1491% | 0 |
| 单次锁定测试 | **13.7034%** | **50.6499%** | **26.2246%** | **691,087.49** | 60.4615% | 60.4325% | 0 |

三个阶段的 `pred_orders`、`pred_rev` 和餐次总量分配均保持一致性；验证集和测试集 `total_allocation_wape` 分别为 `8.09e-17` 和 `8.71e-17`。菜单没有重复菜品，硬约束违规均为 0。

### 迁移验证分析

验证期 Total MAPE 仍然非常高，主要不是普通日期，而是 2026 年 2 月寒假与春节重叠造成的极端分布变化：

| 月份 | Total MAPE | 命中 MAPE | 命中 WAPE | Net benefit |
|---|---:|---:|---:|---:|
| 2026-01 | 29.5659% | 84.1861% | 39.0817% | 164,909.83 |
| 2026-02 | **776.1323%** | 73.8643% | 41.2971% | -5,008.01 |
| 2026-03 | 20.9884% | 86.3715% | 34.7284% | 405,432.00 |

27 个“寒假＋春节”餐次的 Total MAPE 为 2298.2401%，实际菜品数和需求量突然降至极低水平，而模型仍依据此前正常餐次历史预测。去掉这 27 个重叠餐次后，其余 243 个验证餐次的 Total MAPE 约为 32.28%。这说明当前联合状态虽避免了假期因子重复相乘，但对“校园几乎停餐”的制度性断点仍缺少足够历史样本。

验证期容量诊断为：

- `menu_count_mae = 20.9778`，`menu_count_mape = 77.8975%`；
- 菜品数平均偏差为 +3.4444 道，假期偏差为 +15.3141 道；
- 低于最小批量菜品数为 0；
- 寒假与春节重叠时，早餐、午餐、晚餐分别平均多预测 36.56、70.89、52.33 道菜。

相对上一阶段 Trial 108，Trial 18 的验证 Total MAPE 增加 3.2374 个百分点，但净收益增加 15,118.77，Revenue coverage 增加约 1.00 个百分点，命中 WAPE下降 1.0873 个百分点。该结果没有用于回头修改 Trial 18。

### 锁定测试分析

测试期按月份表现：

| 月份 | Total MAPE | 命中 MAPE | 命中 WAPE | Net benefit |
|---|---:|---:|---:|---:|
| 2026-04 | 12.7454% | 52.3305% | **24.2534%** | 304,806.87 |
| 2026-05 | 15.8045% | 53.1174% | 26.2750% | 281,107.03 |
| 2026-06（截至10日） | **9.9387%** | **38.7095%** | 32.3574% | 105,173.60 |

按餐次表现：

| 餐次 | Total MAPE | 命中 MAPE | 命中 WAPE | Net benefit |
|---|---:|---:|---:|---:|
| 早餐 | **11.9341%** | **30.0974%** | **19.9041%** | 87,326.07 |
| 午餐 | 13.7063% | 53.9327% | 26.9615% | **404,179.54** |
| 晚餐 | 15.4951% | 61.6629% | 29.8958% | 199,581.88 |

早餐仍是最稳定餐次；晚餐的总量和命中误差最高。普通工作日 Total MAPE 为 9.4264%，普通周末为 12.7957%；清明节、劳动节和调休工作日分别升至 35.9766%、36.5821% 和 45.7005%。

测试期容量表现：

- `menu_count_mae = 15.1274`，`menu_count_mape = 13.3014%`；
- 整体平均少预测 12.5613 道菜，但法定节假日平均多预测 5.0417 道；
- 超容量餐次率为 21.23%，多余菜品合计 272 道；
- 低于最小批量菜品数为 0，菜单填充率为 99.90%；
- 法定节假日早餐、午餐、晚餐分别平均多预测 0.63、9.63、4.88 道菜。

相对同一测试集上的 Trial 108，Trial 18：

- Total MAPE 从 13.6604% 变为 13.7034%，相对恶化 0.315%；
- 命中菜品 MAPE 从 52.1003% 降至 50.6499%，相对改善 2.78%；
- 命中 WAPE 从 26.8595% 降至 26.2246%；
- Net benefit 增加 16,789.73，提升 2.49%；
- Revenue coverage 提升 1.0769 个百分点；
- Dish hit rate 下降 1.1047 个百分点，Non-hit预测量占比增加 1.0887 个百分点。

因此测试结果符合主要目标的 Pareto 特征：Total MAPE 仅小幅恶化且未超过 1%，命中误差和净收益均明显改善；代价是菜品命中率略有下降，仍需继续关注非命中预测量。

## 训练内分日期状态与餐次结论

- 普通周末三餐的销量 MAPE基本不变；工作日晚餐 MAPE改善约 0.035 个百分点。
- 校历假期午餐/早餐 MAPE小幅改善，晚餐基本持平。直接加入完整或 25% 校历销量因子都使暑假和偏热日期明显退化，因此未保留。
- 法定节假日仍表现为销量高估，早餐/午餐约高估 19.7%/17.3%，晚餐约高估 18.3%。菜单容量明显下降：午餐菜单偏差由 +33.63 道降至 +24.13，道；晚餐由 +36.75 降至 +28.63，道，但尚未完全消除。
- 校历假期菜单原本偏少；层级容量使午餐偏差由 -13.10 改善到 -11.10，道，早餐由 -1.67 改善到 -0.90，道。
- 调休工作日只有 2 个样本/餐次，销量仍明显低估 28%–35%；模型按工作日父状态处理并强收缩，没有针对这 2 天硬编码。

详细结果见 `best_state_meal_diagnostics.csv` 和 `best_menu_capacity_sessions.csv`。

## 成功与失败实验

成功/结构等价保留：Trial 1、3、6、9、10、12、18。

- Trial 3：保守组合因子边界 `[0.50, 1.30]`。
- Trial 6：影子总量校准，容量变化不再改变餐次总量。
- Trial 9：25% 层级容量，菜单 MAE明显改善。
- Trial 10：25% 弱最小批量执行，低批量计数归零。
- Trial 12：假期天气只保留 25% 弱残差，数值等价。
- Trial 18：最小批量尺度降至 20%，回收净收益并保持容量改善。

代表性失败：

- Trial 2：组合因子下限 0.65 放大法定假期高估。
- Trial 4/11：校历假期完整/弱销量修正都使暑假分组退化。
- Trial 5：直接使用未校准独立总量使 Total MAPE恶化 1.44%。
- Trial 7：完整层级容量虽把菜单 MAE降至 9.75，但命中 MAPE恶化约 5.8%。
- Trial 8：完整最小批量使菜单填充降至 69%，命中 MAPE升至 80.62%。
- Trial 13：天气容量修正使菜单 MAE和假期偏差变差。
- Trial 15：35% 容量强度触发批量级联，命中 MAPE恶化 9.84%。
- Trial 16/17：纯状态容量和纯需求容量都不如 50/50 联合容量。
- Trial 19—21：分别移除因子边界、最小批量和层级容量，均未刷新 Pareto 最佳。

全部失败记录保留在 `trials.jsonl`；Trial 9、12、18 的选择器审计更正在 `selection_corrections.jsonl`，解析后的完整账本为 `all_trials.jsonl`。

## 停止与测试集状态

本阶段没有固定实验次数。已探索因子裁剪、日历/天气交互、影子总量、层级容量、状态/需求容量消融、容量天气、最小批量强度和核心模块消融。Trial 18 后连续三个完整有效消融 Trial 19—21 均未刷新 Pareto 最佳，因此停止。

Trial 18 在全部训练实验停止后才进行外部评估。`trial18_external_evaluation.lock.json` 记录 `external_data_read_count = 1`，配置已标记 `external_evaluated = true`；评估入口再次运行会直接拒绝。上一阶段 Trial 108 的旧锁未覆盖。外部结果没有用于追加实验、选择参数或修改固定 Evaluator。

## 主要产物

- `policy.py`：最终推荐与容量核心
- `config.json`：Trial 18 参数和完整指标
- `evaluator.py`：固定评价器
- `adaptive_search.py`：一次一个假设、无固定总次数的训练内实验器
- `trials.jsonl` / `selection_corrections.jsonl` / `all_trials.jsonl`
- `summary.csv` / `factor_capacity_comparison.csv`
- `state_meal_diagnostics.csv` / `best_state_meal_diagnostics.csv`
- `best_menu_capacity_sessions.csv`
- `recommended_menu.csv`：Trial 18 的训练内 expanding-window 推荐菜单
- `trial18_validation_recommended_menu.csv`：Trial 18 冻结迁移验证菜单
- `trial18_test_recommended_menu.csv`：Trial 18 单次锁定测试菜单
- `trial18_external_evaluation.json` / `trial18_external_evaluation.lock.json`
- `trial18_train_validation_test_summary.csv`
- `trial18_external_segment_metrics.csv`
- `trial18_external_state_meal_diagnostics.csv`
- `failure_cases.csv` / `missed_high_revenue_dishes.csv`
- `unmatched_dishes.csv`
- `trial_total_mape_net_benefit.png` / `sample_efficiency.png`
- `stop_audit.json`
- `versions/trial_18/`

## 复现命令

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache \
python3 -m unittest discover -s tests -v

PYTHONPYCACHEPREFIX=/private/tmp/pycache \
python3 finalize_adaptive.py

MPLCONFIGDIR=/private/tmp/smart-canteen-mpl \
/opt/miniconda3/bin/python3 plot_trial_metrics.py
```

`finalize_adaptive.py` 只整理训练内产物，不打开外部销售文件。

Trial 18 外部评估已经完成，禁止再次运行 `evaluate_trial18_external.py`；排他锁会拒绝重复读取。

---

## 仓库与数据说明（2026-10-15 整理）

本仓库只保存**源码、配置、协议、账本与最终交付件**，大体积实验产物留在本地、不入库。

### 已入库

| 内容 | 说明 |
| --- | --- |
| 根目录 `*.py`、`*.json`、`*.jsonl` | 研究脚本、协议、Trial 账本与冻结清单 |
| `tests/` | 根目录研究脚本的单元测试 |
| `smart_canteen_web/` | Web 交付系统（`backend/` FastAPI、`frontend/` Vue、`model/` 冻结模型、`data/datasets/` 两套数据集） |
| `sdk/src/`、`sdk/tests/`、`sdk/examples/` | Python SDK 源码 |
| `数据/*.csv` | 训练 / 外部评估所需的修正后销售明细、菜品库、名称映射、天气 |
| `outputs/` | 甲方汇报 PPTX / DOCX / PDF、A 湖畔一周菜单等最终交付件 |
| `scripts/`、`reports/`、`analysis_artifacts/` | 预处理与审计脚本、报告 |

### 未入库（仅本地保留，可另行归档）

| 内容 | 体积 |
| --- | --- |
| `runs/` | ≈ 18 GB，各阶段回测中间产物（`.pkl` / `.gz` / `.npz` / `.csv`） |
| `outputs/` 部分子目录 | ≈ 1 GB，中间渲染、测试数据更新、历史发布包 |
| `1-长春A食堂/` … `9-长春I食堂/` | ≈ 850 MB，客户原始台账 Excel |
| `数据/*.xlsx` | ≈ 170 MB，用餐者取餐 / 汇总原始表 |
| `archive/`、`versions/` | ≈ 210 MB，历史归档 |
| `node_modules/`、`.venv/`、`__pycache__/`、`dist/` | 依赖与构建产物 |

完整规则见根目录 `.gitignore`。

### 重建开发环境

仓库里的 `.targeted-venv/` 是原开发机遗留的虚拟环境（`pyvenv.cfg` 指向原作者 home，符号链接已损坏），**不要直接使用**，请重建：

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r smart_canteen_web/backend/requirements.txt
```

### 注意

- `数据/长春A食堂_菜品销量_菜名修正后.csv` 等 CSV 是测试与复现的硬依赖，删掉会导致测试失败。
- `policy.py`、`diagnostics.py`、`evaluator.py`、`config.json`、`trial18_freeze_manifest.json`、`trial18_external_evaluation.lock.json`、`adaptive_test_evaluation.lock.json`、`trials.jsonl`、`all_trials.jsonl` 这 9 个文件受 `smart_canteen_web/model/model_manifest.json` 的 SHA-256 锁定，**移动或修改会使模型完整性测试失败**。
- 消费者测试：`smart_canteen_web/backend/tests/test_model_and_context.py::test_frozen_model_hashes_and_source_project_unchanged`

### 接手时的测试基线（2026-10-15）

| 测试套件 | 结果 |
| --- | --- |
| `smart_canteen_web/backend/tests` | **46 passed** ✅ |
| `tests/`（根目录研究脚本） | **76 passed / 9 failed** ❌ |

`tests/` 的 9 个失败**不是整理造成的**，而是两份「预注册哈希清单」与后来被更新的文件对不上，属于接手前的既有状态：

| 清单 | 期望文件 | 实际 mtime | 影响的测试 |
| --- | --- | --- | --- |
| `runs/phase_20260913/registration.json` | `smart_canteen_web/backend/app/active_strategy.py` | 2026-09-18 | `test_phase_runtime.py` ×5 |
| `runs/demand_20260913/preregistration.json` | `smart_canteen_web/data/datasets/a_hupan/sales.csv`、`dish_library.csv` | 2026-09-13 21:15 | `test_demand_runtime.py` ×4 |

两个清单里其余条目（phase 20/21、demand 10/12）哈希均一致。要修复需要在「回退数据文件」和「重跑并重新登记」之间二选一，属于业务决策，整理阶段未改动。

`tests/` 依赖 `runs/` 下的回测产物，而 `runs/` 不入库，因此**从全新克隆直接跑 `tests/` 会因缺少 fixture 而报错**；`runs/` 缺失时 `test_demand_runtime.py` 会 skip，`test_phase_runtime.py` 会 error。CI 建议只跑 `smart_canteen_web/backend/tests`。
