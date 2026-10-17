# 智慧食堂定向优化（2026-09-11）

新策略沿用 Trial 18 特征与评价口径，增加高频菜单前一自然日重复率硬约束、特殊日期需求与菜数独立校准。主入口为 `targeted_policy.make_recommendations`，网页通过 `backend/app/active_strategy.py` 加载经完整性校验的新版本。

## 运行网页

在已有后端依赖环境中：

```bash
cd smart_canteen_web
python run_app.py --no-browser
```

地址：`http://127.0.0.1:8000/`。本轮前端已重新构建。新增提示包括自动高频菜单相邻日重复率、替换数、前日对照来源、缺少对照，以及人工确认菜单超限提醒。

本轮项目内验证环境为 `.targeted-venv`；运行网页时需要原 `backend/requirements.txt` 中的依赖。已有业务数据、冻结 Trial 18 和旧测试锁均保留。

在当前机器也可直接从项目根目录使用：

```bash
.targeted-venv/bin/python smart_canteen_web/run_app.py --no-browser
```

本轮对照运行环境为 Python 3.12.14、pandas 2.2.3、NumPy 2.3.5，详细版本与检查结果保存在 `runs/targeted_20260911/verification.json`。基线和新策略使用完全相同的环境；旧报告数值不代替此次重跑基线。

## 策略调用

```python
import json
from pathlib import Path
import policy
from targeted_policy import make_recommendations

config = json.loads(Path("runs/targeted_20260911/frozen/targeted_config.json").read_text())
train = policy.read_sales(policy.TRAIN_PATH)
attributes, _ = policy.build_library_attributes()
warmup, actual, _ = policy.training_backtest_frames(train)
contexts = policy.build_contexts(warmup, actual, attributes)
audit = []
menu = make_recommendations(contexts, config, policy.BASE_CONSTRAINTS, selection_audit=audit)
```

`audit` 逐餐保留约束状态、原计划数量、替换来源与历史次数、缺口原因和未发布需求。`adjacent_status=missing_previous` 时重复率为 `None`，不能按0处理。`infeasible` 表示当前候选组合未找到可发布菜单；该餐次仍进入评估，不能删除或用潜力菜填充分母。请先检查审计结果，不要把空输出误认为零需求。

替换后保持最低批量筛选确定的菜数；所有类别按标准化菜名去重后参与配额。新候选来自原同餐次合格池，至少3次本地历史；冷启动可回退同餐次跨食堂历史。最终全部通过最低批量和相似菜硬约束检查。

## 实验与复测

开发流程沿用现有训练期，不读取湖畔/基础园测试数据。每次运行只验证一个明确假设，记录在 `runs/targeted_20260911/<实验>/`，包括代码快照、配置、完整推荐、逐餐诊断和日期类型指标。代码版本使用SHA-256，因为本目录不是Git仓库。

冻结之前的实验命令形式为：

```bash
python targeted_search.py <新实验名> --patch '<JSON参数>' --hypothesis '<本轮假设>'
```

本轮完成冻结后，开发入口会拒绝继续调参。原实验不覆盖。新的一轮研发应建立新的独立运行目录与冻结记录。

冻结和复测使用：

```bash
python targeted_release.py freeze final_verified
python targeted_release.py retest hupan
python targeted_release.py retest jichuyuan
python targeted_release.py install
python targeted_report.py
```

以上冻结、复测、安装步骤已执行的目录会拒绝重复覆盖。湖畔原验证期2026-01-01—03-31、原测试期2026-04-01—06-10，基础园原测试期2026-01-01—06-10均标注“复测”。不删除旧一次性测试锁，也不使用复测结果选择新参数。

## 测试

```bash
.targeted-venv/bin/python -m pytest tests smart_canteen_web/backend/tests -q
```

前端在 `smart_canteen_web/frontend` 运行 `npm test` 与 `npm run build`。

## 版本与回退

新文件：`smart_canteen_web/model/targeted_policy.py`、`targeted_config.json`、`targeted_manifest.json`。

原 `model/policy.py`、`best_policy.json`、`model_manifest.json` 保持不变。若需回退，在停止服务后将上述三个 `targeted_*` 文件移到单独的备份目录，再重启；加载器会恢复旧 Trial 18。业务数据与正式推荐历史不要移动。回退后不会继续强制80%约束。

前日对照优先采用最近一次已确认菜单，同日不同版本不合并；无确认菜单时使用 `temp_recommendations/automatic_high_frequency/<dataset_id>/<餐次>/<日期>.json`。此目录仅保存高频对照，不将潜力菜混入高频分母，也不追加正式推荐历史。

## 报告与注意事项

最终报告：`runs/targeted_20260911/新旧策略对比报告.md`。完整失败实验与选择理由见 `experiment_reviews.json`，停止依据见 `stop_audit.json`。

只有长春A训练集参加调参，寒假和寒暑假/法定重叠没有开发期样本。新状态的第一餐可能没有足够历史判断停餐，仍须检查模型提示与人工运营信息。原MAPE不会因真实销量极小而删除样本；WAPE、绝对误差、低估量、超量预测和不可行餐次作为补充诊断。
