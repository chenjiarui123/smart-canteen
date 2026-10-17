# 单菜销量分配：可选运行方式

本模块是在现有菜单生成之后的可选分配层，未安装为网页默认策略。
冻结配置与源码位于 `runs/allocation_20260911/frozen`。
完整结果见该运行目录的 `单菜分配优化对比报告.md`。

## 接口

使用 `allocation_runtime.AllocationRuntime(history_before_target, dataset="a")`
或 `dataset="f"`。每个实例只对应一个逻辑食堂，A/湖畔与F/基础园不能混用历史。

1. 沿用现有 `policy.build_contexts` 接口构建一天所有餐次的上下文，销售统计必须来自日期之前；目标日期的日历和天气可作为已声明输入。
2. 调用 `runtime.recommend(contexts)` 获取菜单与审计。它自动生成原基线菜单、捕获内部历史权重，然后分配单菜销量。
3. 检查所有请求餐次都有菜单，并逐项检查 `audit["allocation"]` 的 `allocation_feasible`。空审计列表不是“所有餐次可行”的证明；已有空菜单也不能作为可发布计划。
4. 当天各餐全部预测结束后，用 `runtime.observe(actual_of_that_day)` 添加已发生销售。不得提前添加当天或未来销量。

可以传入 `initial_menu_history` 接续原菜单对照；菜单历史的确认优先级仍由原网页流程负责。本可选封装器没有自动接入网页存储。

底层 `allocation_policy.allocate_session` 可直接接受冻结基线单餐菜单与完整历史特征。需要内部权重的配置不要手工猜测权重，优先使用上面的封装器。

## 约束及回退

菜单、菜数、总量字段和基线元数据不变；只更新销量、订单和销售额三列。实际分配总和与原基线总和在浮点容差内一致。

不满足历史范围时明确回退基线并标记不可行，不代表回退结果已经满足新范围。基线自身存在的“整餐总量小于最低备餐量”冲突也只能明确报告，不能在固定总量条件下自动解决。

停用分配层、直接使用原冻结 targeted 策略即可回退。网页目前仍是该原策略，不需要执行回滚。

## 复现

使用项目现有 `.targeted-venv/bin/python`。实验准备和运行脚本会拒绝覆盖同名目录，以保护原始实验记录；重新完整复现时应使用项目副本或新的运行根目录，不删除原始记录。

- 历史准备：`allocation_prepare.py a`、`allocation_prepare.py f`。
- 开发实验：`allocation_experiments.py --spec allocation_round5_spec.json`，其余 spec 文件保留各阶段完整参数。
- 最终复核：`allocation_verify_a_spec.json`、`allocation_verify_f_spec.json`；各自对应 `--datasets a` 和 `--datasets f`。
- 冻结前审计：`allocation_selection.py`。
- 冻结与一次复测：`allocation_release.py freeze final_verified_a final_verified_f`，再分别 `retest a` 和 `retest f`。现有冻结及复测目录不可再次覆盖运行。
- 报告重建：`allocation_report.py`，只读取已产生结果，不搜索新策略。
- 回归测试：`python -m pytest -q tests smart_canteen_web/backend/tests`。

各次 `record.json` 包含假设、配置、完整指标、源码SHA256及失败原因；同目录保存源码快照。目录不是Git仓库，因此代码版本由内容哈希标识。

分组边界来自2025开发历史特征并保存于各开发目录的 `group_edges.json`，复测不得重新拟合。2026结果标注为既有区间复测，不宣称独立盲测。
