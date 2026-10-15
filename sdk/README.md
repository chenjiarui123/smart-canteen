# 智慧食堂推荐 SDK 0.1.0

将当前部署的 `targeted_repeat_holiday_20260911` 策略封装为独立 Python 包。核心引擎接收内存数据，不访问数据库，不写入业务文件，不依赖原项目、Vue 或 CSV 路径。模型资源随 wheel 安装。没有发布尚未通过门槛的研究模型。

## 安装与启动

支持 Python 3.10+。本次实际验证环境为 Python 3.12；依赖 NumPy、pandas、Pydantic。

```bash
python -m pip install 'smart_canteen_sdk-0.1.0-py3-none-any.whl[server]'
smart-canteen-server --host 127.0.0.1 --port 8010 --proposal-db ./canteen_sdk_state.sqlite3
```

将 wheel 路径替换为本机文件路径。仅使用 Python 接口时不用安装 `[server]`。`server` 额外安装 FastAPI、Uvicorn。

- 健康检查：<http://127.0.0.1:8010/health>
- 交互文档：<http://127.0.0.1:8010/docs>
- OpenAPI：<http://127.0.0.1:8010/openapi.json>
- 环境变量 `CANTEEN_API_KEY`：启用 `X-API-Key` 校验；健康检查和文档仍公开。
- `--proposal-db` 指定候选方案数据库，可在多次启动间复用；不要放在 Python 包目录。
- 默认候选有效期 24 小时。`--proposal-ttl-seconds` 可调整。同一方案同一选择重复确认返回相同结果，不重复生成历史事件；变更已确认方案返回 409。

默认仅监听本机。接入新后端时由后端调用算法服务；租户权限、业务审批和正式菜单存储由新后端负责。SQLite 在这里仅保存算法候选和确认结果，不是业务销售数据库。多实例跨机器部署需将此存储适配为共享数据库。

## Python 接口

```python
from smart_canteen_sdk import CanteenEngine, Proposal

engine = CanteenEngine()
proposal = engine.recommend(request, data)
frontend_payload = proposal.public_dict()
saved_json = proposal.model_dump_json()  # 后端保存完整方案，包含内部计算状态

restored = Proposal.model_validate_json(saved_json)
result = engine.finalize(restored, {
    "high_frequency_dish_ids": [item.dish_id for item in restored.high_frequency_menu],
    "low_frequency_dish_ids": [],
    "redistribute_to_total_demand": False,
})
print(result.model_dump(mode="json"))
```

核心引擎无业务会话状态，可在重启后恢复完整方案继续确认。状态摘要检测数据损坏，不是数字签名：完整方案只应来自可信后端存储，不能接受浏览器提交的内部状态。HTTP 接口只接受方案 ID，完整状态不返回前端。

其他方法：`engine.validate_data(data)`、`engine.get_model_info()`。校验错误抛出 `CanteenError`（含稳定 `code`）；数据类型/字段错误抛出 Pydantic `ValidationError`。

## request 与 data

`request`：`canteen_id`、`prediction_date`（YYYY-MM-DD）、`meal`（早餐/午餐/晚餐）、`weather`、`calendar`、可选 `opportunities`。

`weather`：`temperature_c`、`humidity_pct`、`precipitation_mm`、`wind_speed_kmh`，均为有限数值。SDK 接受连续数值，不受原网页下拉值限制。

`calendar`：`school_break`、`statutory_holiday`、`adjusted_workday`；假期为 true 时须提供 `school_break_type`（寒假/暑假）或 `statutory_holiday_name`。

`opportunities`：`enabled`（默认 true）、`category`（默认全部）、`count`（1–60，默认12）、`refresh_seed`（默认0）。改变 seed 可刷新潜力菜候选；常用菜不受影响。已选菜跨刷新保留需要业务后端管理，当前接口不自动合并不同候选方案。

`data`：

| 字段 | 内容 |
|---|---|
| snapshot_id | 由业务后端管理的数据快照版本 |
| canteen_id | 单一食堂 ID，所有历史记录必须一致 |
| canteen_name | 可选算法上下文名称；迁移时保留原食堂名称，以复现潜力菜刷新种子 |
| dish_catalog | dish_id、dish_name、category、flavor；可选 main_ingredient、ingredient_family、cooking_method、similarity_group、staple |
| sales_history | date、canteen_id、meal、dish_id、quantity、unit、order_count、revenue；可选 unit_price、average_portion |
| historical_context | date、canteen_id、weather、calendar；可选 meal，无 meal 时应用于当天全部餐次 |
| menu_history | 历史菜单数组，见下文 |

销售数据按“日期＋餐次＋菜品”预先汇总，每个键只允许一条。菜品 ID 与标准化菜名分别唯一；同菜历史单位不得变化。没有记录不是销量0。菜品改名应由后端映射至稳定 ID 和统一名称。未知字段直接报错。

当前原项目少数菜品有历史单位变化。默认 `unit_policy="strict"` 会拒绝这类数据。为复现原系统，可显式传 `unit_policy="legacy_compatibility"`，或在 CSV 适配器设置 `allow_legacy_unit_changes=True`；结果会列出受影响菜品并提示。该模式保留原模型的混合历史统计，不表示已经完成单位清洗。新业务数据应保持 strict 并统一历史口径。

历史输入建议包含当前已掌握的完整历史，随意截断会改变算法结果。预测会排除目标日及未来销量，三餐都不能利用同日实际销量。没有可用历史会报 `INSUFFICIENT_HISTORY`，短历史会在审计中提示冷启动。

`menu_history` 每条是一个菜单版本：`date`、`canteen_id`、`meal`、`menu_id`、带时区 `created_at`、`source`（automatic/finalized）、`dish_ids`、可选 `low_frequency_dish_ids`。重复比较使用每天最新正式版本，缺少昨日正式菜单时回退昨日自动菜单。潜力菜已选次数延续原系统对正式历史记录的累计规则。

## 数量口径与结果

输入 `quantity`、输出 `quantity` 都使用真实售卖单位，例如克就是实际克数。模型内部“克”历史使用 50 克为一个模型数量；SDK 在输入除50、输出乘50，调用方不要再乘50。其他单位比例为1。

每道菜同时保留 `model_quantity` 和 `business_units_per_model_unit` 供兼容审计。`quantity_kind=expected_demand` 的常用菜返回 `expected_demand_qty`；潜力菜为 `trial_preparation`，其 `expected_demand_qty=null`，返回 `suggested_trial_qty`，避免把试餐量标成已验证需求。

`reference_orders` 和 `reference_revenue` 复用原算法值；潜力菜的这两个字段是按试餐量推算的参考值，不是已验证收入预测。汇总中的 `reference_total_revenue` 包含该参考值，`expected_high_frequency_revenue` 仅包含常用菜。数量按单位分别汇总。

`redistribute_to_total_demand=false` 保留常用菜原预测及潜力菜参考量。true 时按原常用菜单的**每单位总量**重新分配，不将克的预算转移为份；若某单位有预算却没有被选中的菜，返回 `UNIT_BUDGET_INFEASIBLE`。此行为有意修正原网页跨单位混合分配。相似菜硬冲突、非法菜品、重复选择、空菜单会报错。第一版不支持手工增加候选外菜品或手工改量。

`proposal` 返回候选、汇总、审计及完整内部状态。`result.status=ready` 表示计算通过，不表示已发布。两个结果都返回 `history_events`：新后端需按 `menu_id` 幂等保存并在后续调用中提供，不能将预测菜单作为真实销售历史。

## HTTP 接口

| 方法与路径 | 输入 | 输出 |
|---|---|---|
| GET /health | 无 | 服务、算法版本及存储状态 |
| GET /v1/model-info | 无 | 版本和完整性摘要 |
| POST /v1/data/validate | data 对象 | 数据有效性 |
| POST /v1/recommendations | `{"request": ..., "data": ...}` | 公开候选方案和 proposal_id |
| POST /v1/recommendations/finalize | `{"proposal_id": "...", "selection": {...}}` | 最终菜单和检查结果 |

HTTP 422 为字段或业务校验错误；404 为候选不存在；410 为候选过期；409 为同方案已按不同选择确认；启用 key 后未授权为401。OpenAPI 明确输入模型，Python 完整模型见 `schemas.py`。

## CSV 兼容与示例

```python
from smart_canteen_sdk.adapters import load_legacy_csv

data = load_legacy_csv(
    "/data/sales.csv", "/data/dish_library.csv",
    canteen_id="a_hupan", snapshot_id="import_v1",
    runtime_sales_path="/data/runtime_actual_sales.csv",
    recommendation_history_path="/data/recommendation_history.csv",
)
```

此适配器只适用于原项目已经预处理的 CSV，原表中的克数是内部模型数量。它会恢复为业务克数。新数据库直接传标准 `DataSnapshot`，不要使用此适配器。`from_legacy_frames` 提供相同转换但接收 DataFrame，并可显式传入自动菜单历史。

`examples/demo.py` 只使用合成数据：

```bash
python examples/demo.py
python examples/demo.py --request-json > request.json
curl -X POST http://127.0.0.1:8010/v1/recommendations -H 'Content-Type: application/json' --data-binary @request.json
```

业务历史、原数据、实验报告、虚拟环境和前端均不包含在 wheel 中。`SOURCE_MANIFEST.json` 记录原始代码来源；`BUNDLE_MANIFEST.json` 校验随包算法资源。

## 构建与测试

```bash
python -m pip install '.[server,test]'
python -m pytest -q
python -m pip wheel . --no-deps -w dist
```

原项目集成回归需要原项目测试依赖，并在仓库根目录运行 `sdk/tests/test_project_parity.py`。独立包测试自动跳过不可访问原项目时的集成检查。
