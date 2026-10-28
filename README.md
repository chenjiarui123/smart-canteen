# 智慧食堂启发式菜单推荐系统

面向高校食堂的本机部署菜单推荐系统。项目由两部分组成：

- **研究/回测代码**：Trial 18 冻结策略、需求预测、结构/容量优化、定向策略与历史实验；
- **交付应用**：Vue 3 + FastAPI 网页系统，以及可独立集成的 Python SDK。

仓库当前为私有仓库，包含真实食堂经营数据与历史实验结果，不应改为公开仓库。

## 项目结构

```text
.
├── research/                    # 研究代码、测试与已筛选的数据/结果（平铺模块）
│   ├── policy.py                # 冻结 Trial 18 核心策略（禁止修改）
│   ├── evaluator.py             # 固定评价器（禁止修改）
│   ├── config.json              # Trial 18 参数与指标
│   ├── demand_*.py / phase_*.py # 各阶段需求与因子实验代码
│   ├── structural_*.py / allocation_*.py  # 结构与容量实验代码
│   ├── targeted_*.py            # 定向策略与发布代码
│   ├── tests/                   # 研究回归测试
│   └── 数据/                    # 研究用销售、菜品库、映射与天气数据
├── smart_canteen_web/           # 可交付的本地网页系统
│   ├── backend/                 # FastAPI 后端与测试
│   ├── frontend/                # Vue 3 前端
│   ├── model/                   # 网页使用的冻结模型副本
│   └── data/                    # 两套逻辑食堂数据
├── sdk/                         # 独立 Python SDK 与可选 HTTP 服务
├── scripts/                     # 数据构建、迁移与验证脚本
├── analysis_artifacts/          # 菜品类别审计脚本与历史产物
├── docs/                        # 研究结果、阶段说明和历史 prompt
├── outputs/                     # 已筛选的审计/报告产物
├── reports/                     # 汇总报告
├── runs/                        # 大体量实验运行记录（仅本地，不入库）
├── archive/                     # 历史归档（仅本地，不入库）
└── versions/                    # 历史版本（仅本地，不入库）
```

研究模块采用平铺布局，是为了保持既有冻结文件的字节哈希和 `runs/` 登记清单可验证。不要把 `research/*.py` 随意移入子包；被 `model_manifest.json` 和运行登记清单锁定的文件尤其不能改动。

## 快速开始：网页系统

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r smart_canteen_web/backend/requirements.txt
cd smart_canteen_web/frontend
npm ci
npm run build
cd ../..
python smart_canteen_web/run_app.py --no-browser
```

打开 <http://127.0.0.1:8000/>。网页系统支持：

- A湖畔与 F基础园两套严格隔离的数据集；
- 高频菜单推荐与低频潜力菜；
- 销售、天气、校历数据的预览、校验、原子写入与备份；
- 菜单确认、CSV/Markdown 报告下载和推荐历史持久化。

更完整的运行、Windows 部署、数据更新和故障排查说明见
[`smart_canteen_web/README.md`](smart_canteen_web/README.md)。

## 测试

### 稳定基线：网页后端（CI 执行）

```bash
python -m pytest smart_canteen_web/backend/tests -q
```

2026-10-24 业务缺陷修复后：**51 passed**。同日隔离真实数据验收 **62/62 通过**，记录见 [`outputs/delivery_acceptance_20261024/web_acceptance.md`](outputs/delivery_acceptance_20261024/web_acceptance.md)。2026-10-19 交付验证为 48 passed，历史记录见 [`outputs/delivery_acceptance_20261019/验收报告.md`](outputs/delivery_acceptance_20261019/验收报告.md)。

### 研究测试

```bash
python -m pytest research/tests -q
```

当前基线：**85 passed**。历史 runtime 测试读取 `runs/demand_20260913/snapshot/` 和 `runs/phase_20260913/snapshot/` 中与预注册清单逐字节一致的冻结输入，不再用后来变化的生产数据和代码冒充历史输入。`active_strategy.py` 的后续 UTF-8-SIG 兼容改动，以及 A湖畔米饭合并后的生产数据，保持现状；生产文件完整性由 Web 后端测试校验。

这些测试依赖本地 `runs/`，因此不放入 CI。`runs/` 不入库；全新克隆缺少冻结快照时，相关历史 runtime 测试会显式 skip，而不是用当前生产文件替代。

### 前端

```bash
cd smart_canteen_web/frontend
npm test
```

### SDK

```bash
cd sdk
python -m pip install -e '.[server,test]'
python -m pytest -q
```

## 数据与产物策略

入库：

- 源码、测试、配置、冻结模型与完整性清单；
- 已筛选的研究数据 CSV；
- 网页运行所需的 `smart_canteen_web/data/` 数据；
- 关键指标、失败案例和审计结果；
- SDK 源码与随包模型资源。

不入库：

- `runs/`、`archive/`、`versions/`；
- 客户原始台账 `1-长春A食堂/` 至 `9-长春I食堂/`；
- 原始供应商 Excel 和大体积推荐菜单 CSV；
- `node_modules/`、虚拟环境、构建目录和缓存；
- 本地兼容软链接 `research/runs`、`research/smart_canteen_web`。

`runs/` 等目录仍保留在本机，暂不删除；待目录重整、测试与推送完成后再决定清理策略。

## 冻结模型与完整性

网页当前启用策略为 `targeted_repeat_holiday_20260911`，基础模型为 Trial 18。以下文件属于字节锁定资产：

- `research/policy.py`
- `research/diagnostics.py`
- `research/evaluator.py`
- `research/config.json`
- `research/trial18_freeze_manifest.json`
- `research/trial18_external_evaluation.lock.json`
- `research/adaptive_test_evaluation.lock.json`
- `research/trials.jsonl`
- `research/all_trials.jsonl`

不要为了让测试通过而修改哈希或重写锁定文件。若策略需要升级，应建立新版本、重新评估并更新清单。

## 研究文档

- [Trial 18 与因子/容量阶段结果](docs/RESEARCH_RESULTS.md)
- [定向策略优化](docs/TARGETED_OPTIMIZATION.md)
- [独立需求建模](docs/INDEPENDENT_DEMAND.md)
- [结构优化](docs/STRUCTURAL_OPTIMIZATION.md)
- [容量分配优化](docs/ALLOCATION_OPTIMIZATION.md)

## 已知工程风险（非业务逻辑，尚未修复）

以下问题不改变当前菜单推荐与人工确认的业务规则，但在共享、部署或正式运维前需要处理：

- **CSV 公式注入风险**：销售上传的菜名未限制以 `=`、`+`、`-`、`@` 开头，下载菜单 CSV 后用 Excel 打开时，菜名可能被解释为公式。涉及 `backend/app/preprocessor.py` 与 `backend/app/history_store.py`。
- **过期暂存不会自动清理**：上传预览文件声明两小时过期，但只在提交时校验；用户只预览不确认时，销售数据和元数据可能长期留在 `data/staging/`。涉及 `backend/app/updater.py`。
- **健康检查缺少降级路径**：`sales.csv` 或 `dish_library.csv` 缺失时，健康接口仍会计算哈希并可能返回 500，而不是返回 `degraded`。涉及 `backend/app/api/routes.py`。
- **写接口没有身份认证或访问控制**：系统默认只监听 `127.0.0.1`，本机单用户运行风险有限；若按 `deploy/nginx-menu-recommend.conf` 代理到外部路径，菜单确认、数据上传和写入接口可被任何能访问站点的人调用。对外部署前必须在反向代理或应用层增加认证和来源限制。

## 安全与隐私

- 仓库必须保持 Private；
- 不要在 issue、PR、日志或截图中粘贴真实销售明细；
- 上传的新数据应先脱敏或在私有环境中处理；
- 网页默认仅监听 `127.0.0.1`，不要直接暴露到公网。
