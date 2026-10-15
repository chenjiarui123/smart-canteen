# 智慧食堂本地菜单推荐系统

这是一个完全在本机运行的 Vue 3 + FastAPI 应用。前端只负责受控输入、展示和下载；销售数据、实时特征、冻结 Trial 18 推理、上传校验及 CSV 写入全部由后端完成。系统不联网发送业务数据，也不使用 LLM 对菜品主观评分。

## 系统范围

系统包含两个严格隔离的逻辑数据集：

| 数据集 ID | 网页名称 | 初始化历史 |
|---|---|---|
| `a_hupan` | A湖畔智慧餐厅 | 长春A食堂 + 吉大湖畔智慧餐厅 |
| `f_jichuyuan` | F基础园智慧餐厅 | 长春F食堂 + 吉大南校基础园智慧餐厅 |

每套数据分别保存在 `data/datasets/<dataset_id>/`。后端按 `dataset_registry.json` 解析目录，不会把 A湖畔与F基础园的销售历史混合。

初始化销售数据 `sales.csv` 只读；网页新增的真实销售记录写入 `runtime_actual_sales.csv`。推荐时合并两者、按日期+餐次+标准化菜名去重，并严格过滤 `date < prediction_date`。

## 冻结模型

当前网页通过 `active_strategy.py` 启用 `targeted_repeat_holiday_20260911`，包含相邻日重复与节假日策略调整。下列 Trial 18 信息描述基础模型；最新启用参数见 `model/targeted_config.json`，完整性信息见 `model/targeted_manifest.json`。

- 策略版本：`factor_capacity_v1_trial18`
- Trial：18
- 策略代码 SHA-256：`40fc1ff02d494059c6db366da37cfca8fac24190993945872313e21a8a3ef9b5`
- 固定 Evaluator SHA-256：`4f5cd9494b40ea021744640bc86551febb4f019eb4a668ddc37f20fcea9fcffc`
- 参数 SHA-256：`43fd47621fd7eb127293a98bdb00dce5279f7b449e1a4d8ddab0614bedf4389e`

`model/policy.py`、`model/diagnostics.py` 和 `model/evaluator.py` 是现有 Trial 18 的字节级副本。网页系统没有重新训练、搜索参数或修改这些文件。

实时适配层只做一件实验回测函数没有做的事：把目标日期以前的 `recommendation_history.csv` 显式注入原函数已经存在的7天、14天和上一菜单重复惩罚状态。空推荐历史时，测试要求适配输出与冻结函数逐字段完全一致。

高频菜单仍由上述冻结策略独立生成。`backend/app/high_frequency_recommender.py` 只负责调用和展示适配，原候选池、预测公式、评分方式、配置与参数均未修改。新增的低频高销量潜力菜来自独立的 `backend/app/low_frequency_recommender.py`，参数单独保存在 `backend/app/low_frequency_config.json`；低频模块异常时会自动降级，高频菜单仍可确认和保存。

低频模块把每道菜历史明确分成普通日期、全部节日、菜品对应节日三套表现。汤圆、粽子、月饼在普通日期仍可进入候选，但普通日期评分只读取非节日记录；对应节日才会使用匹配节日表现，并按样本量收缩。其他节日不会调用不相关节日销量。候选同时计算星期、季节、温度和降水适配，使用可复现的半随机刷新；正式选入最终菜单的次数越多，后续 `selection_penalty` 越低。

潜力菜不再使用固定8道上限。网页可按“全部、荤菜、素菜、主食、甜品、水果”选择菜品库类别，并输入1～60的推荐数量（默认12）。后端先用 `dish_library.csv` 的类别筛选，再在该类别内部计算潜力表现和排名；不会根据菜名推断或覆盖类别，也不会为凑数量降低筛选标准。`low_frequency_config.json` 中的 `max_candidate_limit=60` 仅作为请求安全上限。

菜品库只保留菜名、类别、口味、颜色、营养以及存在时的食材、做法、主食和相似菜等静态字段。历史平均销量、历史销售额、总销量、总销售额和订单数等数值统计没有复制到网页数据集。

## 页面和地址

- 菜单推荐：`http://localhost:8000/`
- 数据更新：`http://localhost:8000/data-update`
- 健康状态：`http://localhost:8000/api/health`
- 本地 API 文档：`http://localhost:8000/api/docs`

生产模式由 FastAPI 提供 `frontend/dist`。所有非 `/api` 页面路由回退到 `index.html`，所以直接刷新 `/data-update` 不会出现404。

## Windows PowerShell

推荐安装：

- Python 3.10～3.12；
- Node.js 20.19+ 或 22.12+（只有首次重新构建前端时需要）。

在项目目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Set-Location frontend
npm ci
npm run test
npm run build
Set-Location ..
python run_app.py
```

如果使用随项目交付的已构建 `frontend/dist`，可以跳过 Node.js、`npm ci` 和 `npm run build`，只安装 Python 依赖后运行 `python run_app.py`。

## macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
cd frontend
npm ci
npm run test
npm run build
cd ..
python run_app.py
```

## Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
cd frontend
npm ci
npm run test
npm run build
cd ..
python run_app.py
```

系统默认只监听 `127.0.0.1:8000`。如需更换端口：

```bash
python run_app.py --port 8080
```

不自动打开浏览器：

```bash
python run_app.py --no-browser
```

所有路径都从 `run_app.py` 和模块文件位置解析，不依赖启动时所在目录，也没有 macOS 用户路径或 Windows 盘符硬编码。

## 开发模式

后端：

```bash
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```

前端：

```bash
cd frontend
npm run dev
```

Vite 把 `/api` 代理到本地 FastAPI。Vue 路由和 API 路由互不混用。

## 菜单推荐流程

1. 用户从后端返回的下拉选项选择食堂、日期、餐次、天气与校历状态；
2. 日期改变时，前端调用 `/api/context/{date}` 自动回填已有天气和校历；
3. 后端只读取目标日期以前的 `sales.csv + runtime_actual_sales.csv`；
4. `build_inference_context()` 独立计算近期、长期、星期、趋势、天气、假期和容量特征，不读取目标日真实菜单或销量；
5. 读取目标日前持久化推荐历史，应用上一菜单、7天和14天重复惩罚；
6. 菜品活跃状态以真实历史末日为参照：`history_max_date - dish_last_date`。预测日期与历史末日之间的未来空白不会被误判为停供；真正截至历史末日已超过 `max_inactive_days` 未出现的非主食仍会过滤；
7. 当预测日距离历史末日超过近期窗口时，页面提示“远期预测、近期特征不足”。现有自适应近期权重会因目标日前近期窗口缺少观测而降低近期权重、增加长期历史占比；
8. 原策略独立生成常用菜品，低频模块按用户选择的菜品库类别和数量生成潜力菜；
9. 常用菜品默认进入独立“已选菜品”缓冲区。缓冲区记录来源、类别、原始预测量、展示量、单位、订单和销售额；表格筛选与潜力菜刷新都不会删除缓冲区内容；
10. 半随机刷新只替换当前类别中尚未选择的潜力菜，已选潜力菜不参与刷新且不占新一批候选数量。前端随请求传递上一草稿编号和全部已选潜力菜，后端新草稿合并保留这些菜，因此连续刷新后仍可确认保存；
11. 点击“确认并保存菜单”后，`POST /api/recommend/finalize` 以缓冲区为最终选择，校验所选菜均来自后端草稿、没有重复菜或相似菜硬冲突；菜单容量只作为菜品数量参考，不是潜力菜硬上限；
12. 潜力菜作为常用菜品之外的附加试餐项，不占用常用菜品名额。默认保留常用菜品原预测量，并额外加入潜力菜建议试餐量；系统对最终菜品数和总备餐量相对参考值的差异给出提醒，但不阻止保存；
13. 用户可选“按当餐总需求重新分配备餐量”。开启后，合并层按常用菜品预测权重和潜力菜建议试餐量重新分配，使最终 `pred_qty` 总和等于餐次预测总需求；关闭时则保留“常用菜品原量 + 潜力试餐量”；
14. 生成包含推荐来源、节日数据使用情况、数量/备餐提醒和整体搭配检查的 Markdown 报告，并在文件锁内保存推荐文件、追加正式推荐历史。

下载的菜单 CSV 仅包含 `日期、食堂名称、餐次、菜品id、菜品名、菜品类别、菜品销量、菜品销售额`。菜品 ID 按标准化菜名从当前逻辑食堂的菜品库读取；克制菜的模型销量以 50 克份计数，下载时乘以 50 输出总克数，其他售卖单位保持原预测数量。该展示转换不修改模型预测、销售额或内部推荐历史。

常用菜品表可按类别和网页展示销量的最低/最高值筛选。该筛选只改变可见行，不重新计算评分、不取消选择。原始销售数据中的“菜品总销量”以50克计量单位记录，因此售卖单位为“克”的菜在网页及下载菜单中按“原始预测量 × 50”换算为总克数；原始 `pred_qty`、模型输入、预测订单、预测销售额和内部推荐历史均不变。气温、降水量和风速下拉值由原选项四舍五入并去重为整数，日期回填也使用整数；天气历史 CSV 不做修改。

生成双栏候选属于草稿，不写入推荐历史。只有用户确认后的菜品才是“正式选中”；其中潜力菜以 `recommendation_source=low_frequency_opportunity` 写入，并用于下一次渐进降权。高频菜记录为 `recommendation_source=high_frequency`。

推荐文件保存在：

```text
temp_recommendations/<dataset_id>/<餐次>/<YYYY-MM-DD>.csv
temp_recommendations/<dataset_id>/<餐次>/<YYYY-MM-DD>.md
```

同名文件存在时自动使用 `_v2`、`_v3`，从不覆盖旧推荐。

## 数据更新

数据更新页支持 UTF-8/UTF-8-SIG CSV 和 `.xlsx`。不接受 `.xls`、`.xlsm`、宏或公式。

销售、天气和校历均采用两阶段流程：

1. 上传并选择工作表；
2. 预览原始行数、有效行、重复、异常、日期范围、未匹配菜品及缺失上下文；
3. 用户明确点击“确认写入”；
4. 后端才把暂存数据写入正式 CSV。

销售业务唯一键为日期+餐次+标准化菜名。相同文件重复上传不会重复写入。销售数据缺少天气或校历日期时会阻止写入。

天气和校历按日期唯一。已有日期内容不同会显示为冲突并保留原记录，不会静默覆盖。

## CSV 安全、备份与恢复

所有写入都集中在 `dataset_repository.py`、`history_store.py` 和 `updater.py`：

- 所有业务写入共用跨平台文件锁；
- 临时文件写在目标文件同目录；
- 临时 CSV 可重新读取且字段一致后才使用 `os.replace()` 原子替换；
- 更新前备份到 `data/backups/`；
- 事务后续步骤失败时恢复原文件；
- 写入前关闭文件句柄；
- 推荐保存与数据更新写操作互斥。

恢复前先停止应用。使用跨平台恢复脚本：

```bash
python scripts/restore_backup.py \
  --backup data/backups/<目录>/<备份文件.csv> \
  --target data/datasets/<dataset_id>/runtime_actual_sales.csv
```

恢复脚本会先为当前目标再创建一份安全副本。初始化 `sales.csv` 禁止通过恢复脚本修改。

## 测试

后端：

```bash
python -m pytest backend/tests -q
```

前端：

```bash
cd frontend
npm test
npm run build
```

测试覆盖：

- 两个逻辑食堂映射与历史隔离；
- 目标日期以前历史过滤；
- 远期预测按真实历史末日判断菜品活跃状态；2026-06-10 历史末日到 2026-07-10 晚餐的未来空白不会误删 2026-06-09 仍供应的非主食，历史末日前真正超期未出现的菜仍会过滤；
- 同日不同餐次独立上下文；
- 冻结函数空历史回归一致；
- 持久化推荐历史影响重复惩罚；
- 推荐文件版本化和确定性；
- CSV/XLSX、UTF-8-SIG 与公式拒绝；
- 重复上传幂等、并发锁和失败回滚；
- 数值天气类型、中文星期与假期联动；
- 页面/API 路由分离及 `/data-update` 刷新；
- 冻结模型和既有项目关键文件哈希未改变；
- 冻结高频候选、预测和评分逐字段回归一致；
- 低频模块独立运行及异常时高频功能降级可用；
- 节日菜普通日期可候选、普通评分不读取节日销量、仅匹配节日使用收缩后的节日表现；
- 正式选中次数渐进降权、潜力菜额外添加、参考容量提醒、可选总需求重分配和整体硬约束检查；
- 潜力菜按菜品库类别先筛选后评分，1～60数量校验及候选不足提示；
- 已选潜力菜连续刷新三次后仍在后端草稿中，并能确认保存且最终菜单无重复；
- 常用菜品类别/展示销量筛选不影响缓冲区，克单位按原始预测量乘以50展示总克数；
- 气温、降水量和风速选项及日期回填值均为去重整数；
- Vue 生产构建和统一启动。

本次页面完善已实际完成：

- 当前执行环境没有可直接调用的 `pytest` 命令，因此未伪造完整 pytest 结果；已直接执行15个相关后端回归用例，并额外完成 API、Schema、真实数据类别数量、候选不足、三次刷新保留及临时目录确认保存检查，全部通过；
- 前端 `13 passed`；
- A湖畔完整逻辑历史完成荤菜10、水果4、全部20和水果60不足场景的不落盘真实推理；三个低频筛选请求的常用菜品名称、预测和评分完全一致；
- `npm audit` 为0个已知生产或开发依赖漏洞；
- Vue生产构建成功；
- `python run_app.py --no-browser` 启动成功；
- `/api/health`、`/`、`/data-update` 均返回200，且后两个地址返回同一Vue入口；
- 两套 `sales.csv`、菜品库及冻结模型健康检查均为完整。

## 常见故障

### 页面提示“前端尚未构建”

在 `frontend` 目录执行 `npm ci` 和 `npm run build`，确认存在 `frontend/dist/index.html`。

### 提示缺少后端依赖

确认已经激活虚拟环境，再执行：

```bash
python -m pip install -r backend/requirements.txt
```

### 推荐日期无法自动回填

这表示 `weather.csv` 或 `calendar.csv` 没有该日期。先在数据更新页上传对应公共特征，或从后端允许的下拉范围手动选择；系统不会虚构缺失上下文。

### 上传销售数据无法确认

查看预览区的缺失字段、异常行、缺失天气和缺失校历日期。必须先补齐所有上下文日期。

### 端口被占用

使用 `python run_app.py --port 8080` 更换端口。

### Windows 文件被其他程序占用

关闭正在打开运行时 CSV 的 Excel 窗口后重试。后端自身会在写入前关闭文件句柄。
