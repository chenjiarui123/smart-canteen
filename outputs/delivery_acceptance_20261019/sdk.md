# SDK 附加交付验收（2026-10-19）

本轮仅进行交付验证，没有执行训练、搜索或参数更新。网页是主交付对象，本报告补充 SDK 的可用性和打包检查。

## 结论

SDK 的推理、输入校验、候选确认、HTTP 应用层流程和独立安装包均通过本机验收；与当前网页推理在 4 组真实数据场景中数值一致。真实 TCP 监听验证未完成，不能将本报告视为端口连通或生产负载验收。

| 项目 | 结果 | 证据 |
|---|---|---|
| SDK 源码全套测试 | 21 通过，40.35 秒 | `sdk_pytest.log`、`sdk_junit.xml` |
| 真实数据与网页推理一致性 | 4 通过：两食堂 × 劳动节午餐/普通日早餐 | `test_real_deployed_prediction_parity`，已包含在 21 项中 |
| wheel 离线安装及脱离源码导入 | 通过；仅安装到临时目录 | `sdk_installed_tests.log` |
| 已安装 wheel 的引擎和 HTTP ASGI 套件 | 17 通过，1.96 秒 | `sdk_installed_tests.log`、`sdk_installed_junit.xml` |
| 打包模型完整性 | 通过，当前策略 `targeted_repeat_holiday_20260911` | 引擎加载日志中的 `integrity_ok: true` |
| wheel、源码压缩包 SHA256 | 均与 `sdk/dist/SHA256SUMS.json` 一致 | `sdk_environment.json` |
| 正式 delivery.zip 的 ZIP CRC 和内含 wheel | 通过；内含 wheel 与本机已验收 wheel 相同 | ZIP 内容检查及安装日志 |
| 真实 TCP 端口 | 未完成，受执行权限限制 | `sdk_wheel_smoke.log`，下文说明 |

两套 suite 都出现 1 条同样的依赖弃用提示：Starlette 引用的 `anyio.abc.BlockingPortal` 别名已弃用。本次未引起功能失败。

## 验证范围

- 双食堂 `a_hupan`、`f_jichuyuan`；日期/餐次为 `2026-05-01` 午餐和 `2026-06-11` 早餐。比较常用菜与潜力菜的名称、顺序、模型数量、参考订单、参考营收、分数，以及确认后的常用菜单数量；数值容差为 `rtol=1e-12, atol=1e-8`。
- 合成数据验证候选序列化恢复、输入不变及不写业务文件、剔除目标日和未来销量、食堂/重复销售/单位/上下文/菜品异常、非有限或负数量、克单位换算、按单位守恒、潜力菜开关、非法选择、状态损坏、相似菜硬冲突、正式菜单版本和自动菜单回退。
- HTTP ASGI 验证健康检查、文档、OpenAPI、数据校验、推理、确认、SQLite 状态跨应用重启保留、重复确认幂等、变更选择冲突、候选缺失/过期、API Key 和非法请求。
- 已安装 wheel 在临时工作目录运行，`PYTHONPATH` 只指向临时安装目录；日志记录实际导入路径，未用 SDK 源码覆盖 wheel。依赖来自现有 `.targeted-venv`，此项不等于从空环境重新下载全部依赖。

## 环境结论

本机已有可运行的 `.targeted-venv/bin/python`，版本 Python 3.12.14；SDK 不需要显卡、CUDA、PyTorch 或训练环境。SDK 仅调用引擎需要 Python、NumPy、pandas、Pydantic；启用 HTTP 服务再加 FastAPI、Uvicorn。SQLite 属于 Python 标准库，无须另装数据库服务器。

本次实际依赖：NumPy 2.3.5、pandas 2.2.3、Pydantic 2.14.0、FastAPI 0.115.12、Uvicorn 0.34.3、pytest 8.4.1、httpx 0.28.1。它们满足 SDK 的版本范围，但部分不等于网页 `requirements.txt` 的精确锁定版本；网页锁定环境验收由主报告记录。

`/usr/bin/python3` 为 3.9.6，不满足 SDK 的 Python 3.10+ 要求。项目附带的 `smart_canteen_web/.venv` 来自另一用户/机器（`/Users/liushilin`、`/opt/miniconda3`），不应当作为可迁移交付环境。用户当前无需为 SDK 验收手工安装软件；新机器部署时应重新创建虚拟环境并安装依赖。

## 交付包及范围边界

推荐交付 `sdk/dist/smart-canteen-sdk-0.1.0-delivery.zip` 或其中的 wheel。该 delivery.zip 包含 wheel、源码包、校验和、说明、示例和历史报告；这些随包历史报告不代替本次日志。

根目录 `sdk.zip` 包含 `.installed`、`.pytest_cache`、macOS 元数据和旧 `sdk/runtime/proposals.sqlite3`，不适合作为干净 SDK 交付包。wheel 内未发现业务 CSV、SQLite 数据库或 `.pyc`。

SDK 的 `SOURCE_MANIFEST.json` 是封装当时的来源记录，最终扫描有 16/18 项与网页源文件完全相同。差异为 `active_strategy.py` 的 UTF-8/BOM 读取修复，以及本轮网页 `report_generator.py` 的报告修复；冻结模型与参数没有改动。SDK 的内部 bundle 自检正常，现有 wheel 与 SDK 源码对应文件一致。网页侧后续修复不会自动进入旧 SDK 包，SDK 没有对外报告生成接口，本轮不重新封装。

真实 TCP 测试在沙箱内创建本机监听 socket 时得到 `PermissionError`。随后对同一动作申请自动审批，审批服务返回 `content-blocked`，明确未执行动作，且说明这是审批失败、不是不安全判定。未绕过审批；继续完成了不监听端口的独立安装包 ASGI 验证。若需补齐此项，需允许执行本机临时端口测试（合成数据、临时 SQLite，结束即清理）。

所有测试写入限定在验收输出或系统临时目录；SDK 未修改业务 CSV、历史记录、冻结模型或 manifest，也未安装到全局环境。长期负载、生产网络部署及 Windows 原生运行未在本轮 SDK 附加验收中执行。

## 可复现命令

在仓库根目录运行：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sdk/src .targeted-venv/bin/python -m pytest sdk/tests -q -ra -p no:cacheprovider
```

其他证据文件：`sdk_environment.json`、`sdk_pytest.log`、`sdk_junit.xml`、`sdk_installed_tests.log`、`sdk_installed_junit.xml`、`sdk_wheel_smoke.log`。
