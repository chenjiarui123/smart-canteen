# 前端交付验收记录

- 验收日期：2026-10-19（Asia/Shanghai）。
- 范围：现有依赖、单元测试、生产构建；未进行训练，未修改模型、业务数据、前端源码或依赖清单。
- 结论：当前 macOS arm64 环境可直接完成前端测试与生产构建，没有安装阻塞。浏览器端实际交互由总验收单独覆盖。

## 环境与依赖

执行位置：`smart_canteen_web/frontend`。

| 检查 | 结果 |
| --- | --- |
| `node --version` | `v23.11.0` |
| `npm --version` | `10.9.2` |
| `uname -m` | `arm64` |
| `npm ls --depth=0` | 退出码 0，9 个直接依赖均正常解析 |
| 原生依赖 | 已存在 `@esbuild/darwin-arm64`、`@rollup/rollup-darwin-arm64`，与当前平台一致 |
| 引擎要求 | Vite 7.3.6 和 Vue 插件 6.0.0 均要求 `^20.19.0 || >=22.12.0`，当前版本满足 |

直接依赖版本：`@element-plus/icons-vue@2.3.1`、`@vitejs/plugin-vue@6.0.0`、`axios@1.18.1`、`element-plus@2.14.3`、`jsdom@26.1.0`、`vite@7.3.6`、`vitest@3.2.7`、`vue-router@4.5.1`、`vue@3.5.17`。

沿用当前 `node_modules`，没有运行 `npm ci`，没有新增软件或联网安装。当前 Node 23 不是适合作为长期交付基线的 LTS 版本；新机器可按 README 使用 Node 22.12+ 的 LTS 分支。使用已构建 `dist` 运行后端服务时不需要 Node.js。迁移到其他操作系统/CPU 架构后，重新构建应在目标机器执行 `npm ci`，不要复用本机原生依赖目录。

## 单元测试

命令：`npm test`。退出码：0。Vitest 3.2.7，7 个测试文件、18 个用例全部通过，耗时 1.02 秒。

| 文件 | 通过用例 |
| --- | ---: |
| `controlled-inputs.spec.js` | 4 |
| `dataUpdateUploadReset.spec.js` | 3 |
| `deploymentUrl.spec.js` | 2 |
| `displayUnits.spec.js` | 2 |
| `selectionBuffer.spec.js` | 2 |
| `formRules.spec.js` | 4 |
| `routes.spec.js` | 1 |

以上为现有 jsdom 单元测试，不代表已经覆盖真实浏览器操作或后端推理。

## 生产构建

命令：`npm run build`。退出码：0。Vite 7.3.6，转换 1,665 个模块，构建耗时 1.71 秒。

构建目录：`/Users/chenjr/Desktop/Smart Canteen Heuristic Learning/smart_canteen_web/frontend/dist`。

| 文件 | 原始大小 | gzip 大小 |
| --- | ---: | ---: |
| `dist/index.html` | 0.53 kB | 0.37 kB |
| `dist/assets/logo-BQygBWA3.png` | 14.77 kB | — |
| `dist/assets/index-Ozv3dPFP.css` | 363.61 kB | 49.43 kB |
| `dist/assets/index-DLxzn-ND.js` | 1,263.28 kB | 405.11 kB |

本次为 README 中的本地根路径部署构建，静态资源使用 `/assets/...`，API 使用 `/api`。没有运行用于 `/menu_recommend/` 子路径部署的 `npm run build:server`。

非阻断警告：

1. `@vueuse/core/dist/index.js` 两处 `/* #__PURE__ */` 注释位置无法被 Rollup 识别，构建器已移除注释并继续成功构建。
2. 主 JavaScript 包超过 500 kB 的默认提示阈值；当前压缩后 405.11 kB。可作为后续加载性能优化项，本地交付构建不受阻。

`git status --short -- smart_canteen_web/frontend` 没有显示受版本控制的前端文件变更。构建产物已刷新。本次未运行联网漏洞审计，不能沿用 README 中历史审计数字作为本次结果。
