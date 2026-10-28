# Web 交付链路验收（2026-10-19）

通过 62 项；失败 0 项。

本次使用当前真实交付数据的临时副本，通过 FastAPI TestClient 执行 HTTP/ASGI 请求；没有占用真实 TCP 端口，没有训练或搜索参数。日期为 2026-06-11，天气为现有 UI 允许选项；这些是验证输入，不声称是当天实测天气。所有销售写入均为标记的合成验收数据，仅写副本。

## 双食堂三餐

| 数据集 | 餐次 | 状态 | 高频数 | 潜力数 | 耗时秒 |
| --- | --- | --- | ---: | ---: | ---: |
| a_hupan | 早餐 | 200 | 60 | 12 | 2.987 |
| a_hupan | 午餐 | 200 | 140 | 12 | 5.753 |
| a_hupan | 晚餐 | 200 | 118 | 12 | 5.008 |
| f_jichuyuan | 早餐 | 200 | 65 | 12 | 2.793 |
| f_jichuyuan | 午餐 | 200 | 126 | 12 | 5.039 |
| f_jichuyuan | 晚餐 | 200 | 114 | 12 | 4.290 |

## 验收范围

- 当前冻结模型与双食堂三餐推理、逐菜有限非负数量、历史日期隔离。
- 同请求重复输出、10 次连续重复、4 路并发请求。
- 不合法日期、目标日前无历史、跨食堂请求、缺必要天气字段拒绝。
- 正式确认、CSV/Markdown 下载、克×50换算、原始模型数量历史持久化。
- 重建服务对象后历史与下载恢复。
- 销售／天气／校历预览及提交、重复提交不新增、缺上下文和跨数据集提交拒绝。
- 原始 data、temp_recommendations 及 model 文件前后内容哈希比较。

连续重复：10 次，平均 5.207 秒，最大 5.297 秒。
并发：4 路，总墙钟 25.289 秒，平均请求 24.859 秒。

## 逐项结果

| 检查 | 结果 |
| --- | --- |
| health_current_data_and_model_integrity | 通过 |
| inference_a_hupan_早餐 | 通过 |
| inference_a_hupan_午餐 | 通过 |
| inference_a_hupan_晚餐 | 通过 |
| inference_f_jichuyuan_早餐 | 通过 |
| inference_f_jichuyuan_午餐 | 通过 |
| inference_f_jichuyuan_晚餐 | 通过 |
| ten_identical_requests_are_stable | 通过 |
| four_concurrent_inferences_are_stable | 通过 |
| invalid_date_rejected | 通过 |
| cross_canteen_request_rejected | 通过 |
| unknown_dataset_rejected | 通过 |
| no_prior_history_rejected | 通过 |
| missing_weather_input_rejected | 通过 |
| finalize_a_hupan_早餐 | 通过 |
| downloads_a_hupan_早餐 | 通过 |
| csv_markdown_business_units_a_hupan_早餐 | 通过 |
| final_menu_constraints_a_hupan_早餐 | 通过 |
| persisted_model_quantities_a_hupan_早餐 | 通过 |
| finalize_a_hupan_午餐 | 通过 |
| downloads_a_hupan_午餐 | 通过 |
| csv_markdown_business_units_a_hupan_午餐 | 通过 |
| final_menu_constraints_a_hupan_午餐 | 通过 |
| persisted_model_quantities_a_hupan_午餐 | 通过 |
| finalize_a_hupan_晚餐 | 通过 |
| downloads_a_hupan_晚餐 | 通过 |
| csv_markdown_business_units_a_hupan_晚餐 | 通过 |
| final_menu_constraints_a_hupan_晚餐 | 通过 |
| persisted_model_quantities_a_hupan_晚餐 | 通过 |
| finalize_f_jichuyuan_早餐 | 通过 |
| downloads_f_jichuyuan_早餐 | 通过 |
| csv_markdown_business_units_f_jichuyuan_早餐 | 通过 |
| final_menu_constraints_f_jichuyuan_早餐 | 通过 |
| persisted_model_quantities_f_jichuyuan_早餐 | 通过 |
| finalize_f_jichuyuan_午餐 | 通过 |
| downloads_f_jichuyuan_午餐 | 通过 |
| csv_markdown_business_units_f_jichuyuan_午餐 | 通过 |
| final_menu_constraints_f_jichuyuan_午餐 | 通过 |
| persisted_model_quantities_f_jichuyuan_午餐 | 通过 |
| finalize_f_jichuyuan_晚餐 | 通过 |
| downloads_f_jichuyuan_晚餐 | 通过 |
| csv_markdown_business_units_f_jichuyuan_晚餐 | 通过 |
| final_menu_constraints_f_jichuyuan_晚餐 | 通过 |
| persisted_model_quantities_f_jichuyuan_晚餐 | 通过 |
| downloads_survive_service_reconstruction | 通过 |
| history_survives_service_reconstruction | 通过 |
| sales_missing_stored_context_rejected | 通过 |
| blocked_preview_cannot_commit | 通过 |
| weather_preview_accepts_valid_row | 通过 |
| weather_commit_and_duplicate_are_idempotent | 通过 |
| calendar_preview_accepts_valid_row | 通过 |
| calendar_commit_and_duplicate_are_idempotent | 通过 |
| data_preview_accepts_valid_row | 通过 |
| sales_cross_dataset_commit_rejected | 通过 |
| data_commit_and_duplicate_are_idempotent | 通过 |
| sales_update_does_not_change_other_canteen | 通过 |
| synthetic_sales_written_once | 通过 |
| committed_weather_calendar_are_readable | 通过 |
| health_after_updates | 通过 |
| original_data_byte_identical | 通过 |
| original_temp_recommendations_byte_identical | 通过 |
| original_model_byte_identical | 通过 |

## 证据与边界

聚合机器证据：`web_acceptance.json`。私有本地报告样例：`exports/`。临时销售副本已自动清理。

本轮是有界烟测与短时重复／并发验收，不等于长时间压测或所有操作系统认证。数据仍截止 2026-06-10，当前业务精度与历史单位变化见 `data_audit.md`。
