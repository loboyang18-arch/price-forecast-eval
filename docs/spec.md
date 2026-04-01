# Metrics Spec

该工具包实现口径基于《电价预测模型量化评估标准 v1.0 附录》：

- 点级：`mae`, `rmse`
- 日级 shape：`profile_corr`, `neg_corr_day_ratio`, `amplitude_err`, `direction_acc`, `normalized_profile_mae`
- 扩展：peak/valley、turning point、block 系列
- 分场景：`tag_weekend`, `tag_vol_class`, `tag_extreme`, `tag_holiday`
- 综合分：`compute_composite_score`（DA/RT 权重）
