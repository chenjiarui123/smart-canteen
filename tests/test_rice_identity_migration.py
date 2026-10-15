"""Tests for the explicitly authorized rice-data migration only."""
import pandas as pd
import pytest

from scripts.rice_merge_retest_20260913 import merge_rice


def sample():
    common = {"年": "2025", "月": "12", "日": "20", "食堂名称": "A湖畔智慧餐厅",
              "餐次": "午餐", "菜品售卖单位": "克", "菜品单价": "0.11元/50克",
              "天气": "晴"}
    return pd.DataFrame([
        dict(common, 菜品名称="米饭", 菜品总销量="6.88", 订单数量="2", 销售额="0.76", 平均取菜量="172"),
        dict(common, 菜品名称="米饭🍚", 菜品总销量="1528.64", 订单数量="357", 销售额="168.22", 平均取菜量="214.095238"),
        dict(common, 菜品名称="二米饭", 菜品总销量="20", 订单数量="10", 销售额="2.2", 平均取菜量="100"),
    ])


def test_merge_preserves_totals_and_unrelated_dish():
    before = sample()
    after, audit = merge_rice(before)
    assert audit["merged_sessions"] == 1
    rice = after.loc[after["菜品名称"].eq("米饭")].iloc[0]
    assert rice["菜品总销量"] == "1535.52"
    assert rice["订单数量"] == "359"
    assert rice["销售额"] == "168.98"
    assert abs(float(rice["平均取菜量"]) - 1535.52 * 50 / 359) < 1e-6
    pd.testing.assert_series_equal(before.iloc[2], after.iloc[1], check_names=False)


def test_merge_is_idempotent():
    once, _ = merge_rice(sample())
    twice, _ = merge_rice(once)
    pd.testing.assert_frame_equal(once, twice)


@pytest.mark.parametrize("field,value", [("菜品售卖单位", "份"), ("菜品单价", "0.22元/50克"), ("天气", "雨")])
def test_conflicting_metadata_is_not_silently_merged(field, value):
    frame = sample()
    frame.loc[1, field] = value
    with pytest.raises(AssertionError):
        merge_rice(frame)


def test_different_sessions_are_not_merged():
    frame = sample()
    frame.loc[1, "餐次"] = "晚餐"
    after, audit = merge_rice(frame)
    assert len(after) == 3
    assert audit["merged_sessions"] == 0
    assert after["菜品名称"].eq("米饭").sum() == 2
