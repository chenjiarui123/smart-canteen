from app.recommendation_copy import high_frequency_reason, low_frequency_reason


def test_high_copy_is_plain_and_preserves_supply_vs_recommendation_distinction():
    text = high_frequency_reason(recent_count=12, history_count=80,
        appeared_last=True, recommended7=3, recommended14=5,
        weather_factor=1.1, calendar_factor=0.8)
    assert "近30天同餐次有12次供应记录" in text
    assert "近7天已推荐3次" in text
    assert "当餐需求有所增加" in text and "当餐需求有所减少" in text
    assert "上次同餐次菜单也选过" in text
    for jargon in ["因子", "权重", "校准", "负荷", "约束", "×"]:
        assert jargon not in text


def test_high_sparse_history_does_not_claim_popularity():
    text = high_frequency_reason(recent_count=0, history_count=2,
        appeared_last=False, recommended7=0, recommended14=0,
        weather_factor=1.0, calendar_factor=1.0)
    assert "主要参考更早的销售表现" in text
    assert "或暂无上次菜单记录" in text
    assert "热销" not in text


def test_matching_festival_without_samples_does_not_claim_holiday_evidence():
    text = low_frequency_reason(offer_count=3, ordinary_count=3,
        dish_holiday="端午节", holiday_matches=True, holiday_score_used=False,
        matching_count=0, low_sample=False, selected_count=0)
    assert "暂无可参考的同节日销售记录" in text
    assert "未使用端午节销量" in text
    assert "参考了0次同节日" not in text


def test_low_repeat_and_cold_start_copy():
    text = low_frequency_reason(offer_count=1, ordinary_count=0,
        dish_holiday="端午节", holiday_matches=False, holiday_score_used=False,
        matching_count=1, low_sample=True, selected_count=2)
    assert "样本不足" in text and "少量试餐" in text
    assert "正式选入潜力菜单2次" in text
    assert "其他潜力菜也有机会" in text
    assert "降权" not in text and "×" not in text


def test_festival_copy_does_not_promise_high_sales():
    text = low_frequency_reason(offer_count=4, ordinary_count=2,
        dish_holiday="端午节", holiday_matches=True, holiday_score_used=True,
        matching_count=2, low_sample=True, selected_count=0)
    assert "应节选择" in text and "没有直接照搬" in text
    assert "2次同节日销售记录" in text
    assert "收缩" not in text
