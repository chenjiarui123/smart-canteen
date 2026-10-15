"""Operator-facing copy only; never used to calculate scores or quantities."""

import math


def demand_note(label: str, factor: float) -> str:
    if not math.isfinite(factor):
        return f"{label}对客流的影响暂不明确"
    if factor > 1.005:
        return f"结合{label}，预计当餐需求有所增加"
    if factor < 0.995:
        return f"结合{label}，预计当餐需求有所减少，备餐宜谨慎"
    return f"{label}未提示当餐需求有明显变化"


def high_frequency_reason(*, recent_count: int, history_count: int,
                          appeared_last: bool, recommended7: int,
                          recommended14: int, weather_factor: float,
                          calendar_factor: float) -> str:
    if recent_count:
        history = f"近30天同餐次有{recent_count}次供应记录，本次结合以往销量选入常用菜单"
    elif history_count:
        history = "近30天同餐次暂无供应记录，本次主要参考更早的销售表现，建议留意实际需求"
    else:
        history = "本食堂同餐次的历史记录较少，本次结合可用历史推荐，建议先观察实际销售情况"
    repeat = (
        "上次同餐次菜单也选过这道菜，可结合现场情况决定是否换菜"
        if appeared_last else "上次同餐次菜单未出现，或暂无上次菜单记录"
    )
    return "。".join([
        history,
        demand_note("当前天气", weather_factor),
        demand_note("校历和节假日安排", calendar_factor),
        repeat,
        f"近7天已推荐{recommended7}次、近14天{recommended14}次，供安排菜单时参考",
    ]) + "。"


def low_frequency_reason(*, offer_count: int, ordinary_count: int,
                         dish_holiday: str | None, holiday_matches: bool,
                         holiday_score_used: bool, matching_count: int,
                         low_sample: bool, selected_count: int) -> str:
    if dish_holiday and holiday_matches and holiday_score_used:
        main = (
            f"今天是{dish_holiday}，这道菜可作为应节选择。"
            f"本次参考了{matching_count}次同节日销售记录和{ordinary_count}次非节日记录，"
            "没有直接照搬节日里的高销量，建议先少量供应、观察销售情况"
        )
    elif dish_holiday:
        timing = (
            f"今天是{dish_holiday}，但暂无可参考的同节日销售记录"
            if holiday_matches else f"这道菜常与{dish_holiday}相关，平时也可作为换口味的选择"
        )
        evidence = (
            "非节日销售样本不足，建议仅作少量试餐"
            if low_sample else f"本次参考{ordinary_count}次非节日销售记录，建议少量试餐"
        )
        main = f"{timing}。{evidence}；未使用{dish_holiday}销量来抬高推荐优先级"
    elif low_sample:
        main = "这道菜平时供应较少，非节日销售样本不足，建议先少量试餐，不宜直接大量备餐"
    else:
        main = (
            f"这道菜平时供应较少，历史有{offer_count}次供应记录。"
            f"从{ordinary_count}次非节日销售记录来看，供应时的平均销量比较可观。"
            "本次也参考了星期、季节和天气，建议作为菜单补充，先少量供应"
        )
    if selected_count:
        main += (
            f"。此前已正式选入潜力菜单{selected_count}次，"
            "本次适当降低推荐优先级，让其他潜力菜也有机会入选"
        )
    return main + "。"
