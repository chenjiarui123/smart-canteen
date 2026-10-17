"""Targeted continuation of the frozen policy, with explicit guarded integration hooks.

The base source remains byte-for-byte reproducible. A private function namespace
reuses its features, pricing, allocation and output contract; no global monkey
patching, training on external data, or runtime modification of model files.
"""
from __future__ import annotations

import inspect
import math
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

import policy as base


def _special(target, config, key):
    category = str(target.get("_calendar_category", "普通日期"))
    categories = config.get(key + "_categories", ["校历假期", "法定节假日", "校历法定重叠", "调休工作日"])
    excluded = config.get(key + "_excluded_states", [])
    return bool(config.get(key, False)) and category in categories and str(target.get("_calendar_state", "")) not in excluded


def holiday_estimate(meta, target, config, metric, parent_value=None):
    """Local exact state -> compatible parent -> same-meal transferred state.

    State levels are pooled in log space.
    Overlap is a single exact joint state backed by the same school-break state;
    holiday/weather/trend factors are not multiplied onto this absolute estimate.
    """
    state = str(target.get("_calendar_state", ""))
    category = str(target.get("_calendar_category", "普通日期"))
    day = "workday" if target.get("_day_type") == "工作日" else "weekend"
    tau = float(config.get("holiday_shrinkage_tau", 3.0))
    statistic = "median"
    value_key = statistic if metric == "qty" else "dish_count_" + statistic
    fallback_key = f"regular_{day}_qty_mean" if metric == "qty" else f"regular_{day}_capacity_median"

    def pool(child, parent, n):
        w = n / (n + tau) if tau else 1.0
        return float(np.expm1(w * np.log1p(max(child, 0)) + (1 - w) * np.log1p(max(parent, 0))))

    def entries(prefix):
        levels = []
        # Different school breaks are not interchangeable. For an overlap,
        # use its own school-break parent, then the joint calendar state.
        if category == "校历法定重叠":
            levels.append(("calendar_state_stats", state.split("|法定:")[0]))
        elif category == "法定节假日":
            levels.append(("calendar_category_stats", category))
        levels.append(("calendar_state_stats", state))
        result, seen = [], set()
        for table, key in levels:
            entry = meta.get(prefix + table, {}).get(key)
            if not entry or float(entry.get("count", 0)) <= 0:
                continue
            value = float(entry.get(value_key, math.nan))
            n = float(entry["count"])
            identity = (value, n)
            if math.isfinite(value) and value >= 0 and identity not in seen:
                result.append((value, n, prefix + table + ":" + key))
                seen.add(identity)
        return result

    fallback = float(meta.get("l_" + fallback_key, math.nan))
    if not math.isfinite(fallback):
        fallback = float(meta.get("g_" + fallback_key, 0))
    if parent_value is not None and math.isfinite(parent_value):
        fallback = float(parent_value)
    current, count, source = max(fallback, 0), 0.0, "regular_same_meal_fallback"
    local, global_entries = entries("l_"), entries("g_")
    # Transfer only when local special history is absent. Do not count the
    # same canteen's history twice as both local and global evidence.
    for value, n, name in local or global_entries:
        current = pool(value, current, n)
        count, source = n, name
    return current, count, source


def group_forecast(meta, metric, config, target_context=None, return_components=False):
    target = target_context or {}
    result = base._group_total_forecast(meta, metric, config, target, True)
    if metric == "qty" and _special(target, config, "holiday_demand"):
        parent = result["group_total_qty"]
        estimate, count, source = holiday_estimate(meta, target, config, "qty", parent)
        if count > 0:
            value = estimate
            # Absolute state estimate: no ordinary-day multiplier floor.
            denominator = max(float(result["base_total_qty"]), 1e-12)
            factor = value / denominator
            result.update(group_total_qty=value, weekday_factor=1.0, group_trend_factor=1.0,
                          group_momentum_factor=1.0, calendar_factor=factor, weather_factor=1.0,
                          weather_residual_factor=1.0, raw_combined_factor=factor,
                          final_combined_factor=factor, factor_clip_applied=False,
                          calendar_sample_count=count, factor_combination_mode="joint_absolute_history",
                          holiday_demand_source=source)
        else:
            result["holiday_demand_source"] = "no_compatible_special_history; legacy fallback"
    return result if return_components else result["group_total_qty"]


def capacity_forecast(meta, config, target, pred_total_qty):
    result = base._capacity_forecast(meta, config, target, pred_total_qty)
    if _special(target, config, "holiday_capacity"):
        parent = result["planned_capacity"]
        estimate, count, source = holiday_estimate(meta, target, config, "count", parent)
        if count > 0:
            value = estimate
            result.update(planned_capacity=max(1, int(round(value))), capacity_state_estimate=estimate,
                          capacity_state_sample_count=count, capacity_weather_factor=1.0,
                          capacity_mode="independent_holiday_history", holiday_capacity_source=source)
    return result


def greedy_select(frame, capacity, constraints, config, *, previous=None, required_new=0):
    """Same soft-diversity objective, evaluated with arrays instead of frame copies.

    Reserving novel selections first makes the quota binding. Duplicate names
    and hard similarity groups are always excluded, including staples.
    """
    if frame.empty:
        return [], set()
    frame = frame.drop_duplicates("_dish_norm").sort_values("菜品名称_库", kind="stable")
    records = frame.to_dict("records")
    score = frame["score"].to_numpy(float)
    norms = frame["_dish_norm"].astype(str).to_numpy()
    sim = frame["相似菜分组"].astype(str).to_numpy()
    hard = frame["_similarity_hard"].fillna(False).to_numpy(bool) & np.array([base._is_known(x) for x in sim])
    new = np.array([n not in (previous or set()) for n in norms])
    specs = [("主要食材", "max_same_main_ingredient", "selection_same_main_penalty", .85),
             ("食材大类", "max_same_ingredient_family", "selection_same_family_penalty", .85),
             ("制作方法", "max_same_cooking_method", "selection_same_method_penalty", .9),
             ("口味", "max_same_flavor", "selection_same_flavor_penalty", .95)]
    values = [frame[s[0]].astype(str).to_numpy() for s in specs]
    known = [np.array([base._is_known(x) for x in a]) for a in values]
    counters = [Counter() for _ in specs]
    sim_counts = Counter()
    used = np.zeros(len(frame), dtype=bool)
    selected, excluded = [], set()
    n_new = 0
    repeat_cap = capacity - required_new
    for _ in range(capacity):
        conflict = hard & np.array([sim_counts[x] >= int(constraints["max_same_similarity_group"]) for x in sim])
        excluded.update(norms[conflict])
        allowed = ~used & ~conflict
        if n_new < required_new:
            allowed &= new
        elif len(selected) - n_new >= repeat_cap:
            allowed &= new
        if not allowed.any():
            break
        utility = score.copy()
        for i, (_, limit_key, penalty_key, default) in enumerate(specs):
            limit = constraints.get("soft_max_same_main_ingredient", constraints[limit_key]) if i == 0 else constraints[limit_key]
            excess = np.maximum(np.array([counters[i][v] for v in values[i]]) + 1 - limit, 0)
            utility *= np.power(float(config.get(penalty_key, default)), np.where(known[i], excess, 0))
        utility[~allowed] = -np.inf
        idx = int(np.argmax(utility))
        used[idx] = True
        selected.append(pd.Series(records[idx]))
        n_new += int(new[idx])
        if hard[idx]:
            sim_counts[sim[idx]] += 1
        for i in range(len(specs)):
            if known[i][idx]:
                counters[i][values[i][idx]] += 1
    return selected, excluded


def enforce_repeat(selected, candidates, total, config, constraints, prior_history, context):
    target = pd.Timestamp(context["date"]).normalize()
    previous = next((set(base.normalize_name(n) for n in names) for d, names in reversed(prior_history)
                     if pd.Timestamp(d).normalize() == target - pd.Timedelta(days=1) and names), None)
    n = len(selected)
    audit = {"adjacent_status": "missing_previous" if previous is None else "not_enforced",
             "adjacent_repeat_rate": None, "replacement_count": 0, "replacement_shortage": 0,
             "requested_high_frequency_count": n, "infeasible_reason": "", "replacements": []}
    context["_targeted_audit"] = audit
    if previous is None or not n:
        return selected
    original = set(selected["_dish_norm"].astype(str))
    cap = (4 * n) // 5
    repeated = len(original & previous)
    audit["adjacent_repeat_rate"] = repeated / n
    audit["adjacent_repeat_cap"] = cap
    if not config.get("adjacent_repeat_hard", False):
        return selected
    if repeated <= cap:
        audit["adjacent_status"] = "feasible"
        return selected
    minimum = int(config.get("replacement_min_history", 3))
    local_count = candidates["l_offer_count"].fillna(0)
    reliable = local_count.ge(minimum)
    if float(context["meta"].get("l_session_count", 0)) < float(config["candidate_warmup_sessions"]):
        reliable |= candidates["g_offer_count"].fillna(0).ge(minimum)
    pool = candidates.loc[reliable | candidates["_dish_norm"].isin(original)].copy()
    # Incumbents without reliable history may be retained, but cannot be used
    # as new replacements. All replacement records came from this meal's pool.
    novelty = pool.loc[~pool["_dish_norm"].isin(previous)]
    audit["eligible_novel_count"] = int(novelty["_dish_norm"].nunique())
    banned = set()
    while True:
        available = pool.loc[~pool["_dish_norm"].isin(banned)].copy()
        replacement_config = dict(config)
        diversity_strength = float(config.get("replacement_diversity_strength", 1))
        for key, default in [("selection_same_main_penalty", .85), ("selection_same_family_penalty", .85),
                             ("selection_same_method_penalty", .9), ("selection_same_flavor_penalty", .95)]:
            replacement_config[key] = float(config.get(key, default)) ** diversity_strength
        rows, _ = greedy_select(available, n, constraints, replacement_config, previous=previous, required_new=n-cap)
        candidate = pd.DataFrame(rows)
        if len(candidate) != n:
            break
        candidate["expected_orders_per_qty"] = candidate["_unit_orders_ratio"].fillna(0)
        candidate["expected_rev_per_qty"] = candidate["_unit_revenue_ratio"].fillna(0)
        allocated, _ = base._allocate_and_prune_min_batch(candidate, total, dict(config, min_batch_mode="none"))
        below = allocated.pred_qty + 1e-9 < allocated._min_batch_qty
        if not below.any():
            chosen = set(allocated["_dish_norm"].astype(str))
            assert len(chosen) == n and len(chosen & previous) <= cap
            audit.update(adjacent_status="feasible", adjacent_repeat_rate=len(chosen & previous) / n,
                         replacement_count=len(chosen-original),
                         replacements=[{"dish": str(r["_dish_norm"]), "local_history_count": float(r.get("l_offer_count", 0)),
                                        "same_meal_global_history_count": float(r.get("g_offer_count", 0)),
                                        "qty_stability": float(r["_qty_stability"]), "pred_qty": float(r["pred_qty"])}
                                       for r in allocated.to_dict("records") if str(r["_dish_norm"]) not in original])
            return allocated
        banned.update(allocated.loc[below, "_dish_norm"].astype(str))
    # No undersized or over-quota menu is released. The full intended demand
    # and count stay in audit, and the failed session remains in evaluation.
    required = n-cap
    independent_new = greedy_select(novelty.loc[~novelty["_dish_norm"].isin(banned)], len(novelty), constraints, config)[0]
    independent_all = greedy_select(pool.loc[~pool["_dish_norm"].isin(banned)], len(pool), constraints, config)[0]
    shortage = max(required-len(independent_new), n-len(independent_all), 1)
    audit.update(adjacent_status="infeasible", adjacent_repeat_rate=None, replacement_shortage=shortage,
                 unreleased_total_qty=total, unreleased_count=n,
                 infeasible_reason=f"计划{n}道需至少{required}道前日未出现菜；当前组合经同餐次可靠历史、相似菜约束和最低备餐量检查后缺少{shortage}道可行替换；未发布自动菜单")
    return selected.iloc[0:0].copy()


@lru_cache(maxsize=1)
def _integrated_function():
    source = inspect.getsource(base.make_recommendations)
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError("Trial18 integration hook mismatch: " + old[:90])
        source = source.replace(old, new)
    replace("    selection_audit: list[dict[str, Any]] | None = None,\n", "    selection_audit: list[dict[str, Any]] | None = None,\n    initial_menu_history: dict | None = None,\n")
    replace("    menu_history: dict[tuple[str, str], list[tuple[pd.Timestamp, set[str]]]] = {}",
            "    menu_history = {k: [(pd.Timestamp(d), set(names)) for d, names in v] for k, v in (initial_menu_history or {}).items()}")
    start = source.index('        if constraint_mode == "penalized_diversity_greedy":')
    end = source.index('        shadow_bottom_up_total =', start)
    source = source[:start] + "        selected, hard_excluded = greedy_select(frame, selection_capacity, constraints, policy)\n" + source[end:]
    replace("        days_inactive = (target_date - last_date).dt.days", "        inactivity_reference = pd.Timestamp(context['max_history_date']) if policy.get('runtime_history_end_inactivity', False) else target_date\n        days_inactive = (inactivity_reference - last_date).dt.days")
    replace("        selected_frame, min_batch_removed_count = _allocate_and_prune_min_batch(",
            "        if total_components.get('factor_combination_mode') == 'joint_absolute_history':\n            target_total = group_total\n        selected_frame, min_batch_removed_count = _allocate_and_prune_min_batch(")
    replace("        below_min_batch_count = int(\n", "        selected_frame = enforce_repeat(selected_frame, frame, target_total, policy, constraints, prior_history, context)\n        below_min_batch_count = int(\n")
    namespace = dict(base.__dict__)
    namespace.update(_group_total_forecast=group_forecast, _capacity_forecast=capacity_forecast,
                     greedy_select=greedy_select, enforce_repeat=enforce_repeat, _special=_special)
    exec(compile(source, __file__ + "::trial18_adapter", "exec"), namespace)
    return namespace["make_recommendations"]


def make_recommendations(contexts, config, constraints, selection_audit=None, initial_menu_history=None):
    contexts = [dict(c) for c in contexts]
    first_date = min((pd.Timestamp(c["date"]) for c in contexts), default=pd.Timestamp.max)
    initial_menu_history = {
        (str(key[0]), str(key[1])): sorted(
            [(pd.Timestamp(d).normalize(), set(base.normalize_name(n) for n in names))
             for d, names in values if pd.Timestamp(d) < first_date], key=lambda item: item[0])
        for key, values in (initial_menu_history or {}).items()
    }
    audit = []
    recs = _integrated_function()(contexts, config, constraints, selection_audit=audit, initial_menu_history=initial_menu_history)
    indexed = {(pd.Timestamp(c["date"]), c["食堂名称"], c["餐次"]): c for c in contexts}
    for a in audit:
        c = indexed[(pd.Timestamp(a["date"]), a["食堂名称"], a["餐次"])]
        a.update(c.get("_targeted_audit", {"adjacent_status": "infeasible", "infeasible_reason": "无合格候选", "replacement_shortage": a["planned_capacity"]}))
    if selection_audit is not None:
        selection_audit.extend(audit)
    return recs
