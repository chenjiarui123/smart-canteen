"""Historical offered-quantity features; never materialize missing sales as zero."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

import policy


class OfferedHistory:
    def __init__(self):
        self.items = defaultdict(list)
        self.meal_dates = defaultdict(set)
        self.max_date = None

    def append(self, actual):
        for r in actual.to_dict("records"):
            date = pd.Timestamp(r["date"])
            self.meal_dates[str(r["餐次"])].add(date)
            self.items[(str(r["餐次"]), str(r["_dish_norm"]))].append(
                (date, float(r["菜品总销量"]), str(r.get("菜品售卖单位", "unknown")).strip(),
                 str(r.get("_calendar_state", "")), str(r.get("_calendar_category", "")),
                 str(r.get("_temperature_band", "")), str(r.get("_rain_band", ""))))
            self.max_date = max(date, self.max_date) if self.max_date is not None else date

    def features(self, menu, target):
        if menu.empty:
            return pd.DataFrame()
        date = pd.Timestamp(menu.date.iloc[0])
        if self.max_date is not None and self.max_date >= date:
            raise AssertionError("Feature history includes target/future actuals")
        rows = []
        for r in menu.to_dict("records"):
            dish = policy.normalize_name(r["菜品名称"])
            meal = str(r["餐次"])
            hist = self.items[(meal, dish)]
            latest_unit = hist[-1][2] if hist else "unknown"
            compatible = [x for x in hist if x[2] == latest_unit]
            positive = [x for x in compatible if x[1] > 0]
            qty = np.array([x[1] for x in positive], dtype=float)
            ages = np.array([(date-x[0]).days for x in positive], dtype=float)
            base = float(r["pred_qty"])
            row = {"date": date, "食堂名称": r["食堂名称"], "餐次": meal, "菜品名称": r["菜品名称"],
                   "_dish_norm": dish, "baseline_qty": base, "sale_unit": latest_unit,
                   "sample_count": len(qty), "record_count": len(compatible),
                   "explicit_zero_count": sum(x[1] == 0 for x in compatible),
                   "incompatible_unit_records": len(hist)-len(compatible),
                   "offer_frequency": len({x[0] for x in positive})/max(len(self.meal_dates[meal]), 1),
                   "last_observation_age": float(ages[-1]) if len(ages) else None,
                   "history_max_date": self.max_date,
                   "calendar_state": target.get("_calendar_state", ""),
                   "calendar_category": target.get("_calendar_category", ""),
                   "temperature_band": target.get("_temperature_band", ""),
                   "rain_band": target.get("_rain_band", "")}
            if len(qty):
                row.update(mean=float(qty.mean()), logmean=float(np.expm1(np.log1p(qty).mean())),
                           median=float(np.median(qty)), minimum=float(qty.min()), maximum=float(qty.max()),
                           std=float(qty.std()), last=float(qty[-1]))
                for q in [.1,.25,.4,.6,.75,.9,.95,.99]:
                    row[f"q{int(q*100)}"] = float(np.quantile(qty,q))
                row["trimmed_mean"] = float(np.clip(qty, row["q10"], row["q90"]).mean())
            else:
                row.update({k:base for k in ["mean","logmean","median","minimum","maximum","last","trimmed_mean",
                                            "q10","q25","q40","q60","q75","q90","q95","q99"]})
                row["std"] = 0.
            for days in [7,14,30,60]:
                values = qty[ages <= days]
                row[f"n{days}"] = len(values)
                for statistic in ["mean", "median", "logmean"]:
                    row[f"w{days}_{statistic}"] = self.stat(values, statistic, row[statistic])
            for kind, mask in [
                ("weekday", np.array([x[0].weekday()==date.weekday() for x in positive], dtype=bool)),
                ("state", np.array([x[3]==target.get("_calendar_state", "") for x in positive], dtype=bool)),
                ("category", np.array([x[4]==target.get("_calendar_category", "") for x in positive], dtype=bool)),
                ("weather", np.array([x[5]==target.get("_temperature_band", "") and x[6]==target.get("_rain_band", "") for x in positive], dtype=bool)),
                ("state_recent", np.array([x[3]==target.get("_calendar_state", "") and (date-x[0]).days<=30 for x in positive], dtype=bool)),
            ]:
                values = qty[mask]
                row[kind+"_n"] = len(values)
                for statistic in ["mean", "median", "logmean"]:
                    row[kind+"_"+statistic] = self.stat(values,statistic,row[statistic])
            # Inverse-demand weighted median is the empirical MAPE minimizer;
            # positive tiny observations remain, never clipped from scoring.
            if len(qty):
                order = np.argsort(qty)
                weights = 1/qty[order]
                row["mape_median"] = float(qty[order][np.searchsorted(np.cumsum(weights), weights.sum()/2)])
            else:
                row["mape_median"] = base
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def stat(values, statistic, fallback):
        if not len(values):
            return float(fallback)
        if statistic == "mean":
            return float(values.mean())
        if statistic == "median":
            return float(np.median(values))
        return float(np.expm1(np.log1p(values).mean()))
