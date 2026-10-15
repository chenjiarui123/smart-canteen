"""Presentation-only weather option normalization.

The source CSV stays untouched.  Temperature, precipitation and wind are
rounded to integer dropdown values using conventional half-up rounding.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


INTEGER_WEATHER_KEYS = {
    "temperature_c",
    "precipitation_mm",
    "wind_speed_kmh",
}


def round_weather_integer(value: Any) -> int:
    return int(
        Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )


def normalized_weather_options(weather: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(weather)
    for key, item in result.items():
        values = item.get("values", [])
        if key in INTEGER_WEATHER_KEYS:
            item["values"] = sorted(
                {round_weather_integer(value) for value in values}
            )
        elif key == "humidity_pct":
            item["values"] = sorted({int(round(float(value))) for value in values})
    return result
