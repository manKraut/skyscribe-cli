from __future__ import annotations
from statistics import mean
from collections import defaultdict

def _window(hours, vals, start, end):
    # inclusive start, exclusive end indices
    out = []
    for t, v in zip(hours[start:end], vals[start:end]):
        out.append((t, v))
    return out

def _minmax(vals):
    return (min(vals), max(vals)) if vals else (None, None)

def _rain_window(hours, probs, threshold=40, idxs=None):
    # return the earliest contiguous block where precip prob crosses threshold
    if idxs is None:
        idxs = range(len(hours))
    blocks, start = [], None
    for i in idxs:
        p = probs[i] if i < len(probs) else None
        if p is not None and p >= threshold and start is None:
            start = i
        if (p is None or p < threshold) and start is not None:
            blocks.append((start, i))
            start = None
    if start is not None:
        blocks.append((start, idxs[-1] + 1))
    if not blocks:
        return None
    blocks.sort(key=lambda b: (b[1]-b[0]), reverse=True)
    s, e = blocks[0]
    return (hours[s], hours[e-1])

def _group_indices_by_date(hours):
    by_date = defaultdict(list)
    for i, t in enumerate(hours):
        date = t.split("T", 1)[0]  # 'YYYY-MM-DD'
        by_date[date].append(i)
    # preserve order as first-seen
    ordered = []
    seen = set()
    for t in hours:
        d = t.split("T", 1)[0]
        if d not in seen:
            ordered.append((d, by_date[d]))
            seen.add(d)
    return ordered  # [(date, [idx...]), ...]

def facts_for_day(fx: dict, day: str = "today", units: str = "metric") -> dict:
    """Returns structured facts for 'today' or 'tomorrow' based on local calendar day in fx['hourly']['time']."""
    h = fx["hourly"]
    hours = h["time"]
    grouped = _group_indices_by_date(hours)
    if not grouped:
        return {}

    if day.lower() == "tomorrow" and len(grouped) >= 2:
        _, idxs = grouped[1]
        target_date = grouped[1][0]
    else:
        _, idxs = grouped[0]
        target_date = grouped[0][0]

    def take(key):
        return [h.get(key, [None]*len(hours))[i] for i in idxs]

    temp = take("temperature_2m")
    ppop = take("precipitation_probability")
    wind = take("wind_speed_10m")
    gust = take("wind_gusts_10m")
    hum  = take("relative_humidity_2m")

    tmin, tmax = _minmax([v for v in temp if v is not None])
    wavg = round(mean([v for v in wind if v is not None]), 1) if any(v is not None for v in wind) else None
    gmax = max([v for v in gust if v is not None], default=None)
    rain_block = _rain_window(hours, h.get("precipitation_probability", []), threshold=40, idxs=idxs)

    # compute max precip prob for the day
    maxp = None
    if any(v is not None for v in ppop):
        maxp = max([v for v in ppop if v is not None], default=None)

    return {
        "date": target_date,
        "units": "metric" if units == "metric" else "imperial",
        "temp_min": None if tmin is None else round(tmin),
        "temp_max": None if tmax is None else round(tmax),
        "wind_avg": wavg,
        "gust_max": None if gmax is None else round(gmax),
        "humidity_avg": None if not hum else round(mean([v for v in hum if v is not None])),
        "rain_window": None if not rain_block else {
            "start_local": rain_block[0][-5:], "end_local": rain_block[1][-5:]
        },
        "rain_prob_max": maxp,
    }

def summarize_day(place: str, fx: dict, day: str = "today", units: str = "metric") -> list[str]:
    facts = facts_for_day(fx, day=day, units=units)
    if not facts:
        return ["No data available."]
    u_temp = "°C" if units=="metric" else "°F"
    u_wind = "km/h" if units=="metric" else "mph"

    lines = []
    if facts["temp_min"] is not None and facts["temp_max"] is not None:
        lines.append(f"{day.capitalize()} ranges {facts['temp_min']}–{facts['temp_max']} {u_temp}.")
    if facts["rain_window"]:
        rw = facts["rain_window"]
        p = facts["rain_prob_max"]
        lines.append(f"Showers most likely {rw['start_local']}–{rw['end_local']} (≈{p}% peak).")
    else:
        lines.append("Low rain chance most hours (<40%).")
    if facts["wind_avg"] is not None:
        if facts["gust_max"] and facts["gust_max"] >= (55 if units=="metric" else 35):
            lines.append(f"Breezy: avg wind ~{facts['wind_avg']} {u_wind}, gusts up to {facts['gust_max']} {u_wind}.")
        else:
            lines.append(f"Light winds on average (~{facts['wind_avg']} {u_wind}).")
    if facts["humidity_avg"] is not None:
        lines.append(f"Humidity around {facts['humidity_avg']}% on average.")
    lines.append("Tip: umbrella if out during the peak shower window." if facts["rain_window"] else "Tip: umbrella not needed for most plans.")
    return lines