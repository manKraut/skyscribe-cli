from __future__ import annotations
from statistics import mean

def _window(hours, vals, start, end):
    # inclusive start, exclusive end indices
    out = []
    for t, v in zip(hours[start:end], vals[start:end]):
        out.append((t, v))
    return out

def _minmax(vals):
    return (min(vals), max(vals)) if vals else (None, None)

def _rain_window(hours, probs, threshold=40):
    # return the earliest contiguous block where precip prob crosses threshold
    blocks = []
    start = None
    for i, p in enumerate(probs):
        if p is not None and p >= threshold and start is None:
            start = i
        if (p is None or p < threshold) and start is not None:
            blocks.append((start, i))
            start = None
    if start is not None:
        blocks.append((start, len(probs)))
    # pick the longest block
    if not blocks: return None
    blocks.sort(key=lambda b: (b[1]-b[0]), reverse=True)
    s, e = blocks[0]
    return (hours[s], hours[e-1])

def summarize_today(place: str, fx: dict, units: str="metric") -> list[str]:
    h = fx["hourly"]
    hours = h["time"]
    temp = h.get("temperature_2m", [])
    ppop = h.get("precipitation_probability", [])
    wind = h.get("wind_speed_10m", [])
    gust = h.get("wind_gusts_10m", [])
    hum  = h.get("relative_humidity_2m", [])

    tmin, tmax = _minmax(temp)
    wavg = round(mean([v for v in wind if v is not None]), 1) if wind else None
    gmax = max([v for v in gust if v is not None]) if gust else None
    rain_block = _rain_window(hours, ppop, threshold=40)

    u_temp = "°C" if units=="metric" else "°F"
    u_wind = "km/h" if units=="metric" else "mph"

    lines = []
    if tmin is not None and tmax is not None:
        lines.append(f"Temperatures between {round(tmin)}–{round(tmax)} {u_temp}.")
    if rain_block:
        start_local = rain_block[0][-5:]  # HH:MM from ISO
        end_local   = rain_block[1][-5:]
        # approximate max prob in that window
        s_idx = hours.index(rain_block[0]); e_idx = hours.index(rain_block[1])+1
        maxp = max([p for p in ppop[s_idx:e_idx] if p is not None], default=None)
        lines.append(f"Highest shower risk {start_local}–{end_local} (≈{maxp}% probability).")
    else:
        lines.append("Low chance of rain (most hours < 40%).")
    if wavg is not None:
        if gmax and gmax >= (55 if units=="metric" else 35):
            lines.append(f"Breezy: avg wind ~{wavg} {u_wind}, gusts up to {round(gmax)} {u_wind}.")
        else:
            lines.append(f"Light winds on average (~{wavg} {u_wind}).")
    if hum:
        hmean = round(mean([v for v in hum if v is not None]))
        lines.append(f"Humidity around {hmean}% on average.")
    lines.append("Tip: carry an umbrella if out during the peak shower window." if rain_block else "Tip: no umbrella needed for most plans.")
    return lines
