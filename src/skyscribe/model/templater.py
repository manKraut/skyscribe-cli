from __future__ import annotations
import random

ADVICE = [
    "Carry an umbrella if you’ll be out during the shower window.",
    "Light layer recommended in the morning.",
    "Sunscreen advised when skies brighten.",
    "Wind-sensitive activities should be cautious during stronger gusts.",
]

def render_brief(facts: dict) -> str:
    d = facts
    u_temp = "°C" if d.get("units","metric") == "metric" else "°F"
    u_wind = "km/h" if d.get("units","metric") == "metric" else "mph"

    lines = []
    if d.get("temp_min") is not None and d.get("temp_max") is not None:
        lines.append(f"Temperatures range {d['temp_min']}–{d['temp_max']} {u_temp}.")
    if d.get("rain_window"):
        rw = d["rain_window"]
        p  = d.get("rain_prob_max")
        lines.append(f"Highest rain risk {rw['start_local']}–{rw['end_local']} (≈{p}%).")
    else:
        lines.append("Rain is unlikely for most hours.")
    if d.get("wind_avg") is not None:
        g = d.get("gust_max")
        if g and g >= (55 if d.get("units")=="metric" else 35):
            lines.append(f"Breezy at times: average wind ~{d['wind_avg']} {u_wind}, gusts up to {g} {u_wind}.")
        else:
            lines.append(f"Light winds on average (~{d['wind_avg']} {u_wind}).")
    if d.get("humidity_avg") is not None:
        lines.append(f"Humidity averages around {d['humidity_avg']}%.")

    lines.append(random.choice(ADVICE))
    return " ".join(lines)