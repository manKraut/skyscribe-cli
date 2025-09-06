from __future__ import annotations
import httpx, re

UA = "skyscribe-cli/0.0.1 (github demo)"

def _parse_latlon(s: str):
    """Parse 'lat,lon' string into (lat, lon) tuple or None."""
    m = re.match(r"^\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*$", s)
    if m: return float(m.group(1)), float(m.group(3))
    return None

def resolve_location(place: str) -> dict | None:
    latlon = _parse_latlon(place)
    if latlon:
        return {"name": f"{latlon[0]:.4f},{latlon[1]:.4f}", "lat": latlon[0], "lon": latlon[1]}
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": place, "count": 1, "language": "en", "format": "json"}
    with httpx.Client(timeout=10, headers={"User-Agent": UA}) as c:
        r = c.get(url, params=params); r.raise_for_status()
        j = r.json()
    if not j.get("results"): return None
    # Take the first result
    hit = j["results"][0]
    name_parts = [hit.get("name"), hit.get("admin1"), hit.get("country")]
    name = ", ".join([p for p in name_parts if p])
    return {"name": name, "lat": hit["latitude"], "lon": hit["longitude"]}

def fetch_forecast(lat: float, lon: float, units: str = "metric", days: int = 1) -> dict:
    """Fetch hourly forecast data for the next `days` days (max 7) from Open-Meteo. Enforces passing args "units" and "days" with keyword."""
    temp_unit = "celsius" if units == "metric" else "fahrenheit"
    wind_unit = "kmh" if units == "metric" else "mph"
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "hourly": "temperature_2m,precipitation_probability,precipitation,wind_speed_10m,wind_gusts_10m,relative_humidity_2m",
        "temperature_unit": temp_unit, "wind_speed_unit": wind_unit, "forecast_days": min(days, 7) # max 7 days
    }
    with httpx.Client(timeout=15, headers={"User-Agent": UA}) as c:
        r = c.get(url, params=params); r.raise_for_status()
        return r.json()
