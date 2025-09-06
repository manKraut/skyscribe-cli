from __future__ import annotations
import typer, sys
from datetime import datetime, timezone
from skyscribe.agents.ingestion import resolve_location, fetch_forecast
from skyscribe.agents.features import summarize_day
from skyscribe.utils.printing import hr, green, cyan

app = typer.Typer(add_completion=False, help="SkyScribe — CLI weather briefs from open data")

@app.command()
def forecast(place: str = typer.Argument(..., help="City or 'lat,lon'"),
             when: str = typer.Option("today", "--when", "-w", case_sensitive=False, help="today|tomorrow or a number of days (1–7)"),
             units: str = typer.Option("metric", "--units", case_sensitive=False, help="metric|imperial")):
    """Print a concise brief for today."""
    loc = resolve_location(place)
    if loc is None:
        typer.echo("Could not resolve location. Try a more specific place name.", err=True)
        raise typer.Exit(1)

    # Normalize --when input
    when_value = str(when).lower().strip()

    if when_value == "today":
        days = 1
        label = "today"
    elif when_value == "tomorrow":
        days = 2
        label = "tomorrow"
    elif when_value.isdigit():
        days = int(when_value)
        if days < 1 or days > 7:
            typer.echo("Error: please provide a number between 1 and 7 for --when", err=True)
            raise typer.Exit(2)
        label = f"next {days} days"
    else:
        typer.echo("Error: --when must be 'today', 'tomorrow', or a number (1–7).", err=True)
        raise typer.Exit(2)
    
    # Fetch forecast data
    fx = fetch_forecast(loc["lat"], loc["lon"], units=units, days=days)

    if when_value in ("today", "tomorrow"):
        lines = summarize_day(loc["name"], fx, day=label, units=units)
        typer.echo(hr())
        typer.echo(f"{cyan(loc['name'])} — {label.capitalize()}")
        typer.echo(hr())
        for line in lines:
            typer.echo("• " + line)
            typer.echo(hr())
    else:
        from skyscribe.agents.features import _group_indices_by_date
        dates = _group_indices_by_date(fx["hourly"]["time"])[:days]
        for date, _ in dates:
            lines = summarize_day(loc["name"], fx, day=date, units=units)
            typer.echo(hr())
            typer.echo(f"{cyan(loc['name'])} — {date}")
            typer.echo(hr())
            for line in lines:
                typer.echo("• " + line)

@app.command()
def raw(place: str):
    """Dump the first few hourly rows (debugging)."""
    import pandas as pd
    loc = resolve_location(place)
    if not loc:
        typer.echo("Could not resolve location", err=True); raise typer.Exit(1)
    fx = fetch_forecast_today(loc["lat"], loc["lon"])
    df = pd.DataFrame(fx["hourly"])
    typer.echo(df.head(12).to_string(index=False))

if __name__ == "__main__":
    app()
