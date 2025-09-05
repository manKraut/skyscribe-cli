from __future__ import annotations
import typer, sys
from datetime import datetime, timezone
from skyscribe.agents.ingestion import resolve_location, fetch_forecast_today
from skyscribe.agents.features import summarize_today
from skyscribe.utils.printing import hr, green, cyan

app = typer.Typer(add_completion=False, help="SkyScribe — CLI weather briefs from open data")

@app.command()
def forecast(place: str = typer.Argument(..., help="City or 'lat,lon'"),
             units: str = typer.Option("metric", "--units", case_sensitive=False, help="metric|imperial")):
    """Print a concise brief for today."""
    loc = resolve_location(place)
    if loc is None:
        typer.echo("Could not resolve location. Try a more specific place name.", err=True)
        raise typer.Exit(1)

    data = fetch_forecast_today(loc["lat"], loc["lon"], units=units)
    brief = summarize_today(loc["name"], data, units=units)

    now = datetime.now(timezone.utc).astimezone()
    typer.echo(hr())
    typer.echo(f"{cyan(loc['name'])} — Today ({now.strftime('%Y-%m-%d')}, local)")
    typer.echo(hr())
    for line in brief:
        typer.echo("• " + line)
    typer.echo(hr())

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
