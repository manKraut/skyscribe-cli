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
             units: str = typer.Option("metric", "--units", case_sensitive=False, help="metric|imperial"),
             llm: bool = typer.Option(False, "--llm", help="Use the LLM (with optional LoRA) instead of rule-based summary")):
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

    if llm:
        from skyscribe.agents.writer import write_brief_from_facts
        from skyscribe.agents.features import facts_for_day
        facts = facts_for_day(fx, day=when_value, units=units)
        text = write_brief_from_facts({"location": loc["name"], **facts})
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    else:
        from skyscribe.agents.features import summarize_day
        lines = summarize_day(loc["name"], fx, day=when_value, units=units)

    typer.echo("─" * 50)
    typer.echo(f"{loc['name']} — {when_value.capitalize()}")
    typer.echo("─" * 50)
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

@app.command()
def train(
    cities: str = typer.Option("Paris,Athens,London,Oslo,Berlin", "--cities", help="Comma-separated city names"),
    days: int = typer.Option(2, "--days", min=1, max=7, help="Forecast days to collect per city"),
    out: str = typer.Option("adapters/tinyllama-lora", "--out", help="Where to save the LoRA adapter"),
    base_model: str = typer.Option("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "--base-model", help="HF model id"),
    quant4bit: bool = typer.Option(False, "--quant4bit", help="Use 4-bit during training (GPU + bitsandbytes)"),
    epochs: float = typer.Option(1.0, "--epochs"),
    lr: float = typer.Option(2e-4, "--lr"),
):
    """Generate a tiny dataset and fine-tune a LoRA adapter."""
    from skyscribe.model.train import run_training
    city_list = [c.strip() for c in cities.split(",") if c.strip()]
    run_training(
        cities=city_list, out_dir=out, base_model=base_model,
        days=days, quantize_4bit=quant4bit, epochs=epochs, lr=lr
    )
