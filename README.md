# SkyScribe-CLI 🌤️  

**SkyScribe** is a CLI-based multi-agent weather app that fetches live forecast data from open APIs, summarizes it into human-friendly briefs, and can optionally generate narrative forecasts using a fine-tuned LoRA adapter on a small LLM.  

---

## ✨ Features
- Fetches weather data from [Open-Meteo](https://open-meteo.com/) (no API key required).  
- Multiple modes:  
  - `forecast` → Today, tomorrow, or multi-day summaries.  
  - `raw` → Inspect the raw hourly data in tabular form.   
- **Rule-based summarizer** (lightweight, works out of the box).  
- **LLM summarizer** (optional):  
  - Uses TinyLlama (or another open model).  
  - Supports LoRA adapters for domain-specific tuning (e.g., weather phrasing).  

---

### Local (Python venv)
```bash
git clone https://github.com/manKraut/skyscribe-cli.git
cd skyscribe-cli

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .

# Forecast for today
skyscribe forecast "Paris"

# Tomorrow's forecast
skyscribe forecast "Berlin" --when tomorrow

# Multi-day forecast (next 3 days)
skyscribe forecast "Athens" --when 3

# Natural-language Q&A
skyscribe ask "Will it rain before 18:00 in Madrid?"

# Inspect raw data
skyscribe raw "London"
```

🧠 Optional: LLM + LoRA Adapter
```bash
# Install ML dependencies:

pip install "transformers>=4.44" "accelerate>=0.33" "peft>=0.12" "datasets>=2.20" torch
Train a tiny LoRA adapter:

skyscribe train --cities "Paris,Athens,London,Oslo" --days 2 --out adapters/tinyllama-lora --epochs 1
Run with your adapter:

export SKYSCRIBE_LORA_ADAPTER="adapters/tinyllama-lora"
skyscribe forecast "Berlin" --when today --llm
