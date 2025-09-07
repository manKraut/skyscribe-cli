# =========================
# SkyScribe-CLI Dockerfile
# =========================
# Features:
# - CPU by default (safe everywhere)
# - Optional GPU (CUDA) via TORCH_INDEX_URL build-arg
# - Includes transformers, accelerate, peft, datasets for LoRA training
# - Installs your package (editable-style install baked into image)
# - Non-root runtime; caches models to /models (mountable volume)

# ---- Base (slim) ----
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Hugging Face cache; mount this as a volume for persistence
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models/hf \
    # Keep tokenizers quiet
    TOKENIZERS_PARALLELISM=false

# System deps: CA certs, git (optional, for HF & debugging), build tools (light)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- App sources ----
# Copy only project files first to leverage Docker layer caching
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

# ---- Python deps ----
# Torch wheels differ by platform (CPU vs CUDA).
# Use build-arg to select the index. Defaults to CPU wheels.
# For CUDA 12.1 you can pass:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

# Upgrade pip first
RUN python -m pip install --upgrade pip setuptools wheel

# Install core app deps (your package)
RUN pip install --no-cache-dir -e .

# Install ML stack for LLM + LoRA training
RUN pip install --no-cache-dir \
    "torch" --index-url ${TORCH_INDEX_URL} && \
    pip install --no-cache-dir \
    "transformers>=4.44" \
    "accelerate>=0.33" \
    "peft>=0.12" \
    "datasets>=2.20" \
    "pandas>=2.2.2" \
    "numpy>=1.26" \
    "httpx>=0.27.0" \
    "typer>=0.12.3" \
    "pydantic>=2.8.2"

# NOTE: bitsandbytes is ONLY for NVIDIA GPUs with CUDA; do NOT install on CPU.
# If you need 4-bit training on CUDA, build from a CUDA-enabled base and add:
#   pip install --no-cache-dir "bitsandbytes>=0.43"
# …but only when using a CUDA base image + nvidia-container-runtime.

# ---- Non-root user ----
RUN useradd -m appuser && \
    mkdir -p /models && chown -R appuser:appuser /models /app
USER appuser

VOLUME ["/models"]        # HF/transformers cache and any downloaded weights
# You can also mount adapters dir from host when running:
#   -v $PWD/adapters:/adapters  (then set SKYSCRIBE_LORA_ADAPTER=/adapters/your-adapter)

# Optional: pre-pull a small base model to warm cache (commented to keep image smaller)
# RUN python - <<'PY'
# from transformers import AutoTokenizer, AutoModelForCausalLM
# m = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# AutoTokenizer.from_pretrained(m)
# AutoModelForCausalLM.from_pretrained(m)
# PY

# Default environment (you can override at runtime)
ENV SKYSCRIBE_BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Entrypoint runs the CLI
ENTRYPOINT ["skyscribe"]

