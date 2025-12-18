#!/usr/bin/env bash
set -e

# HuggingFace cache (persist on /data)
export HF_HOME=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

# Model config
export INTERNVL_MODEL_ID="${INTERNVL_MODEL_ID:-OpenGVLab/InternVL3_5-4B}"
export INTERNVL_DTYPE="${INTERNVL_DTYPE:-float16}"
export MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-8000}"

uvicorn model.server.app:app --host 0.0.0.0 --port "${MODEL_SERVER_PORT}"
