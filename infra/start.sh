#!/usr/bin/env bash
set -euo pipefail

ollama serve &          # start API
sleep 3                 # let it boot

MODEL="${OLLAMA_MODEL:-llama3:8b}"
echo "Pulling model: $MODEL"
ollama pull "$MODEL" || true

wait -n                 # keep container in foreground
