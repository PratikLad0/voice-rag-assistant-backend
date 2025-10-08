#!/usr/bin/env bash
set -euo pipefail

# Start the server in the background
ollama serve &

# Give it a moment to come up
sleep 3

# Pull the model (uses persistent disk so it only downloads once)
MODEL="${OLLAMA_MODEL:-llama3:8b}"
echo "Pulling model: $MODEL"
ollama pull "$MODEL" || true

# Keep the server process in the foreground (wait on any child)
wait -n
