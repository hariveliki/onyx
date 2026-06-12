#!/usr/bin/env bash
# Run retrieval evaluation inside the Onyx API container.
# Usage: ONYX_API_KEY=<key> ./run_retrieval_eval.sh [testset.jsonl]
set -euo pipefail

CONTAINER="${CONTAINER:-docker_compose_api_server_1}"
ONYX_API_URL="${ONYX_API_URL:-http://localhost:8080}"
ONYX_API_KEY="${ONYX_API_KEY:?ONYX_API_KEY must be set}"
TESTSET="${1:-testset.jsonl}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

podman cp "$SCRIPT_DIR/retrieval_eval.py" "$CONTAINER:/app/retrieval_eval.py"
podman cp "$TESTSET" "$CONTAINER:/app/testset.jsonl"

podman exec -i "$CONTAINER" \
  env ONYX_API_URL="$ONYX_API_URL" ONYX_API_KEY="$ONYX_API_KEY" \
  python3 /app/retrieval_eval.py --testset /app/testset.jsonl --output-dir /app/eval_out

podman cp "$CONTAINER:/app/eval_out" ./eval_out
echo "Results written to ./eval_out/"
