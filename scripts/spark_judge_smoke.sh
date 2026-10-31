#!/usr/bin/env bash
# Smoke test of the Spark services from any machine that reaches the Spark (default localhost: run it on the Spark):
#     bash lerobot/scripts/spark_judge_smoke.sh [host] [ports...]      # default ports 8011 (gemma) 8013 (assigner)
# Per port: lists the served model, then one chat completion with a 2x2 red PNG asking for a JSON object (thinking off, as the
# clients send it). Robometer (:8012, optional) is checked only when 8012 is among the ports.
set -uo pipefail
HOST="${1:-localhost}"; shift || true
PORTS=("${@:-8011 8013}"); read -r -a PORTS <<< "${PORTS[*]}"
PNG="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAADklEQVR4nGP4z8DwHwYABP4D/VrW2TgAAAAASUVORK5CYII="
for PORT in "${PORTS[@]}"; do
  if [[ $PORT == 8012 ]]; then
    echo "== robometer :8012 health"; curl -s "http://$HOST:8012/health"; echo
    curl -s "http://$HOST:8012/score" -H 'Content-Type: application/json' -d "{\"task\": \"Put the bottle in the brown basket\", \"frames\": [\"$PNG\", \"$PNG\"]}"; echo
    continue
  fi
  MODEL=$(curl -s "http://$HOST:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])') || { echo "== :$PORT not answering"; continue; }
  echo "== :$PORT $MODEL"
  curl -s "http://$HOST:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @- <<EOF2 | python3 -c 'import json,sys; b=json.load(sys.stdin); print(b["choices"][0]["message"]["content"]); print(b.get("usage"))'
{"model": "$MODEL", "max_tokens": 60, "temperature": 0, "response_format": {"type": "json_object"},
 "chat_template_kwargs": {"enable_thinking": false},
 "messages": [{"role": "user", "content": [
   {"type": "text", "text": "What colour is this image? Answer with one JSON object {\"colour\": str, \"confidence\": float} and nothing else."},
   {"type": "image_url", "image_url": {"url": "$PNG"}}]}]}
EOF2
done
