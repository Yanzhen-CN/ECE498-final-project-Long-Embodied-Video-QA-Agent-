#!/usr/bin/env bash
set -e

python - <<'EOF'
from PIL import Image
img = Image.new("RGB",(448,448),(255,255,255))
img.save("/data/test.jpg")
print("saved /data/test.jpg")
EOF

curl -s http://127.0.0.1:8000/health
echo

curl -s -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths":["/data/test.jpg"],
    "prompt":"Return JSON with keys: summary, objects, actions.",
    "schema_hint":"{\"summary\":\"string\",\"objects\":[\"string\"],\"actions\":[\"string\"]}",
    "max_new_tokens":128
  }'
echo
