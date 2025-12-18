from model.client.client import ModelClient
import json

SERVER = "http://<server-ip>:8000"
client = ModelClient(SERVER)

schema = json.dumps({
  "chunk_id": "string",
  "summary": "string",
  "actions": ["string"],
  "objects": ["string"]
}, ensure_ascii=False)

resp = client.generate(
  image_paths=["/data/test.jpg"],
  prompt="Summarize this chunk as JSON.",
  schema_hint=schema,
  max_new_tokens=256
)

print(resp["json"] or resp["text"])
