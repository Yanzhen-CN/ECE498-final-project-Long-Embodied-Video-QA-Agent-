import requests
from typing import List, Optional, Dict, Any

class ModelClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def generate(self, image_paths: List[str], prompt: str, schema_hint: Optional[str] = None,
                 max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        payload = {
            "image_paths": image_paths,
            "prompt": prompt,
            "schema_hint": schema_hint,
            "max_new_tokens": max_new_tokens,
        }
        r = requests.post(f"{self.base_url}/v1/generate", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
