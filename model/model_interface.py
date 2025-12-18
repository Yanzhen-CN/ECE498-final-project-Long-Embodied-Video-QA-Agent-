# model/interface.py
"""
Unified model interface for pipeline.

Pipeline imports:
    from model.interface import model_interface

This interface does NOT enforce output format.
Pipeline should enforce format via prompt (e.g., ask for JSON).
"""

from typing import List, Optional, Dict, Any
import os
import requests

# 允许用环境变量覆盖（服务器部署/本地调试都方便）
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8000").rstrip("/")
GENERATE_ENDPOINT = f"{MODEL_SERVER_URL}/v1/generate"


def model_interface(
    image_paths: Optional[List[str]],
    prompt: str,
    max_new_tokens: int = 512,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Args:
        image_paths: None or [] means text-only; non-empty list means multimodal.
        prompt: prompt text (pipeline controls output format).
        max_new_tokens: generation limit.
        timeout: http timeout seconds.

    Returns:
        {
          "text": "<raw model output>",
          "meta": {...}
        }
    """
    if image_paths is None:
        image_paths = []

    payload = {
        "image_paths": image_paths,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
    }

    try:
        resp = requests.post(GENERATE_ENDPOINT, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Model server request failed: {e}")

    data = resp.json()

    # 兼容：如果 server 还没改成只返回 text/meta，也能跑
    if "text" not in data:
        # 例如旧版本返回 {"text":..., "json":...}
        # 或者仅返回 string（很少见）
        if isinstance(data, str):
            return {"text": data, "meta": {"model_server_url": MODEL_SERVER_URL, "num_images": len(image_paths)}}
        raise RuntimeError(f"Unexpected response from model server: {data}")

    return data
