"""
Unified model interface for pipeline.

This file provides a single entry point:
    model_interface(image_paths, prompt)

- If image_paths is empty or None:
    -> treat as LLM usage (text-only)
- If image_paths is provided:
    -> treat as multimodal usage (InternVL)

Pipeline should ONLY import this file.
"""

from typing import List, Optional, Union, Dict, Any
import requests


# ===== configuration =====
MODEL_SERVER_URL = "http://localhost:8000"  # 改成服务器地址即可
GENERATE_ENDPOINT = f"{MODEL_SERVER_URL}/v1/generate"


def model_interface(
    image_paths: Optional[List[str]],
    prompt: str,
    schema_hint: Optional[str] = None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    Unified model interface.

    Args:
        image_paths:
            - None or []  -> text-only (LLM-style)
            - List[str]   -> multimodal (VLM-style)
        prompt:
            Task instruction.
        schema_hint:
            Optional JSON schema hint.
        max_new_tokens:
            Generation length control.

    Returns:
        Dict with keys:
            - text: raw model output
            - json: parsed JSON if available, else None
    """

    # normalize image_paths
    if not image_paths:
        image_paths = []

    payload = {
        "image_paths": image_paths,
        "prompt": prompt,
        "schema_hint": schema_hint,
        "max_new_tokens": max_new_tokens,
    }

    try:
        resp = requests.post(GENERATE_ENDPOINT, json=payload, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Model server request failed: {e}")

    return resp.json()
