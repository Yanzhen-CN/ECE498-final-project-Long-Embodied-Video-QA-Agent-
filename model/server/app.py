# model/server/app.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import time

from .runner_internvl import get_runner

app = FastAPI(title="InternVL Model Server", version="0.2")

class GenerateRequest(BaseModel):
    image_paths: List[str] = Field(..., description="List of local image paths on the server")
    prompt: str = Field(..., description="User prompt")
    max_new_tokens: Optional[int] = None

class GenerateResponse(BaseModel):
    text: str
    meta: Dict[str, Any]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    runner = get_runner()
    t0 = time.time()
    text = runner.generate(req.image_paths, req.prompt, req.max_new_tokens)
    dt = int((time.time() - t0) * 1000)

    return GenerateResponse(
        text=text,
        meta={
            "model_id": runner.model_id if hasattr(runner, "model_id") else None,
            "num_images": len(req.image_paths),
            "latency_ms": dt,
        },
    )

