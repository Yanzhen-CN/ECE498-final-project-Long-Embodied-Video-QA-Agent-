from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from .prompts import build_prompt
from .runner_internvl import get_runner
from .utils_json import ensure_json

app = FastAPI(title="InternVL Model Server", version="0.1")

class GenerateRequest(BaseModel):
    image_paths: List[str] = Field(..., description="List of local image paths on the server")
    prompt: str = Field(..., description="User prompt (will be wrapped to enforce JSON output)")
    schema_hint: Optional[str] = Field(None, description="Optional schema hint to help model output JSON")
    max_new_tokens: Optional[int] = None

class GenerateResponse(BaseModel):
    text: str
    json: Optional[Dict[str, Any]] = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    runner = get_runner()
    final_prompt = build_prompt(req.prompt, req.schema_hint)
    text = runner.generate(req.image_paths, final_prompt, req.max_new_tokens)

    parsed = None
    try:
        parsed = ensure_json(text)
    except Exception:
        parsed = None

    return GenerateResponse(text=text, json=parsed)
