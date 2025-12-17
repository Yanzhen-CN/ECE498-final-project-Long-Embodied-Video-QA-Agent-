# video_summary_pipeline.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Memory ingest: memory 同学提供的接口（内部怎么存/建索引由他封装）
from memory.interface import memory_ingest


# =========================
# 1) Data: read manifest
# =========================
def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def seconds_to_mmss(t: int) -> str:
    return f"{t//60:02d}:{t%60:02d}"


@dataclass
class ChunkSpec:
    video_id: str
    chunk_id: int
    t_start: int
    t_end: int
    image_paths: List[str]


def read_chunks(manifest_path: Path) -> List[ChunkSpec]:
    m = load_json(manifest_path)
    video_id = m["video_id"]
    chunks = sorted(m.get("chunks", []), key=lambda x: int(x["chunk_id"]))
    return [
        ChunkSpec(
            video_id=video_id,
            chunk_id=int(ch["chunk_id"]),
            t_start=int(ch["t_start"]),
            t_end=int(ch["t_end"]),
            image_paths=list(ch.get("chunk_images", [])),
        )
        for ch in chunks
    ]


# =========================
# 2) Prompt builder
# =========================
# NOTE: 这个 token 属于“模型后端协议”，不是 data.json 的一部分。
# 现在先用占位符，等你们接 LMDeploy 再替换成真正 IMAGE_TOKEN。
IMAGE_TOKEN = "<IMAGE_TOKEN>"


def build_prompt(chunk: ChunkSpec, prev_summary: str) -> str:
    lines = [f"Image-{i+1}: {IMAGE_TOKEN}" for i in range(len(chunk.image_paths))]
    lines.append(f"This chunk is from {seconds_to_mmss(chunk.t_start)} to {seconds_to_mmss(chunk.t_end)}.")
    lines.append(f"Previous chunk summary: {prev_summary}")
    lines.append("Return ONLY JSON with keys: chunk_id, t_start, t_end, summary, entities, events, state_update.")
    lines.append('Where events is a list of objects like {"verb":"...","obj":"...","detail":""}.')
    lines.append("Return ONLY JSON. No extra text.")
    return "\n".join(lines)

'''
prompt = (
  f"Image-1: {IMAGE_TOKEN}\n"
  f"Image-2: {IMAGE_TOKEN}\n"
  ...
  f"Image-8: {IMAGE_TOKEN}\n"
  "This chunk is from 00:30 to 01:00.\n"
  "Previous chunk summary: ...\n"
  "Return ONLY JSON with keys: chunk_id, t_start, t_end, summary, entities, events, state_update.\n"
)
response = pipe((prompt, images))  # images 是一个 list
'''


# =========================
# 3) Model I/O (stub now)
# =========================
def load_images_stub(image_paths: List[str]) -> List[Any]:
    """
    TODO: Replace with real image loading for your chosen backend.

    LMDeploy example:
      from lmdeploy.vl import load_image
      return [load_image(p) for p in image_paths]

    HF example:
      from PIL import Image
      return [Image.open(p).convert("RGB") for p in image_paths]
    """
    # stub: keep paths as placeholders
    return list(image_paths)


def model_infer_stub(prompt: str, images: List[Any]) -> str:
    """
    TODO: Replace with real InternVL inference.

    LMDeploy example:
      resp = pipe((prompt, images))
      return resp.text
    """
    dummy = {
        "chunk_id": -1,
        "t_start": -1,
        "t_end": -1,
        "summary": "STUB: replace model_infer_stub() with real InternVL inference.",
        "entities": [],
        "events": [],
        "state_update": {}
    }
    return json.dumps(dummy, ensure_ascii=False)


# =========================
# 4) Parse & normalize to ChunkMemory
# =========================
def extract_outer_json(text: str) -> Optional[str]:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        return s[l:r+1]
    return None


def normalize_record(raw_text: str, chunk: ChunkSpec, evidence_per_chunk: int = 2) -> Dict[str, Any]:
    js = extract_outer_json(raw_text)
    if js is None:
        rec: Dict[str, Any] = {"parse_error": True, "raw_model_output": raw_text}
    else:
        try:
            rec = json.loads(js)
            rec["parse_error"] = False
        except Exception:
            rec = {"parse_error": True, "raw_model_output": raw_text}

    # enforce truth from data.json
    rec["video_id"] = chunk.video_id
    rec["chunk_id"] = chunk.chunk_id
    rec["t_start"] = chunk.t_start
    rec["t_end"] = chunk.t_end

    # required keys default
    rec.setdefault("summary", "")
    rec.setdefault("entities", [])
    rec.setdefault("events", [])
    rec.setdefault("state_update", {})

    # evidence chosen by agent (stable)
    rec["evidence_frames"] = chunk.image_paths[:evidence_per_chunk]
    return rec


# =========================
# 5) Pipeline (ingest)
# =========================
def run_video_summary_pipeline(
    manifest_path: Path,
    *,
    use_prev_summary: bool = True,
    evidence_per_chunk: int = 2,
) -> None:
    chunks = read_chunks(manifest_path)
    prev_summary = ""

    for chunk in chunks:
        prompt = build_prompt(chunk, prev_summary=(prev_summary if use_prev_summary else ""))

        images = load_images_stub(chunk.image_paths)
        raw_text = model_infer_stub(prompt=prompt, images=images)

        record = normalize_record(raw_text, chunk, evidence_per_chunk=evidence_per_chunk)

        # Hand off to memory module (store/index inside memory module)
        memory_ingest(record)

        # Update for next chunk
        if use_prev_summary and record.get("summary", "").strip():
            prev_summary = record["summary"].strip()

    print("[OK] video_summary_pipeline finished (records ingested via memory_ingest).")


if __name__ == "__main__":
    run_video_summary_pipeline(
        manifest_path=Path("agent/data.json"),
        use_prev_summary=True,
        evidence_per_chunk=2,
    )
