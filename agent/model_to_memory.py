from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.data_to_model import ChunkSpec


def extract_outer_json(text: str) -> Optional[str]:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        return s[l:r+1]
    return None


# -------------------------
# TODO 1: Model inference (stub now)
# -------------------------
def model_infer(prompt: str, image_paths: List[str]) -> str:
    """
    Replace later with InternVL inference.
    Input:
      - prompt: contains IMAGE_TOKEN placeholders + prev_summary
      - image_paths: list of image file paths for this chunk
    Output:
      - raw model text (ideally JSON string)
    """
    dummy = {
        "chunk_id": -1,
        "t_start": -1,
        "t_end": -1,
        "summary": "STUB: replace model_infer() with real model call.",
        "entities": [],
        "events": [],
        "state_update": {}
    }
    return json.dumps(dummy, ensure_ascii=False)


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

    # enforce ground truth from data.json
    rec["video_id"] = chunk.video_id
    rec["chunk_id"] = chunk.chunk_id
    rec["t_start"] = chunk.t_start
    rec["t_end"] = chunk.t_end

    # required fields default
    rec.setdefault("summary", "")
    rec.setdefault("entities", [])
    rec.setdefault("events", [])
    rec.setdefault("state_update", {})

    # evidence chosen by agent (stable)
    rec["evidence_frames"] = chunk.image_paths[:evidence_per_chunk]
    return rec


# -------------------------
# TODO 2: Memory ingest (stub now)
# -------------------------
def memory_ingest(record: Dict[str, Any], out_jsonl: Path) -> None:
    """
    This represents "hand to memory module to store".
    For now, the fastest integration is: append to a jsonl file (ingest log).
    Later, memory teammate can replace this body with:
      - write into their memory bank
      - build index
      - etc.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
