from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


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


def build_prompt(
    *,
    chunk: ChunkSpec,
    image_token: str,
    prev_summary: str,
) -> str:
    lines = [f"Image-{i+1}: {image_token}" for i in range(len(chunk.image_paths))]
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