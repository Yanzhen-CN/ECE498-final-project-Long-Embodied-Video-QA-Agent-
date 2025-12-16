from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def seconds_to_mmss(t: int) -> str:
    return f"{t//60:02d}:{t%60:02d}"


def build_image_prefix(n: int) -> str:
    # InternVL 多图输入习惯：Frame1: <image> ...
    return "".join([f"Frame{i+1}: <image>\n" for i in range(n)])


MEMORY_WRITER_PROMPT = """You are building episodic memory for a long video.
This chunk covers time range [{t0}, {t1}].

Previous chunk summary (for consistency):
{prev_summary}

Output ONLY valid JSON with keys:
chunk_id, t_start, t_end, summary, entities, events, state_update

Where:
- entities: list[str]
- events: list of objects like {{"verb":"...", "obj":"...", "detail":""}}
- state_update: object (can be empty)
Return ONLY JSON. No extra text.
"""


def make_memory_prompt(
    images: List[str],
    t_start: int,
    t_end: int,
    prev_summary: str,
) -> str:
    return (
        build_image_prefix(len(images))
        + MEMORY_WRITER_PROMPT.format(
            t0=seconds_to_mmss(t_start),
            t1=seconds_to_mmss(t_end),
            prev_summary=(prev_summary or "")
        )
    )


def build_jobs_from_manifest(data_json: Path, use_prev_summary: bool = True) -> List[Dict[str, Any]]:
    """
    Input: VideoManifest json (your data.json)
    Output: list of jobs:
      {
        "video_id": str,
        "chunk_id": int,
        "t_start": int,
        "t_end": int,
        "images": list[str],
        "prompt": str
      }
    NOTE: 这里先把 prev_summary 占位为空；真正 prev_summary 的更新在 model_to_memory 里完成。
    """
    manifest = load_json(data_json)
    video_id = manifest["video_id"]

    chunks = manifest.get("chunks", [])
    chunks = sorted(chunks, key=lambda x: int(x["chunk_id"]))

    jobs: List[Dict[str, Any]] = []
    prev_summary = ""  # placeholder

    for ch in chunks:
        chunk_id = int(ch["chunk_id"])
        t_start = int(ch["t_start"])
        t_end = int(ch["t_end"])
        images = list(ch.get("chunk_images", []))

        prompt = make_memory_prompt(
            images=images,
            t_start=t_start,
            t_end=t_end,
            prev_summary=(prev_summary if use_prev_summary else ""),
        )

        jobs.append({
            "video_id": video_id,
            "chunk_id": chunk_id,
            "t_start": t_start,
            "t_end": t_end,
            "images": images,
            "prompt": prompt,
        })

    return jobs
