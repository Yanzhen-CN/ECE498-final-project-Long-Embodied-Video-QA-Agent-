from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# 由 memory提供：负责存储（内部怎么落盘/建索引由他决定）
from memory.interface import memory_ingest

# 由 model提供：负责多模态推理（内部怎么 load 图、用什么 token、用什么引擎都封装）
# from model.interface import model_interface  # model_interface(image_paths: list[str], prompt: str) -> str


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
def build_prompt(chunk: ChunkSpec, prev_summary: str) -> str:
    """
    注意：这里不再放 IMAGE_TOKEN 了，因为你们的 model_interface 已经封装了
    “多图怎么喂模型”的实现。prompt 只写任务与约束。
    """
    lines = []
    lines.append(f"Time range: {seconds_to_mmss(chunk.t_start)} to {seconds_to_mmss(chunk.t_end)}.")
    lines.append(f"Previous chunk summary: {prev_summary}")
    lines.append(
        "Task: summarize this chunk using the provided images in chronological order."
    )
    lines.append(
        "Return ONLY valid JSON with keys: chunk_id, t_start, t_end, summary, entities, events, state_update."
    )
    lines.append(
        'Where events is a list of objects like {"verb":"...","obj":"...","detail":""}.'
    )
    lines.append("Return ONLY JSON. No extra text.")
    return "\n".join(lines)


# =========================
# 3) Parse & normalize to ChunkMemory
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

    # enforce truth from data.json (agent-controlled)
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
# 4) Pipeline (ingest)
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

        # Model teammate interface: (image_paths, prompt) -> raw_text
        raw_text = model_interface(image_paths=chunk.image_paths, prompt=prompt)

        record = normalize_record(raw_text, chunk, evidence_per_chunk=evidence_per_chunk)

        # hand off to memory module
        memory_ingest(record)

        # update prev_summary for next chunk
        if use_prev_summary and record.get("summary", "").strip():
            prev_summary = record["summary"].strip()

    print("[OK] video_summary_pipeline finished (records ingested via memory_ingest).")


if __name__ == "__main__":
    run_video_summary_pipeline(
        manifest_path=Path("agent/data.json"),
        use_prev_summary=True,
        evidence_per_chunk=2,
    )
