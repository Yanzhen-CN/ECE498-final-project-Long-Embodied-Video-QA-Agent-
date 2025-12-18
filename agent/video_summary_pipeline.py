from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.scripts.video_slicer import slice_video          # returns (manifest_path, manifest_json)
# from memory.memory_interface import memory_ingest                   # memory_ingest(record: dict) -> None
from model.model_interface import model_interface                  # ✅ 你们队友提供的：model_interface(image_paths, prompt)->str


# =========================
# Types
# =========================
@dataclass
class ChunkSpec:
    video_id: str
    chunk_id: int
    t_start: int
    t_end: int
    image_paths: List[str]


# =========================
# Helpers
# =========================
def seconds_to_mmss(t: int) -> str:
    return f"{t // 60:02d}:{t % 60:02d}"


def read_chunks_from_manifest(manifest: Dict[str, Any]) -> List[ChunkSpec]:
    video_id = str(manifest.get("video_id", "unknown"))
    chunks = sorted(manifest.get("chunks", []), key=lambda x: int(x.get("chunk_id", 0)))

    out: List[ChunkSpec] = []
    for ch in chunks:
        out.append(
            ChunkSpec(
                video_id=video_id,
                chunk_id=int(ch["chunk_id"]),
                t_start=int(ch["t_start"]),
                t_end=int(ch["t_end"]),
                image_paths=list(ch.get("chunk_images", [])),
            )
        )
    return out


def extract_outer_json(text: str) -> Optional[str]:
    s = text.strip()

    # strip fenced code blocks
    if s.startswith("```"):
        s = s.strip()
        # remove leading ```json or ```
        if s.startswith("```json"):
            s = s[len("```json"):].strip()
        elif s.startswith("```"):
            s = s[len("```"):].strip()
        # remove trailing ```
        if s.endswith("```"):
            s = s[: -len("```")].strip()

    if s.startswith("{") and s.endswith("}"):
        return s

    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        return s[l : r + 1]
    return None


# =========================
# Prompt builder
# =========================
def build_prompt(chunk: ChunkSpec, prev_summary: str) -> str:
    time_str = f"{seconds_to_mmss(chunk.t_start)}–{seconds_to_mmss(chunk.t_end)}"

    lines: List[str] = []
    lines.append(f"Chunk info: chunk_id={chunk.chunk_id}, time_range={time_str}.")
    lines.append(
        f"Previous chunk summary (for continuity): {prev_summary.strip() if prev_summary.strip() else '(none)'}"
    )
    lines.append("You are given multiple keyframes from this video chunk in chronological order.")
    lines.append("Task: produce a concise, faithful summary of what happens in this chunk. Do NOT guess.")
    lines.append(
        "Output MUST be valid JSON ONLY (no markdown, no extra text) with keys:\n"
        "  chunk_id (int), t_start (int), t_end (int),\n"
        "  summary (string),\n"
        "  entities (list of strings),\n"
        "  events (list of objects {\"verb\": str, \"obj\": str, \"detail\": str}),\n"
        "  state_update (object)."
    )
    lines.append("Keep summary <= 4 sentences. events <= 6. entities <= 10.")
    lines.append("Return ONLY JSON.")
    return "\n".join(lines)


# =========================
# Normalize record for memory
# =========================
def normalize_record(
    raw_text: str,
    chunk: ChunkSpec,
    *,
    evidence_per_chunk: int = 2,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    js = extract_outer_json(raw_text)

    if js is None:
        rec: Dict[str, Any] = {"parse_error": True, "raw_model_output": raw_text}
    else:
        try:
            rec = json.loads(js)
            rec["parse_error"] = False
        except Exception:
            rec = {"parse_error": True, "raw_model_output": raw_text}

    # enforce authoritative fields
    rec["video_id"] = chunk.video_id
    rec["chunk_id"] = chunk.chunk_id
    rec["t_start"] = chunk.t_start
    rec["t_end"] = chunk.t_end

    # defaults
    rec.setdefault("summary", "")
    rec.setdefault("entities", [])
    rec.setdefault("events", [])
    rec.setdefault("state_update", {})

    # stable evidence frames chosen by agent
    rec["evidence_frames"] = chunk.image_paths[: max(0, evidence_per_chunk)]

    # provenance
    if manifest_path is not None:
        rec["manifest_path"] = str(manifest_path)

    return rec


# =========================
# Public API for CLI
# =========================
def summarize_video_for_cli(
    video_path: str | Path,
    *,
    use_prev_summary: bool = True,
    evidence_per_chunk: int = 2,
) -> str:
    manifest_path, manifest_json = slice_video(str(video_path))
    manifest_path = Path(manifest_path)

    chunks = read_chunks_from_manifest(manifest_json)
    if not chunks:
        return f"[WARN] No chunks found in manifest. video='{Path(video_path).name}'"

    prev_summary = ""
    chunk_summaries: List[str] = []

    for chunk in chunks:
        # skip empty chunks (no frames)
        if not chunk.image_paths:
            continue

        prompt = build_prompt(chunk, prev_summary if use_prev_summary else "")
        raw_text = model_interface(image_paths=chunk.image_paths, prompt=prompt)

        record = normalize_record(
            raw_text,
            chunk,
            evidence_per_chunk=evidence_per_chunk,
            manifest_path=manifest_path,
        )

        memory_ingest(record)

        s = (record.get("summary") or "").strip()
        if s:
            time_str = f"{seconds_to_mmss(chunk.t_start)}–{seconds_to_mmss(chunk.t_end)}"
            chunk_summaries.append(f"[{time_str}] {s}")
            if use_prev_summary:
                prev_summary = s

    if not chunk_summaries:
        return f"[WARN] No summaries produced. video='{Path(video_path).name}'"

    header = f"Video: {Path(video_path).name} (video_id={chunks[0].video_id})"
    return header + "\n" + "\n".join(chunk_summaries)
