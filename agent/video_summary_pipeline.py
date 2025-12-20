# agent/video_summary_pipeline.py
from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from data.video_interface import slice_video
from memory.memory_interface import memory_ingest
from model.model_interface import init_model, model_interface, InferConfig

from agent.analysis_store import has_summary, load_summary, save_summary, append_run


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


@dataclass(frozen=True)
class ModeConfig:
    name: str
    chunk_seconds: int
    keyframes_per_chunk: int  # fixed to 6 in all modes (your requirement)
    infer_cfg: InferConfig
    two_pass: bool


# =========================
# Modes (keyframes_per_chunk fixed = 6; mode mainly changes chunk_seconds)
# =========================
KEYFRAMES_PER_CHUNK = 6

MODES: Dict[str, ModeConfig] = {
    "fast": ModeConfig(
        name="Fast",
        chunk_seconds=60,
        keyframes_per_chunk=KEYFRAMES_PER_CHUNK,
        infer_cfg=InferConfig(max_new_tokens=128, max_num=2, use_thumbnail=False),
        two_pass=False,
    ),
    "standard": ModeConfig(
        name="Standard",
        chunk_seconds=30,
        keyframes_per_chunk=KEYFRAMES_PER_CHUNK,
        infer_cfg=InferConfig(max_new_tokens=192, max_num=2, use_thumbnail=False),
        two_pass=True,  # 3+3
    ),
    "detailed": ModeConfig(
        name="Detailed",
        chunk_seconds=15,
        keyframes_per_chunk=KEYFRAMES_PER_CHUNK,
        infer_cfg=InferConfig(max_new_tokens=192, max_num=2, use_thumbnail=False),
        two_pass=True,  # 3+3
    ),
}


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
    if s.startswith("```"):
        if s.startswith("```json"):
            s = s[len("```json") :].strip()
        else:
            s = s[len("```") :].strip()
        if s.endswith("```"):
            s = s[: -len("```")].strip()

    if s.startswith("{") and s.endswith("}"):
        return s

    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        return s[l : r + 1]
    return None


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

    # authoritative fields
    rec["video_id"] = chunk.video_id
    rec["chunk_id"] = chunk.chunk_id
    rec["t_start"] = chunk.t_start
    rec["t_end"] = chunk.t_end

    # defaults
    rec.setdefault("summary", "")
    rec.setdefault("entities", [])
    rec.setdefault("events", [])
    rec.setdefault("state_update", {})

    # evidence frames
    rec["evidence_frames"] = chunk.image_paths[: max(0, evidence_per_chunk)]

    if manifest_path is not None:
        rec["manifest_path"] = str(manifest_path)

    return rec


def _two_pass_infer(frames: List[str], base_prompt: str, infer_cfg: InferConfig) -> str:
    """
    Reduce peak VRAM: split frames into 3+3 (since you fixed 6 per chunk).
    Pass1: first 3 frames -> plain text
    Pass2: last 3 frames + partial -> FINAL JSON
    """
    if len(frames) <= 3:
        return model_interface(image_paths=frames, prompt=base_prompt, cfg=infer_cfg)

    first = frames[:3]
    second = frames[3:]

    prompt1 = base_prompt + "\n\nYou only see the FIRST part of frames. Summarize briefly in plain text (no JSON)."
    partial = model_interface(image_paths=first, prompt=prompt1, cfg=infer_cfg)

    prompt2 = (
        base_prompt
        + "\n\nYou only see the SECOND part of frames.\n"
        + "First-part summary (may be imperfect):\n"
        + partial.strip()
        + "\n\nNow output FINAL JSON for the whole chunk."
    )
    return model_interface(image_paths=second, prompt=prompt2, cfg=infer_cfg)


def _post_infer_cleanup() -> None:
    # Helps with fragmentation / reserved memory staying high
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================
# Public API
# =========================
def summarize_video_for_cli(
    video_path: str | Path,
    *,
    mode: str = "standard",
    project_root: str | Path = ".",
    local_files_only: bool = False,
    evidence_per_chunk: int = 2,
    use_prev_summary: bool = True,
    overwrite_slice: bool = True,
    reuse_if_cached: bool = True,
) -> str:
    """
    - Produces per-chunk summaries + memory_ingest
    - Writes a cached full summary to data/analysis/<video_id>/summary_<mode>.txt
    - Appends a run record to data/analysis/<video_id>/runs.jsonl

    reuse_if_cached=True:
      if cached summary exists for (video_id, mode), return it immediately.
      (no slicing, no model call)
    """
    mode_key = (mode or "standard").strip().lower()
    if mode_key not in MODES:
        mode_key = "standard"
    mc = MODES[mode_key]

    project_root_p = Path(project_root).resolve()
    video_path_p = Path(video_path)
    vid = video_path_p.stem

    # cache hit
    if reuse_if_cached and has_summary(project_root_p, vid, mode_key):
        cached = load_summary(project_root_p, vid, mode_key)
        if cached is not None:
            return cached

    # ensure model loaded once per process
    init_model(local_files_only=local_files_only)

    t0 = datetime.now(timezone.utc)

    manifest_path, manifest_json = slice_video(
        str(video_path_p),
        chunk_seconds=mc.chunk_seconds,
        keyframes_per_chunk=mc.keyframes_per_chunk,  # fixed 6
        overwrite=overwrite_slice,
    )
    manifest_path = Path(manifest_path)

    chunks = read_chunks_from_manifest(manifest_json)
    if not chunks:
        out = f"[WARN] No chunks found in manifest. video='{video_path_p.name}'"
        # still cache the warning (so CLI shows “analyzed” but it's just a warn)
        save_summary(project_root_p, vid, mode_key, out)
        return out

    prev_summary = ""
    chunk_summaries: List[str] = []

    for chunk in chunks:
        if not chunk.image_paths:
            continue

        prompt = build_prompt(chunk, prev_summary if use_prev_summary else "")

        if mc.two_pass:
            raw_text = _two_pass_infer(chunk.image_paths, prompt, mc.infer_cfg)
        else:
            raw_text = model_interface(image_paths=chunk.image_paths, prompt=prompt, cfg=mc.infer_cfg)

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

        _post_infer_cleanup()

    if not chunk_summaries:
        out = f"[WARN] No summaries produced. video='{video_path_p.name}'"
        save_summary(project_root_p, vid, mode_key, out)
        return out

    header = (
        f"Video: {video_path_p.name} | Mode: {mc.name} | "
        f"chunk_seconds={mc.chunk_seconds} | frames_per_chunk={mc.keyframes_per_chunk} | video_id={chunks[0].video_id}"
    )
    out = header + "\n" + "\n".join(chunk_summaries)

    # save cache
    save_summary(project_root_p, vid, mode_key, out)

    # append run record
    t1 = datetime.now(timezone.utc)
    run_rec = {
        "video_id": vid,
        "video_path": str(video_path_p).replace("\\", "/"),
        "mode": mode_key,
        "slice": {
            "chunk_seconds": mc.chunk_seconds,
            "keyframes_per_chunk": mc.keyframes_per_chunk,
        },
        "infer": {
            "max_new_tokens": int(mc.infer_cfg.max_new_tokens),
            "max_num": int(mc.infer_cfg.max_num),
            "use_thumbnail": bool(mc.infer_cfg.use_thumbnail),
            "two_pass": bool(mc.two_pass),
        },
        "time": {
            "started_at": t0.isoformat(),
            "ended_at": t1.isoformat(),
        },
        "artifacts": {
            "manifest_path": str(manifest_path).replace("\\", "/"),
            "summary_txt": f"data/analysis/{vid}/summary_{mode_key}.txt",
        },
        "status": "ok",
    }
    append_run(project_root_p, vid, run_rec)

    return out
