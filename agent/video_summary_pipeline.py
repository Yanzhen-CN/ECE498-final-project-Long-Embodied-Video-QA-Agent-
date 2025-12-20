from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.video_interface import slice_video
from memory.memory_interface import memory_ingest

from model.model_interface import init_model, model_interface, InferConfig


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
class SliceConfig:
    chunk_seconds: int
    keyframes_per_chunk: int


@dataclass(frozen=True)
class ModeConfig:
    name: str
    slice_cfg: SliceConfig
    infer_cfg: InferConfig
    two_pass: bool


# =========================
# Analysis modes (3 presets)
# =========================
MODES: Dict[str, ModeConfig] = {
    # Fast: fewer chunks + fewer frames, single pass
    "fast": ModeConfig(
        name="Fast",
        slice_cfg=SliceConfig(chunk_seconds=60, keyframes_per_chunk=4),
        infer_cfg=InferConfig(max_new_tokens=128, max_num=2, use_thumbnail=False),
        two_pass=False,
    ),
    # Standard: your current slicing (30s, 8 frames) but two-pass to avoid OOM
    "standard": ModeConfig(
        name="Standard",
        slice_cfg=SliceConfig(chunk_seconds=30, keyframes_per_chunk=8),
        infer_cfg=InferConfig(max_new_tokens=256, max_num=2, use_thumbnail=False),
        two_pass=True,
    ),
    # Detailed: more temporal resolution, keep frames=8 but still two-pass
    "detailed": ModeConfig(
        name="Detailed",
        slice_cfg=SliceConfig(chunk_seconds=15, keyframes_per_chunk=8),
        infer_cfg=InferConfig(max_new_tokens=256, max_num=2, use_thumbnail=False),
        two_pass=True,
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

    rec["video_id"] = chunk.video_id
    rec["chunk_id"] = chunk.chunk_id
    rec["t_start"] = chunk.t_start
    rec["t_end"] = chunk.t_end

    rec.setdefault("summary", "")
    rec.setdefault("entities", [])
    rec.setdefault("events", [])
    rec.setdefault("state_update", {})

    rec["evidence_frames"] = chunk.image_paths[: max(0, evidence_per_chunk)]

    if manifest_path is not None:
        rec["manifest_path"] = str(manifest_path)

    return rec


def _two_pass_infer(frames: List[str], base_prompt: str, infer_cfg: InferConfig) -> str:
    """
    Keep chunk fixed (8 frames), but reduce peak VRAM by splitting inference 4+4.
    Pass1 returns plain text. Pass2 returns final JSON for the whole chunk.
    """
    if len(frames) <= 4:
        return model_interface(image_paths=frames, prompt=base_prompt, cfg=infer_cfg)

    mid = len(frames) // 2
    first = frames[:mid]
    second = frames[mid:]

    prompt1 = base_prompt + "\n\nYou only see the FIRST half frames. Summarize briefly in plain text (no JSON)."
    partial = model_interface(image_paths=first, prompt=prompt1, cfg=infer_cfg)

    prompt2 = (
        base_prompt
        + "\n\nYou only see the SECOND half frames.\n"
        + "First-half summary (may be imperfect):\n"
        + partial.strip()
        + "\n\nNow output FINAL JSON for the whole chunk."
    )
    return model_interface(image_paths=second, prompt=prompt2, cfg=infer_cfg)


# =========================
# Public API for CLI
# =========================
def summarize_video_for_cli(
    video_path: str | Path,
    *,
    mode: str = "standard",
    local_files_only: bool = False,
    evidence_per_chunk: int = 2,
    use_prev_summary: bool = True,
) -> str:
    mode_key = (mode or "standard").strip().lower()
    if mode_key not in MODES:
        mode_key = "standard"

    cfg = MODES[mode_key]

    # load model once (cache reuse)
    init_model(local_files_only=local_files_only)

    manifest_path, manifest_json = slice_video(
        str(video_path),
        chunk_seconds=cfg.slice_cfg.chunk_seconds,
        keyframes_per_chunk=cfg.slice_cfg.keyframes_per_chunk,
        overwrite=True,
    )
    manifest_path = Path(manifest_path)

    chunks = read_chunks_from_manifest(manifest_json)
    if not chunks:
        return f"[WARN] No chunks found in manifest. video='{Path(video_path).name}'"

    prev_summary = ""
    chunk_summaries: List[str] = []

    for chunk in chunks:
        if not chunk.image_paths:
            continue

        prompt = build_prompt(chunk, prev_summary if use_prev_summary else "")

        if cfg.two_pass:
            raw_text = _two_pass_infer(chunk.image_paths, prompt, cfg.infer_cfg)
        else:
            raw_text = model_interface(image_paths=chunk.image_paths, prompt=prompt, cfg=cfg.infer_cfg)

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

    header = f"Video: {Path(video_path).name} | Mode: {cfg.name} | video_id={chunks[0].video_id}"
    return header + "\n" + "\n".join(chunk_summaries)
