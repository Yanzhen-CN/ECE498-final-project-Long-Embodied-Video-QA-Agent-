from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.video_interface import slice_video
from memory.memory_interface import memory_ingest
from model.model_interface import InferConfig, init_model, model_interface
from agent.video_management import register_analysis_run

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


@dataclass(frozen=True)
class ModeConfig:
    name: str
    slice_cfg: SliceConfig
    infer_cfg: InferConfig


# =========================
# Analysis modes
# =========================
KEYFRAMES_PER_CHUNK = 6

MODES: Dict[str, ModeConfig] = {
    "fast": ModeConfig(
        name="Fast",
        slice_cfg=SliceConfig(chunk_seconds=60),
        infer_cfg=InferConfig(max_new_tokens=128, max_num=2, use_thumbnail=False),
    ),
    "standard": ModeConfig(
        name="Standard",
        slice_cfg=SliceConfig(chunk_seconds=30),
        infer_cfg=InferConfig(max_new_tokens=256, max_num=2, use_thumbnail=False),
    ),
    "detailed": ModeConfig(
        name="Detailed",
        slice_cfg=SliceConfig(chunk_seconds=15),
        infer_cfg=InferConfig(max_new_tokens=256, max_num=2, use_thumbnail=False),
    ),
}


# =========================
# Analysis registry
# =========================
ROOT_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
REGISTRY_PATH = ANALYSIS_DIR / "registry.json"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

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

    video_path = Path(video_path)
    video_name = video_path.name
    video_stem = video_path.stem
    run_id = f"{video_stem}__{mode_key}"

    # load model once (cache reuse)
    init_model(local_files_only=local_files_only)

    # slice: store under run_id so modes don't overwrite
    manifest_path, manifest_json = slice_video(
        str(video_path),
        video_id=run_id,  # ✅ key change
        chunk_seconds=cfg.slice_cfg.chunk_seconds,
        keyframes_per_chunk=KEYFRAMES_PER_CHUNK,
        overwrite=True,
    )
    manifest_path = Path(manifest_path)

    chunks = read_chunks_from_manifest(manifest_json)
    if not chunks:
        return f"[WARN] No chunks found in manifest. video='{video_name}'"

    prev_summary = ""
    chunk_summaries: List[str] = []

    for chunk in chunks:
        if not chunk.image_paths:
            continue

        prompt = build_prompt(chunk, prev_summary if use_prev_summary else "")


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
        return f"[WARN] No summaries produced. video='{video_name}'"

    header = f"Video: {video_name} | Mode: {cfg.name} | run_id={run_id}"
    context = header + "\n" + "\n".join(chunk_summaries)
    print(f"Run summary pipeline success, {video_name}__{mode} registering")
    # registry
    try:
        register_analysis_run(
            video_name=video_name,
            mode=mode_key,  # Automatically handled
            run_id = run_id
        )
    except:
        print("register fail, but the summary finish, you can start QA now")

    return context
