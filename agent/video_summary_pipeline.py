from __future__ import annotations

import json
import re
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
    if prev_summary:
        lines.append(
            f"Previous chunk summary (for continuity): {prev_summary.strip() if prev_summary.strip() else '(none)'}"
        )
    lines.append("You are given multiple keyframes from this video chunk in chronological order.")
    lines.append("Task: produce a concise, faithful summary of what happens in this chunk. Do NOT guess.")
    lines.append(
        "Output MUST be valid JSON ONLY (no markdown, no extra text) with keys:\n"
        "summary,entities,state_update"
    )
    lines.append("Keep summary <= 4 sentences. events <= 6. entities <= 10.")
    lines.append("Return ONLY JSON.")
    return "\n".join(lines)



import json
import re

def normalize_record(
    raw_text: str,
    chunk: ChunkSpec,
    *,
    evidence_per_chunk: int = 2,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    # 尝试通过正则提取关键信息
    raw_json = slice_and_reconstruct(raw_text)
    if raw_json is None:
        raw_json = {"parse_error": True, "raw_model_output": raw_text}

    # 确保每个字段都存在，使用默认值填充
    raw_json.setdefault("summary", "")
    raw_json.setdefault("entities", [])
    raw_json.setdefault("events", [])
    raw_json.setdefault("state_update", {})


    # 格式化 state_update 字段
    if isinstance(raw_json["state_update"], dict):
        raw_json["state_update"].setdefault("holding", "")
        raw_json["state_update"].setdefault("red cup location", "")

    # 创建最终的 record 字典，返回标准格式
    record = {
        "video_id": chunk.video_id,
        "chunk_id": chunk.chunk_id,
        "t_start": chunk.t_start,
        "t_end": chunk.t_end,
        "summary": raw_json["summary"],
        "entities": raw_json["entities"],
        "events": raw_json["events"],
        "state_update": raw_json["state_update"],
        "evidence_frames": chunk.image_paths[:max(0, evidence_per_chunk)],
    }

    if manifest_path is not None:
        record["manifest_path"] = str(manifest_path)

    return record


def slice_and_reconstruct(raw_text: str) -> Dict[str, Any]:
    """
    使用字符串切割和正则表达式提取关键信息并重构 JSON 数据。
    """
    # 尝试通过正则表达式提取字段（例如 summary, entities, events, state_update）
    summary_pattern = r'"summary":\s*"([^"]+)"'
    entities_pattern = r'"entities":\s*\[([^\]]+)\]'
    events_pattern = r'"events":\s*\[([^\]]+)\]'
    state_update_pattern = r'"state_update":\s*({[^}]+})'

    summary = re.search(summary_pattern, raw_text)
    entities = re.search(entities_pattern, raw_text)
    events = re.search(events_pattern, raw_text)
    state_update = re.search(state_update_pattern, raw_text)

    # 处理提取的数据，避免 `None` 值
    reconstructed = {
        "summary": summary.group(1) if summary else "",  # 提取 summary，如果没有则为空字符串
        "entities": [entity.strip() for entity in entities.group(1).split(",")] if entities else [],  # 提取并分割 entities
        
        # 提取并解析 state_update
        "state_update": json.loads(state_update.group(1)) if state_update else {}  # 提取并解析 state_update
    }

    return reconstructed



def slice_and_reconstruct(raw_text: str) -> Dict[str, Any]:
    """
    使用字符串切割和正则表达式提取关键信息并重构 JSON 数据。
    """
    # 尝试通过正则表达式提取字段（例如 summary, entities, events, state_update）
    summary_pattern = r'"summary":\s*"([^"]+)"'
    entities_pattern = r'"entities":\s*\[([^\]]+)\]'
    events_pattern = r'"events":\s*\[([^\]]+)\]'
    state_update_pattern = r'"state_update":\s*({[^}]+})'

    summary = re.search(summary_pattern, raw_text)
    entities = re.search(entities_pattern, raw_text)
    events = re.search(events_pattern, raw_text)
    state_update = re.search(state_update_pattern, raw_text)

    # 处理提取的数据，避免 `None` 值
    reconstructed = {
        "summary": summary.group(1) if summary else "",  # 提取 summary，如果没有则为空字符串
        "entities": [entity.strip() for entity in entities.group(1).split(",")] if entities else [],  # 提取并分割 entities
        
        # 检查 events 是否存在，存在时拆分处理
        "events": [
            {
                "verb": event.split(":")[0].strip(),  # 提取 verb
                "obj": event.split(":")[1].strip(),   # 提取 obj
                "detail": event.split(":")[2].strip() if len(event.split(":")) > 2 else ""  # 提取 detail（如果存在）
            }
            for event in (events.group(1).split("},") if events else [])  # 分割事件并处理
        ],
        
        # 提取并解析 state_update
        "state_update": json.loads(state_update.group(1)) if state_update else {}  # 提取并解析 state_update
    }



    return reconstructed


# =========================
# Public API for CLI
# =========================
def summarize_video_for_cli(
    video_path: str | Path,
    *,
    mode: str = "standard",
    local_files_only: bool = False,
    evidence_per_chunk: int = 2,
    use_prev_summary: bool = False,
) -> str:
    mode_key = (mode or "standard").strip().lower()
    if mode_key not in MODES:
        mode_key = "standard"
    cfg = MODES[mode_key]

    video_path = Path(video_path)
    video_name = video_path.name
    video_stem = video_path.stem
    video_id = f"{video_stem}__{mode_key}"

    # load model once (cache reuse)
    init_model(local_files_only=local_files_only)

    # slice: store under video_id so modes don't overwrite
    manifest_path, manifest_json = slice_video(
        str(video_path),
        video_id=video_id,  # ✅ key change
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
        store_path = Path(f"data/temp/raw_response_{chunk.chunk_id}.json")
        Store = {"respond": raw_text}
        with open(store_path, 'w') as f:
            json.dump(Store, f, indent=4)
        memory_ingest(record)

        s = (record.get("summary") or "").strip()
        if s:
            time_str = f"{seconds_to_mmss(chunk.t_start)}–{seconds_to_mmss(chunk.t_end)}"
            chunk_summaries.append(f"[{time_str}] {s}")
            if use_prev_summary:
                prev_summary = s

    if not chunk_summaries:
        return f"[WARN] No summaries produced. video='{video_name}'"

    header = f"Video: {video_name} | Mode: {cfg.name} | video_id={video_id}"
    context = header + "\n" + "\n".join(chunk_summaries)
    print(f"Run summary pipeline success, {video_id} registering")
    # registry
    register_analysis_run(
        video_name=video_name,
        mode=mode_key,  # Automatically handled
        video_id = video_id
    )

    return context
