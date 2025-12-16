from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List


# =========================
# Stage A: DATA interfaces
# =========================

def data_prepare_video(
    video_id: str,
    video_path: Path,
    data_root: Path,
    data_cfg: Dict[str, Any],
) -> Path:
    """
    Prepare all data artifacts for a video and return a DataManifest path.

    Must create:
      data/manifests/<video_id>.json
      data/keyframes/<video_id>/index.jsonl
      data/keyframes/<video_id>/chunk_XXX/*.jpg

    DataManifest JSON must include at least:
      {
        "video_id": "...",
        "video_path": "...",
        "keyframes_index": "data/keyframes/<video_id>/index.jsonl",
        "keyframes_dir": "data/keyframes/<video_id>/",
        "fps": <float>,
        "duration_sec": <int>,
        "chunk_seconds": <int>,
        "keyframes_per_chunk": <int>,
        "num_chunks": <int>
      }
    """
    raise NotImplementedError("TODO: implement in data/ module and wire here.")


# =========================
# Stage A/B: MODEL interface
# =========================

def model_generate(
    image_paths: List[str],
    prompt: str,
    generation_config: Dict[str, Any],
    max_num_tiles: int,
) -> str:
    """
    Call InternVL (or any VLM) to generate text output.
    Expected: returns JSON text (preferred) or raw text.
    """
    raise NotImplementedError("TODO: implement in model/ module and wire here.")


# =========================
# Stage A/B: MEMORY interfaces
# =========================

def memory_append_chunk_memory(video_id: str, chunk_memory: Dict[str, Any], chunks_jsonl: Path) -> None:
    """Append one ChunkMemory record to memory/chunks/<video_id>.jsonl"""
    raise NotImplementedError("TODO: implement in memory/ module and wire here.")


def memory_build_index(chunks_jsonl: Path, index_path: Path) -> None:
    """Build retrieval index from chunks_jsonl -> index_path"""
    raise NotImplementedError("TODO: implement in memory/ module and wire here.")


def memory_retrieve(
    video_id: str,
    query: str,
    top_k: int,
    chunks_jsonl: Path,
    index_path: Path
) -> List[Dict[str, Any]]:
    """
    Return retrieved chunk memories:
      [
        {"chunk_id":..., "score":..., "t_start":..., "t_end":...,
         "memory": <ChunkMemory dict>, "evidence_frames":[...]}
      ]
    """
    raise NotImplementedError("TODO: implement in memory/ module and wire here.")
