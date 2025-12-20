import json
from pathlib import Path
from typing import Dict, List, Union
import shutil

# ===== ChunkMemory schema (based on your JSON) =====
REQUIRED_FIELDS = {
    "video_id",
    "chunk_id",
    "t_start",
    "t_end",
    "summary",
    "entities",
    "events",
    "state_update",
    "evidence_frames",
}


def _validate_chunk_memory(chunk: Dict) -> None:
    """Lightweight validation for ChunkMemory."""
    missing = REQUIRED_FIELDS - chunk.keys()
    if missing:
        raise ValueError(f"ChunkMemory missing fields: {missing}")

    if not isinstance(chunk["events"], list):
        raise ValueError("events must be a list")

    if not isinstance(chunk["evidence_frames"], list):
        raise ValueError("evidence_frames must be a list")


def _save_chunks(
    chunks: List[Dict],
    memory_root: str = "memory/saved_videos",
) -> List[Path]:
    """
    Save each chunk into its own JSONL file under:
    memory/{video_id}/{t_start}-{t_end}.jsonl

    File name does NOT include video_id, only time range.
    Time is formatted as 4 digits (seconds): 0570-0600
    """
    out_paths: List[Path] = []

    for c in chunks:
        video_id = c["video_id"]
        t_start = int(c["t_start"])
        t_end = int(c["t_end"])

        # 每个 video_id 一个子文件夹
        video_dir = Path(memory_root) / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        # 文件名：四位时间范围，不含 video_id
        out_path = video_dir / f"{t_start:04d}-{t_end:04d}.jsonl"

        # 一个文件只存一个 chunk（覆盖写）
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(c, ensure_ascii=False))
            f.write("\n")

        out_paths.append(out_path)

    return out_paths




# ==================================================
# ⭐ Unified public interface
# ==================================================
def memory_ingest(record: Union[Dict, List[Dict]], memory_root: str = "memory/saved_videos") -> Dict:
    if isinstance(record, dict):
        chunks = [record]
    elif isinstance(record, list):
        chunks = record
    else:
        raise TypeError("record must be dict or list[dict]")

    if not chunks:
        raise ValueError("Empty memory record")

    video_id = chunks[0].get("video_id")
    if not video_id:
        raise ValueError("video_id missing")

    for chunk in chunks:
        _validate_chunk_memory(chunk)
        if chunk["video_id"] != video_id:
            raise ValueError("Mixed video_id in one ingest call")

    out_paths = _save_chunks(chunks, memory_root=memory_root)

    return {
        "status": "ok",
        "video_id": video_id,
        "num_chunks": len(chunks),
        "saved_dir": str(Path(memory_root) / video_id),
        "saved_to": [str(p) for p in out_paths],
    }

def clean_memory(video_id: str, memory_root: str = "memory/saved_videos") -> bool:
    """
    Delete all chunk memories for a given video_id by removing:
      {memory_root}/{video_id}/

    Returns:
        True  -> directory existed and was deleted
        False -> directory not found (nothing to delete)
    """
    if not video_id or not video_id.strip():
        raise ValueError("video_id is required")

    # Safety: forbid path traversal / nested paths
    if Path(video_id).name != video_id or "/" in video_id or "\\" in video_id:
        raise ValueError(f"Invalid video_id: {video_id}")

    root = Path(memory_root).resolve()
    target_dir = (root / video_id).resolve()

    # Safety: ensure target_dir is inside root
    if root not in target_dir.parents and target_dir != root:
        raise ValueError("Refusing to delete: target directory is outside memory_root")

    if not target_dir.exists():
        return False

    if not target_dir.is_dir():
        raise ValueError(f"Expected a directory for video_id, but got file: {target_dir}")

    shutil.rmtree(target_dir)
    return True