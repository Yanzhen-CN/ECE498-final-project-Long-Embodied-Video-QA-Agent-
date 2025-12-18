import json
from pathlib import Path
from typing import Dict, List, Union


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
    memory_root: str = "memory/chunks",
) -> Path:
    """Append chunks to memory/chunks/{video_id}.jsonl"""

    video_id = chunks[0]["video_id"]

    memory_dir = Path(memory_root)
    memory_dir.mkdir(parents=True, exist_ok=True)

    out_path = memory_dir / f"{video_id}.jsonl"

    with out_path.open("a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False))
            f.write("\n")

    return out_path


# ==================================================
# ⭐ Unified public interface
# ==================================================
def memory_ingest(record: Union[Dict, List[Dict]]) -> Dict:
    """
    Unified memory ingestion entry.

    Args:
        record: single ChunkMemory dict or list of ChunkMemory dicts

    Returns:
        metadata dict (for logging / API response)
    """

    # Normalize to list
    if isinstance(record, dict):
        chunks = [record]
    elif isinstance(record, list):
        chunks = record
    else:
        raise TypeError("record must be dict or list[dict]")

    if not chunks:
        raise ValueError("Empty memory record")

    # Validate & consistency check
    video_id = chunks[0].get("video_id")
    if not video_id:
        raise ValueError("video_id missing")

    for chunk in chunks:
        _validate_chunk_memory(chunk)
        if chunk["video_id"] != video_id:
            raise ValueError("Mixed video_id in one ingest call")

    # Save
    out_path = _save_chunks(chunks)

    return {
        "status": "ok",
        "video_id": video_id,
        "num_chunks": len(chunks),
        "saved_to": str(out_path),
    }
