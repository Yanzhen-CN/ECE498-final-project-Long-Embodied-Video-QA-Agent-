import json
import shutil
from pathlib import Path
from typing import Dict

# 改成你的源代码模块名（不带 .py）
from memory_interface import memory_ingest


def main() -> None:
    # sample 1: 单个 record
    record1 = {
        "video_id": "vid_demo_001",
        "chunk_id": "chunk_0001",
        "t_start": 570,
        "t_end": 600,
        "summary": "Demo chunk: person enters the room.",
        "entities": [{"name": "Alice", "type": "person"}],
        "events": [{"type": "action", "text": "Alice enters."}],
        "state_update": {"location": "indoors"},
        "evidence_frames": [{"frame": 123, "ts": 571.2}],
    }

    # sample 2: 多个 record（同一个 video_id）
    record2 = [
        {
            "video_id": "vid_demo_002",
            "chunk_id": "chunk_0001",
            "t_start": 0,
            "t_end": 30,
            "summary": "Intro segment.",
            "entities": [],
            "events": [],
            "state_update": {},
            "evidence_frames": [],
        },
        {
            "video_id": "vid_demo_002",
            "chunk_id": "chunk_0002",
            "t_start": 30,
            "t_end": 60,
            "summary": "Second segment with speech.",
            "entities": [{"name": "Bob", "type": "person"}],
            "events": [{"type": "speech", "text": "Hello"}],
            "state_update": {"topic": "greeting"},
            "evidence_frames": [{"frame": 50, "ts": 32.0}],
        },
    ]

    print("=== Ingest record1 (single) ===")
    res1 = memory_ingest(record1)  # 不传 memory_root，使用默认 saved_videos
    print(json.dumps(res1, ensure_ascii=False, indent=2))

    print("\n=== Ingest record2 (list) ===")
    res2 = memory_ingest(record2)  # 不传 memory_root，使用默认 saved_videos
    print(json.dumps(res2, ensure_ascii=False, indent=2))

    print("\nNow check the folder: ./saved_videos/")
    print("Expected files:")
    print(" - ./saved_videos/vid_demo_001/0570-0600.jsonl")
    print(" - ./saved_videos/vid_demo_002/0000-0030.jsonl")
    print(" - ./saved_videos/vid_demo_002/0030-0060.jsonl")


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


if __name__ == "__main__":
    main()
