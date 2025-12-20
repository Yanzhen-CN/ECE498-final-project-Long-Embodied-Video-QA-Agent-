from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def slice_video(
    video: str,
    *,
    video_id: Optional[str] = None,
    data_root: str = "data",
    chunk_seconds: int = 30,
    keyframes_per_chunk: int = 6,
    jpeg_quality: int = 90,
    overwrite: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Slice a long video into chunk skeyframes and produce:
      - manifest JSON (chunk->image)
      - metadata CSV (frame-level index)

    IMPORTANT:
      video_id controls storage folder/name:
        keyframes/<video_id>/..., manifests/<video_id>.json, metadata/<video_id>.csv

      So if you pass video_id = f"{video_stem}__{mode}",
      different modes will NOT overwrite each other.
    """
    video_path = Path(video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    vid = video_id or video_path.stem
    data_root_p = Path(data_root)

    base = data_root_p / "processed_videos" / vid
    keyframes_dir = base / "keyframes" / vid
    manifests_dir = base / "manifests"
    metadata_dir = base / "metadata"

    manifest_path = manifests_dir / f"{vid}.json"
    csv_path = metadata_dir / f"{vid}.csv"

    if overwrite:
        base = data_root_p / "processed_videos" / vid
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)
        _ensure_dir(keyframes_dir)
        _ensure_dir(manifest_path.parent)
        _ensure_dir(metadata_path.parent)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = (frame_count / fps) if frame_count > 0 else 0.0
    if duration_sec <= 0:
        duration_sec = float(chunk_seconds)

    n_chunks = max(1, math.ceil(duration_sec / chunk_seconds))

    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["video_id", "chunk_id", "img_idx", "t_sec", "image_path"])

    manifest: Dict[str, Any] = {
        "manifest_version": "1.0",
        "video_id": vid,
        "source_video_name": video_path.name,
        "source_video_path": str(video_path).replace("\\", "/"),
        "fps": float(fps),
        "duration_sec": float(round(duration_sec, 3)),
        "chunk_seconds": int(chunk_seconds),
        "keyframes_per_chunk": int(keyframes_per_chunk),
        "chunks": [],
    }

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    def save_frame_at_time(t_sec: float, out_path: Path) -> bool:
        frame_idx = int(round(t_sec * fps))
        frame_idx = max(0, min(frame_idx, max(0, frame_count - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return False
        return bool(cv2.imwrite(str(out_path), frame, encode_params))

    for ci in range(n_chunks):
        t_start = ci * chunk_seconds
        t_end = min((ci + 1) * chunk_seconds, duration_sec)

        chunk_dir = keyframes_root / f"chunk_{ci:03d}"
        _ensure_dir(chunk_dir)

        chunk_images = []
        for j in range(keyframes_per_chunk):
            alpha = (j + 0.5) / keyframes_per_chunk
            t = t_start + alpha * max(0.001, (t_end - t_start))

            img_name = f"img_{j:02d}.jpg"
            img_path = chunk_dir / img_name

            if not save_frame_at_time(t, img_path):
                continue

            rel_img = f"{data_root_p.name}/keyframes/{vid}/chunk_{ci:03d}/{img_name}"
            rel_img = rel_img.replace("\\", "/")
            chunk_images.append(rel_img)

            csv_w.writerow([vid, ci, j, round(t, 3), rel_img])

        manifest["chunks"].append(
            {
                "chunk_id": ci,
                "t_start": int(round(t_start)),
                "t_end": int(round(t_end)),
                "chunk_images": chunk_images,
            }
        )

    csv_f.close()
    cap.release()

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest_path, manifest

def clean_processed_video(
    video_id: str,
    *,
    data_root: str = "data",
    dry_run: bool = False,
) -> bool:
    """
    Delete processed video artifacts under:

      data/processed_videos/<video_id>/

    Returns:
        True  -> folder existed (and deleted if not dry_run)
        False -> folder did not exist
    """
    base_dir = Path(data_root) / "processed_videos" / video_id

    if not base_dir.exists():
        print(f"[Clean] Not found: {base_dir}")
        return False

    if dry_run:
        print(f"[Dry-run] Would delete: {base_dir}")
        return True

    shutil.rmtree(base_dir, ignore_errors=True)
    print(f"[Clean] Deleted: {base_dir}")
    return True
