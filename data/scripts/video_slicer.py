from __future__ import annotations
import os, json, math, csv
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import shutil
import cv2


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def slice_video(
    video: str,
    *,
    video_id: Optional[str] = None,
    data_root: str = "data",
    chunk_seconds: int = 30,
    keyframes_per_chunk: int = 8,
    jpeg_quality: int = 90,
    overwrite: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Slice a long video into chunk keyframes and produce:
      - manifest JSON (chunk->images)
      - metadata CSV (frame-level index)

    Returns:
      (manifest_path, manifest_dict)
    """
    video_path = Path(video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    vid = video_id or video_path.stem

    data_root_p = Path(data_root)
    manifests_dir = data_root_p / "manifests"
    metadata_dir = data_root_p / "metadata"
    keyframes_root = data_root_p / "keyframes" / vid

    manifest_path = manifests_dir / f"{vid}.json"
    csv_path = metadata_dir / f"{vid}.csv"

    if overwrite:
        if keyframes_root.exists():
            shutil.rmtree(keyframes_root)  # 删除整个 keyframes/<vid> 文件夹
        if manifest_path.exists():
            manifest_path.unlink()         # 删除旧 manifest
        if csv_path.exists():
            csv_path.unlink()              # 删除旧 csv
    else:
        # 不允许覆盖：若存在则直接报错，防止误删
        if keyframes_root.exists() or manifest_path.exists() or csv_path.exists():
            raise FileExistsError(
                f"Outputs for video_id='{vid}' already exist. "
                f"Set overwrite=True to replace."
            )

    _ensure_dir(manifests_dir)
    _ensure_dir(metadata_dir)
    _ensure_dir(keyframes_root)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = (frame_count / fps) if frame_count > 0 else 0.0
    if duration_sec <= 0:
        # 兜底：仍然按 chunk_seconds 切至少 1 chunk
        duration_sec = float(chunk_seconds)

    n_chunks = max(1, math.ceil(duration_sec / chunk_seconds))

    # CSV: 每张图一行（索引/标签）
    csv_path = metadata_dir / f"{vid}.csv"
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["video_id", "chunk_id", "img_idx", "t_sec", "image_path"])

    # manifest（chunk级）
    # video_path：建议存相对路径或原路径，你们组里统一即可
    manifest: Dict[str, Any] = {
        "manifest_version": "1.0",
        "video_id": vid,
        "video_path": str(video_path).replace("\\", "/"),
        "fps": float(fps),
        "duration_sec": float(round(duration_sec, 3)),
        "chunk_seconds": int(chunk_seconds),
        "keyframes_per_chunk": int(keyframes_per_chunk),
        "chunks": []
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
            # 中心采样，避免边界
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

        manifest["chunks"].append({
            "chunk_id": ci,
            "t_start": int(round(t_start)),
            "t_end": int(round(t_end)),
            "chunk_images": chunk_images
        })

    csv_f.close()
    cap.release()

    manifest_path = manifests_dir / f"{vid}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest_path, manifest
