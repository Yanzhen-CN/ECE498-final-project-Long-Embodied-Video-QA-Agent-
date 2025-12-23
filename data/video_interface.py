from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import cv2

# -----------------------------
# Paths
# -----------------------------
DATA_ROOT = Path("data")
PROCESSED_ROOT = Path(DATA_ROOT / "processed_videos")
VIDEOS_ROOT = Path(DATA_ROOT / "videos")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_mp4(p: Path) -> bool:
    return p.suffix.lower() == ".mp4"


def _sanitize_id(s: str, *, name: str = "id") -> str:
    """Forbid path traversal / nested paths."""
    s = (s or "").strip()
    if not s:
        raise ValueError(f"{name} is empty")
    if Path(s).name != s or "/" in s or "\\" in s:
        raise ValueError(f"Invalid {name}: {s}")
    return s


# -----------------------------
# Video slicing (processed artifacts)
# -----------------------------
def slice_video(
    video: str,
    *,
    video_id: Optional[str] = None,
    data_root: str | Path = DATA_ROOT,
    chunk_seconds: int = 30,
    keyframes_per_chunk: int = 6,
    jpeg_quality: int = 90,
    overwrite: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Slice a long video into chunk keyframes and produce:
      - manifest JSON (chunk->image)
      - metadata CSV (frame-level index)

    Layout:
      {data_root}/processed_videos/{video_id}/
        keyframes/chunk_000/img_00.jpg ...
        manifests/{video_id}.json
        metadata/{video_id}.csv
    """
    video_path = Path(video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    vid = (video_id or video_path.stem).strip()
    data_root_p = Path(data_root)

    base = data_root_p / "processed_videos" / vid
    keyframes_dir = base / "keyframes"
    manifests_dir = base / "manifests"
    metadata_dir = base / "metadata"

    manifest_path = manifests_dir / f"{vid}.json"
    csv_path = metadata_dir / f"{vid}.csv"

    if overwrite:
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)
        _ensure_dir(keyframes_dir)
        _ensure_dir(manifest_path.parent)
        _ensure_dir(csv_path.parent)
    else:
        _ensure_dir(keyframes_dir)
        _ensure_dir(manifest_path.parent)
        _ensure_dir(csv_path.parent)

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

        chunk_dir = keyframes_dir / f"chunk_{ci:03d}"
        _ensure_dir(chunk_dir)

        chunk_images: List[str] = []
        for j in range(keyframes_per_chunk):
            alpha = (j + 0.5) / keyframes_per_chunk
            t = t_start + alpha * max(0.001, (t_end - t_start))

            img_name = f"img_{j:02d}.jpg"
            img_path = chunk_dir / img_name

            if not save_frame_at_time(t, img_path):
                continue

            rel_img = f"{data_root_p.name}/processed_videos/{vid}/keyframes/chunk_{ci:03d}/{img_name}"
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


def clean_processed_video(video_id: str | None = None, processed_root: str | Path = PROCESSED_ROOT) -> bool:
    """
    Delete processed artifacts under:
      {processed_root}/{video_id}/

    If video_id is None/empty -> delete entire processed_root.
    """
    root = Path(processed_root)

    # clear all
    if not video_id or not str(video_id).strip():
        if not root.exists():
            print("[Clean] processed_videos already cleaned.")
            return False
        shutil.rmtree(root, ignore_errors=True)
        print(f"[Clean] Deleted: {root}")
        return True

    vid = _sanitize_id(str(video_id), name="video_id")
    target = root / vid

    if not target.exists():
        print(f"[Clean] Not found: {target}")
        return False

    shutil.rmtree(target, ignore_errors=True)
    print(f"[Clean] Deleted: {target}")
    return True


# -----------------------------
# Uploaded videos (raw mp4)
# -----------------------------
@dataclass
class VideoStore:
    """Manage uploaded mp4 files under a directory (default: data/videos)."""
    dir_path: Path = VIDEOS_ROOT

    def __post_init__(self) -> None:
        self.dir_path = Path(self.dir_path)  # Convert string to Path if necessary
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def list_names(self) -> List[str]:
        files = sorted([p for p in self.dir_path.iterdir() if p.is_file() and _is_mp4(p)])
        return [p.name for p in files]

    def get_path_by_name(self, name: str) -> Optional[Path]:
        p = self.dir_path / name
        if p.exists() and p.is_file() and _is_mp4(p):
            return p
        return None

    def _src_path(self, src_input: str) -> Path:
        """
        Normalize user input into an existing .mp4 Path.
        - supports: absolute/relative path, with or without .mp4 suffix
        - expands ~ and resolves
        """
        s = (src_input or "").strip()
        if not s:
            raise ValueError("Empty src path.")

        # If user typed without suffix, append ".mp4"
        # NOTE: do NOT use endswith on the raw string when it has trailing spaces
        if not s.lower().endswith(".mp4"):
            print("add suffix")
            s = s + ".mp4"

        p = Path(s).expanduser()

        # On Windows, resolve() can throw if path is weird; keep it safe
        try:
            p = p.resolve()
        except Exception:
            p = p.absolute()

        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")

        if p.suffix.lower() != ".mp4":
            raise ValueError(f"Only .mp4 supported, got: {p.suffix}")

        return p
    
    def _dst_path(self, src: Path, dst_name: Optional[str]) -> Path:
        if dst_name is not None and dst_name.strip():
            name = Path(dst_name.strip()).name
            if not name.lower().endswith(".mp4"):
                name += ".mp4"
                print("add suffix")
            return self.dir_path / name
        return self.dir_path / src.name

    def upload_from_path(
        self,
        src_path: str,
        dst_name: Optional[str] = None,
        *,
        overwrite: bool = False,
    ) -> Tuple[bool, str]:
        try:
            src = self._src_path(src_path)
        except Exception as e:
            return False, f"Upload failed: {e}"

        dst = self._dst_path(src, dst_name)
        if dst.exists() and not overwrite:
            return False, f"Upload blocked: '{dst.name}' already exists."

        try:
            shutil.copy2(src, dst)
            return True, f"Upload success{' (overwritten)' if overwrite else ''}: {dst.name}"
        except Exception as e:
            return False, f"Upload failed: {e}"
    def delete_from_name(self, video_name: str) -> bool:
        """Delete the specified video."""
        video_path = self.dir_path / video_name
        if video_path.exists() and video_path.is_file():
            video_path.unlink()
            print(f"[Clean] Deleted uploaded: {video_path.name}")
            return True
        print(f"[Clean] Video not found: {video_path.name}")
        return False
    def delete_all(self) -> int:
        """Delete all mp4 videos in the directory."""
        deleted_count = 0
        for video_path in self.dir_path.glob("*.mp4"):
            if video_path.exists() and video_path.is_file():
                video_path.unlink()
                deleted_count += 1
                print(f"[Clean] Deleted uploaded: {video_path.name}")
        return deleted_count
def upload_video(
    src_path: str,
    *,
    dst_name: Optional[str] = None,
    overwrite: bool = False,
    videos_root: Path = VIDEOS_ROOT,
) -> Tuple[bool, str]:
    """Thin wrapper (kept for compatibility): upload using VideoStore."""
    store = VideoStore(videos_root)
    return store.upload_from_path(src_path, dst_name=dst_name, overwrite=overwrite)


def list_uploaded_videos(videos_root: Path = VIDEOS_ROOT) -> List[str]:
    store = VideoStore(videos_root)
    return store.list_names()


def clean_one_uploaded(video_name: str, videos_root: Path = VIDEOS_ROOT) -> bool:
    store = VideoStore(videos_root)
    name = (video_name or "").strip()

    if not name:
        return False

    if not name.lower().endswith(".mp4"):
        name += ".mp4"

    return store.delete(name)

def clean_all_uploaded(videos_root: Path = VIDEOS_ROOT) -> int:
    store = VideoStore(videos_root)
    deleted_count = store.delete_all()  # Call the delete_all method
    print(f"[Clean] Deleted {deleted_count} uploaded videos under: {store.dir_path}")
    return deleted_count