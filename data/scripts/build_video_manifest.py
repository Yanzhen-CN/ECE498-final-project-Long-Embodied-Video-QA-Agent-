import argparse, os, json, math, csv
from pathlib import Path
import cv2

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to input mp4")
    ap.add_argument("--video_id", default=None, help="default: filename without suffix")
    ap.add_argument("--data_root", default="data", help="data root directory")
    ap.add_argument("--chunk_seconds", type=int, default=30)
    ap.add_argument("--keyframes_per_chunk", type=int, default=8)
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    video_id = args.video_id or video_path.stem

    data_root = Path(args.data_root)
    videos_dir = data_root / "videos"
    kf_root = data_root / "keyframes" / video_id
    manifests_dir = data_root / "manifests"
    metadata_dir = data_root / "metadata"

    ensure_dir(videos_dir)
    ensure_dir(kf_root)
    ensure_dir(manifests_dir)
    ensure_dir(metadata_dir)

    # (可选) 把视频复制/移动到 data/videos 下：这里不自动移动，避免误操作
    # 你可以手动把视频放到 data/videos/<video_id>.mp4
    rel_video_path = f"data/videos/{video_id}{video_path.suffix}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video with OpenCV")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if frame_count > 0 else 0.0
    if duration_sec <= 0:
        # 兜底：尝试读到结尾估计
        duration_sec = 0.0

    chunk_seconds = args.chunk_seconds
    k = args.keyframes_per_chunk

    n_chunks = max(1, math.ceil(duration_sec / chunk_seconds)) if duration_sec > 0 else 1

    # CSV：每张图一行，作为“标签表”
    csv_path = metadata_dir / f"{video_id}.csv"
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["video_id", "chunk_id", "img_idx", "t_sec", "image_path"])

    manifest = {
        "manifest_version": "1.0",
        "video_id": video_id,
        "video_path": rel_video_path,
        "fps": float(fps),
        "duration_sec": float(round(duration_sec, 3)),
        "chunk_seconds": int(chunk_seconds),
        "keyframes_per_chunk": int(k),
        "chunks": []
    }

    # 质量参数
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    def save_frame_at_time(t_sec: float, out_path: Path):
        # 定位到目标帧
        frame_idx = int(round(t_sec * fps))
        frame_idx = max(0, min(frame_idx, frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return False
        ok2 = cv2.imwrite(str(out_path), frame, encode_params)
        return ok2

    for ci in range(n_chunks):
        t_start = ci * chunk_seconds
        t_end = min((ci + 1) * chunk_seconds, duration_sec) if duration_sec > 0 else (ci + 1) * chunk_seconds

        chunk_dir = kf_root / f"chunk_{ci:03d}"
        ensure_dir(chunk_dir)

        # 在 [t_start, t_end) 内均匀采样 k 个时间点（避开边界）
        # 用中心采样：((j+0.5)/k)
        chunk_images = []
        for j in range(k):
            alpha = (j + 0.5) / k
            t = t_start + alpha * max(0.001, (t_end - t_start))
            img_name = f"img_{j:02d}.jpg"
            img_path = chunk_dir / img_name

            ok = save_frame_at_time(t, img_path)
            if not ok:
                continue

            rel_img = f"data/keyframes/{video_id}/chunk_{ci:03d}/{img_name}"
            chunk_images.append(rel_img)
            csv_w.writerow([video_id, ci, j, round(t, 3), rel_img])

        manifest["chunks"].append({
            "chunk_id": ci,
            "t_start": int(round(t_start)),
            "t_end": int(round(t_end)),
            "chunk_images": chunk_images
        })

    csv_f.close()
    cap.release()

    manifest_path = manifests_dir / f"{video_id}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] labels(csv): {csv_path}")
    print(f"[OK] keyframes dir: {kf_root}")
    print(f"[INFO] duration_sec={duration_sec:.2f}, chunks={n_chunks}, total_imgs≈{n_chunks*k}")

if __name__ == "__main__":
    main()
