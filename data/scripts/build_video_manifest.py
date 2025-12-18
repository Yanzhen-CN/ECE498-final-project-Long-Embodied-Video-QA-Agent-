import argparse
from data.scripts.video_slicer import slice_video

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--video_id", default=None)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--chunk_seconds", type=int, default=30)
    ap.add_argument("--keyframes_per_chunk", type=int, default=8)
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args()

    manifest_path, manifest = slice_video(
        args.video,
        video_id=args.video_id,
        data_root=args.data_root,
        chunk_seconds=args.chunk_seconds,
        keyframes_per_chunk=args.keyframes_per_chunk,
        jpeg_quality=args.jpeg_quality,
    )

    print(f"[OK] manifest: {manifest_path}")
    print(f"[INFO] video_id={manifest['video_id']} duration={manifest['duration_sec']} chunks={len(manifest['chunks'])}")

if __name__ == "__main__":
    main()
