#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cli_agent.py

Terminal CLI agent:
- Main:
    -1: exit
     1: test mode (placeholder)
     2: free QA mode

- Free QA mode:
    -1: exit
     0: back to main mode selection
     1: list uploaded videos (names)
     2: upload video (input path; must exist and be .mp4)

- If choose list uploaded videos:
    0: back
    1: input a video name -> run summary pipeline -> then enter QA loop

- QA loop:
    -1: exit
     0: back to main mode selection
     1: ask a question (repeat)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# ✅ Real pipeline entry (you created this file)
# e.g., pipeline/video_summary_pipeline.py contains summarize_video_for_cli()
from agent.video_summary_pipeline import summarize_video_for_cli


# -----------------------------
# Configuration
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploaded_videos"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Integration hooks
# -----------------------------
def summary_pipeline(video_path: str) -> str:
    """
    Real summary pipeline.
    Input: local path to uploaded .mp4
    Output: CLI-friendly summary string
    """
    return summarize_video_for_cli(
        video_path,
        use_prev_summary=True,
        evidence_per_chunk=2,
    )


def answer_question(context: str, question: str) -> str:
    """
    QA hook.
    For now, we use `context` (summary text) as input.
    Later you can replace this with memory retrieval + model answer, e.g.:
      - memory_query(video_id, question) -> evidence
      - model_interface(evidence_images, prompt) -> answer
    """
    # Placeholder behavior (kept simple & predictable):
    return (
        "[TODO] Replace answer_question() with your QA module.\n"
        f"Question: {question}\n"
        f"Context (summary, truncated): {context[:400]}..."
    )


# -----------------------------
# Utilities
# -----------------------------
def _is_exit(s: str) -> bool:
    return s.strip() == "-1"


def _safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return "-1"


def _is_mp4(path: Path) -> bool:
    return path.suffix.lower() == ".mp4"


def _dedup_name(dst_dir: Path, filename: str) -> Path:
    """If filename exists, append _1, _2, ... before suffix."""
    candidate = dst_dir / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    i = 1
    while True:
        candidate = dst_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


# -----------------------------
# Video store
# -----------------------------
@dataclass
class VideoStore:
    """Simple local store: uploaded videos are copied into UPLOAD_DIR."""
    dir_path: Path

    def list_names(self) -> list[str]:
        files = sorted([p for p in self.dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])
        return [p.name for p in files]

    def get_path_by_name(self, name: str) -> Optional[Path]:
        p = self.dir_path / name
        if p.exists() and p.is_file() and p.suffix.lower() == ".mp4":
            return p
        return None

    def upload_from_path(self, src_path: str) -> Tuple[bool, str]:
        """Copy src video into upload dir. Returns (success, message)."""
        src = Path(src_path).expanduser().resolve()
        if not src.exists() or not src.is_file():
            return False, f"Upload failed: file not found: {src}"

        if not _is_mp4(src):
            return False, f"Upload failed: only .mp4 is supported, got: {src.suffix}"

        dst = _dedup_name(self.dir_path, src.name)
        try:
            print("Uploading... (copying file to local upload dir)")
            shutil.copy2(src, dst)
            return True, f"Upload success: {dst.name}"
        except Exception as e:
            return False, f"Upload failed: {e}"


# -----------------------------
# Menus
# -----------------------------
def main_menu() -> str:
    print("\n========== CLI Agent ==========")
    print("(-1) Exit")
    print("( 1) Test mode")
    print("( 2) Free QA mode")
    return _safe_input("Select: ").strip()


def free_mode_menu() -> str:
    print("\n========== Free QA Mode ==========")
    print("(-1) Exit")
    print("( 0) Back to mode selection")
    print("( 1) List uploaded videos (names)")
    print("( 2) Upload video (.mp4)")
    return _safe_input("Select: ").strip()


def list_videos_menu() -> str:
    print("\n========== Uploaded Videos ==========")
    print("( 0) Back")
    print("( 1) Choose a video by name to run summary -> QA")
    print("(-1) Exit")
    return _safe_input("Select: ").strip()


def qa_menu() -> str:
    print("\n========== QA ==========")
    print("(-1) Exit")
    print("( 0) Back to mode selection")
    print("( 1) Ask a question")
    return _safe_input("Select: ").strip()


# -----------------------------
# Flows
# -----------------------------
def run_test_mode() -> None:
    print("\n[Test mode] Placeholder (TODO): run preset video + question demo.")
    print("Returning to main menu...")


def run_qa_loop(context: str) -> None:
    while True:
        choice = qa_menu()
        if _is_exit(choice):
            raise SystemExit(0)

        if choice == "0":
            return

        if choice == "1":
            q = _safe_input("Your question (-1 to exit): ").strip()
            if _is_exit(q):
                raise SystemExit(0)

            resp = answer_question(context, q)
            print("\n----- Agent Response -----")
            print(resp)
            print("--------------------------")
        else:
            print("Invalid option. Please choose 0/1/-1.")


def run_list_and_select_video(store: VideoStore) -> None:
    names = store.list_names()
    if not names:
        print("No uploaded videos yet. Please upload one first.")
        return

    print("\nUploaded videos:")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")

    while True:
        choice = list_videos_menu()
        if _is_exit(choice):
            raise SystemExit(0)

        if choice == "0":
            return

        if choice == "1":
            name = _safe_input("Enter video name exactly (0 back, -1 exit): ").strip()
            if _is_exit(name):
                raise SystemExit(0)
            if name == "0":
                continue

            path = store.get_path_by_name(name)
            if path is None:
                print("Video not found. Please check the name and try again.")
                continue

            print("\nRunning summary pipeline...")
            try:
                context = summary_pipeline(str(path))
            except Exception as e:
                print(f"Summary pipeline failed: {e}")
                continue

            print("\nSummary ready. Entering QA...")
            run_qa_loop(context)
            return

        print("Invalid option. Please choose 0/1/-1.")


def run_free_mode(store: VideoStore) -> None:
    while True:
        choice = free_mode_menu()
        if _is_exit(choice):
            raise SystemExit(0)

        if choice == "0":
            return

        if choice == "1":
            run_list_and_select_video(store)
            continue

        if choice == "2":
            src = _safe_input("Enter video file path (.mp4) (0 back, -1 exit): ").strip()
            if _is_exit(src):
                raise SystemExit(0)
            if src == "0":
                continue

            ok, msg = store.upload_from_path(src)
            print(msg)
            continue

        print("Invalid option. Please choose 0/1/2/-1.")


def main() -> None:
    store = VideoStore(UPLOAD_DIR)

    while True:
        choice = main_menu()
        if _is_exit(choice):
            break

        if choice == "1":
            run_test_mode()
        elif choice == "2":
            run_free_mode(store)
        else:
            print("Invalid option. Please choose 1/2/-1.")

    print("Bye.")


if __name__ == "__main__":
    main()
