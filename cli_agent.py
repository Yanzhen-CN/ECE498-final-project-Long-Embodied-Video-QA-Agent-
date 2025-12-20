#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from agent.video_summary_pipeline import summarize_video_for_cli, MODES
from model.model_interface import init_model, is_model_loaded


# -----------------------------
# Configuration
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR
UPLOAD_DIR = ROOT_DIR / "data" / "videos"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Model preload behavior
PRELOAD_MODEL_ON_START = True
CACHE_ONLY = True  # True: offline/cache-only
MODEL_NAME = "OpenGVLab/InternVL3_5-4B"


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


def _preload_model_or_warn() -> None:
    if not PRELOAD_MODEL_ON_START:
        return
    if is_model_loaded():
        return

    print("\n[Model] Preloading model at agent startup...")
    try:
        init_model(model_name=MODEL_NAME, local_files_only=CACHE_ONLY)
        print("[Model] Ready.")
    except Exception as e:
        print(f"[Model] Preload failed: {e}")
        if CACHE_ONLY:
            print(
                "[Hint] CACHE_ONLY=True but cache may be incomplete.\n"
                "       Run once with CACHE_ONLY=False on a machine that can reach HuggingFace to populate model/hf_cache/."
            )


# -----------------------------
# Video store
# -----------------------------
@dataclass
class VideoStore:
    dir_path: Path

    def list_names(self) -> list[str]:
        files = sorted([p for p in self.dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])
        return [p.name for p in files]

    def get_path_by_name(self, name: str) -> Optional[Path]:
        p = self.dir_path / name
        if p.exists() and p.is_file() and p.suffix.lower() == ".mp4":
            return p
        return None

    def _dst_path(self, src: Path, dst_name: Optional[str]) -> Path:
        if dst_name is not None and dst_name.strip():
            name = Path(dst_name.strip()).name
            if not name.lower().endswith(".mp4"):
                name += ".mp4"
            return self.dir_path / name
        return self.dir_path / src.name

    def upload_from_path(
        self,
        src_path: str,
        dst_name: Optional[str] = None,
        *,
        overwrite: bool = False,
    ) -> Tuple[bool, str]:
        src = Path(src_path).expanduser().resolve()
        if not src.exists() or not src.is_file():
            return False, f"Upload failed: file not found: {src}"
        if not _is_mp4(src):
            return False, f"Upload failed: only .mp4 is supported, got: {src.suffix}"

        dst = self._dst_path(src, dst_name)
        if dst.exists() and not overwrite:
            return False, f"Upload blocked: '{dst.name}' already exists."

        try:
            print("Uploading... (copying file to local upload dir)")
            shutil.copy2(src, dst)
            return True, f"Upload success{' (overwritten)' if overwrite else ''}: {dst.name}"
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


def analysis_mode_menu() -> str:
    print("\n========== Analysis Mode ==========")
    print("(-1) Exit")
    print("( 0) Back")
    print("( 1) Fast     (chunk=60s, frames=4, single-pass)")
    print("( 2) Standard (chunk=30s, frames=8, TWO-pass 4+4 to avoid OOM)")
    print("( 3) Detailed (chunk=15s, frames=8, TWO-pass 4+4)")
    return _safe_input("Select: ").strip()


def qa_menu() -> str:
    print("\n========== QA ==========")
    print("(-1) Exit")
    print("( 0) Back to mode selection")
    print("( 1) Ask a question")
    return _safe_input("Select: ").strip()


# -----------------------------
# Hooks
# -----------------------------
def answer_question(context: str, question: str) -> str:
    return (
        "[TODO] Replace answer_question() with your QA module.\n"
        f"Question: {question}\n"
        f"Context (summary, truncated): {context[:400]}..."
    )


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


def _pick_mode() -> Optional[str]:
    while True:
        c = analysis_mode_menu()
        if _is_exit(c):
            raise SystemExit(0)
        if c == "0":
            return None
        if c == "1":
            return "fast"
        if c == "2":
            return "standard"
        if c == "3":
            return "detailed"
        print("Invalid option. Please choose 0/1/2/3/-1.")


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

            mode = _pick_mode()
            if mode is None:
                continue

            print(f"\nRunning summary pipeline... mode='{MODES[mode].name}'")
            try:
                context = summarize_video_for_cli(str(path), mode=mode, local_files_only=CACHE_ONLY)
            except RuntimeError as e:
                msg = str(e)
                print(f"Summary pipeline failed: {msg}")
                if "CUDA out of memory" in msg:
                    print("[Hint] OOM: try mode=Fast first, or keep Standard (two-pass) but ensure max_num=2 + thumbnail off.")
                return
            except Exception as e:
                print(f"Summary pipeline failed: {e}")
                return

            print("\n[OK] Video analyzed. Entering QA...")
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

            new_name = _safe_input("Save as (Enter to keep original name) (0 back, -1 exit): ").strip()
            if _is_exit(new_name):
                raise SystemExit(0)
            if new_name == "0":
                continue
            if new_name == "":
                new_name = None

            src_p = Path(src).expanduser().resolve()
            dst_p = store._dst_path(src_p, new_name)
            if dst_p.exists():
                ans = _safe_input(f"'{dst_p.name}' already exists. Overwrite? (y/N): ").strip().lower()
                if ans != "y":
                    print("Upload cancelled (no overwrite).")
                    continue
                ok, msg = store.upload_from_path(src, dst_name=new_name, overwrite=True)
            else:
                ok, msg = store.upload_from_path(src, dst_name=new_name, overwrite=False)

            print(msg)
            continue

        print("Invalid option. Please choose 0/1/2/-1.")


def main() -> None:
    store = VideoStore(UPLOAD_DIR)

    _preload_model_or_warn()

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
