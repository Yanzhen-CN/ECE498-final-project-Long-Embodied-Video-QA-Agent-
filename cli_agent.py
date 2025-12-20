#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from agent.video_summary_pipeline import (
    MODES,
    list_analysis_runs,
    summarize_video_for_cli,
    analysis_summary_path,
)
from model.model_interface import init_model, is_model_loaded


# -----------------------------
# Configuration
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR
UPLOAD_DIR = ROOT_DIR / "data" / "videos"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PRELOAD_MODEL_ON_START = True
CACHE_ONLY = True  # offline/cache-only
MODEL_NAME = "OpenGVLab/InternVL3_5-4B"


# If CACHE_ONLY, force offline (prevents accidental remote-code download)
if CACHE_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


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
                "       Populate model/hf_cache/ on a machine that can reach HuggingFace, then copy it here."
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
    print("( 2) Free mode")
    return _safe_input("Select: ").strip()


def free_menu() -> str:
    print("\n========== Free Mode ==========")
    print("(-1) Exit")
    print("( 0) Back")
    print("( 1) Upload video (.mp4)")
    print("( 2) List uploaded videos")
    print("( 3) Analyze a video (run summary pipeline)")
    print("( 4) List analyzed runs (video + mode)")
    print("( 5) QA on analyzed run")
    return _safe_input("Select: ").strip()


def analysis_mode_menu() -> str:
    print("\n========== Analysis Mode ==========")
    print("(-1) Exit")
    print("( 0) Back")
    print("( 1) Fast     (chunk=45s, frames=6)")
    print("( 2) Standard (chunk=30s, frames=6)")
    print("( 3) Detailed (chunk=15s, frames=6)")
    return _safe_input("Select: ").strip()


def qa_menu() -> str:
    print("\n========== QA ==========")
    print("(-1) Exit")
    print("( 0) Back")
    print("( 1) Ask a question")
    return _safe_input("Select: ").strip()


# -----------------------------
# Hooks (placeholder)
# -----------------------------
def answer_question(context: str, question: str) -> str:
    return (
        "[TODO] Replace answer_question() with your QA module.\n"
        f"Question: {question}\n"
        f"Context (summary, truncated): {context[:600]}..."
    )


# -----------------------------
# Helpers
# -----------------------------
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


def _print_uploaded(store: VideoStore) -> None:
    names = store.list_names()
    if not names:
        print("No uploaded videos yet.")
        return
    print("\nUploaded videos:")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")


def _print_analyzed() -> list[dict]:
    runs = list_analysis_runs()
    if not runs:
        print("No analyzed runs yet.")
        return []
    print("\nAnalyzed runs:")
    for i, r in enumerate(runs):
        print(f"  [{i}] {r.get('video_name')} | mode={r.get('mode')} | run_id={r.get('run_id')}")
    return runs


def _run_analysis(video_path: Path, mode: str) -> Optional[str]:
    print(f"\nRunning summary pipeline... mode='{MODES[mode].name}'")
    try:
        context = summarize_video_for_cli(
            str(video_path),
            mode=mode,
            local_files_only=CACHE_ONLY,
        )
        print("[OK] Analysis done. Summary saved.")
        return context
    except RuntimeError as e:
        msg = str(e)
        print(f"[ERR] Summary pipeline failed: {msg}")
        if "CUDA out of memory" in msg:
            print("[Hint] OOM: try mode=Fast first (or reduce max_new_tokens / max_num).")
        return None
    except Exception as e:
        print(f"[ERR] Summary pipeline failed: {e}")
        return None


# -----------------------------
# Flows
# -----------------------------
def run_test_mode() -> None:
    print("\n[Test mode] Placeholder (TODO): run preset video + 3 modes + compare QA quality.")
    print("Returning to main menu...")


def run_qa_loop(context: str) -> None:
    while True:
        c = qa_menu()
        if _is_exit(c):
            raise SystemExit(0)
        if c == "0":
            return
        if c == "1":
            q = _safe_input("Your question (-1 to exit): ").strip()
            if _is_exit(q):
                raise SystemExit(0)
            resp = answer_question(context, q)
            print("\n----- Agent Response -----")
            print(resp)
            print("--------------------------")
        else:
            print("Invalid option. Please choose 0/1/-1.")


def run_free_mode(store: VideoStore) -> None:
    while True:
        choice = free_menu()
        if _is_exit(choice):
            raise SystemExit(0)
        if choice == "0":
            return

        # (1) upload
        if choice == "1":
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
            if not ok:
                continue

            # Optional: analyze immediately
            ans2 = _safe_input("Analyze this video now? (y/N): ").strip().lower()
            if ans2 == "y":
                mode = _pick_mode()
                if mode is None:
                    continue
                vp = store.get_path_by_name(dst_p.name)
                if vp is None:
                    print("[ERR] Uploaded video not found in store (unexpected).")
                    continue
                _run_analysis(vp, mode)
            continue

        # (2) list uploaded
        if choice == "2":
            _print_uploaded(store)
            continue

        # (3) analyze a video
        if choice == "3":
            _print_uploaded(store)
            name = _safe_input("Enter video name to analyze (0 back, -1 exit): ").strip()
            if _is_exit(name):
                raise SystemExit(0)
            if name == "0":
                continue
            vp = store.get_path_by_name(name)
            if vp is None:
                print("Video not found. Please check the name.")
                continue
            mode = _pick_mode()
            if mode is None:
                continue
            _run_analysis(vp, mode)
            continue

        # (4) list analyzed runs
        if choice == "4":
            _print_analyzed()
            continue

        # (5) QA on analyzed run
        if choice == "5":
            runs = _print_analyzed()
            if not runs:
                continue
            idx_s = _safe_input("Pick run index for QA (0 back, -1 exit): ").strip()
            if _is_exit(idx_s):
                raise SystemExit(0)
            if idx_s == "0":
                continue
            try:
                idx = int(idx_s)
                run = runs[idx]
            except Exception:
                print("Invalid index.")
                continue

            run_id = str(run.get("run_id", ""))
            if not run_id:
                print("[ERR] run_id missing in registry.")
                continue

            sp = analysis_summary_path(run_id)
            if not sp.exists():
                print(f"[ERR] summary.txt not found for run_id={run_id}.")
                continue

            context = sp.read_text(encoding="utf-8")
            print(f"\n[OK] Loaded analyzed context: run_id={run_id}. Entering QA...")
            run_qa_loop(context)
            continue

        print("Invalid option. Please choose 0/1/2/3/4/5/-1.")


def main() -> None:
    store = VideoStore(UPLOAD_DIR)
    _preload_model_or_warn()

    while True:
        c = main_menu()
        if _is_exit(c):
            break
        if c == "1":
            run_test_mode()
        elif c == "2":
            run_free_mode(store)
        else:
            print("Invalid option. Please choose 1/2/-1.")

    print("Bye.")


if __name__ == "__main__":
    main()
