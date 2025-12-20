#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from agent.video_summary_pipeline import summarize_video_for_cli, MODES
from agent.analysis_store import has_summary, load_summary
from model.model_interface import init_model, is_model_loaded


# -----------------------------
# Paths / Config
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR
UPLOAD_DIR = ROOT_DIR / "data" / "videos"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PRELOAD_MODEL_ON_START = True
CACHE_ONLY = True              # True: offline/cache-only
MODEL_NAME = "OpenGVLab/InternVL3_5-4B"


# -----------------------------
# State
# -----------------------------
active_video_name: Optional[str] = None  # e.g., "xxx.mp4"
active_mode: str = "standard"            # fast/standard/detailed


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
    print("( 1) Video manager (upload/list/set active)")
    print("( 2) Analyze / QA (manual)")
    print("( 3) Test demo (auto run all modes)")
    return _safe_input("Select: ").strip()


def video_menu() -> str:
    global active_video_name
    print("\n========== Video Manager ==========")
    print(f"Active video: {active_video_name if active_video_name else '(none)'}")
    print("(-1) Exit")
    print("( 0) Back")
    print("( 1) List uploaded videos")
    print("( 2) Upload video (.mp4)")
    print("( 3) Set active video")
    return _safe_input("Select: ").strip()


def manual_menu() -> str:
    global active_video_name, active_mode
    print("\n========== Manual Analyze / QA ==========")
    print(f"Active video: {active_video_name if active_video_name else '(none)'} | mode={active_mode}")
    print("(-1) Exit")
    print("( 0) Back")
    print("( 1) Analyze now (choose mode)")
    print("( 2) QA with cached summary (choose mode)")
    print("( 3) Switch active video")
    return _safe_input("Select: ").strip()


def mode_menu() -> str:
    print("\n========== Analysis Mode ==========")
    print("(-1) Exit")
    print("( 0) Back")
    print("( 1) fast     (chunk=60s, frames_per_chunk=6)")
    print("( 2) standard (chunk=30s, frames_per_chunk=6, two-pass 3+3)")
    print("( 3) detailed (chunk=15s, frames_per_chunk=6, two-pass 3+3)")
    return _safe_input("Select: ").strip()


def qa_menu() -> str:
    print("\n========== QA ==========")
    print("(-1) Exit")
    print("( 0) Back")
    print("( 1) Ask a question")
    return _safe_input("Select: ").strip()


# -----------------------------
# Hooks
# -----------------------------
def answer_question(context: str, question: str) -> str:
    # TODO: replace with your real QA module (memory retrieval etc.)
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
        c = mode_menu()
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


def _pick_video_interactively(store: VideoStore) -> Optional[str]:
    names = store.list_names()
    if not names:
        print("No uploaded videos yet. Please upload one first.")
        return None

    print("\nUploaded videos:")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")

    name = _safe_input("Enter video name exactly (0 back, -1 exit): ").strip()
    if _is_exit(name):
        raise SystemExit(0)
    if name == "0":
        return None
    if store.get_path_by_name(name) is None:
        print("Video not found. Please check the name.")
        return None
    return name


def _print_cache_status(video_id: str) -> None:
    print("\n[Cache] Summary availability:")
    for k, mc in MODES.items():
        ok = has_summary(ROOT_DIR, video_id, k)
        print(f"  - {k:<8} ({mc.name:<8}): {'YES' if ok else 'NO'}")


# -----------------------------
# Flows
# -----------------------------
def run_video_manager(store: VideoStore) -> None:
    global active_video_name

    while True:
        c = video_menu()
        if _is_exit(c):
            raise SystemExit(0)
        if c == "0":
            return

        if c == "1":
            names = store.list_names()
            if not names:
                print("No uploaded videos yet.")
            else:
                print("\nUploaded videos:")
                for i, n in enumerate(names):
                    print(f"  [{i}] {n}")
            continue

        if c == "2":
            src = _safe_input("Enter video file path (.mp4) (0 back, -1 exit): ").strip()
            if _is_exit(src):
                raise SystemExit(0)
            if src == "0":
                continue

            new_name = _safe_input("Save as (Enter keep original) (0 back, -1 exit): ").strip()
            if _is_exit(new_name):
                raise SystemExit(0)
            if new_name == "0":
                continue
            if new_name == "":
                new_name = None

            src_p = Path(src).expanduser().resolve()
            dst_p = store._dst_path(src_p, new_name)
            if dst_p.exists():
                ans = _safe_input(f"'{dst_p.name}' exists. Overwrite? (y/N): ").strip().lower()
                if ans != "y":
                    print("Upload cancelled.")
                    continue
                ok, msg = store.upload_from_path(src, dst_name=new_name, overwrite=True)
            else:
                ok, msg = store.upload_from_path(src, dst_name=new_name, overwrite=False)

            print(msg)
            if ok and active_video_name is None:
                # auto-set active for first upload
                active_video_name = dst_p.name
                print(f"[Info] Active video set to: {active_video_name}")
            continue

        if c == "3":
            picked = _pick_video_interactively(store)
            if picked:
                active_video_name = picked
                print(f"[Info] Active video set to: {active_video_name}")
                _print_cache_status(Path(active_video_name).stem)
            continue

        print("Invalid option. Please choose 0/1/2/3/-1.")


def run_qa_loop(context: str) -> None:
    while True:
        c = qa_menu()
        if _is_exit(c):
            raise SystemExit(0)
        if c == "0":
            return
        if c == "1":
            q = _safe_input("Your question (-1 exit): ").strip()
            if _is_exit(q):
                raise SystemExit(0)
            resp = answer_question(context, q)
            print("\n----- Agent Response -----")
            print(resp)
            print("--------------------------")
            continue

        print("Invalid option. Please choose 0/1/-1.")


def run_manual(store: VideoStore) -> None:
    global active_video_name, active_mode

    while True:
        c = manual_menu()
        if _is_exit(c):
            raise SystemExit(0)
        if c == "0":
            return

        if c == "3":
            picked = _pick_video_interactively(store)
            if picked:
                active_video_name = picked
                print(f"[Info] Active video set to: {active_video_name}")
                _print_cache_status(Path(active_video_name).stem)
            continue

        if active_video_name is None:
            print("[Hint] No active video. Please set active video first.")
            continue

        vpath = store.get_path_by_name(active_video_name)
        if vpath is None:
            print("[ERR] Active video missing on disk. Please re-select.")
            active_video_name = None
            continue

        if c == "1":
            mode = _pick_mode()
            if mode is None:
                continue
            active_mode = mode

            print(f"\nRunning analysis... video='{active_video_name}', mode='{mode}'")
            try:
                out = summarize_video_for_cli(
                    str(vpath),
                    mode=mode,
                    project_root=ROOT_DIR,
                    local_files_only=CACHE_ONLY,
                    overwrite_slice=True,
                    reuse_if_cached=False,  # force re-run when user selects Analyze now
                )
                print("\n[OK] Analysis done. Cached summary saved.")
                _print_cache_status(vpath.stem)
                # optional: print result
                print("\n----- Summary -----")
                print(out)
                print("-------------------")
            except Exception as e:
                print(f"[ERR] Analysis failed: {e}")
            continue

        if c == "2":
            mode = _pick_mode()
            if mode is None:
                continue
            active_mode = mode

            vid = vpath.stem
            ctx = load_summary(ROOT_DIR, vid, mode)
            if ctx is None:
                print(f"[Hint] No cached summary for mode='{mode}'. Run Analyze now first.")
                _print_cache_status(vid)
                continue

            print(f"\n[OK] Using cached summary. Entering QA... video='{active_video_name}', mode='{mode}'")
            run_qa_loop(ctx)
            continue

        print("Invalid option. Please choose 0/1/2/3/-1.")


def run_test_demo(store: VideoStore) -> None:
    global active_video_name

    print("\n========== Test Demo ==========")
    picked = _pick_video_interactively(store)
    if not picked:
        return
    active_video_name = picked

    vpath = store.get_path_by_name(active_video_name)
    assert vpath is not None

    print(f"\n[Test] Running all modes for video='{active_video_name}' ...")
    for mode in ["fast", "standard", "detailed"]:
        print(f"\n--- Mode: {mode} ({MODES[mode].name}) ---")
        try:
            out = summarize_video_for_cli(
                str(vpath),
                mode=mode,
                project_root=ROOT_DIR,
                local_files_only=CACHE_ONLY,
                overwrite_slice=True,
                reuse_if_cached=False,  # demo wants fresh run
            )
            print(out)
        except Exception as e:
            print(f"[ERR] Mode '{mode}' failed: {e}")

    print("\n[Test] Done. Cache status:")
    _print_cache_status(vpath.stem)


def main() -> None:
    store = VideoStore(UPLOAD_DIR)

    _preload_model_or_warn()

    while True:
        c = main_menu()
        if _is_exit(c):
            break
        if c == "1":
            run_video_manager(store)
        elif c == "2":
            run_manual(store)
        elif c == "3":
            run_test_demo(store)
        else:
            print("Invalid option. Please choose 1/2/3/-1.")

    print("Bye.")


if __name__ == "__main__":
    main()
