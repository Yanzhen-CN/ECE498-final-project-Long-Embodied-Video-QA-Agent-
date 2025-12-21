#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
import json
import re
from agent.video_summary_pipeline import summarize_video_for_cli
from agent.address_questions_evaluation import run_qa_system  
from model.model_interface import model_interface

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from data.video_interface import clean_processed_video
from memory.memory_interface import clean_saved_memory
from agent.video_management import list_analysis_runs, check_analysis_runs, delete_analysis_run, clear_all_analysis_runs
from agent.address_questions_evaluation import run_qa_system
from agent.video_summary_pipeline import MODES,summarize_video_for_cli
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
    print("( 4) List analyzed videos (video + mode)")
    print("( 5) QA on analyzed video")
    print("( 6) Clean analyzed videos")
    return _safe_input("Select: ").strip()


def analysis_mode_menu() -> str:
    print("\n========== Analysis Mode ==========")
    print("(-1) Exit")
    print("( 0) Back")
    print("( 1) Fast     (chunk=60s, frames=6)")
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

    #TODO: integrate your existing model inference function here(model_inference_fn)

    response = run_qa_system(
        mode="interactive",
        model_inference_fn=model_inference_fn,
        context=context,
        question=question
    )
    return response


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
        print(f"  [{i}] {r.get('video_name')} | mode={r.get('mode')} | video_id={r.get('video_id')}")
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

# -----------------------------
# Test Mode Logic
# -----------------------------
def _load_question_bank(json_path: str) -> list:
    # this is a help function for test mode
    """Helper: Load QA benchmark from JSON."""
    path = Path(json_path)
    if not path.exists():
        print(f"[ERR] Question bank file not found: {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 简单校验数据格式
            if isinstance(data, list) and "question" in data[0] and "ground_truth" in data[0]:
                return data
            else:
                print("[ERR] JSON format invalid. Expected a list of dicts with 'question' and 'ground_truth'.")
                return []
    except Exception as e:
        print(f"[ERR] Failed to load JSON: {e}")
        return []

def _evaluate_context_accuracy(context: str, question_bank: list) -> float:
    # a help function for test mode
    """
    针对生成的 Summary Context 跑一遍题库，计算准确率。
    这里临时定义一个 Wrapper，因为 run_qa_system 的 evaluation 模式是针对 memory dict 的，
    而这里我们只有 string context，所以我们模拟 Interactive 模式但自动判题。
    """
    
    # 定义一个适配器，把 Prompt 传给你的底层模型
    def _text_inference_adapter(prompt_text):
        return model_interface(image_paths=[], prompt=prompt_text, cfg=None)

    correct_count = 0
    total = len(question_bank)
    
    print(f"    -> Evaluating {total} questions against generated summary...")

    for i, q_item in enumerate(question_bank):
        # 构造一个专门用于“做选择题”的 Prompt
        options_str = "\n".join([f"{k}. {v}" for k, v in q_item['options'].items()])
        
        prompt = f"""
Video Summary:
{context}

Question: {q_item['question']}

Options:
{options_str}

Instruction: Based on the Video Summary, select the correct option (A, B, C, or D). 
Return ONLY the letter.
"""
        # 调用模型
        response = _text_inference_adapter(prompt)
        
        # 简单的答案提取 (提取 A/B/C/D)
        match = re.search(r'\b([A-D])\b', response.strip().upper())
        pred = match.group(1) if match else "Unknown"
        
        ground_truth = q_item['ground_truth']
        if pred == ground_truth:
            correct_count += 1
        
        # 可选：打印每题详情
        # print(f"       Q{i+1}: Pred={pred} | Truth={ground_truth} | {'✓' if pred==ground_truth else '✗'}")

    return (correct_count / total) * 100 if total > 0 else 0.0


def run_test_mode(store: VideoStore) -> None:
    print("\n========== Test Mode: Fast vs Detailed Comparison ==========")
    print("This mode runs the pipeline twice on the same video to compare speed and accuracy.")
    
    # 1. 选择视频
    _print_uploaded(store)
    vid_name = _safe_input("Enter video name from list (0 back): ").strip()
    if vid_name == "0": return
    
    video_path = store.get_path_by_name(vid_name)
    if not video_path:
        print("[ERR] Video not found.")
        return

    # 2. 输入题库路径 (Ground Truth)
    # 你可以硬编码一个默认路径方便测试
    default_json = ROOT_DIR / "data" / "benchmarks" / "test_questions.json"
    json_input = _safe_input(f"Enter path to Question Bank JSON (Enter for default: {default_json.name}): ").strip()
    
    json_path = json_input if json_input else str(default_json)
    question_bank = _load_question_bank(json_path)
    
    if not question_bank:
        print("[ERR] No questions loaded. Aborting test.")
        return

    # 3. 开始对比测试
    modes_to_test = ["fast", "detailed"]
    results = {}

    print(f"\n[Test] Starting comparison on '{vid_name}'...")

    for mode in modes_to_test:
        print(f"\n--- Running Mode: {mode.upper()} ---")
        
        # A. 计时 - 生成摘要
        t0 = time.time()
        context = summarize_video_for_cli(str(video_path), mode=mode, local_files_only=True)
        t1 = time.time()
        
        if not context:
            print(f"[ERR] Failed to generate summary for {mode}.")
            results[mode] = {"time": 0, "len": 0, "acc": 0.0}
            continue

        analysis_time = t1 - t0
        summary_len = len(context)
        
        # B. 跑分 - 计算准确率
        accuracy = _evaluate_context_accuracy(context, question_bank)
        
        results[mode] = {
            "time": analysis_time,
            "len": summary_len,
            "acc": accuracy
        }
        print(f"--- Finished {mode.upper()}: Time={analysis_time:.1f}s | Acc={accuracy:.1f}% ---")

    # 4. 输出对比表格
    print("\n\n==================== Final Comparison ====================")
    header = f"| {'Mode':<10} | {'Time (s)':<10} | {'Sum Len (char)':<15} | {'Accuracy (%)':<12} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for mode in modes_to_test:
        res = results[mode]
        row = f"| {mode.capitalize():<10} | {res['time']:<10.2f} | {res['len']:<15} | {res['acc']:<12.1f} |"
        print(row)
    print("-" * len(header))
    
    # 简单的分析结论
    fast = results["fast"]
    detailed = results["detailed"]
    
    print("\nAnalysis:")
    if detailed["acc"] > fast["acc"]:
        gain = detailed["acc"] - fast["acc"]
        cost = detailed["time"] - fast["time"]
        print(f"* Detailed mode improved accuracy by +{gain:.1f}% but took {cost:.1f}s longer.")
    else:
        print("* Detailed mode did not improve accuracy in this test case.")
    
    print("==========================================================\n")
    _safe_input("Press Enter to return to menu...")

# functino for free mode QA
def run_qa_loop(context: str) -> None:
    while True:
        c = qa_menu()
        if _is_exit(c):
            raise SystemExit(0)
        if c == "0":
            return
        if c == "1":
            q = _safe_input("Enter your question (-1 to exit): ").strip()
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
            if not name.endswith(".mp4"):
                name += ".mp4"
                print(name)
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

        # (4) list analyzed videos
        if choice == "4":
            _print_analyzed()
            continue

        # (5) QA on analyzed video
        if choice == "5":
            runs = _print_analyzed()
            if not runs:
                continue

            video_id = _safe_input("Enter video_id (video_name__mode) for QA (0 back, -1 exit): ").strip()
            if _is_exit(video_id):
                raise SystemExit(0)
            if video_id == "0":
                continue

            # strict registry check
            if not check_analysis_runs(video_id):
                print()
                continue

            # load context from disk (memory)
            # TODO
            print("TODO: compelete this function")
            '''
            context = load_context_from_memory(video_id, memory_root="memory/saved_videos")
            if not context:
                print(f"[ERR] Context not found on disk for video_id={video_id}. "
                    f"Expected memory at memory/saved_videos/{video_id}/")
                continue

            print(f"\n[OK] Loaded analyzed context: video_id={video_id}. Entering QA...")
            run_qa_loop(context)
            '''
            continue


        # (6) Clean on analyzed run
        if choice == "6":
            sub = _safe_input(
                "Clean analyzed runs:\n"
                "  1) Delete ONE run (by video_id)\n"
                "  2) Clear ALL runs\n"
                "Choose (0 back, -1 exit): "
            )

            if _is_exit(sub):
                raise SystemExit(0)
            if sub == "0":
                continue

            # ---- clear all ----
            if sub == "2":
                n = clear_all_analysis_runs()
                ok_proc = clean_processed_video(None, root="data/processed_videos")
                ok_mem = clean_saved_memory(None, memory_root="memory/saved_videos")
                print(f"[OK] Cleared {n} registry records | processed={ok_proc} | memory={ok_mem}")
                continue

            # ---- delete one ----
            if sub != "1":
                print("Invalid option. Please choose 0/1/2/-1")
                continue
            _print_analyzed()  # give user a copyable list

            video_id = _safe_input("Enter video_id to clean (video_name__mode) (0 back, -1 exit): ")
            if _is_exit(video_id):
                raise SystemExit(0)
            if video_id == "0":
                continue

            # strict registry check
            if not check_analysis_runs(video_id):
                continue

            ok_reg = delete_analysis_run(video_id)
            ok_proc = clean_processed_video(video_id, root="data/processed_videos")
            ok_mem = clean_saved_memory(video_id, memory_root="memory/saved_videos")

            print(f"[OK] Cleaned video_id={video_id} | registry={ok_reg} | processed={ok_proc} | memory={ok_mem}")
            continue


        print("Invalid option. Please choose 0/1/2/3/4/5/-1.")


def main() -> None:
    store = VideoStore(UPLOAD_DIR)
    #_preload_model_or_warn()

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