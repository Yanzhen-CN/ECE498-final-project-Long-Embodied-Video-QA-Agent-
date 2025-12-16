from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.data_to_model import build_jobs_from_manifest


def save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_outer_json(text: str) -> Optional[str]:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        return s[l:r+1]
    return None


# --------- model call stub (replace later) ----------
def model_generate(images: List[str], prompt: str) -> str:
    """
    TODO: Replace with InternVL call.
    Must return a JSON string with keys:
      chunk_id, t_start, t_end, summary, entities, events, state_update
    """
    dummy = {
        "chunk_id": -1,
        "t_start": -1,
        "t_end": -1,
        "summary": "TODO: replace with real model output.",
        "entities": [],
        "events": [],
        "state_update": {}
    }
    return json.dumps(dummy, ensure_ascii=False)


def run_stage_a_build_memory(
    data_json: Path,
    out_memory_json: Path,
    use_prev_summary: bool = True,
) -> None:
    """
    Reads data.json (VideoManifest), runs model per chunk, writes memory.json:
      {
        "memory_version": "1.0",
        "video_id": "...",
        "chunks": [ChunkMemory, ...]
      }
    """
    jobs = build_jobs_from_manifest(data_json=data_json, use_prev_summary=use_prev_summary)
    if not jobs:
        raise ValueError("No chunks/jobs found in data.json")

    video_id = jobs[0]["video_id"]
    out: Dict[str, Any] = {
        "memory_version": "1.0",
        "video_id": video_id,
        "chunks": []
    }

    prev_summary = ""

    for job in jobs:
        # 如果你们“只回输上一段 summary”，就在这里替换 prompt 中的 prev_summary
        # 简单做法：重新构造 prompt（避免搞字符串 replace 的坑）
        # 这里为了简洁：直接把 job["prompt"] 里的 placeholder用 format 时就已填了 prev_summary，
        # 但 prev_summary 在 build_jobs_from_manifest 里尚未更新，所以我们这里重建一次 prompt 更严谨。
        # （不用担心性能，字符串而已）
        images = job["images"]
        prompt = job["prompt"]
        if use_prev_summary:
            # 把 prompt 里 "Previous chunk summary..." 后面的内容替换成当前 prev_summary
            # 更稳的方式：你可以在 data_to_model 里暴露 make_memory_prompt 再在这里调用
            # 这里用简单分割，避免误伤其它字段
            marker = "Previous chunk summary (for consistency):\n"
            if marker in prompt:
                before, after = prompt.split(marker, 1)
                # after starts with old prev_summary + "\n\nOutput ONLY..."
                tail_marker = "\n\nOutput ONLY valid JSON"
                if tail_marker in after:
                    _, tail = after.split(tail_marker, 1)
                    prompt = before + marker + (prev_summary or "") + "\n\nOutput ONLY valid JSON" + tail

        raw = model_generate(images=images, prompt=prompt)

        parsed: Dict[str, Any]
        js = extract_outer_json(raw)
        if js is None:
            parsed = {"parse_error": True, "raw_model_output": raw}
        else:
            try:
                parsed = json.loads(js)
                parsed["parse_error"] = False
            except Exception:
                parsed = {"parse_error": True, "raw_model_output": raw}

        # 强制覆盖：以 data.json 为真
        parsed["video_id"] = video_id
        parsed["chunk_id"] = int(job["chunk_id"])
        parsed["t_start"] = int(job["t_start"])
        parsed["t_end"] = int(job["t_end"])

        # 保证字段存在
        parsed.setdefault("summary", "")
        parsed.setdefault("entities", [])
        parsed.setdefault("events", [])
        parsed.setdefault("state_update", {})

        # 证据帧：agent 选 1-2 张保存下来
        parsed["evidence_frames"] = images[:2]

        out["chunks"].append(parsed)

        # 更新 prev_summary（下一段回输用）
        if isinstance(parsed.get("summary"), str) and parsed["summary"].strip():
            prev_summary = parsed["summary"].strip()

    save_json(out_memory_json, out)
    print(f"[OK] wrote memory to: {out_memory_json}")


if __name__ == "__main__":
    # minimal CLI
    data_json = Path("agent/data.json")
    out_memory_json = Path("agent/memory.json")
    run_stage_a_build_memory(data_json=data_json, out_memory_json=out_memory_json, use_prev_summary=True)
