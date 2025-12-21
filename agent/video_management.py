import json
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = BASE_DIR / "registry.json"


def _migrate_if_needed(reg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Old:
      {"runs": [ {"video_name":..., "mode":..., "video_id":...}, ... ]}
    New:
      {"runs": { video_id: run_dict, ... }, "order": [video_id,...]}
    """
    runs = reg.get("runs")

    # already new format
    if isinstance(runs, dict):
        if "order" not in reg or not isinstance(reg["order"], list):
            reg["order"] = list(runs.keys())
        return reg

    # old format (list) -> migrate
    if isinstance(runs, list):
        new_runs: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for r in runs:
            if not isinstance(r, dict):
                continue
            vid = r.get("video_id")
            if not vid:
                continue
            # last one wins if duplicates exist
            new_runs[vid] = r
            if vid in order:
                order.remove(vid)
            order.append(vid)
        return {"runs": new_runs, "order": order}

    # empty / corrupted -> reset
    return {"runs": {}, "order": []}


def _load_registry() -> Dict[str, Any]:
    """Load analysis registry (auto-migrate if needed)."""
    if not REGISTRY_PATH.exists():
        return {"runs": {}, "order": []}
    try:
        reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if not isinstance(reg, dict):
            return {"runs": {}, "order": []}
        reg = _migrate_if_needed(reg)
        return reg
    except Exception:
        return {"runs": {}, "order": []}


def _save_registry(reg: Dict[str, Any]) -> None:
    """Save analysis registry."""
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def register_analysis_run(video_name: str, mode: str, video_id: str | None = None) -> None:
    reg = _load_registry()

    video_name = Path(video_name).stem
    mode = (mode or "").strip().lower()

    if not video_id:
        video_id = f"{video_name}__{mode}"

    runs: Dict[str, Dict[str, Any]] = reg.get("runs", {})
    order: List[str] = reg.get("order", [])

    run_obj = {"video_name": video_name, "mode": mode, "video_id": video_id}
    runs[video_id] = run_obj

    # maintain order: move to end if re-register
    if video_id in order:
        order.remove(video_id)
    order.append(video_id)

    reg["runs"] = runs
    reg["order"] = order
    _save_registry(reg)


def list_analysis_runs() -> List[Dict[str, Any]]:
    """List analyzed runs in insertion order."""
    reg = _load_registry()
    runs: Dict[str, Dict[str, Any]] = reg.get("runs", {})
    order: List[str] = reg.get("order", [])

    out: List[Dict[str, Any]] = []
    for vid in order:
        r = runs.get(vid)
        if r:
            out.append(r)

    # fallback: in case order got lost
    if not out and runs:
        out = list(runs.values())

    return out


def check_analysis_runs(video_id: str) -> bool:
    reg = _load_registry()
    runs: Dict[str, Dict[str, Any]] = reg.get("runs", {})
    if video_id in runs:
        return True
    print(f"[ERROR] No analyzed runs found with video_id={video_id}.")
    return False


def delete_analysis_run(video_id: str) -> bool:
    reg = _load_registry()
    runs: Dict[str, Dict[str, Any]] = reg.get("runs", {})
    order: List[str] = reg.get("order", [])

    if video_id not in runs:
        return False

    runs.pop(video_id, None)
    if video_id in order:
        order.remove(video_id)

    reg["runs"] = runs
    reg["order"] = order
    _save_registry(reg)
    return True


def clear_all_analysis_runs() -> int:
    reg = _load_registry()
    runs: Dict[str, Dict[str, Any]] = reg.get("runs", {})
    n = len(runs)
    reg["runs"] = {}
    reg["order"] = []
    _save_registry(reg)
    return n
