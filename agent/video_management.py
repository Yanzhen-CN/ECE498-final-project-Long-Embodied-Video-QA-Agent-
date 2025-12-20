import json
import time
from pathlib import Path
from typing import List, Dict, Optional

# 分析注册表路径
ANALYSIS_DIR = Path(__file__).resolve().parent / "data" / "analysis"
REGISTRY_PATH = ANALYSIS_DIR / "registry.json"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# 读取注册表
def _load_registry() -> Dict[str, any]:
    """加载分析注册表"""
    if not REGISTRY_PATH.exists():
        return {"version": "1.0", "runs": []}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"version": "1.0", "runs": []}

# 保存注册表
def _save_registry(reg: Dict[str, any]) -> None:
    """保存分析注册表"""
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")

# 注册视频分析记录
def register_analysis_run(
    *,
    video_name: str,
    mode: str,
    run_id: str,
    manifest_path: str,
    summary_file_path: Optional[str] = None,  # 存储总结文件路径
    chunk_seconds: int,
    keyframes_per_chunk: int,
) -> None:
    """注册视频分析记录"""
    reg = _load_registry()
    reg.setdefault("runs", [])
    reg["runs"].append(
        {
            "ts": int(time.time()),
            "video_name": video_name,
            "mode": mode,
            "run_id": run_id,
            "manifest_path": manifest_path,
            "summary_file_path": summary_file_path,  # 保存总结文件路径
            "chunk_seconds": chunk_seconds,
            "keyframes_per_chunk": keyframes_per_chunk,
        }
    )
    _save_registry(reg)

# 列出所有已分析的记录
def list_analysis_runs() -> List[Dict[str, any]]:
    """列出已分析的运行记录"""
    reg = _load_registry()
    return reg.get("runs", [])

# 删除指定的分析记录
def delete_analysis_run(run_id: str) -> bool:
    """根据 run_id 删除指定的分析记录"""
    reg = _load_registry()
    runs = reg.get("runs", [])
    for i, run in enumerate(runs):
        if run.get("run_id") == run_id:
            del runs[i]
            _save_registry(reg)
            return True
    return False

# 删除所有分析记录
def clear_all_analysis_runs() -> None:
    """删除所有的分析记录"""
    reg = _load_registry()
    reg["runs"] = []
    _save_registry(reg)
