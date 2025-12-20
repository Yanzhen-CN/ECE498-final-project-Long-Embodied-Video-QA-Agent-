import json
from pathlib import Path
from typing import List, Dict

# 统一存储位置：只使用 agent 中的 data 文件夹
BASE_DIR = Path(__file__).resolve().parent / "data"
ANALYSIS_DIR = BASE_DIR / "analysis"
REGISTRY_PATH = ANALYSIS_DIR / "registry.json"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# 读取注册表
def _load_registry() -> Dict[str, any]:
    """加载分析注册表"""
    if not REGISTRY_PATH.exists():
        return {"runs": []}  # 直接返回空 runs 列表
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"runs": []}  # 若读取失败，也返回空列表

# 保存注册表
def _save_registry(reg: Dict[str, any]) -> None:
    """保存分析注册表"""
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")

# 注册视频分析记录
def register_analysis_run(
    video_name: str,
    mode: str
) -> None:
    """注册视频分析记录"""
    reg = _load_registry()
    reg["runs"] = []  # 每次覆盖已有记录

    # 简化的 run_id 格式：video_name_mode
    run_id = f"{video_name}_{mode}"

    reg["runs"].append(
        {
            "video_name": video_name,
            "mode": mode,
            "run_id": run_id,  # 自动生成 run_id，不需要传递
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
    reg["runs"] = []  # 清空所有记录
    _save_registry(reg)
