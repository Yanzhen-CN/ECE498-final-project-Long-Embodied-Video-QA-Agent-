import json
from pathlib import Path
from typing import List, Dict

# 统一存储位置：只使用 agent 中的 data 文件夹
BASE_DIR = Path(__file__).resolve().parent / "data"
REGISTRY_PATH = BASE_DIR / "registry.json"

# 读取注册表
def _load_registry() -> Dict[str, any]:
    """加载分析注册表"""
    if not REGISTRY_PATH.exists():
        return {"runs": []}  # 若注册表不存在，返回空的 runs 列表
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"runs": []}  # 若读取失败，也返回空列表

# 保存注册表
def _save_registry(reg: Dict[str, any]) -> None:
    """保存分析注册表"""
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")

# 注册视频分析记录
def register_analysis_run(video_name: str, mode: str, run_id: str | None = None, **kwargs) -> str:
    """注册视频分析记录；run_id 可选，不传则自动生成。返回 run_id。"""
    reg = _load_registry()
    reg.setdefault("runs", [])   # 不要覆盖

    if run_id is None:
        run_id = f"{Path(video_name).stem}__{mode}"  # 用 stem，更一致

    reg["runs"].append(
        {"video_name": video_name, "mode": mode, "run_id": run_id}
    )
    _save_registry(reg)
    return run_id

def register__analysis_run(video_name: str, mode: str, run_id: str | None = None, **kwargs) -> str:
    return register_analysis_run(video_name=video_name, mode=mode, run_id=run_id, **kwargs)

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
            del runs[i]  # 删除该记录
            _save_registry(reg)  # 保存更新后的注册表
            return True  # 删除成功
    return False  # 没有找到对应的记录

# 删除所有分析记录
def clear_all_analysis_runs() -> None:
    """删除所有的分析记录"""
    reg = _load_registry()
    reg["runs"] = []  # 清空所有记录
    _save_registry(reg)
