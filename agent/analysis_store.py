# agent/analysis_store.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AnalysisPaths:
    root: Path  # project root

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def analysis_dir(self) -> Path:
        return self.data_dir / "analysis"

    def video_dir(self, video_id: str) -> Path:
        return self.analysis_dir / video_id

    def summary_path(self, video_id: str, mode: str) -> Path:
        return self.video_dir(video_id) / f"summary_{mode}.txt"

    def runs_jsonl(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "runs.jsonl"

    def latest_json(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "latest.json"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def has_summary(root: Path, video_id: str, mode: str) -> bool:
    ap = AnalysisPaths(root)
    return ap.summary_path(video_id, mode).exists()


def load_summary(root: Path, video_id: str, mode: str) -> Optional[str]:
    ap = AnalysisPaths(root)
    p = ap.summary_path(video_id, mode)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="ignore")


def save_summary(root: Path, video_id: str, mode: str, text: str) -> Path:
    ap = AnalysisPaths(root)
    vdir = ap.video_dir(video_id)
    ensure_dir(vdir)
    p = ap.summary_path(video_id, mode)
    p.write_text(text, encoding="utf-8")
    return p


def append_run(root: Path, video_id: str, record: Dict[str, Any]) -> None:
    ap = AnalysisPaths(root)
    vdir = ap.video_dir(video_id)
    ensure_dir(vdir)
    runs = ap.runs_jsonl(video_id)
    with runs.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # update latest pointer
    latest = ap.latest_json(video_id)
    latest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
