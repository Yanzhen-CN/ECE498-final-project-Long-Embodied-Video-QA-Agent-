# memory/schema.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal
import time

from .exceptions import SchemaValidationError


SchemaVersion = Literal["v1"]


@dataclass(frozen=True)
class MemoryRecordV1:
    """
    Record stored by memory layer. Produced by agent.

    - video_id/chunk_id: stable identifiers
    - start_time/end_time: seconds (float) in video timeline
    - keyframes: list of paths/URIs to evidence frames
    - memory_text: main textual memory (summary / events / objects / actions)
    - extra: optional structured payload from agent/model (JSON-serializable)
    """
    schema_version: SchemaVersion
    video_id: str
    chunk_id: str
    start_time: float
    end_time: float
    keyframes: List[str]
    memory_text: str

    created_at: float  # unix seconds
    agent_version: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    @property
    def record_key(self) -> str:
        return f"{self.video_id}::{self.chunk_id}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["record_key"] = self.record_key
        return d


def now_ts() -> float:
    return time.time()


def validate_record_dict(d: Dict[str, Any]) -> None:
    """
    Strict-ish validation without extra dependencies.
    Raises SchemaValidationError with actionable messages.
    """
    required = [
        "schema_version", "video_id", "chunk_id", "start_time", "end_time",
        "keyframes", "memory_text", "created_at"
    ]
    missing = [k for k in required if k not in d]
    if missing:
        raise SchemaValidationError(f"Missing required fields: {missing}")

    if d["schema_version"] != "v1":
        raise SchemaValidationError(f"Unsupported schema_version: {d['schema_version']}")

    for k in ["video_id", "chunk_id", "memory_text"]:
        if not isinstance(d[k], str) or not d[k].strip():
            raise SchemaValidationError(f"Field '{k}' must be a non-empty string")

    for k in ["start_time", "end_time", "created_at"]:
        if not isinstance(d[k], (int, float)):
            raise SchemaValidationError(f"Field '{k}' must be a number")

    if d["start_time"] < 0 or d["end_time"] < 0:
        raise SchemaValidationError("start_time/end_time must be >= 0")

    if d["end_time"] < d["start_time"]:
        raise SchemaValidationError("end_time must be >= start_time")

    if not isinstance(d["keyframes"], list) or any(not isinstance(x, str) for x in d["keyframes"]):
        raise SchemaValidationError("keyframes must be a list[str]")

    # extra must be JSON-serializable; we check only basic type here
    if "extra" in d and d["extra"] is not None and not isinstance(d["extra"], dict):
        raise SchemaValidationError("extra must be a dict or null")


def coerce_to_record(d: Dict[str, Any]) -> MemoryRecordV1:
    """
    Convert dict -> MemoryRecordV1 after validation.
    """
    validate_record_dict(d)
    return MemoryRecordV1(
        schema_version="v1",
        video_id=d["video_id"],
        chunk_id=d["chunk_id"],
        start_time=float(d["start_time"]),
        end_time=float(d["end_time"]),
        keyframes=list(d["keyframes"]),
        memory_text=d["memory_text"],
        created_at=float(d["created_at"]),
        agent_version=d.get("agent_version"),
        extra=d.get("extra"),
    )
