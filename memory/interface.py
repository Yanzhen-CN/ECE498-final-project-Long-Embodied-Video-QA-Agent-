# memory/interface.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Literal

from .schema import MemoryRecordV1

UpsertPolicy = Literal["overwrite", "skip", "error"]


class MemoryStore(ABC):
    """
    Storage-only memory module (per your clarified responsibility).
    No retrieval ranking. Only stable persistence & reads.
    """

    @abstractmethod
    def put(self, record: MemoryRecordV1, policy: UpsertPolicy = "overwrite") -> None:
        ...

    @abstractmethod
    def put_many(self, records: Iterable[MemoryRecordV1], policy: UpsertPolicy = "overwrite") -> None:
        ...

    @abstractmethod
    def exists(self, video_id: str, chunk_id: str) -> bool:
        ...

    @abstractmethod
    def get(self, video_id: str, chunk_id: str) -> MemoryRecordV1:
        ...

    @abstractmethod
    def list_by_video(self, video_id: str) -> List[MemoryRecordV1]:
        ...

    @abstractmethod
    def get_range(self, video_id: str, start_time: float, end_time: float) -> List[MemoryRecordV1]:
        """
        Optional but useful: return chunks that overlap [start_time, end_time].
        """
        ...

    @abstractmethod
    def export_jsonl(self, video_id: str, out_path: str) -> None:
        ...
