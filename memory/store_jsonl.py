# memory/store_jsonl.py
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

from .exceptions import DuplicateRecordError, RecordNotFound, StorageIOError
from .interface import MemoryStore, UpsertPolicy
from .schema import MemoryRecordV1, coerce_to_record
from .utils import read_jsonl, write_jsonl


class JsonlMemoryStore(MemoryStore):
    """
    JSONL storage. One file per video_id.
    """
    def __init__(self, root_dir: str = "memory_bank") -> None:
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _path(self, video_id: str) -> str:
        return os.path.join(self.root_dir, f"{video_id}.jsonl")

    def _load_map(self, video_id: str) -> Dict[str, MemoryRecordV1]:
        p = self._path(video_id)
        if not os.path.exists(p):
            return {}
        try:
            out: Dict[str, MemoryRecordV1] = {}
            for d in read_jsonl(p):
                rec = coerce_to_record(d)
                out[rec.record_key] = rec
            return out
        except Exception as e:
            raise StorageIOError(f"Failed to read JSONL: {p}. Error: {e}") from e

    def _save_map(self, video_id: str, recs: Dict[str, MemoryRecordV1]) -> None:
        p = self._path(video_id)
        try:
            # stable ordering: by start_time then chunk_id
            rows = [r.to_dict() for r in sorted(
                recs.values(),
                key=lambda x: (x.start_time, x.chunk_id)
            )]
            write_jsonl(p, rows)
        except Exception as e:
            raise StorageIOError(f"Failed to write JSONL: {p}. Error: {e}") from e

    def put(self, record: MemoryRecordV1, policy: UpsertPolicy = "overwrite") -> None:
        self.put_many([record], policy=policy)

    def put_many(self, records: Iterable[MemoryRecordV1], policy: UpsertPolicy = "overwrite") -> None:
        # group by video_id to reduce IO
        grouped: Dict[str, List[MemoryRecordV1]] = {}
        for r in records:
            grouped.setdefault(r.video_id, []).append(r)

        for video_id, rec_list in grouped.items():
            rec_map = self._load_map(video_id)

            for r in rec_list:
                key = r.record_key
                exists = key in rec_map

                if exists and policy == "error":
                    raise DuplicateRecordError(f"Record already exists: {key}")
                if exists and policy == "skip":
                    continue
                # overwrite or new
                rec_map[key] = r

            self._save_map(video_id, rec_map)

    def exists(self, video_id: str, chunk_id: str) -> bool:
        key = f"{video_id}::{chunk_id}"
        rec_map = self._load_map(video_id)
        return key in rec_map

    def get(self, video_id: str, chunk_id: str) -> MemoryRecordV1:
        key = f"{video_id}::{chunk_id}"
        rec_map = self._load_map(video_id)
        if key not in rec_map:
            raise RecordNotFound(f"Record not found: {key}")
        return rec_map[key]

    def list_by_video(self, video_id: str) -> List[MemoryRecordV1]:
        rec_map = self._load_map(video_id)
        return sorted(rec_map.values(), key=lambda x: (x.start_time, x.chunk_id))

    def get_range(self, video_id: str, start_time: float, end_time: float) -> List[MemoryRecordV1]:
        # overlap test: [a,b] overlaps [c,d] if b>=c and d>=a
        recs = self.list_by_video(video_id)
        out = []
        for r in recs:
            if r.end_time >= start_time and end_time >= r.start_time:
                out.append(r)
        return out

    def export_jsonl(self, video_id: str, out_path: str) -> None:
        recs = self.list_by_video(video_id)
        try:
            rows = [r.to_dict() for r in recs]
            write_jsonl(out_path, rows)
        except Exception as e:
            raise StorageIOError(f"Failed to export JSONL: {out_path}. Error: {e}") from e
