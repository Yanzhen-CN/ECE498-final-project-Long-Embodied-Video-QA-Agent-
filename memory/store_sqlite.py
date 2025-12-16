# memory/store_sqlite.py
from __future__ import annotations

import json
import os
import sqlite3
from typing import Iterable, List

from .exceptions import DuplicateRecordError, RecordNotFound, StorageIOError
from .interface import MemoryStore, UpsertPolicy
from .schema import MemoryRecordV1


class SqliteMemoryStore(MemoryStore):
    def __init__(self, db_path: str = "memory_bank/memory.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        try:
            with self._conn() as con:
                con.execute("""
                CREATE TABLE IF NOT EXISTS memory_records (
                    record_key TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    memory_text TEXT NOT NULL,
                    keyframes_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    agent_version TEXT,
                    extra_json TEXT,
                    schema_version TEXT NOT NULL
                );
                """)
                con.execute("CREATE INDEX IF NOT EXISTS idx_video_time ON memory_records(video_id, start_time, end_time);")
        except Exception as e:
            raise StorageIOError(f"Failed to init sqlite db: {self.db_path}. Error: {e}") from e

    def put(self, record: MemoryRecordV1, policy: UpsertPolicy = "overwrite") -> None:
        self.put_many([record], policy=policy)

    def put_many(self, records: Iterable[MemoryRecordV1], policy: UpsertPolicy = "overwrite") -> None:
        try:
            with self._conn() as con:
                for r in records:
                    key = r.record_key
                    if policy in ("skip", "error"):
                        cur = con.execute("SELECT 1 FROM memory_records WHERE record_key = ?", (key,))
                        exists = cur.fetchone() is not None
                        if exists and policy == "error":
                            raise DuplicateRecordError(f"Record already exists: {key}")
                        if exists and policy == "skip":
                            continue

                    con.execute("""
                    INSERT INTO memory_records (
                        record_key, video_id, chunk_id, start_time, end_time,
                        memory_text, keyframes_json, created_at, agent_version,
                        extra_json, schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(record_key) DO UPDATE SET
                        start_time=excluded.start_time,
                        end_time=excluded.end_time,
                        memory_text=excluded.memory_text,
                        keyframes_json=excluded.keyframes_json,
                        created_at=excluded.created_at,
                        agent_version=excluded.agent_version,
                        extra_json=excluded.extra_json,
                        schema_version=excluded.schema_version
                    """, (
                        key, r.video_id, r.chunk_id, r.start_time, r.end_time,
                        r.memory_text, json.dumps(r.keyframes, ensure_ascii=False),
                        r.created_at, r.agent_version,
                        json.dumps(r.extra, ensure_ascii=False) if r.extra is not None else None,
                        r.schema_version
                    ))
        except DuplicateRecordError:
            raise
        except Exception as e:
            raise StorageIOError(f"Failed sqlite put_many. Error: {e}") from e

    def exists(self, video_id: str, chunk_id: str) -> bool:
        key = f"{video_id}::{chunk_id}"
        try:
            with self._conn() as con:
                cur = con.execute("SELECT 1 FROM memory_records WHERE record_key = ?", (key,))
                return cur.fetchone() is not None
        except Exception as e:
            raise StorageIOError(f"Failed sqlite exists. Error: {e}") from e

    def get(self, video_id: str, chunk_id: str) -> MemoryRecordV1:
        key = f"{video_id}::{chunk_id}"
        try:
            with self._conn() as con:
                cur = con.execute("""
                SELECT record_key, video_id, chunk_id, start_time, end_time, memory_text,
                       keyframes_json, created_at, agent_version, extra_json, schema_version
                FROM memory_records WHERE record_key = ?
                """, (key,))
                row = cur.fetchone()
                if not row:
                    raise RecordNotFound(f"Record not found: {key}")
                return self._row_to_record(row)
        except RecordNotFound:
            raise
        except Exception as e:
            raise StorageIOError(f"Failed sqlite get. Error: {e}") from e

    def list_by_video(self, video_id: str) -> List[MemoryRecordV1]:
        try:
            with self._conn() as con:
                cur = con.execute("""
                SELECT record_key, video_id, chunk_id, start_time, end_time, memory_text,
                       keyframes_json, created_at, agent_version, extra_json, schema_version
                FROM memory_records
                WHERE video_id = ?
                ORDER BY start_time ASC, chunk_id ASC
                """, (video_id,))
                return [self._row_to_record(r) for r in cur.fetchall()]
        except Exception as e:
            raise StorageIOError(f"Failed sqlite list_by_video. Error: {e}") from e

    def get_range(self, video_id: str, start_time: float, end_time: float) -> List[MemoryRecordV1]:
        # overlap: end_time >= start AND start_time <= end
        try:
            with self._conn() as con:
                cur = con.execute("""
                SELECT record_key, video_id, chunk_id, start_time, end_time, memory_text,
                       keyframes_json, created_at, agent_version, extra_json, schema_version
                FROM memory_records
                WHERE video_id = ?
                  AND end_time >= ?
                  AND start_time <= ?
                ORDER BY start_time ASC, chunk_id ASC
                """, (video_id, start_time, end_time))
                return [self._row_to_record(r) for r in cur.fetchall()]
        except Exception as e:
            raise StorageIOError(f"Failed sqlite get_range. Error: {e}") from e

    def export_jsonl(self, video_id: str, out_path: str) -> None:
        import os
        from .utils import write_jsonl
        recs = self.list_by_video(video_id)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_jsonl(out_path, [r.to_dict() for r in recs])

    @staticmethod
    def _row_to_record(row) -> MemoryRecordV1:
        (_, video_id, chunk_id, start_time, end_time, memory_text,
         keyframes_json, created_at, agent_version, extra_json, schema_version) = row
        return MemoryRecordV1(
            schema_version=schema_version,
            video_id=video_id,
            chunk_id=chunk_id,
            start_time=float(start_time),
            end_time=float(end_time),
            keyframes=json.loads(keyframes_json),
            memory_text=memory_text,
            created_at=float(created_at),
            agent_version=agent_version,
            extra=json.loads(extra_json) if extra_json else None,
        )
