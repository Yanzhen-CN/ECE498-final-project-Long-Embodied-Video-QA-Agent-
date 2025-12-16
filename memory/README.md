# memory module

Scope: storage only (agent produces records; memory persists and serves them back).

## Record schema (v1)
- video_id (str)
- chunk_id (str)
- start_time/end_time (float seconds)
- keyframes (list[str]) paths/URIs
- memory_text (str)
- created_at (float unix seconds)
- agent_version (optional)
- extra (optional dict)

## Stores
- JsonlMemoryStore(root_dir="memory_bank")
- SqliteMemoryStore(db_path="memory_bank/memory.db")

## Upsert policies
- overwrite (default)
- skip
- error
