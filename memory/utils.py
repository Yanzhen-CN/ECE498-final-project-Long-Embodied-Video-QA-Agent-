# memory/utils.py
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Iterable


def atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)  # atomic on POSIX, best-effort on Windows
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    # atomic rewrite
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    atomic_write_text(path, text)
