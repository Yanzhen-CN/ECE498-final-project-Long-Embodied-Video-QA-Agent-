#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_memory.py

Hard-clean generated artifacts under ./data/:
- ./data/manifests/
- ./data/keyframes/
- ./data/metadata/

Safe by default:
- dry-run prints what would be deleted
- requires confirmation unless --yes

Usage:
  python clean_memory.py --dry-run
  python clean_memory.py
  python clean_memory.py --yes
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Tuple


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}" if u != "B" else f"{int(x)} {u}"
        x /= 1024.0
    return f"{n} B"


def walk_stats(p: Path) -> Tuple[int, int]:
    if not p.exists():
        return (0, 0)
    if p.is_file():
        try:
            return (1, p.stat().st_size)
        except OSError:
            return (1, 0)

    num_files = 0
    total_bytes = 0
    for f in p.rglob("*"):
        if f.is_file():
            num_files += 1
            try:
                total_bytes += f.stat().st_size
            except OSError:
                pass
    return (num_files, total_bytes)


def delete_dir(p: Path, *, dry_run: bool) -> None:
    if not p.exists():
        return
    if dry_run:
        return
    shutil.rmtree(p, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be deleted")
    ap.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    args = ap.parse_args()

    # Project root = directory where this script lives
    root = Path(__file__).resolve().parent
    data_dir = root / "data"

    if not data_dir.exists():
        print(f"[ERR] data/ not found under: {root}")
        return 2

    targets = [
        data_dir / "manifests",
        data_dir / "keyframes",
        data_dir / "metadata",
    ]

    existing = [t for t in targets if t.exists()]
    if not existing:
        print("[OK] Nothing to clean. None of these exist:")
        for t in targets:
            print(f"  - {t.relative_to(root)}")
        return 0

    print(f"[Info] Project root: {root}")
    print("[Info] Targets:")
    total_files = 0
    total_bytes = 0
    for t in existing:
        nfiles, nbytes = walk_stats(t)
        total_files += nfiles
        total_bytes += nbytes
        print(f"  - {t.relative_to(root)}  ({nfiles} files, {human_bytes(nbytes)})")

    if args.dry_run:
        print(f"\n[Dry-run] Would delete {total_files} files ({human_bytes(total_bytes)}).")
        return 0

    if not args.yes:
        ans = input(f"\nDelete these folders now? This will remove {total_files} files. [y/N]: ").strip().lower()
        if ans != "y":
            print("[Cancelled] No files were deleted.")
            return 0

    for t in existing:
        print(f"[Delete] {t.relative_to(root)}")
        delete_dir(t, dry_run=False)

    print(f"\n[OK] Deleted up to {total_files} files ({human_bytes(total_bytes)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
