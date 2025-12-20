#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_memory.py

Delete generated artifacts under data/:
- data/manifests/
- data/keyframes/
- data/meta_data/   (also supports: "meta data", "metadata")

Safe by default:
- dry-run mode prints what would be deleted
- requires confirmation unless --yes

Usage:
  python scripts/clean_memory.py --dry-run
  python scripts/clean_memory.py
  python scripts/clean_memory.py --yes
  python scripts/clean_memory.py --root /data/ece498/final
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, Tuple


CANDIDATE_DIR_NAMES = [
    "manifests",
    "keyframes",
    "meta_data",
    "meta data",
    "metadata",
]


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}" if u != "B" else f"{int(x)} {u}"
        x /= 1024.0
    return f"{n} B"


def walk_stats(p: Path) -> Tuple[int, int]:
    """Return (num_files, total_bytes) for a directory/file path."""
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


def resolve_targets(root: Path) -> list[Path]:
    data_dir = root / "data"
    targets: list[Path] = []
    for name in CANDIDATE_DIR_NAMES:
        p = data_dir / name
        if p.exists():
            targets.append(p)
    # de-dup (in case of symlinks or same path)
    uniq = []
    seen = set()
    for t in targets:
        rp = str(t.resolve())
        if rp not in seen:
            uniq.append(t)
            seen.add(rp)
    return uniq


def delete_path(p: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if p.is_symlink() or p.is_file():
        p.unlink(missing_ok=True)
    else:
        shutil.rmtree(p, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Project root (contains data/)")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be deleted")
    ap.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    data_dir = root / "data"
    if not data_dir.exists():
        print(f"[ERR] data/ not found under root: {root}")
        return 2

    targets = resolve_targets(root)
    if not targets:
        print("[OK] Nothing to clean. None of these folders exist:")
        for n in CANDIDATE_DIR_NAMES:
            print(f"  - data/{n}/")
        return 0

    print(f"[Info] Project root: {root}")
    print("[Info] Targets found:")
    total_files = 0
    total_bytes = 0
    for t in targets:
        nfiles, nbytes = walk_stats(t)
        total_files += nfiles
        total_bytes += nbytes
        rel = t.relative_to(root)
        print(f"  - {rel}  ({nfiles} files, {human_bytes(nbytes)})")

    if args.dry_run:
        print(f"\n[Dry-run] Would delete {total_files} files ({human_bytes(total_bytes)}).")
        return 0

    if not args.yes:
        ans = input(f"\nDelete these folders now? This will remove {total_files} files. [y/N]: ").strip().lower()
        if ans != "y":
            print("[Cancelled] No files were deleted.")
            return 0

    for t in targets:
        rel = t.relative_to(root)
        print(f"[Delete] {rel}")
        delete_path(t, dry_run=False)

    print(f"\n[OK] Deleted up to {total_files} files ({human_bytes(total_bytes)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
