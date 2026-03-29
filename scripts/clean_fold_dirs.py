#!/usr/bin/env python3
"""
Safely find and delete fold_X directories (X in [0..4]) under a parent directory.

Defaults to dry-run for safety. Use --execute to actually delete. You can also
use --yes to skip the confirmation prompt when executing.

Examples:
  Dry-run (list what would be deleted):
    python scripts/clean_fold_dirs.py /path/to/parent

  Actually delete (with confirmation prompt):
    python scripts/clean_fold_dirs.py /path/to/parent --execute

  Delete without prompt:
    python scripts/clean_fold_dirs.py /path/to/parent --execute --yes
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List


FOLD_PATTERN = re.compile(r"^fold_[0-4]$")


def find_target_dirs(parent: Path) -> List[Path]:
    matches: List[Path] = []
    parent_resolved = parent.resolve()

    # Walk top-down so we can prevent descending into matched dirs
    for root, dirs, _files in os.walk(parent_resolved, topdown=True, followlinks=False):
        # Copy since we'll mutate dirs to avoid descending into matched directories
        for d in list(dirs):
            if FOLD_PATTERN.match(d):
                p = Path(root) / d

                # Safety: ensure under parent, and not a symlink
                try:
                    p_resolved = p.resolve()
                    _ = p_resolved.relative_to(parent_resolved)
                except Exception:
                    # If resolution or relative path fails, skip
                    continue

                if p.is_symlink():
                    # Skip symlinks by default for safety
                    continue

                if p.is_dir():
                    matches.append(p)

                # Do not traverse into this directory once matched
                try:
                    dirs.remove(d)
                except ValueError:
                    pass

    return matches


def delete_dirs(paths: List[Path]) -> tuple[int, int]:
    ok = 0
    err = 0
    for p in paths:
        # Double-check safety before deletion
        if not p.exists() or not p.is_dir():
            err += 1
            continue
        if not FOLD_PATTERN.match(p.name):
            err += 1
            continue
        if p.is_symlink():
            # Shouldn't happen due to earlier filter, but be safe
            err += 1
            continue
        try:
            shutil.rmtree(p)
            ok += 1
        except Exception:
            err += 1
    return ok, err


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delete fold_X (X in 0..4) directories under a parent directory.")
    parser.add_argument("parent", type=Path, help="Parent directory to scan")
    parser.add_argument("--execute", action="store_true", help="Actually delete matched directories (default is dry-run)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt when using --execute")
    parser.add_argument("--max-output", type=int, default=200, help="Max directories to list in dry-run output")
    parser.add_argument("--absolute", action="store_true", help="Show absolute paths instead of relative to parent")

    args = parser.parse_args(argv)

    parent: Path = args.parent
    if not parent.exists() or not parent.is_dir():
        print(f"Error: parent directory not found or not a directory: {parent}", file=sys.stderr)
        return 2

    matches = find_target_dirs(parent)
    matches_sorted = sorted(matches)

    if not args.execute:
        # Dry-run output
        print(f"Found {len(matches_sorted)} fold_X directories under: {parent}")
        if len(matches_sorted) == 0:
            return 0
        to_show = matches_sorted[: args.max_output]
        for p in to_show:
            s = str(p if args.absolute else p.relative_to(parent))
            print(s)
        if len(matches_sorted) > len(to_show):
            print(f"... and {len(matches_sorted) - len(to_show)} more")
        print("\nDry-run only. Re-run with --execute to delete.")
        return 0

    # Execute path
    print(f"About to delete {len(matches_sorted)} directories under: {parent}")
    if len(matches_sorted) > 0:
        to_show = matches_sorted[: args.max_output]
        for p in to_show:
            s = str(p if args.absolute else p.relative_to(parent))
            print(f" - {s}")
        if len(matches_sorted) > len(to_show):
            print(f"... and {len(matches_sorted) - len(to_show)} more")

    if not args.yes:
        try:
            resp = input("Proceed with deletion? Type 'yes' to confirm: ").strip().lower()
        except EOFError:
            resp = ""
        if resp != "yes":
            print("Aborted. Nothing deleted.")
            return 0

    ok, err = delete_dirs(matches_sorted)
    print(f"Deletion complete. Deleted: {ok}, Skipped/Errors: {err}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

