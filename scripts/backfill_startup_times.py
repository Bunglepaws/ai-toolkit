"""
One-time migration: recover startup (model-load) time for training sessions
that predate the startup_seconds tracking feature.

For each job under the training output folder whose logs/ dir was touched in
the last 30 days, this scans logs/*_log.txt for the "Time to first step: Xs"
line the trainer already prints, and matches each occurrence to the closest
training_sessions row (by file mtime vs session start_time) in that job's
loss_log.db. Only confident matches (within TOLERANCE_SECONDS) are written;
everything else is skipped and reported.

This never creates a training_sessions row - it only fills in
startup_seconds on existing rows recorded by the app itself.

Usage (from WSL):
    venv/bin/python3 scripts/backfill_startup_times.py [--dry-run] [--output-dir PATH]
"""

import argparse
import os
import re
import sqlite3
import time

DEFAULT_OUTPUT_DIR = "/mnt/c/Data/AIToolkit-StagingArea/output"
LOOKBACK_SECONDS = 30 * 24 * 60 * 60
TOLERANCE_SECONDS = 15 * 60

TIME_TO_FIRST_STEP_RE = re.compile(r"Time to first step:\s*(\d+)s")


def ensure_startup_column(con: sqlite3.Connection) -> None:
    cols = {row[1] for row in con.execute("PRAGMA table_info(training_sessions);").fetchall()}
    if "startup_seconds" not in cols:
        con.execute("ALTER TABLE training_sessions ADD COLUMN startup_seconds REAL;")


def find_startup_seconds(log_file: str) -> float | None:
    try:
        with open(log_file, "r", errors="ignore") as f:
            content = f.read()
    except OSError:
        return None
    match = TIME_TO_FIRST_STEP_RE.search(content)
    if not match:
        return None
    return float(match.group(1))


def log_file_index(fname: str) -> int | None:
    """Extract the numeric prefix from Nlog.txt filenames, e.g. '3_log.txt' → 3."""
    m = re.match(r"^(\d+)_log\.txt$", fname)
    return int(m.group(1)) if m else None


def backfill_job(job_dir: str, cutoff: float, dry_run: bool) -> tuple[int, int, str]:
    logs_dir = os.path.join(job_dir, "logs")
    if not os.path.isdir(logs_dir):
        return 0, 0, "no logs dir"

    # Build a map of log-file-index → startup_seconds for files in the cutoff window.
    # We use the log-file index (0, 1, 2 …) to match against the Nth training session
    # (sorted by start_time), because file mtime = last write = end of session, which
    # can be hours after session start — too far off for timestamp-based matching.
    indexed_files: list[tuple[int, float]] = []  # (log_index, seconds)
    all_log_files = sorted(
        (f for f in os.listdir(logs_dir) if f.endswith("_log.txt")),
        key=lambda f: (log_file_index(f) or 0, f),
    )
    if not all_log_files:
        return 0, 0, "no log files"

    any_in_range = False
    for fname in all_log_files:
        fpath = os.path.join(logs_dir, fname)
        mtime = os.path.getmtime(fpath)
        if mtime < cutoff:
            continue
        any_in_range = True
        idx = log_file_index(fname)
        if idx is None:
            continue
        seconds = find_startup_seconds(fpath)
        if seconds is None:
            continue
        indexed_files.append((idx, seconds))

    if not any_in_range:
        return 0, 0, "no log files in 30-day window"
    if not indexed_files:
        return 0, 0, "no 'Time to first step' lines in range"

    db_path = os.path.join(job_dir, "loss_log.db")
    if not os.path.isfile(db_path):
        return 0, len(indexed_files), "no loss_log.db"

    con = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        if "training_sessions" not in tables:
            return 0, len(indexed_files), "no training_sessions table"

        cols = {row[1] for row in con.execute("PRAGMA table_info(training_sessions);").fetchall()}
        has_startup_col = "startup_seconds" in cols

        if not dry_run and not has_startup_col:
            ensure_startup_column(con)
            con.commit()
            has_startup_col = True

        if has_startup_col:
            session_rows = con.execute(
                "SELECT id, startup_seconds FROM training_sessions ORDER BY start_time ASC;"
            ).fetchall()
        else:
            session_rows = [
                (row[0], None)
                for row in con.execute(
                    "SELECT id FROM training_sessions ORDER BY start_time ASC;"
                ).fetchall()
            ]

        matched = 0
        skipped = 0
        for log_idx, seconds in indexed_files:
            if log_idx >= len(session_rows):
                skipped += 1
                continue
            sid, existing_startup = session_rows[log_idx]
            if existing_startup is not None:
                skipped += 1
                continue
            if not dry_run:
                con.execute(
                    "UPDATE training_sessions SET startup_seconds=? WHERE id=?;",
                    (seconds, sid),
                )
            matched += 1

        if not dry_run:
            con.commit()

        return matched, skipped, "ok"
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    cutoff = time.time() - LOOKBACK_SECONDS

    if not os.path.isdir(args.output_dir):
        print(f"Output dir not found: {args.output_dir}")
        return

    total_matched = 0
    total_skipped = 0

    for name in sorted(os.listdir(args.output_dir)):
        job_dir = os.path.join(args.output_dir, name)
        if not os.path.isdir(job_dir):
            continue
        matched, skipped, status = backfill_job(job_dir, cutoff, args.dry_run)
        if matched == 0 and skipped == 0 and status in ("no logs dir", "no log files"):
            continue
        print(f"{name}: matched={matched} skipped={skipped} ({status})")
        total_matched += matched
        total_skipped += skipped

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"\n{mode}Total: matched={total_matched} skipped={total_skipped}")


if __name__ == "__main__":
    main()
