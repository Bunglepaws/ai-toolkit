"""Smoke test for the queue's job-launch path.

The queue has broken twice in ways that compiled cleanly and reviewed fine, because the
bugs live in the window *between* two database writes rather than in the logic. A launch is
not atomic: `startJob` flips the row to 'running' first and can only record the pid once the
Windows relay reports it back, which is longer than the worker's one-second tick. Anything
that reads the row in between sees 'running' with no live pid -- indistinguishable from a
trainer that just died.

So this does not test functions, it watches a real launch and fails on what actually goes
wrong:

  * a healthy, still-loading job reconciled away as "trainer process gone"
  * a row parked at 'running' with no pid for longer than a launch could plausibly take
  * two live trainers on the same GPU (the reconcile handing an occupied slot to the next job)
  * the trainer process dying outright during startup

Usage:

    python scripts/queue_launch_smoke.py --watch            # then start a job from the UI
    python scripts/queue_launch_smoke.py --start "<job name>"

Exit code is 0 on pass, 1 on failure, so it can gate a commit.
"""

import argparse
import ctypes
import os
import sqlite3
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(REPO_ROOT, 'aitk_db.db')

# Longest a launch may sit at 'running' with no pid before we call it broken. The worker's
# own grace window is 90s; this has to be at least that or the test fails jobs the worker
# would still (correctly) be waiting on.
PID_DEADLINE_SEC = 90

# How long a job must hold 'running' with a live pid to count as launched. Long enough to
# cover the reconcile pass that used to kill jobs, short enough not to spend real GPU time:
# the failure always fired within seconds of the status flip.
HOLD_SEC = 60

DEAD_INFO_MARKERS = ('trainer process gone', 'did not report a pid', 'exited unexpectedly')

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def process_alive(pid):
    """True if the pid is a live process. Windows-native; the stack no longer runs in WSL."""
    if pid is None:
        return False
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def connect(db_path):
    # Read-mostly, and the worker/trainer are writing constantly -- a busy timeout keeps a
    # sampled read from failing the test for reasons that have nothing to do with the queue.
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def snapshot(conn):
    return {
        row['id']: row
        for row in conn.execute(
            "SELECT id, name, status, pid, gpu_ids, info, step FROM Job WHERE job_type = 'train'"
        )
    }


def describe(row):
    return f"{row['name']} (status={row['status']} pid={row['pid']} step={row['step']})"


class Failure(Exception):
    pass


def check_gpu_exclusivity(conn, gpu_ids):
    """No two live trainers may share a GPU. This is the damage the reconcile bug caused:
    a healthy job was marked stopped, its slot handed out, and both runs were wrecked."""
    live = [
        row
        for row in conn.execute(
            "SELECT id, name, status, pid, step FROM Job WHERE gpu_ids = ? AND pid IS NOT NULL",
            (gpu_ids,),
        )
        if process_alive(row['pid'])
    ]
    if len(live) > 1:
        names = ', '.join(f"{r['name']} (pid {r['pid']})" for r in live)
        raise Failure(f"two live trainers on GPU(s) {gpu_ids}: {names}")


def wait_for_launch(conn, timeout_sec, job_id=None):
    """Block until a job transitions into 'running', and return its row.

    Watching for the transition rather than polling for a name means this works whether the
    job was started from the UI, the worker, or --start.
    """
    before = snapshot(conn)
    if job_id is not None and job_id in before and before[job_id]['status'] == 'running':
        return before[job_id]

    deadline = time.time() + timeout_sec
    print(f"Waiting up to {timeout_sec}s for a job to start...", flush=True)
    while time.time() < deadline:
        now = snapshot(conn)
        for jid, row in now.items():
            if job_id is not None and jid != job_id:
                continue
            was = before.get(jid)
            if row['status'] == 'running' and (was is None or was['status'] != 'running'):
                print(f"\nLaunch detected: {describe(row)}", flush=True)
                return row
        before = now
        time.sleep(0.25)
    raise Failure(f"no job entered 'running' within {timeout_sec}s")


def watch_launch(conn, job_id, gpu_ids):
    """Follow one launch from the status flip until it has held 'running' with a live pid."""
    started = time.time()
    pid_seen_at = None
    last_report = 0.0

    while True:
        row = conn.execute(
            "SELECT id, name, status, pid, gpu_ids, info, step FROM Job WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise Failure('job row disappeared mid-launch')

        elapsed = time.time() - started
        info = (row['info'] or '').lower()

        for marker in DEAD_INFO_MARKERS:
            if marker in info:
                raise Failure(
                    f"queue declared the job dead {elapsed:.1f}s into the launch: "
                    f"info={row['info']!r}. This is the reconcile race -- the job was almost "
                    f"certainly still loading."
                )

        if row['status'] != 'running':
            raise Failure(
                f"status left 'running' after {elapsed:.1f}s (now {row['status']!r}, "
                f"info={row['info']!r}) without the job ever holding its slot"
            )

        if row['pid'] is None:
            if elapsed > PID_DEADLINE_SEC:
                raise Failure(
                    f"still no pid {elapsed:.1f}s after the row went 'running' -- the launch "
                    f"never reported one back"
                )
        else:
            if pid_seen_at is None:
                pid_seen_at = time.time()
                print(f"  pid {row['pid']} recorded after {elapsed:.1f}s", flush=True)
            if not process_alive(row['pid']):
                raise Failure(
                    f"trainer pid {row['pid']} is gone {elapsed:.1f}s into the launch -- the "
                    f"process died during startup (check the job's log.txt tail)"
                )

        check_gpu_exclusivity(conn, gpu_ids)

        if pid_seen_at is not None:
            held = time.time() - pid_seen_at
            if held >= HOLD_SEC:
                print(f"  held 'running' with a live pid for {held:.0f}s", flush=True)
                return row
            if time.time() - last_report >= 10:
                last_report = time.time()
                print(f"  holding... {held:.0f}/{HOLD_SEC}s (step {row['step']})", flush=True)

        time.sleep(1)


def queue_job(conn, name):
    row = conn.execute("SELECT id, name, status, gpu_ids FROM Job WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise Failure(f"no job named {name!r}")
    if row['status'] == 'running':
        raise Failure(f"{name!r} is already running -- nothing to launch")

    # Same field set as the UI's start route, so this exercises the real path rather than a
    # shortcut the worker would treat differently.
    conn.execute(
        "UPDATE Job SET status='queued', stop=0, return_to_queue=0, save_now=0, "
        "info='Job queued' WHERE id=?",
        (row['id'],),
    )
    conn.execute("UPDATE Queue SET is_running=1 WHERE gpu_ids=?", (row['gpu_ids'],))
    conn.commit()
    print(f"Queued {name!r} on GPU(s) {row['gpu_ids']} and started that queue.", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--watch', action='store_true',
                       help='watch for the next launch from any source (default)')
    group.add_argument('--start', metavar='JOB_NAME',
                       help='queue this job and watch it launch')
    parser.add_argument('--db', default=DEFAULT_DB, help=f'path to aitk_db.db (default: {DEFAULT_DB})')
    parser.add_argument('--timeout', type=int, default=300,
                       help='seconds to wait for a launch to begin (default: 300)')
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"FAIL: no database at {args.db}", file=sys.stderr)
        return 1

    conn = connect(args.db)
    try:
        job_id = None
        if args.start:
            job_id = queue_job(conn, args.start)['id']
        else:
            print("Watching for the next job launch. Start a job from the UI now.", flush=True)

        row = wait_for_launch(conn, args.timeout, job_id)
        watch_launch(conn, row['id'], row['gpu_ids'])
    except Failure as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nInterrupted -- no verdict.', file=sys.stderr)
        return 1
    finally:
        conn.close()

    print('\nPASS: the job launched, kept its pid, and held the GPU alone.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
