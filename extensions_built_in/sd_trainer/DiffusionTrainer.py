from collections import OrderedDict, deque
import glob as _glob
import json
import os
import shutil
import sqlite3
import statistics
import asyncio
import concurrent.futures
import traceback
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import SampleConfig
from toolkit.ui_utils import JobStoppedException, SampleAbortedException, SampleSkippedException
from typing import Literal, Optional
import threading
import time
import signal
from toolkit.basic import flush
from toolkit.print import print_acc

AITK_Status = Literal["running", "stopped", "error", "completed"]


class DiffusionTrainer(SDTrainer):
    # How long the stop watcher waits for an in-flight checkpoint write before
    # interrupting anyway. A write is seconds; this only exists so a wedged one
    # can never make a job impossible to stop.
    STOP_WATCHER_SAVE_GRACE_SEC = 300

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(DiffusionTrainer, self).__init__(process_id, job, config, **kwargs)
        self.sqlite_db_path = self.config.get("sqlite_db_path", "./aitk_db.db")
        self.job_id = os.environ.get("AITK_JOB_ID", None)
        self.job_id = self.job_id.strip() if self.job_id is not None else None
        self.is_ui_trainer = True
        if not os.path.exists(self.sqlite_db_path):
            self.is_ui_trainer = False
        else:
            print(f"Using SQLite database at {self.sqlite_db_path}")
        if self.job_id is None:
            self.is_ui_trainer = False
        else:
            print(f"Job ID: \"{self.job_id}\"")

        # >0 while a checkpoint write is pending or in flight. The stop watcher
        # refuses to interrupt the main thread while this is raised, so a
        # save-and-pause always gets its checkpoint. Set before the watcher
        # thread can start, and unconditionally (save() runs for periodic saves
        # even when this isn't a UI trainer).
        self._save_guard = 0

        if self.is_ui_trainer:
            self.is_stopping = False
            # Create a thread pool for database operations
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            # Track all async tasks
            self._async_tasks = []
            # Initialize the status
            self._run_async_operation(self._update_status("running", "Starting"))
            # A job that is only now starting cannot have a checkpoint write
            # pending, so `stop_after_save` here can only be a leftover from a
            # previous run -- and a leftover is fatal: hook_before_model_load()
            # calls maybe_stop() before the model is even loaded, so a stale flag
            # stops the job dead within milliseconds and it silently drops out of
            # the queue with no error to show for it. Clear it here rather than
            # trusting whoever launched us: this is the one point every launch
            # path (Node's startJob and run_ui.py's hot-model handoff) shares.
            self.update_db_key("stop_after_save", 0)
            self._stop_watcher_started = False
            if os.name == "nt":
                # On Windows the stop route cannot send us SIGINT from outside
                # (no console to deliver a Ctrl+C to), so watch the stop flag
                # and raise the interrupt from inside. On Linux the route
                # sends a real SIGINT to the pid and this is unnecessary.
                self.start_stop_watcher(interval_sec=2.0)

        # Alert detection state (maintained regardless of is_ui_trainer so the
        # rolling history works even for non-UI runs — only DB writes are gated)
        self._loss_history: deque = deque(maxlen=50)
        self._last_step_loss: float = 0.0
        self._last_spike_step: int = -1
        self._spike_streak: int = 0
        self._baseline_sample_avg_bytes: float | None = None
        # Stall detection state (separate, longer horizon than spike detection above —
        # a spike is a single bad step, a stall is loss never improving over thousands
        # of steps, e.g. automagic3's per-tensor LR decaying to near-zero before it
        # fits anything). See _check_loss_stall.
        self._stall_window: deque = deque(maxlen=300)
        self._stall_best_median: float | None = None
        self._stall_best_step: int = 0
        self._last_stall_alert_step: int = -1

    def start_stop_watcher(self, interval_sec: float = 5.0):
        """
        Start a daemon thread that periodically checks should_stop()
        and terminates the process immediately when triggered.
        """
        if not self.is_ui_trainer:
            return
        if getattr(self, "_stop_watcher_started", False):
            return
        self._stop_watcher_started = True
        t = threading.Thread(
            target=self._stop_watcher_thread, args=(interval_sec,), daemon=True
        )
        t.start()

    def _stop_watcher_thread(self, interval_sec: float):
        deferred_since: float | None = None
        while True:
            try:
                if self.should_stop():
                    if self.is_stopping:
                        # maybe_stop() already started the graceful shutdown;
                        # a second interrupt would only break its cleanup.
                        return
                    # `stop` now unambiguously means "interrupt now" -- the
                    # save-then-stop buttons use stop_after_save, which this
                    # thread never looks at. The only thing worth waiting for
                    # is a checkpoint write already in flight, since killing
                    # mid-write leaves a truncated file. Bounded purely so a
                    # wedged write can't make a job unstoppable.
                    if self._save_guard > 0:
                        now = time.time()
                        if deferred_since is None:
                            deferred_since = now
                        if now - deferred_since < self.STOP_WATCHER_SAVE_GRACE_SEC:
                            time.sleep(interval_sec)
                            continue
                        print("")
                        print(
                            f"Checkpoint write still running after "
                            f"{self.STOP_WATCHER_SAVE_GRACE_SEC}s; stopping anyway."
                        )
                    print("")
                    print("****************************************************")
                    print("    Stop signal received; terminating process.      ")
                    print("****************************************************")
                    # Deliver a real KeyboardInterrupt to the main thread so
                    # on_error runs the normal shutdown (final DB write, last
                    # log). os.kill(pid, SIGINT) must not be used here: on
                    # Windows it is TerminateProcess and kills us instantly.
                    # Leave the thread pool alone -- on_error still needs it.
                    signal.raise_signal(signal.SIGINT)
                    return
                time.sleep(interval_sec)
            except Exception:
                time.sleep(interval_sec)

    def _run_async_operation(self, coro):
        """Helper method to run an async coroutine and track the task."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Create a task and track it
        if loop.is_running():
            task = asyncio.run_coroutine_threadsafe(coro, loop)
            self._async_tasks.append(asyncio.wrap_future(task))
        else:
            task = loop.create_task(coro)
            self._async_tasks.append(task)
            try:
                loop.run_until_complete(task)
            except Exception as e:
                # Status/step updates are informational only — a DB hiccup (lock,
                # transient I/O error on WSL/NTFS, etc.) must never crash the job.
                print(f"[AITK] Warning: DB update failed (non-fatal): {e}")

    async def _execute_db_operation(self, operation_func):
        """Execute a database operation in a separate thread with retry on lock."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool, lambda: self._retry_db_operation(operation_func)
        )

    def _db_connect(self):
        """Create a new connection for each operation to avoid locking."""
        conn = sqlite3.connect(self.sqlite_db_path, timeout=30.0)
        conn.isolation_level = None  # Enable autocommit mode
        return conn

    def _retry_db_operation(self, operation_func, max_retries=3, base_delay=2.0):
        """Retry a database operation with exponential backoff on lock/transient I/O errors."""
        last_error = None
        retryable = ("database is locked", "disk i/o error")
        for attempt in range(max_retries + 1):
            try:
                return operation_func()
            except sqlite3.OperationalError as e:
                if any(r in str(e).lower() for r in retryable):
                    last_error = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)  # 2s, 4s, 8s
                        print(f"[AITK] Database error ({e}) (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        print(f"[AITK] Database error persisted after {max_retries + 1} attempts, giving up.")
                else:
                    raise
        raise last_error

    def should_stop(self):
        if not self.is_ui_trainer:
            return False
        def _check_stop():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT stop FROM Job WHERE id = ?", (self.job_id,))
                stop = cursor.fetchone()
                return False if stop is None else stop[0] == 1

        return self._retry_db_operation(_check_stop)

    def should_return_to_queue(self):
        if not self.is_ui_trainer:
            return False
        def _check_return_to_queue():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT return_to_queue FROM Job WHERE id = ?", (self.job_id,))
                return_to_queue = cursor.fetchone()
                return False if return_to_queue is None else return_to_queue[0] == 1

        return self._retry_db_operation(_check_return_to_queue)

    def should_stop_after_save(self):
        """Cooperative 'save then stop' (our save-and-pause). Deliberately a
        separate flag from `stop`: the stop-watcher thread raises SIGINT the
        moment it sees `stop`, which killed the training step before the
        checkpoint was ever written. Nothing watches this one but maybe_stop()."""
        if not self.is_ui_trainer:
            return False
        def _check():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT stop_after_save FROM Job WHERE id = ?", (self.job_id,))
                row = cursor.fetchone()
                return False if row is None else row[0] == 1

        return self._retry_db_operation(_check)

    def reset_stop_after_save(self):
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self.update_db_key("stop_after_save", 0)

    def should_save(self):
        # Reads the `save_now` flag (ostris' canonical on-demand-save schema).
        # Save-and-pause pairs it with `stop_after_save`; see maybe_save/maybe_stop.
        if not self.is_ui_trainer:
            return False
        def _check_save():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT save_now FROM Job WHERE id = ?", (self.job_id,))
                save_now = cursor.fetchone()
                return False if save_now is None else save_now[0] == 1

        return self._retry_db_operation(_check_save)

    def reset_save(self):
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self.update_db_key("save_now", 0)

    def reset_sample(self):
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self.update_db_key("sample", False)

    def should_stop_sample(self):
        """The 'Return to Training' button: abandon the sample, keep training."""
        if not self.is_ui_trainer:
            return False

        def _check():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT stop_sample FROM Job WHERE id = ?", (self.job_id,))
                row = cursor.fetchone()
                return False if row is None else row[0] == 1

        return self._retry_db_operation(_check)

    def reset_stop_sample(self):
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self.update_db_key("stop_sample", False)

    def should_skip_sample(self):
        """The 'Skip' button on the live preview: abandon only the clip
        currently rendering, unlike stop_sample which abandons the whole
        batch. Checked every denoise step via the maybe_skip hook below, not
        just once per image, so it lands within one step of being clicked."""
        if not self.is_ui_trainer:
            return False

        def _check():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT skip_sample FROM Job WHERE id = ?", (self.job_id,))
                row = cursor.fetchone()
                return False if row is None else row[0] == 1

        return self._retry_db_operation(_check)

    def reset_skip_sample(self):
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self.update_db_key("skip_sample", False)

    def _maybe_skip_sample_hook(self):
        """Registered on the model via add_maybe_skip_hook; raises to unwind
        out of the current clip's denoise loop. See BaseModel.maybe_skip_sample
        and generate_images, which catches this around the single-image call."""
        if self.should_skip_sample():
            self.reset_skip_sample()
            raise SampleSkippedException("Sample skipped by user")

    def _reset_control_flags(self):
        """Clear the on-demand control flags (and the launcher-owned `pid`) once
        this job has committed to a terminal state (stopped/queued). Node never
        clears these on a clean, self-reported exit -- see the comment on
        watchDetachedJob in ui/cron/actions/startJob.ts -- so if we don't do it
        here they persist forever and the UI keeps treating the row as busy/in
        flight (stale pid, stop/save/save_now still set)."""
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self.update_db_key("stop", False)
            self.update_db_key("save", False)
            self.update_db_key("save_now", 0)
            self.update_db_key("return_to_queue", False)
            # Must be cleared here, not only in the should_stop_after_save()
            # branch below. Save and Stop Queue sets return_to_queue AND
            # stop_after_save together, and maybe_stop() checks return_to_queue
            # first -- so that branch raises and the reset in the stop_after_save
            # branch is never reached, leaving the flag set forever.
            self.update_db_key("stop_after_save", 0)
            self.update_db_key("pid", None)

    def maybe_stop(self):
        if not self.is_ui_trainer:
            return
        # Hard stop: the user asked to stop now, nothing to wait for.
        if self.should_stop():
            self._run_async_operation(
                self._update_status("stopped", "Job stopped"))
            self._reset_control_flags()
            self.is_stopping = True
            raise JobStoppedException("Job stopped")
        # Cooperative stops below must never pre-empt a pending save --
        # save-and-pause / save-and-requeue set save_now alongside them and the
        # checkpoint has to land first. maybe_save() clears save_now, so these
        # fire on the very next call (including save()'s own trailing one).
        if self.should_save():
            return
        if self.should_return_to_queue():
            self._run_async_operation(
                self._update_status("queued", "Job queued"))
            self._reset_control_flags()
            self.is_stopping = True
            raise JobStoppedException("Job returning to queue")
        if self.should_stop_after_save():
            self.reset_stop_after_save()
            self._run_async_operation(
                self._update_status("stopped", "Job stopped"))
            self._reset_control_flags()
            self.is_stopping = True
            raise JobStoppedException("Job stopped")

    def maybe_save(self):
        """Returns True if a checkpoint was actually written this step, so the
        caller can tell maybe_sample() not to write an identical one again."""
        if not self.is_ui_trainer:
            return False
        if self.should_save():
            # raise the guard before reset_save() clears the flag, so there is no
            # window where the watcher sees `stop` with no save pending and kills
            # the step before the write starts
            self._save_guard += 1
            try:
                self.reset_save()
                if self.progress_bar is not None:
                    self.progress_bar.pause()
                print_acc(f"\nSaving at step {self.step_num}")
                self.optimizer.zero_grad()
                self.save(self.step_num)
                self.ensure_params_requires_grad()
                flush()
                if self.progress_bar is not None:
                    self.progress_bar.unpause()
            finally:
                self._save_guard -= 1
            return True
        return False

    def reload_sample_config(self):
        """Re-read sample config from the DB in case prompts were edited while running."""
        if not self.is_ui_trainer:
            return
        try:
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT job_config FROM Job WHERE id = ?", (self.job_id,))
                row = cursor.fetchone()
                if row:
                    job_cfg = json.loads(row[0])
                    sample_conf = job_cfg.get('config', {}).get('process', [{}])[0].get('sample', {})
                    if sample_conf:
                        self.sample_config = SampleConfig(**sample_conf)
                        # prompt embeds are precomputed and indexed by position when the
                        # text encoder is unloaded/cached; rebuild that cache so newly
                        # added prompts have a matching entry
                        if self.sd.sample_prompts_cache is not None:
                            self.cache_sample_prompts()
        except Exception as e:
            print(f"Warning: Could not reload sample config from DB: {e}")

    def should_sample(self):
        if not self.is_ui_trainer:
            return False
        def _check_sample():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT sample FROM Job WHERE id = ?", (self.job_id,))
                sample = cursor.fetchone()
                return False if sample is None else sample[0] == 1

        return self._retry_db_operation(_check_sample)

    def should_sample_now(self):
        """Check the lightweight sample_now flag (no config reload or save)."""
        if not self.is_ui_trainer:
            return False
        def _check_sample_now():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT sample_now FROM Job WHERE id = ?", (self.job_id,))
                sample_now = cursor.fetchone()
                return False if sample_now is None else sample_now[0] == 1

        return self._retry_db_operation(_check_sample_now)

    def maybe_sample(self, already_saved: bool = False):
        if not self.is_ui_trainer:
            return
        if self.should_sample():
            self.reload_sample_config()
            self.reset_sample()
            # clear any abort/skip left over from a previous sample. Without
            # this a single stale flag would abort or skip every future
            # sample, and nothing else ever clears it.
            self.reset_stop_sample()
            self.reset_skip_sample()
            # save model and optimizer first as requested, UNLESS maybe_save()
            # already wrote this exact step (both flags land together whenever
            # Save Snapshot is clicked while an on-demand sample is pending).
            # Saving twice rewrote the same checkpoint and made save() archive
            # the optimizer it had just written as optimizer_<thisstep>.pt.
            if not already_saved:
                self.save(self.step_num)
            # then sample
            self.sample(self.step_num)

    def maybe_sample_now(self):
        """Lightweight on-demand sample triggered by the 'Sample Next Step' gear menu item.
        Unlike maybe_sample(), does not reload config or save first."""
        if not self.is_ui_trainer:
            return
        if self.should_sample_now():
            self.update_db_key("sample_now", 0)
            self.reset_stop_sample()
            self.reset_skip_sample()
            if self.progress_bar is not None:
                self.progress_bar.pause()
            print_acc(f"\nSampling at step {self.step_num}")
            self.optimizer.zero_grad()
            if self.train_config.free_u:
                self.sd.pipeline.disable_freeu()
            self.sample(self.step_num)
            if self.train_config.unload_text_encoder:
                self.sd.text_encoder_to('cpu')
            self.ensure_params_requires_grad()
            flush()
            if self.progress_bar is not None:
                self.progress_bar.unpause()

    async def _update_key(self, key, value):
        if not self.accelerator.is_main_process:
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    # Convert the value to appropriate SQLite type
                    if isinstance(value, bool):
                        value_to_insert = 1 if value else 0
                    elif isinstance(value, (int, float, str)) or value is None:
                        value_to_insert = value
                    else:
                        value_to_insert = str(value)

                    # Use parameterized query for both the column name and value
                    update_query = f"UPDATE Job SET {key} = ? WHERE id = ?"
                    cursor.execute(
                        update_query, (value_to_insert, self.job_id))
                except Exception:
                    try:
                        cursor.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
                else:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    async def _update_step_and_epoch(self):
        if not self.accelerator.is_main_process:
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    cursor.execute(
                        "UPDATE Job SET step = ?, epoch = ? WHERE id = ?",
                        (int(self.step_num), int(self.epoch_num), self.job_id)
                    )
                except Exception:
                    try:
                        cursor.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
                else:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    def update_step(self):
        """Non-blocking update of the step count (and the epoch it belongs to).

        Both columns go out in a single UPDATE so the UI can never read a step
        from one moment paired with an epoch from another.
        """
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self._note_epoch_progress()
            self._run_async_operation(self._update_step_and_epoch())

    def _persist_steps_per_epoch(self):
        """Mark epoch tracking as live for this session but not yet measured.

        -1 is a sentinel meaning "this job reports epochs, length still unknown";
        0 means "no epoch info at all" (a job that has not run since the epoch
        columns were added). The UI distinguishes the two so it can show the epoch
        number immediately and add the projected total once one is measured."""
        try:
            self.update_db_key("steps_per_epoch", -1.0)
        except Exception as e:
            print(f"[AITK] Warning: could not flag epoch tracking: {e}")

    # --- epoch length measurement -------------------------------------------
    # len(dataloader) is NOT the number of iterations the train loop pulls in a
    # pass (bucketed datasets batch inside the dataset, reg datasets alternate,
    # and the dataset can change size between sessions), so epoch length is
    # measured from observed rollovers instead: the gap between two consecutive
    # increments of epoch_num is exactly one epoch, by definition.
    _prev_rollover_step = None
    _last_epoch_seen = None

    def _note_epoch_progress(self):
        """Called once per step. Writes steps_per_epoch after a full epoch is observed."""
        try:
            ep = int(self.epoch_num)
            st = int(self.step_num)
            if self._last_epoch_seen is None:
                self._last_epoch_seen = ep
                return
            if ep == self._last_epoch_seen:
                return
            self._last_epoch_seen = ep
            if self._prev_rollover_step is None:
                # We joined this epoch partway through (fresh start or resume), so
                # its span is a lower bound, not a measurement. Anchor and measure
                # from the next rollover onward.
                self._prev_rollover_step = st
                return
            span = st - self._prev_rollover_step
            self._prev_rollover_step = st
            if span > 0:
                # Most recent epoch only, not an average: the dataset can grow
                # mid-run (datasets get added between sessions) and the latest
                # span is the one that projects the remaining steps correctly.
                self.update_db_key("steps_per_epoch", float(span))
        except Exception as e:
            print(f"[AITK] Warning: could not measure epoch length: {e}")

    def load_training_state_from_metadata_if_available(self):
        """Read step/epoch from the latest checkpoint metadata without loading weights.
        Safe to call early (e.g. before model load) so on_error() has the right step
        if training is stopped during the loading / quantization phase."""
        try:
            latest = self.get_latest_save_path()
            if latest is not None:
                self.load_training_state_from_metadata(latest)
        except Exception:
            pass

    def update_db_key(self, key, value):
        """Non-blocking update a key in the database."""
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self._run_async_operation(self._update_key(key, value))

    async def _update_status(self, status: AITK_Status, info: Optional[str] = None):
        if not self.accelerator.is_main_process or not self.is_ui_trainer:
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    if info is not None:
                        cursor.execute(
                            "UPDATE Job SET status = ?, info = ? WHERE id = ?",
                            (status, info, self.job_id)
                        )
                    else:
                        cursor.execute(
                            "UPDATE Job SET status = ? WHERE id = ?",
                            (status, self.job_id)
                        )
                except Exception:
                    try:
                        cursor.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
                else:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    def update_status(self, status: AITK_Status, info: Optional[str] = None):
        """Non-blocking update of status."""
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self._run_async_operation(self._update_status(status, info))

    def append_alert(self, alert_type: str, message: str, data: dict = None):
        """Append one alert to the Job.alerts JSON column (capped at 50 entries).
        Synchronous with BEGIN IMMEDIATE so concurrent alert writes can't corrupt the JSON."""
        if not self.accelerator.is_main_process or not self.is_ui_trainer:
            return
        import datetime
        alert = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "step": self.step_num,
            "type": alert_type,
            "message": message,
            "data": data or {},
        }
        try:
            def _do_append():
                with self._db_connect() as conn:
                    cursor = conn.cursor()
                    cursor.execute("BEGIN IMMEDIATE")
                    try:
                        row = cursor.execute(
                            "SELECT alerts FROM Job WHERE id = ?", (self.job_id,)
                        ).fetchone()
                        existing = json.loads(row[0]) if row and row[0] else []
                        existing.append(alert)
                        if len(existing) > 50:
                            existing = existing[-50:]
                        cursor.execute(
                            "UPDATE Job SET alerts = ? WHERE id = ?",
                            (json.dumps(existing), self.job_id)
                        )
                    except Exception:
                        try:
                            cursor.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise
                    else:
                        cursor.execute("COMMIT")
            self._retry_db_operation(_do_append)
        except Exception as e:
            print(f"[AITK] Warning: could not write alert to DB: {e}")

    def preserve_safe_snapshot(self, reason: str):
        """Copy the most recent checkpoint's .safetensors and .pt files to
        {save_root}/_safe_snapshots/{reason}_step_{N}/ so they survive cleanup."""
        if not self.accelerator.is_main_process or not self.is_ui_trainer:
            return
        try:
            save_root = getattr(self, 'save_root', None)
            if not save_root:
                return
            # Find the most recent step_* checkpoint directory
            step_dirs = sorted(
                _glob.glob(os.path.join(save_root, "step_*")),
                key=lambda p: int(p.rsplit("_", 1)[-1]) if p.rsplit("_", 1)[-1].isdigit() else -1
            )
            if not step_dirs:
                print(f"[AITK] Safe snapshot: no checkpoint found to snapshot (step {self.step_num})")
                return
            src_dir = step_dirs[-1]
            snap_name = f"{reason}_step_{self.step_num}"
            dst_dir = os.path.join(save_root, "_safe_snapshots", snap_name)
            os.makedirs(dst_dir, exist_ok=True)

            copied = []
            for pattern in ("*.safetensors", "*.pt"):
                for fpath in _glob.glob(os.path.join(src_dir, pattern)):
                    dst = os.path.join(dst_dir, os.path.basename(fpath))
                    shutil.copy2(fpath, dst)
                    copied.append(os.path.basename(fpath))

            if not copied:
                print(f"[AITK] Safe snapshot: no .safetensors/.pt files found in {src_dir}")
                return

            import datetime
            readme = (
                f"Safe snapshot — training anomaly detected\n"
                f"Reason: {reason}\n"
                f"Detected at step: {self.step_num}\n"
                f"Timestamp: {datetime.datetime.utcnow().isoformat()}Z\n"
                f"Source checkpoint: {src_dir}\n"
                f"Files preserved: {', '.join(copied)}\n\n"
                f"To restore:\n"
                f"  1. Copy the .safetensors file(s) back to the output directory.\n"
                f"  2. Set 'resume_lora_model' in your config to point at the .safetensors.\n"
                f"  3. For WAN 2.2: both high and low noise files must be present.\n"
                f"  4. Restart training from the UI.\n"
            )
            with open(os.path.join(dst_dir, "README.txt"), "w") as f:
                f.write(readme)

            print(f"[AITK] Safe snapshot saved to {dst_dir} ({len(copied)} file(s): {', '.join(copied)})")
        except Exception as e:
            print(f"[AITK] Warning: safe snapshot failed: {e}")

    async def wait_for_all_async(self):
        """Wait for all tracked async operations to complete."""
        if not self._async_tasks:
            return

        try:
            await asyncio.gather(*self._async_tasks)
        except Exception as e:
            pass
        finally:
            # Clear the task list after completion
            self._async_tasks.clear()

    def _is_oom_error(self, e: Exception) -> bool:
        msg = str(e).lower()
        return (
            isinstance(e, (MemoryError,))
            or "out of memory" in msg
            or "cuda out of memory" in msg
            or "cudaerroroutofmemory" in msg
        )

    def on_error(self, e: Exception):
        super(DiffusionTrainer, self).on_error(e)
        if self.is_ui_trainer:
            try:
                if self.accelerator.is_main_process and not self.is_stopping:
                    self.update_status("error", str(e))
                    self.update_db_key("step", self.last_save_step)
                else:
                    # If it's a KeyboardInterrupt, mark as stopped instead of error
                    if not self.is_stopping and (isinstance(e, KeyboardInterrupt) or "Job stopped" in str(e)):
                        self.update_status("stopped", "Job stopped by user")
                    if isinstance(e, KeyboardInterrupt):
                        # silence the bar so tqdm doesn't repaint it at interpreter exit
                        progress_bar = getattr(self, "progress_bar", None)
                        if progress_bar is not None:
                            progress_bar.disable = True
                            progress_bar.close()
                    # On intentional stop/pause (including SIGINT), preserve the current step count
                    self.update_db_key("step", self.step_num)
                asyncio.run(self.wait_for_all_async())
            except Exception as db_err:
                print(f"[AITK] Warning: failed to update DB during error handling: {db_err}")
            finally:
                self.thread_pool.shutdown(wait=True)

    def handle_timing_print_hook(self, timing_dict):
        # pull the rate from the progress bar's EMA so the UI matches it exactly
        rate = None  # iter/sec
        if self.progress_bar is not None:
            rate = self.progress_bar.format_dict.get("rate")
        if rate:
            if rate >= 1:
                self.update_db_key("speed_string", f"{rate:.2f} iter/sec")
            else:
                self.update_db_key("speed_string", f"{1 / rate:.2f} sec/iter")
            return
        # fallback: bar not available yet (no rate until its first refresh)
        if "train_loop" not in timing_dict:
            print("train_loop not found in timing_dict", timing_dict)
            return
        seconds_per_iter = timing_dict["train_loop"]
        # determine iter/sec or sec/iter
        if seconds_per_iter < 1:
            iters_per_sec = 1 / seconds_per_iter
            self.update_db_key("speed_string", f"{iters_per_sec:.2f} iter/sec")
        else:
            self.update_db_key(
                "speed_string", f"{seconds_per_iter:.2f} sec/iter")

    def done_hook(self):
        super(DiffusionTrainer, self).done_hook()
        if self.is_ui_trainer:
            if self.sample_only:
                # Restore the status the job had before sample-only mode started
                previous_status = os.environ.get("AITK_PREVIOUS_STATUS", "stopped")
                self.update_status(previous_status, "Sampling complete")
            else:
                self.update_status("completed", "Training completed")
            # Wait for all async operations to finish before shutting down
            asyncio.run(self.wait_for_all_async())
            self.thread_pool.shutdown(wait=True)

    def _check_loss_spike(self):
        """Update rolling loss history and fire an alert if a sustained spike is detected.

        A single high-loss step is normal per-sample variance (small dataset, diverse
        bucket sizes, weighted timestep sampling).  We only alert when the elevated loss
        persists for 3+ consecutive steps, which indicates a genuine training instability
        rather than one hard image.
        """
        loss = getattr(self, '_last_step_loss', 0.0)
        # Compute rolling avg from history BEFORE appending so the spike doesn't
        # inflate its own detection threshold.
        if len(self._loss_history) >= 10:
            rolling_avg = sum(self._loss_history) / len(self._loss_history)
        else:
            rolling_avg = None
        self._loss_history.append(loss)
        if rolling_avg is None:
            return
        if loss > rolling_avg * 3 and loss > 0.4:
            self._spike_streak += 1
        else:
            self._spike_streak = 0
        if (self._spike_streak >= 3
                and self.step_num - self._last_spike_step > 10):
            self._last_spike_step = self.step_num
            msg = (f"Sustained loss spike at step {self.step_num}: {loss:.4f} "
                   f"(rolling avg {rolling_avg:.4f}, {loss/rolling_avg:.1f}×, "
                   f"{self._spike_streak} consecutive steps)")
            print(f"\n[AITK] ⚠ {msg}")
            self.append_alert("loss_spike", msg, {
                "current_loss": loss,
                "rolling_avg": rolling_avg,
                "ratio": round(loss / rolling_avg, 2),
                "streak": self._spike_streak,
            })
            self.preserve_safe_snapshot("loss_spike")

    # Rolling-median window for stall detection, how long loss can go without a
    # new low before we call it stalled, how much improvement counts as a genuine
    # new low (filters noise so a run doesn't reset its own clock every step), and
    # how early training can start being evaluated. Calibrated against this
    # project's own run history: every labelled-good ACE-Step run (including
    # noisy automagic2/3 runs that dip and partially recover mid-run) went at
    # most ~2,600 steps without a new low; every automagic3 run that flatlined
    # and never recovered went >=3,750 steps. 3000 sits in that gap.
    STALL_WINDOW = 300
    STALL_PATIENCE = 3000
    STALL_MARGIN = 0.01
    STALL_MIN_STEP = 2500

    def _check_loss_stall(self):
        """Fire an alert if the loss hasn't set a new (meaningfully lower) rolling
        median in STALL_PATIENCE steps.

        Unlike _check_loss_spike (a single bad step), this catches a run that never
        diverges but also never learns — e.g. automagic3's per-tensor LR decaying
        toward its floor before the model fits anything, which produces a loss curve
        that looks calm (no spikes) but never moves. A plain "loss went up over
        window X" check would false-alarm on good automagic2/3 runs, which are
        noisy and dip-then-partially-recover mid-run without ever being stalled;
        tracking the best-seen rolling median (like early-stopping patience) avoids
        that because those runs keep setting new lows overall, just non-monotonically.
        """
        loss = getattr(self, '_last_step_loss', 0.0)
        self._stall_window.append(loss)
        if self.step_num < self.STALL_MIN_STEP or len(self._stall_window) < self.STALL_WINDOW:
            return
        median = statistics.median(self._stall_window)
        if self._stall_best_median is None or median < self._stall_best_median * (1 - self.STALL_MARGIN):
            self._stall_best_median = median
            self._stall_best_step = self.step_num
            return
        gap = self.step_num - self._stall_best_step
        if (gap >= self.STALL_PATIENCE
                and self.step_num - self._last_stall_alert_step >= self.STALL_PATIENCE):
            self._last_stall_alert_step = self.step_num
            msg = (f"Loss hasn't improved in {gap} steps (best {self._stall_best_median:.4f} "
                   f"at step {self._stall_best_step}, currently {median:.4f}). This matches "
                   f"the stall pattern seen in past runs that never recovered — worth checking "
                   f"the optimizer/LR rather than waiting it out.")
            print(f"\n[AITK] ⚠ {msg}")
            self.append_alert("loss_stalled", msg, {
                "best_median": self._stall_best_median,
                "best_step": self._stall_best_step,
                "current_median": median,
                "gap_steps": gap,
            })

    def end_step_hook(self):
        super(DiffusionTrainer, self).end_step_hook()
        self._check_loss_spike()
        self._check_loss_stall()
        if self.is_ui_trainer:
            self.update_step()
            # Order matters: maybe_save() runs before maybe_stop() so that
            # save-and-pause (save_now + stop set together) writes the checkpoint
            # before the stop is raised. save()'s trailing maybe_stop() then stops
            # cleanly. maybe_sample() is our on-demand sample feature.
            saved_this_step = self.maybe_save()
            self.maybe_sample(already_saved=saved_this_step)
            self.maybe_sample_now()
            self.maybe_stop()

    def hook_before_model_load(self):
        super().hook_before_model_load()
        # Validate sample LoRA paths before loading anything — fail fast rather than
        # discovering a bad path only when the first sample is triggered mid-training.
        from toolkit.util.get_model import get_model_class
        ModelClass = get_model_class(self.model_config)
        if hasattr(ModelClass, 'validate_sample_lora_paths'):
            sample_configs = list({id(c): c for c in [self.sample_config, self.first_sample_config] if c is not None}.values())
            ModelClass.validate_sample_lora_paths(self.model_config, *sample_configs)
        if self.is_ui_trainer:
            # Pre-load step from checkpoint before the first maybe_stop() call so
            # on_error() has the right step even if we stop during model
            # loading / quantization (checkpoint weights aren't loaded until much later).
            if self.step_num == 0:
                self.load_training_state_from_metadata_if_available()
            self.maybe_stop()
            self.update_status("running", "Loading model")

    def before_dataset_load(self):
        super().before_dataset_load()
        if self.is_ui_trainer:
            self.maybe_stop()
            self.update_status("running", "Loading dataset")

    def _persist_dataset_stats(self):
        """Write image count + bucket distribution to Job.dataset_stats after caching."""
        try:
            all_datasets = []
            if self.datasets:
                all_datasets.extend(self.datasets)
            if self.datasets_reg:
                all_datasets.extend(self.datasets_reg)
            if not all_datasets:
                return
            total_images = sum(len(ds.file_list) for ds in all_datasets if hasattr(ds, 'file_list'))
            buckets: dict = {}
            for ds in all_datasets:
                if not hasattr(ds, 'buckets') or not isinstance(ds.buckets, dict) or not ds.buckets:
                    continue
                for key, bucket in ds.buckets.items():
                    count = len(getattr(bucket, 'file_list_idx', []))
                    buckets[key] = buckets.get(key, 0) + count
            stats = {"total_images": total_images, "buckets": buckets}
            self.update_db_key("dataset_stats", json.dumps(stats))
        except Exception as e:
            print(f"[AITK] Warning: could not persist dataset stats: {e}")

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        if self.is_ui_trainer:
            self.maybe_stop()
            # Clear any stale save flag left over from a previous session that was
            # stopped before completing a step (e.g. killed during model loading /
            # quantization).  No steps have run yet this session, so there is nothing
            # new to save.
            self.reset_save()
            self.update_step()
            self.update_status("running", "Training")
            self.timer.add_after_print_hook(self.handle_timing_print_hook)
            self._persist_dataset_stats()
            self._persist_steps_per_epoch()

    def status_update_hook_func(self, string):
        self.update_status("running", string)

    def hook_after_sd_init_before_load(self):
        super().hook_after_sd_init_before_load()
        if self.is_ui_trainer:
            self.maybe_stop()
            self.sd.add_status_update_hook(self.status_update_hook_func)
            self.sd.add_maybe_stop_hook(self.maybe_stop)
            self.sd.add_maybe_skip_hook(self._maybe_skip_sample_hook)

    def sample_step_hook(self, img_num, total_imgs):
        super().sample_step_hook(img_num, total_imgs)
        if self.is_ui_trainer:
            self.maybe_stop()
            if self.should_stop_sample():
                raise SampleAbortedException(
                    "Sample generation aborted by user")
            self.update_status(
                "running", f"Generating images - {img_num + 1}/{total_imgs}")

    def sample(self, step=None, is_first=False):
        self.maybe_stop()
        total_imgs = len(self.sample_config.prompts)
        self.update_status("running", f"Generating images - 0/{total_imgs}")
        if self.is_ui_trainer:
            self.logger.record_sample_start()
        try:
            try:
                super().sample(step, is_first)
            except JobStoppedException:
                raise
            except SampleAbortedException:
                # user hit Return to Training: drop the remaining prompts and
                # carry on. Clearing the flag here is what makes the next
                # sample possible at all.
                print_acc(f"\nSample generation aborted by user at step {step}")
                self.reset_stop_sample()
                self.sd._after_sample_failure()
            except Exception as e:
                if self.sample_only:
                    raise
                print(f"\nWarning: Sample generation failed at step {step}, continuing training:\n{e}")
                traceback.print_exc()
                import torch as _torch
                if self._is_oom_error(e):
                    vram_info = ""
                    try:
                        reserved = _torch.cuda.memory_reserved() / 1024**3
                        total = _torch.cuda.get_device_properties(0).total_memory / 1024**3
                        vram_info = f" ({reserved:.1f}GB reserved / {total:.1f}GB total VRAM)"
                    except Exception:
                        pass
                    self.append_alert("oom", f"Out of memory during sample generation at step {step}{vram_info}", {
                        "step": step,
                        "context": "sample_generation",
                        "error": str(e)[:300],
                    })
                self.update_status("running", f"Sample failed (step {step}), continuing training")
                _torch.cuda.empty_cache()
                self.sd._after_sample_failure()
        finally:
            if self.is_ui_trainer:
                self.logger.record_sample_end()
        self.maybe_stop()
        if self.sample_only:
            # reset sample flag in DB
            self.update_db_key("sample", False)
        else:
            self.update_status("running", "Training")

    def save(self, step=None):
        # NOTE: ostris' upstream save() leads with maybe_stop(); we intentionally
        # do NOT. Save-and-pause sets save_now + stop together, so a leading
        # maybe_stop() would raise before the model is ever written to disk. The
        # trailing maybe_stop() below handles the stop cleanly after the save.
        # The guard keeps the stop watcher from interrupting the write half-done.
        self._save_guard += 1
        try:
            self.update_status("running", "Saving model")
            super().save(step)
        finally:
            self._save_guard -= 1
        self.maybe_stop()
        self.update_status("running", "Training")
