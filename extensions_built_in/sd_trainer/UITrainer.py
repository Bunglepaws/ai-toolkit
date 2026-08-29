from collections import OrderedDict
import json
import os
import sqlite3
import asyncio
import concurrent.futures
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import SampleConfig
from toolkit.ui_utils import JobStoppedException, SampleAbortedException
from typing import Literal, Optional
import threading
import time
import signal

AITK_Status = Literal["running", "stopped", "error", "completed"]


class UITrainer(SDTrainer):
    # See DiffusionTrainer.STOP_WATCHER_SAVE_GRACE_SEC.
    STOP_WATCHER_SAVE_GRACE_SEC = 300

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(UITrainer, self).__init__(process_id, job, config, **kwargs)
        self.sqlite_db_path = self.config.get("sqlite_db_path", "./aitk_db.db")
        if not os.path.exists(self.sqlite_db_path):
            raise Exception(
                f"SQLite database not found at {self.sqlite_db_path}")
        print(f"Using SQLite database at {self.sqlite_db_path}")
        self.job_id = os.environ.get("AITK_JOB_ID", None)
        self.job_id = self.job_id.strip() if self.job_id is not None else None
        print(f"Job ID: \"{self.job_id}\"")
        if self.job_id is None:
            raise Exception("AITK_JOB_ID not set")
        self.is_stopping = False
        # >0 while a checkpoint write is pending or in flight; the stop watcher
        # refuses to interrupt while raised. See DiffusionTrainer for the full note.
        self._save_guard = 0
        # Create a thread pool for database operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Track all async tasks
        self._async_tasks = []
        # Initialize the status
        self._run_async_operation(self._update_status("running", "Starting"))
        # Clear any `stop_after_save` left over from a previous run before the
        # first maybe_stop() can act on it. See DiffusionTrainer for the full note.
        self.update_db_key("stop_after_save", 0)
        self._stop_watcher_started = False
        if os.name == "nt":
            # On Windows the stop route cannot send us SIGINT from outside
            # (no console to deliver a Ctrl+C to), so watch the stop flag
            # and raise the interrupt from inside. On Linux the route
            # sends a real SIGINT to the pid and this is unnecessary.
            self.start_stop_watcher(interval_sec=2.0)
    
    def start_stop_watcher(self, interval_sec: float = 5.0):
        """
        Start a daemon thread that periodically checks should_stop()
        and terminates the process immediately when triggered.
        """
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
                    # `stop` now unambiguously means "interrupt now"; save-then-stop
                    # uses stop_after_save, which this thread never reads. Only an
                    # in-flight write is worth waiting for. See DiffusionTrainer.
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
            loop.run_until_complete(task)

    async def _execute_db_operation(self, operation_func):
        """Execute a database operation in a separate thread to avoid blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, operation_func)

    def _db_connect(self):
        """Create a new connection for each operation to avoid locking."""
        conn = sqlite3.connect(self.sqlite_db_path, timeout=10.0)
        conn.isolation_level = None  # Enable autocommit mode
        return conn

    def should_stop(self):
        def _check_stop():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT stop FROM Job WHERE id = ?", (self.job_id,))
                stop = cursor.fetchone()
                return False if stop is None else stop[0] == 1

        return _check_stop()
    
    def should_return_to_queue(self):
        def _check_return_to_queue():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT return_to_queue FROM Job WHERE id = ?", (self.job_id,))
                return_to_queue = cursor.fetchone()
                return False if return_to_queue is None else return_to_queue[0] == 1

        return _check_return_to_queue()

    def should_stop_after_save(self):
        """Cooperative 'save then stop'. Separate from `stop` on purpose -- the
        stop-watcher raises SIGINT on `stop` and would kill the step before the
        checkpoint was written. See DiffusionTrainer for the full note."""
        def _check():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT stop_after_save FROM Job WHERE id = ?", (self.job_id,))
                row = cursor.fetchone()
                return False if row is None else row[0] == 1

        return _check()

    def reset_stop_after_save(self):
        self.update_db_key("stop_after_save", 0)

    def should_save(self):
        # Reads `save_now` (ostris' canonical on-demand-save schema). Save-and-pause
        # pairs it with `stop_after_save`; see maybe_save/maybe_stop.
        def _check_save():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT save_now FROM Job WHERE id = ?", (self.job_id,))
                save_now = cursor.fetchone()
                return False if save_now is None else save_now[0] == 1

        return _check_save()

    def reset_save(self):
        self.update_db_key("save_now", 0)

    def maybe_save(self):
        """Returns True if a checkpoint was actually written this step, so the
        caller can tell maybe_sample() not to write an identical one again."""
        if self.should_save():
            # raise before reset_save() clears the flag so the watcher never sees
            # `stop` with no save pending mid-window
            self._save_guard += 1
            try:
                self.reset_save()
                self.save(self.step_num)
            finally:
                self._save_guard -= 1
            return True
        return False

    def should_sample(self):
        def _check_sample():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT sample FROM Job WHERE id = ?", (self.job_id,))
                sample = cursor.fetchone()
                return False if sample is None else sample[0] == 1

        return _check_sample()

    def reset_sample(self):
        self.update_db_key("sample", False)

    def should_stop_sample(self):
        def _check():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT stop_sample FROM Job WHERE id = ?", (self.job_id,))
                row = cursor.fetchone()
                return False if row is None else row[0] == 1
        return _check()

    def reset_stop_sample(self):
        self.update_db_key("stop_sample", False)

    def reload_sample_config(self):
        """Re-read sample config from the DB in case prompts were edited while running."""
        try:
            def _read():
                with self._db_connect() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT job_config FROM Job WHERE id = ?", (self.job_id,))
                    row = cursor.fetchone()
                    return row[0] if row else None
            raw = _read()
            if raw:
                job_cfg = json.loads(raw)
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

    def maybe_sample(self, already_saved: bool = False):
        if self.should_sample():
            self.reload_sample_config()
            self.reset_sample()
            self.reset_stop_sample()  # clear any stale abort request from a previous sample
            # skip the save if maybe_save() already wrote this exact step -- see
            # the matching note in DiffusionTrainer.maybe_sample()
            if not already_saved:
                self.save(self.step_num)
            self.sample(self.step_num)

    def _reset_control_flags(self):
        """Clear the on-demand control flags (and the launcher-owned `pid`) once
        this job has committed to a terminal state (stopped/queued). Node never
        clears these on a clean, self-reported exit -- see the comment on
        watchDetachedJob in ui/cron/actions/startJob.ts -- so if we don't do it
        here they persist forever and the UI keeps treating the row as busy/in
        flight (stale pid, stop/save/save_now still set)."""
        if self.accelerator.is_main_process:
            self.update_db_key("stop", False)
            self.update_db_key("save", False)
            self.update_db_key("save_now", 0)
            self.update_db_key("return_to_queue", False)
            # The return_to_queue branch raises before the stop_after_save branch
            # can reset it. See DiffusionTrainer for the full note.
            self.update_db_key("stop_after_save", 0)
            self.update_db_key("pid", None)

    def maybe_stop(self):
        # Hard stop: the user asked to stop now, nothing to wait for.
        if self.should_stop():
            self.is_stopping = True
            self._run_async_operation(
                self._update_status("stopped", "Job stopped"))
            self._reset_control_flags()
            raise JobStoppedException("Job stopped")
        # Cooperative stops must never pre-empt a pending save -- see the
        # matching note in DiffusionTrainer.maybe_stop().
        if self.should_save():
            return
        if self.should_return_to_queue():
            self.is_stopping = True
            self._run_async_operation(
                self._update_status("queued", "Job queued"))
            self._reset_control_flags()
            raise JobStoppedException("Job returning to queue")
        if self.should_stop_after_save():
            self.reset_stop_after_save()
            self.is_stopping = True
            self._run_async_operation(
                self._update_status("stopped", "Job stopped"))
            self._reset_control_flags()
            raise JobStoppedException("Job stopped")

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
                finally:
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
                finally:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    def update_step(self):
        """Non-blocking update of the step count (and the epoch it belongs to).

        Both columns go out in a single UPDATE so the UI can never read a step
        from one moment paired with an epoch from another.
        """
        if self.accelerator.is_main_process:
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

    def update_db_key(self, key, value):
        """Non-blocking update a key in the database."""
        if self.accelerator.is_main_process:
            self._run_async_operation(self._update_key(key, value))

    async def _update_status(self, status: AITK_Status, info: Optional[str] = None):
        if not self.accelerator.is_main_process:
            return

        def _do_update():
            # Re-assert our pid whenever we report running. The UI clears pid as
            # soon as a stop is requested, but this process can stay alive for
            # minutes inside an uninterruptible section (model load, quantization,
            # sampling). Without this the row ends up running-with-no-pid, which
            # hides us from the queue's liveness check and lets a second job start
            # on the same GPU.
            sets = ["status = ?"]
            values = [status]
            if info is not None:
                sets.append("info = ?")
                values.append(info)
            if status == "running":
                sets.append("pid = ?")
                values.append(os.getpid())
            values.append(self.job_id)
            update_query = f"UPDATE Job SET {', '.join(sets)} WHERE id = ?"

            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    cursor.execute(update_query, tuple(values))
                finally:
                    cursor.execute("COMMIT")

        await self._execute_db_operation(_do_update)

    def update_status(self, status: AITK_Status, info: Optional[str] = None):
        """Non-blocking update of status."""
        if self.accelerator.is_main_process:
            self._run_async_operation(self._update_status(status, info))

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

    def on_error(self, e: Exception):
        super(UITrainer, self).on_error(e)
        # Close the progress bar so it doesn't linger in the console output
        if getattr(self, "progress_bar", None) is not None:
            self.progress_bar.close()
            self.progress_bar = None
        # A bare KeyboardInterrupt (ctrl+c, or the Windows stop-watcher's
        # raise_signal) reaches here without maybe_stop() having run, so
        # is_stopping/status aren't set yet -- do it here. JobStoppedException
        # means maybe_stop() already set both (status may be "queued" rather
        # than "stopped" for return-to-queue), so don't touch status for it.
        if isinstance(e, KeyboardInterrupt) and not self.is_stopping:
            self.is_stopping = True
            if self.accelerator.is_main_process:
                self.update_status("stopped", "Job stopped")
        is_intentional = self.is_stopping or isinstance(e, (KeyboardInterrupt, JobStoppedException))
        if self.accelerator.is_main_process and not is_intentional:
            self.update_status("error", str(e))
            # On actual error, roll back displayed step to last known good save
            self.update_db_key("step", self.last_save_step)
        else:
            # On intentional stop/pause (including SIGINT), preserve the current step count
            self.update_db_key("step", self.step_num)
        asyncio.run(self.wait_for_all_async())
        self.thread_pool.shutdown(wait=True)

    def handle_timing_print_hook(self, timing_dict):
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
        super(UITrainer, self).done_hook()
        if self.sample_only:
            previous_status = os.environ.get("AITK_PREVIOUS_STATUS", "stopped")
            self.update_status(previous_status, "Sampling complete")
        else:
            self.update_status("completed", "Training completed")
        # Wait for all async operations to finish before shutting down
        asyncio.run(self.wait_for_all_async())
        self.thread_pool.shutdown(wait=True)

    def end_step_hook(self):
        super(UITrainer, self).end_step_hook()
        self.update_step()
        saved_this_step = self.maybe_save()
        self.maybe_sample(already_saved=saved_this_step)
        self.maybe_stop()

    def hook_before_model_load(self):
        super().hook_before_model_load()
        # Pre-load step from checkpoint before the first maybe_stop() call so
        # on_error() has the right step even if stopped during loading/quantization.
        if self.step_num == 0:
            try:
                latest = self.get_latest_save_path()
                if latest is not None:
                    self.load_training_state_from_metadata(latest)
            except Exception:
                pass
        self.maybe_stop()
        self.update_status("running", "Loading model")

    def before_dataset_load(self):
        super().before_dataset_load()
        self.maybe_stop()
        self.update_status("running", "Loading dataset")

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        self.maybe_stop()
        # Clear any stale save flag left over from a previous session that was
        # stopped before completing a step (e.g. killed during model loading /
        # quantization).  No steps have run yet this session, so there is nothing
        # new to save.
        self.reset_save()
        self.update_step()
        self.update_status("running", "Training")
        self.timer.add_after_print_hook(self.handle_timing_print_hook)
        self._persist_steps_per_epoch()

    def status_update_hook_func(self, string):
        self.update_status("running", string)

    def hook_after_sd_init_before_load(self):
        super().hook_after_sd_init_before_load()
        self.maybe_stop()
        self.sd.add_status_update_hook(self.status_update_hook_func)
        self.sd.add_maybe_stop_hook(self.maybe_stop)

    def sample_step_hook(self, img_num, total_imgs):
        super().sample_step_hook(img_num, total_imgs)
        self.maybe_stop()
        if self.should_stop_sample():
            raise SampleAbortedException("Sample generation aborted by user")
        self.update_status(
            "running", f"Generating images - {img_num + 1}/{total_imgs}")

    def sample(self, step=None, is_first=False):
        self.maybe_stop()
        total_imgs = len(self.sample_config.prompts)
        self.update_status("running", f"Generating images - 0/{total_imgs}")
        self.logger.record_sample_start()
        try:
            super().sample(step, is_first)
        except SampleAbortedException:
            # User requested early exit from sampling — reset flag and resume training
            self.reset_stop_sample()
        finally:
            self.logger.record_sample_end()
        self.maybe_stop()
        self.update_status("running", "Training")

    def save(self, step=None):
        # NOTE: do NOT call maybe_stop() here before the save begins.
        # When save_and_pause sets both save=true and stop=true, calling maybe_stop()
        # first would raise "Job stopped" before the model is ever written to disk.
        # The stop check at the end (and in end_step_hook) handles the stop cleanly
        # after the save completes.
        # The guard keeps the stop watcher from interrupting a half-written file.
        self._save_guard += 1
        try:
            self.update_status("running", "Saving model")
            super().save(step)
        finally:
            self._save_guard -= 1
        self.maybe_stop()
        self.update_status("running", "Training")
