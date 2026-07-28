from collections import OrderedDict, deque
import glob as _glob
import json
import os
import shutil
import sqlite3
import asyncio
import concurrent.futures
import traceback
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import SampleConfig
from toolkit.ui_utils import JobStoppedException
from typing import Literal, Optional
import threading
import time
import signal
from toolkit.basic import flush
from toolkit.print import print_acc

AITK_Status = Literal["running", "stopped", "error", "completed"]


class DiffusionTrainer(SDTrainer):
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
        
        if self.is_ui_trainer:
            self.is_stopping = False
            # Create a thread pool for database operations
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            # Track all async tasks
            self._async_tasks = []
            # Initialize the status
            self._run_async_operation(self._update_status("running", "Starting"))
            self._stop_watcher_started = False
            # self.start_stop_watcher(interval_sec=2.0)

        # Alert detection state (maintained regardless of is_ui_trainer so the
        # rolling history works even for non-UI runs — only DB writes are gated)
        self._loss_history: deque = deque(maxlen=50)
        self._last_step_loss: float = 0.0
        self._last_spike_step: int = -1
        self._spike_streak: int = 0
        self._baseline_sample_avg_bytes: float | None = None

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
        while True:
            try:
                if self.should_stop():
                    # Mark and update status (non-blocking; uses existing infra)
                    self.is_stopping = True
                    self._run_async_operation(
                        self._update_status("stopped", "Job stopped (remote)")
                    )
                    # Best-effort flush pending async ops
                    try:
                        asyncio.run(self.wait_for_all_async())
                    except RuntimeError:
                        pass
                    # Try to stop DB thread pool quickly
                    try:
                        self.thread_pool.shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        self.thread_pool.shutdown(wait=False)
                    print("")
                    print("****************************************************")
                    print("    Stop signal received; terminating process.      ")
                    print("****************************************************")
                    os.kill(os.getpid(), signal.SIGINT)
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

    def should_save(self):
        # Reads the `save_now` flag (ostris' canonical on-demand-save schema).
        # Save-and-pause sets `save_now` + `stop` together; see maybe_save/save.
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

        return _check_sample()

    def reset_sample(self):
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self.update_db_key("sample", False)

    def maybe_stop(self):
        if not self.is_ui_trainer:
            return
        if self.should_stop():
            self._run_async_operation(
                self._update_status("stopped", "Job stopped"))
            self.is_stopping = True
            raise JobStoppedException("Job stopped")
        if self.should_return_to_queue():
            self._run_async_operation(
                self._update_status("queued", "Job queued"))
            self.is_stopping = True
            raise JobStoppedException("Job returning to queue")

    def maybe_save(self):
        if not self.is_ui_trainer:
            return
        if self.should_save():
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
    def maybe_sample(self):
        if not self.is_ui_trainer:
            return
        if self.should_sample():
            self.reload_sample_config()
            self.reset_sample()
            # save model and optimizer first as requested
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
            if self.progress_bar is not None:
                self.progress_bar.pause()
            print_acc(f"\nSampling at step {self.step_num}")
            # clear any grads
            self.optimizer.zero_grad()
            if self.train_config.free_u:
                self.sd.pipeline.disable_freeu()

            # The actual sample call
            self.sample(self.step_num)

            # From HEAD (sampling specific teardown)
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

    def update_step(self):
        """Non-blocking update of the step count."""
        if self.accelerator.is_main_process and self.is_ui_trainer:
            self._run_async_operation(self._update_key("step", self.step_num))

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

    def end_step_hook(self):
        super(DiffusionTrainer, self).end_step_hook()
        self._check_loss_spike()
        if self.is_ui_trainer:
            self.update_step()
            # Order matters: maybe_save() runs before maybe_stop() so that
            # save-and-pause (save_now + stop set together) writes the checkpoint
            # before the stop is raised. save()'s trailing maybe_stop() then stops
            # cleanly. maybe_sample() is our on-demand sample feature.
            self.maybe_save()
            self.maybe_sample()
            self.maybe_sample_now()
            self.maybe_stop()
            self.maybe_sample()

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

    def status_update_hook_func(self, string):
        self.update_status("running", string)

    def hook_after_sd_init_before_load(self):
        super().hook_after_sd_init_before_load()
        if self.is_ui_trainer:
            self.maybe_stop()
            self.sd.add_status_update_hook(self.status_update_hook_func)
            self.sd.add_maybe_stop_hook(self.maybe_stop)

    def sample_step_hook(self, img_num, total_imgs):
        super().sample_step_hook(img_num, total_imgs)
        if self.is_ui_trainer:
            self.maybe_stop()
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
        self.update_status("running", "Saving model")
        super().save(step)
        self.maybe_stop()
        self.update_status("running", "Training")
