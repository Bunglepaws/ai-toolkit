import os
import sys
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

seed = None
if "SEED" in os.environ:
    try:
        seed = int(os.environ["SEED"])
    except ValueError:
        print(f"Invalid SEED value: {os.environ['SEED']}. SEED must be an integer.")

sys.path.insert(0, os.getcwd())
os.environ['DISABLE_TELEMETRY'] = 'YES'

import time as _time
os.environ['AITK_PROCESS_START'] = str(_time.time())
del _time


def _early_log(msg: str):
    """Write a line directly to the --log file before setup_log_to_file takes over.

    setup_log_to_file opens in append mode so these lines are preserved.
    """
    try:
        argv = sys.argv
        for i, arg in enumerate(argv[:-1]):
            if arg in ('--log', '-l'):
                log_path = argv[i + 1]
                os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
                with open(log_path, 'a') as _f:
                    _f.write(msg + '\n')
                break
    except Exception:
        pass
    print(msg, flush=True)


_early_log("AI Toolkit: loading libraries (torch / diffusers)...")

import gc
import json
import sqlite3
import argparse
import torch

_early_log("AI Toolkit: libraries loaded, initializing CUDA / accelerator...")

if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    torch.autograd.set_detect_anomaly(True)

if seed is not None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file
from toolkit.ui_utils import update_job_status_to_ui, JobStoppedException

accelerator = get_accelerator()
_early_log("AI Toolkit: accelerator ready, reading job config...")


def _get_db_path(config_file: str) -> str:
    try:
        with open(config_file) as f:
            cfg = json.load(f)
        return cfg.get('config', {}).get('process', [{}])[0].get(
            'sqlite_db_path',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aitk_db.db')
        )
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aitk_db.db')


def _restore_lora_hooks(trainer):
    """Restore original module forward methods that training LoRA hooks replaced."""
    network = getattr(trainer, 'network', None)
    if network is None:
        return 0
    restored = 0
    for lora_list in (getattr(network, 'unet_loras', []), getattr(network, 'text_encoder_loras', [])):
        for lora in lora_list:
            if hasattr(lora, 'org_forward') and hasattr(lora, 'org_module'):
                try:
                    lora.org_module[0].forward = lora.org_forward
                    restored += 1
                except Exception as e:
                    print_acc(f" - Model cache: warning: could not restore forward for {getattr(lora, 'lora_name', '?')}: {e}")
    return restored


def _effective_qtype(model_cfg: dict) -> str:
    """Apply the same qtype override logic as ModelConfig.__init__ to a raw config dict."""
    qtype = model_cfg.get('qtype', 'qfloat8')
    layer_offloading = model_cfg.get('layer_offloading', model_cfg.get('auto_memory', False))
    compile_ = model_cfg.get('compile', False)
    if layer_offloading and qtype == 'qfloat8':
        qtype = 'float8'
    if compile_ and qtype == 'qfloat8':
        qtype = 'float8'
    return qtype


def _query_next_matching_job(db_path: str, hot_sd, gpu_ids: str):
    """Return the next queued job row that can reuse hot_sd, or None."""
    # Use instance arch first (StableDiffusion stores it on self); fall back to class attr
    hot_arch = getattr(hot_sd, 'arch', None) or getattr(type(hot_sd), 'arch', None)
    if hot_arch is None:
        return None

    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, name, job_config FROM Job "
                "WHERE status='queued' AND gpu_ids=? AND job_type='train' "
                "ORDER BY queue_position ASC LIMIT 10",
                (gpu_ids,)
            ).fetchall()
        finally:
            conn.close()
    except Exception as e:
        print_acc(f" - Model cache: DB query failed ({e})")
        return None

    for row in rows:
        try:
            cfg = json.loads(row['job_config'])
            if cfg.get('job') != 'extension':
                continue
            proc = cfg.get('config', {}).get('process', [{}])[0]
            m = proc.get('model', {})

            if m.get('arch', '') != hot_arch:
                continue
            if bool(m.get('quantize', False)) != bool(hot_sd.model_config.quantize):
                continue
            if m.get('name_or_path', '') != hot_sd.model_config.name_or_path:
                continue
            if _effective_qtype(m) != hot_sd.model_config.qtype:
                continue

            return dict(row)
        except Exception:
            continue
    return None


def _claim_and_prepare_job(row: dict, db_path: str, training_root: str) -> tuple:
    """Atomically claim a queued job and write its config file.

    Returns (config_path, job_id) on success, (None, None) if the race was lost.
    """
    job_id = row['id']
    job_name = row['name']

    cfg = json.loads(row['job_config'])
    cfg['config']['process'][0]['sqlite_db_path'] = db_path

    training_folder = os.path.join(training_root, job_name)
    os.makedirs(training_folder, exist_ok=True)
    config_path = os.path.join(training_folder, '.job_config.json')
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    # Write pid.txt for the UI
    try:
        with open(os.path.join(training_folder, 'pid.txt'), 'w') as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    # Atomic claim: only succeeds if the row is still 'queued'
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            conn.execute(
                "UPDATE Job SET status='running', pid=?, info='Starting job...' "
                "WHERE id=? AND status='queued'",
                (os.getpid(), job_id)
            )
            conn.commit()
            changed = conn.execute("SELECT changes()").fetchone()[0]
        finally:
            conn.close()
    except Exception as e:
        print_acc(f" - Model cache: failed to claim job {job_id}: {e}")
        return None, None

    if changed == 0:
        return None, None  # Lost the race to another process

    return config_path, job_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file_list', nargs='+', type=str)
    parser.add_argument('-r', '--recover', action='store_true')
    parser.add_argument('-n', '--name', type=str, default=None)
    parser.add_argument('-l', '--log', type=str, default=None)
    args = parser.parse_args()

    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    is_ui = os.getenv("IS_AI_TOOLKIT_UI", "0") == "1"
    job_id = os.getenv("AITK_JOB_ID", None)
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    jobs_completed = 0
    jobs_failed = 0

    # Persistent-mode state
    config_file = config_file_list[0]
    db_path = _get_db_path(config_file)
    # config lives at <training_root>/<job_name>/.job_config.json
    training_root = os.path.dirname(os.path.dirname(os.path.abspath(config_file)))

    # If multiple config files were passed (direct CLI use), fall through to simple loop
    multi_config = len(config_file_list) > 1

    if multi_config:
        # Non-persistent: run all configs sequentially, same as run.py
        if accelerator.is_main_process:
            print_acc(f"Running {len(config_file_list)} jobs")
        for config_file in config_file_list:
            try:
                job = get_job(config_file, args.name)
                job.run()
                job.cleanup()
                jobs_completed += 1
            except JobStoppedException as e:
                print_acc(f"Job intentionally stopped: {e}")
                try:
                    job.process[0].on_error(e)
                except Exception:
                    pass
                if is_ui:
                    sys.exit(0)
            except Exception as e:
                print_acc(f"Error running job: {e}")
                jobs_failed += 1
                if is_ui and job_id:
                    update_job_status_to_ui(job_id, 'error', f"Error: {str(e)}")
                try:
                    job.process[0].on_error(e)
                except Exception:
                    pass
                if not args.recover:
                    raise e
        return

    # --- Persistent single-job loop ---
    while True:
        hot_sd = None
        try:
            job = get_job(config_file, args.name)
            job.run()

            trainer = job.process[0]

            # Capture model and clean up LoRA hooks before cleanup() deletes the trainer
            hot_sd = getattr(trainer, 'sd', None)
            if hot_sd is not None:
                restored = _restore_lora_hooks(trainer)
                if restored:
                    print_acc(f" - Model cache: cleaned up {restored} LoRA hooks from transformer")
                # Release LoRA weight tensors promptly
                trainer.network = None

            job.cleanup()
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            jobs_completed += 1

        except JobStoppedException as e:
            print_acc(f"Job intentionally stopped: {e}")
            trainer = None
            try:
                trainer = job.process[0]
                trainer.on_error(e)
            except Exception:
                pass

            # Even on a user stop, try to hand off the model to the next queued job
            # so it doesn't have to re-quantize. Only skip this for return_to_queue
            # (which re-queues the current job — we'd pick it up immediately otherwise).
            hot_sd = None
            if trainer is not None and is_ui and accelerator.is_main_process:
                # Check the DB: if the current job is now 'queued', it was return_to_queue
                is_return_to_queue = False
                try:
                    _conn = sqlite3.connect(db_path, timeout=5)
                    _row = _conn.execute("SELECT status FROM Job WHERE id=?", (job_id,)).fetchone()
                    _conn.close()
                    is_return_to_queue = bool(_row and _row[0] == 'queued')
                except Exception:
                    pass
                if not is_return_to_queue:
                    hot_sd = getattr(trainer, 'sd', None)
                    if hot_sd is not None:
                        restored = _restore_lora_hooks(trainer)
                        if restored:
                            print_acc(f" - Model cache: cleaned up {restored} LoRA hooks after stop")
                        trainer.network = None
                        del trainer
                        gc.collect()
                        torch.cuda.empty_cache()

                        next_row = _query_next_matching_job(db_path, hot_sd, gpu_ids)
                        if next_row is not None:
                            next_config_path, next_job_id = _claim_and_prepare_job(
                                next_row, db_path, training_root
                            )
                            if next_config_path is not None:
                                from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
                                BaseSDTrainProcess._hot_model = hot_sd
                                config_file = next_config_path
                                job_id = next_job_id
                                os.environ["AITK_JOB_ID"] = next_job_id
                                job_name = next_row['name']
                                log_path = os.path.join(training_root, job_name, 'log.txt')
                                setup_log_to_file(log_path)
                                print_acc(f" - Model cache: transitioning to '{job_name}' after stop (skipping load+quantize)")
                                continue

            if is_ui:
                sys.exit(0)
            break

        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            if is_ui and job_id:
                update_job_status_to_ui(job_id, 'error', f"Error: {str(e)}")
            try:
                job.process[0].on_error(e)
            except Exception:
                pass
            if not args.recover:
                raise e
            break

        # Try to pick up next queued job with same model (persistent caching)
        if hot_sd is not None and is_ui and accelerator.is_main_process:
            next_row = _query_next_matching_job(db_path, hot_sd, gpu_ids)
            if next_row is not None:
                next_config_path, next_job_id = _claim_and_prepare_job(
                    next_row, db_path, training_root
                )
                if next_config_path is not None:
                    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
                    BaseSDTrainProcess._hot_model = hot_sd

                    config_file = next_config_path
                    job_id = next_job_id
                    os.environ["AITK_JOB_ID"] = next_job_id

                    # Rotate log to new job's folder
                    job_name = next_row['name']
                    log_path = os.path.join(training_root, job_name, 'log.txt')
                    setup_log_to_file(log_path)

                    print_acc(f" - Model cache: transitioning to '{job_name}' (skipping load+quantize)")
                    continue  # Loop back — no model reload

        break  # No more matching jobs, exit normally


if __name__ == '__main__':
    main()
