import os
import sys
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
# We set these before huggingface_hub imports: Xet's parallel reconstructor
# dies on large files with "Background writer channel closed". Sequential
# writes keep the fast Xet transfer and stabilize the disk side.
os.environ["HF_XET_HIGH_PERFORMANCE"] = os.getenv("HF_XET_HIGH_PERFORMANCE", "1")
os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = os.getenv(
    "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY", "1"
)
os.environ["HF_HUB_DISABLE_XET"] = os.getenv("HF_HUB_DISABLE_XET", "0")
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

    jobs_completed = 0
    jobs_failed = 0

    config_file = config_file_list[0]

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

    try:
        job = get_job(config_file, args.name)
        job.run()

        job.cleanup()
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


if __name__ == '__main__':
    main()
