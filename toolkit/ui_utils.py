import os
import requests
import json


class JobStoppedException(BaseException):
    """Raised when a job is intentionally stopped or returned to queue.
    Inherits from BaseException (not Exception) so run.py's generic
    'except Exception' handler does not catch it and overwrite the status."""
    pass


class SampleAbortedException(BaseException):
    """Raised mid-sample ('Return to Training') to abort the whole sample
    batch and resume training. Inherits from BaseException so it bypasses
    DiffusionTrainer.sample()'s broad 'except Exception' guard; that method
    catches it specifically and continues the training loop normally."""
    pass


class SampleSkippedException(BaseException):
    """Raised mid-denoise ('Skip') to abandon only the clip currently
    rendering and move on to the next prompt in the batch, unlike
    SampleAbortedException which abandons the whole batch. Inherits from
    BaseException so it passes through any model pipeline's denoise loop
    without being caught by a broad 'except Exception' there; BaseModel.
    generate_images catches it around the single-image call."""
    pass


def update_job_status_to_ui(job_id: str, status: str, info: str = None):
    ui_url = os.getenv("AITK_UI_URL", "http://localhost:8675")
    if not job_id:
        print("AITK_JOB_ID not found in environment, cannot update UI status.")
        return

    try:
        # Construct the URL for the status update endpoint
        url = f"{ui_url}/api/jobs/{job_id}/update_status"
        headers = {"Content-Type": "application/json"}
        payload = {"status": status, "info": info}

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for HTTP errors
        print(f"Successfully updated job {job_id} status to '{status}' in UI.")
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to UI at {ui_url}. Is the UI server running?")
    except requests.exceptions.RequestException as e:
        print(f"Error updating job status to UI: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while updating job status to UI: {e}")

