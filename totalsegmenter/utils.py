import contextlib
import io
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from functools import wraps
from pathlib import Path
from time import time

import requests

from totalsegmenter.logger import logger
from totalsegmenter.settings import settings

log = logger.getChild(__name__)

global durations
durations = defaultdict(list)
global metrics_buffer
metrics_buffer = {}


@contextlib.contextmanager
def suppress_stdout():
    class OutputBuffer:
        def __init__(self):
            self.buffer = io.StringIO()

        def write(self, x):
            self.buffer.write(x)

        def flush(self):
            pass

    dummy_file = OutputBuffer()
    save_stdout = sys.stdout
    sys.stdout = dummy_file

    try:
        yield
    except Exception as e:
        sys.stdout = save_stdout
        print("*" * 80)
        print("An exception occurred. Suppressed output:")
        print(dummy_file.buffer.getvalue())
        print("*" * 80)
        raise e
    else:
        sys.stdout = save_stdout


def log_duration(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        if settings.LOG_DURATIONS:
            log.debug(f"Function '{func.__qualname__}' took {int(time()-start)} sec")
            global durations
            durations[func.__qualname__].append(time() - start)
        return result

    return wrap


def log_all_durations(to_file: bool = False) -> None:
    global durations
    log.info("Function durations:")
    for func_name, fn_durations in durations.items():
        total_seconds = sum(fn_durations)
        total_minutes = total_seconds / 60
        avg_seconds = total_seconds / len(fn_durations)
        log.debug(f"- {func_name}: {int(total_seconds)} sec ({int(total_minutes)} min), avg: {avg_seconds:.2f} sec")

    if to_file:
        filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_function_durations.csv"
        with open(settings.LOG_DIR / filename, "w") as f:
            f.write("function,total_seconds,avg_seconds\n")
            for func_name, fn_durations in durations.items():
                total_seconds = sum(fn_durations)
                total_minutes = total_seconds / 60
                avg_seconds = total_seconds / len(fn_durations)
                f.write(f"{func_name},{int(total_seconds)},{avg_seconds:.2f}\n")
                log_metrics({f"{func_name}_total_seconds": total_seconds, f"{func_name}_avg_seconds": avg_seconds})


def log_nora_tag(tag: str | None, output_path: Path) -> None:
    if tag is None:
        return

    if not settings.NORA_BINARY_PATH.exists():
        log.warning("Nora binary not found. Skipping Nora tag logging.")
        return

    try:
        subprocess.call(f"{settings.NORA_BINARY_PATH} -p {tag} --add {output_path} --addtag atlas", shell=True)
    except Exception as e:
        log.warning(f"Failed to log Nora tag: {e}")


def init_experiment_tracking():
    if not settings.WANDB_ENTITY or not settings.WANDB_PROJECT:
        log.error("WANDB_ENTITY and WANDB_PROJECT must be set to log stats to Weights & Biases")
        return

    try:
        import wandb

        relevant_config = settings.dict(
            include={
                "NN_UNET_MIXED_PRECISION",
                "NN_UNET_TEST_TIME_AUGMENTATION",
                "NN_UNET_SAVE_NPZ",
                "NN_UNET_PART_ID",
                "NN_UNET_NUM_PARTS",
                "NN_UNET_STEP_SIZE",
                "NN_UNET_ALL_IN_GPU",
                "NN_UNET_PREDICTION_MODE",
            }
        )

        wandb.login(key=settings.WANDB_API_KEY, relogin=True)
        wandb.init(
            entity=settings.WANDB_ENTITY,
            project=settings.WANDB_PROJECT,
            group=settings.WANDB_GROUP,
            config=relevant_config,
        )

        log.info("Logging stats to Weights & Biases")
    except Exception as e:
        log.warning(f"Failed to initialize Weights & Biases: {e}")


def log_metrics(metrics: dict, buffer: bool = False) -> None:
    if not settings.EXPERIMENT_TRACKING_ENABLED:
        return

    if buffer:
        global metrics_buffer
        metrics_buffer.update(metrics)
        return

    # TODO: consider writing to a CSV file for building up local stats
    try:
        import wandb

        wandb.log(metrics)
    except Exception as e:
        log.warning(f"Failed to log metrics to Weights & Biases: {e}")


def notify(message: str) -> None:
    if not settings.NOTIFICATIONS_ENABLED:
        return

    if not settings.SLACK_WEBHOOK_URL:
        log.warning("Slack webhook URL not set. Skipping notification.")
        return

    try:
        headers = {"Content-type": "application/json"}
        data = {"text": message}
        log.debug(f"Sending notification to slack: {data}")
        requests.post(settings.SLACK_WEBHOOK_URL, json=data, headers=headers)
    except Exception as e:
        log.error(f"Failed to send slack notification: {e}")
