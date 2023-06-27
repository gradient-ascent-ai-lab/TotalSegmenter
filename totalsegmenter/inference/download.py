import http.client
import os
import zipfile
from pathlib import Path

import requests

from totalsegmenter.logger import logger
from totalsegmenter.settings import settings
from totalsegmenter.tasks import Task
from totalsegmenter.utils import log_duration

log = logger.getChild(__name__)


if not settings.DOWNLOAD_DISABLE_HTTP1:
    # NOTE: helps to solve incomplete read erros
    #       https://stackoverflow.com/questions/37816596/restrict-request-to-only-ask-for-http-1-0-to-prevent-chunking-error
    http.client.HTTPConnection._http_vsn = 10  # type: ignore
    http.client.HTTPConnection._http_vsn_str = "HTTP/1.0"  # type: ignore


def download_pretrained_weights(task: Task) -> None:
    weights_path = settings.WEIGHTS_DIR / task.weights_subdir
    if task.weights_url is None:
        log.info(f"Task {task.name} (id={task.task_id}) does not have a weights_url. Skipping download.")
        return

    if weights_path.exists():
        log.debug(f"Pretrained weights for Task {task.name} (id={task.task_id}) already exist. Skipping download.")
        return

    log.info(f"Downloading pretrained weights for Task {task.name} (id={task.task_id}) ...")
    download_url_and_unpack(url=task.weights_url, output_dir=weights_path)


@log_duration
def download_url_and_unpack(url: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_file = output_dir / "tmp_weights_file.zip"
    try:
        with open(temp_file, "wb") as f:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                chunk_size = settings.DOWNLOAD_CHUNK_SIZE or None
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)

        log.info("Download finished. Extracting...")
        with zipfile.ZipFile(temp_file, "r") as zip_f:
            zip_f.extractall(output_dir)

    finally:
        if temp_file.exists():
            os.remove(temp_file)
