import logging
from datetime import datetime

from typer import style

from totalsegmenter.settings import settings

# formatter = logging.Formatter("%(asctime)s %(levelname)-14s %(message)s", "%H:%M:%S")
formatter = logging.Formatter("%(asctime)s %(levelname)-14s %(name)-12s:%(lineno)-3d %(message)s", "%H:%M:%S")
logging.addLevelName(logging.DEBUG, style(str(logging.getLevelName(logging.DEBUG)), fg="cyan"))
logging.addLevelName(logging.INFO, style(str(logging.getLevelName(logging.INFO)), fg="green"))
logging.addLevelName(logging.WARNING, style(str(logging.getLevelName(logging.WARNING)), fg="yellow"))
logging.addLevelName(logging.ERROR, style(str(logging.getLevelName(logging.ERROR)), fg="red"))
logging.addLevelName(logging.CRITICAL, style(str(logging.getLevelName(logging.CRITICAL)), fg="bright_red"))
logger = logging.getLogger()
logger.setLevel(settings.LOG_LEVEL)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
if settings.LOG_TO_PATH:
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    handler = logging.FileHandler(settings.LOG_DIR / filename)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logging.getLogger("git.cmd").setLevel(logging.CRITICAL)
logging.getLogger("h5py._conv").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("radiomics.featureextractor").setLevel(logging.CRITICAL)
logging.getLogger("radiomics.glcm").setLevel(logging.CRITICAL)
logging.getLogger("radiomics.imageoperations").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("sentry_sdk").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("wandb").setLevel(logging.CRITICAL)
