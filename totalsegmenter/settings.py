from multiprocessing import cpu_count
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseSettings, Field, root_validator

from totalsegmenter.enums import NnUnetPredictionMode

load_dotenv("secrets.env")
load_dotenv("vars.env")


# NOTE: BaseSettings objects allow setting all attributes via environment variables by default
class Settings(BaseSettings):
    DOWNLOAD_CHUNK_SIZE: int = 8192 * 16
    DOWNLOAD_DISABLE_HTTP1: bool = False

    LOG_LEVEL: str = "INFO"
    LOG_TO_PATH: bool = False
    LOG_DIR: Path = Path("logs")
    LOG_DURATIONS: bool = True

    DATA_DIR: Path = Path("data")
    OUTPUTS_DIR: Path = Path("outputs")
    RADIOMICS_FEATURES_DIR: Path = Path("config/radiomics")
    TASKS_DIR: Path = Path("config/tasks")
    WEIGHTS_DIR: Path = Path("pretrained_weights")

    BATCH_PREDICT_MAX_CONSECUTIVE_ERRORS: int = 5
    BATCH_PREDICT_MOVE_ERROR_FILES: bool = True
    BATCH_PREDICT_ERROR_FILES_DIR: Path = Path("data_errors")
    BATCH_PREDICT_COPY_INPUT_IMAGE_TO_OUTPUT_DIR: bool = True

    NN_UNET_MIXED_PRECISION: bool = True
    # NOTE: disabling test-time augmentation can give a 8x speedup at the cost of reduced segmentation quality
    NN_UNET_TEST_TIME_AUGMENTATION: bool = False
    NN_UNET_SAVE_NPZ: bool = False
    NN_UNET_PART_ID: int = 0
    NN_UNET_NUM_PARTS: int = 1
    NN_UNET_OVERWRITE_EXISTING: bool = False
    NN_UNET_STEP_SIZE: float = 0.5
    NN_UNET_ALL_IN_GPU: bool = False
    NN_UNET_PREDICTION_MODE: NnUnetPredictionMode = NnUnetPredictionMode.FASTEST
    NN_UNET_CHECKPOINT_NAME: str = "model_final_checkpoint"

    NORA_BINARY_PATH: Path = Path("/opt/nora/src/node/nora")

    RADIOMICS_FILENAME: str = "statistics_radiomics.json"
    STATISTICS_FILENAME: str = "statistics.json"
    MULTI_LABEL_SEGMENTATION_FILENAME: str = "multi_label.nii.gz"
    SINGLE_LABEL_SEGMENTATIONS_SUBDIR: str = "segmentations"
    VERIFIED_IMAGES_PATH: Path = Path("verified_images.txt")

    NUM_CORES_RESAMPLING: int = Field(6, ge=1, le=6)
    NUM_CORES_PREPROCESSING: int = Field(6, ge=1)
    NUM_CORES_SAVING: int = Field(6, ge=1)
    NUM_CORES_STATISTICS: int = 6
    NUM_CORES_RADIOMICS: int = 6
    NUM_CORES_DYNAMIC: bool = True

    PREVIEW_SMOOTHING_FACTOR: int = Field(20, ge=0)
    PREVIEW_WINDOW_WIDTH: int = Field(1800, ge=0)
    PREVIEW_WINDOW_HEIGHT: int = Field(400, ge=0)
    PREVIEW_NUM_COLUMNS: int = Field(10, ge=1)
    PREVIEW_SUBJECT_WIDTH: int = Field(330, ge=0)
    PREVIEW_SUBJECT_HEIGHT: int = Field(700, ge=0)
    PREVIEW_USE_SUBJECT_HEIGHT: bool = False

    RADIOMICS_RESAMPLED_PIXEL_SPACING: list[float] = [3.0, 3.0, 3.0]
    # NOTE: pyradiomics default geometry tolerance: 1e-6
    RADIOMICS_GEOMETRY_TOLERANCE: float = 1e-3
    RADIOMICS_VALUES_ROUNDING_NDIGITS: int = 4

    NOTIFICATIONS_ENABLED: bool = False
    SLACK_WEBHOOK_URL: str = None

    EXPERIMENT_TRACKING_ENABLED: bool = False
    WANDB_ENTITY: str = None
    WANDB_PROJECT: str = "totalsegmenter"
    WANDB_API_KEY: str = None
    WANDB_GROUP: str = "default"

    @root_validator()
    def set_num_cores_if_dynamic(cls, values):  # noqa: N805
        if "NUM_CORES_DYNAMIC" in values and values["NUM_CORES_DYNAMIC"]:
            # NOTE: -1 to leave one core for the OS
            num_cores = cpu_count() - 1
            values["NUM_CORES_PREPROCESSING"] = num_cores
            values["NUM_CORES_RESAMPLING"] = min(num_cores, 6)
            # NOTE: high values for NUM_CORES_SAVING can cause memory issues
            # TODO: consider making NUM_CORES_SAVING a percentage of available memory
            values["NUM_CORES_SAVING"] = num_cores
            values["NUM_CORES_STATISTICS"] = num_cores
            values["NUM_CORES_RADIOMICS"] = num_cores
        return values


settings = Settings()
