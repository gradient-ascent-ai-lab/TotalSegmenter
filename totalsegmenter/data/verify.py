from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from p_tqdm import p_map

from totalsegmenter.logger import logger
from totalsegmenter.settings import settings
from totalsegmenter.utils import log_duration

log = logger.getChild(__name__)


@log_duration
def verify_images(image_paths: list[Path]) -> None:
    verified_image_lines: list[str] = []
    if settings.VERIFIED_IMAGES_PATH.exists():
        verified_image_lines = settings.VERIFIED_IMAGES_PATH.read_text().splitlines()

    verified_image_paths = [Path(line) for line in verified_image_lines if Path(line).exists()]
    images_to_verify = [image_path for image_path in image_paths if image_path not in verified_image_paths]
    if not images_to_verify:
        log.info("All images already verified")
        return

    log.info(f"Verifying {len(images_to_verify)} images...")

    newly_verified_image_paths = []
    error_images = defaultdict(list)
    paths_and_errors = p_map(verify_image, images_to_verify)
    for path, errors in paths_and_errors:
        for error in errors:
            error_images[error].append(path)
        else:
            newly_verified_image_paths.append(path)

    for error_type, error_image_paths in error_images.items():
        log.warning(f"{error_type}: {len(error_image_paths)}")
        for error_image in error_image_paths:
            log.warning(f" - {error_image}")

    if newly_verified_image_paths:
        all_verified_image_paths = verified_image_paths + newly_verified_image_paths
        settings.VERIFIED_IMAGES_PATH.write_text("\n".join(str(path) for path in all_verified_image_paths))

    if error_images:
        raise ValueError("Found invalid images")


def verify_image(image_path: Path) -> tuple[Path, list[str]]:
    try:
        image_nii = nib.load(str(image_path))
        image_np = image_nii.get_fdata()
    except Exception:
        return image_path, ["load_error"]

    errors = []
    if np.any(np.isnan(image_np)):
        errors.append("nan_values")

    if np.any(np.isinf(image_np)):
        errors.append("inf_values")

    if np.all(image_np == 0):
        errors.append("all_zero")
    elif np.all(image_np == 1):
        errors.append("all_one")
    elif np.all(image_np == image_np[0]):
        errors.append("all_same_value")

    return image_path, errors
