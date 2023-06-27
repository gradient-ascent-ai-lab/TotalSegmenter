import json
import logging
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
from p_tqdm import p_map

from totalsegmenter.data.nifti import get_file_stem_from_nifti_path
from totalsegmenter.logger import logger
from totalsegmenter.settings import settings
from totalsegmenter.utils import log_duration

log = logger.getChild(__name__)


@log_duration
def calculate_basic_statistics_for_all_labels(image_path: Path, output_dir: Path):
    log.info("Calculating basic statistics...")
    output_path = output_dir / settings.STATISTICS_FILENAME
    stats: dict[str, dict[str, float]] = {}
    if output_path.exists():
        stats = json.loads(output_path.read_text())

    segmentations_dir = output_dir / settings.SINGLE_LABEL_SEGMENTATIONS_SUBDIR
    seg_paths = [f for f in segmentations_dir.iterdir() if f.is_file() and f.name.endswith(".nii.gz")]
    log.debug(f"Found {len(seg_paths)} masks in {segmentations_dir}")
    if len(seg_paths) == 0:
        log.error(f"No segmentations found in {segmentations_dir}! Cannot calculate radiomics features")
        return

    seg_paths = [seg_path for seg_path in seg_paths if get_file_stem_from_nifti_path(seg_path) not in stats]
    log.debug(f"Found {len(seg_paths)} segmentations to process")

    partial_func = partial(calculate_basic_statistics_for_single_mask, image_path=image_path)
    results = p_map(
        partial_func,
        seg_paths,
        num_cpus=settings.NUM_CORES_STATISTICS,
        disable=logger.getEffectiveLevel() != logging.INFO,
    )

    for mask_name, result in results:
        stats[mask_name] = result

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)


def calculate_basic_statistics_for_single_mask(seg_path: Path, image_path: Path) -> tuple[str, dict[str, float]]:
    pred_nii = nib.load(seg_path)
    pred_np = pred_nii.get_fdata()
    image_nii = nib.load(image_path)
    image_np = image_nii.get_fdata()
    spacing = image_nii.header.get_zooms()
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

    stats: dict[str, float] = {}
    data = pred_np > 0  # NOTE: single segmentation masks are binary, so all non-zero values belong to the mask
    stats["volume"] = data.sum() * voxel_volume_mm3
    roi_mask = data.astype(np.uint8)
    stats["intensity"] = np.average(image_np, weights=roi_mask).round(2) if roi_mask.sum() > 0 else 0.0
    return get_file_stem_from_nifti_path(seg_path), stats
