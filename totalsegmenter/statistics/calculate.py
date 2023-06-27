from __future__ import annotations

from pathlib import Path

import nibabel as nib

from totalsegmenter.logger import logger
from totalsegmenter.statistics.basic import calculate_basic_statistics_for_all_labels
from totalsegmenter.statistics.radiomics import calculate_radiomics_features_for_all_segmentations

log = logger.getChild(__name__)


def calculate_statistics(
    image_path: Path,
    output_dir: Path,
    statistics: bool,
    radiomics: bool,
    radiomics_feature_set: dict[str, list[str]] | None = None,
    pred_nii: nib.Nifti1Image | None = None,
    pred_path: Path | None = None,
):
    if pred_nii is None and pred_path is None:
        raise ValueError("Either pred_nii or pred_path must be given.")

    if pred_nii is None:
        pred_nii = nib.load(pred_path)

    if statistics:
        calculate_basic_statistics_for_all_labels(
            image_path=image_path,
            output_dir=output_dir,
        )

    if radiomics:
        log.info("Calculating radiomics...")

        calculate_radiomics_features_for_all_segmentations(
            image_path=image_path,
            output_dir=output_dir,
            radiomics_feature_set=radiomics_feature_set,
        )
