from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from totalsegmenter import consts
from totalsegmenter.data.alignment import undo_canonical
from totalsegmenter.data.cropping import undo_crop
from totalsegmenter.data.nifti import check_if_shape_and_affine_identical
from totalsegmenter.data.resampling import change_spacing
from totalsegmenter.logger import logger
from totalsegmenter.tasks import Task

log = logger.getChild(__name__)


def post_process_prediction(
    task: Task,
    pred_nii: nib.Nifti1Image,
    original_input_nii: nib.Nifti1Image,
    save_binary: bool,
    bbox: list[list[int]] | None,
) -> tuple[nib.Nifti1Image, np.ndarray]:
    log.info("Post-processing prediction...")
    if task.resample is not None:
        pred_nii = resample_prediction_to_input_spacing(
            pred_nii=pred_nii,
            original_input_nii=original_input_nii,
            resample=task.resample,
        )

    pred_nii = undo_canonical(canonical_nii=pred_nii, original_nii=original_input_nii)
    if bbox is not None:
        pred_nii = undo_crop(input_nii=pred_nii, reference_nii=original_input_nii, bbox=bbox)

    check_if_shape_and_affine_identical(original_input_nii, pred_nii)
    pred_np = get_prediction_np(pred_nii=pred_nii, save_binary=save_binary)
    return pred_nii, pred_np


def resample_prediction_to_input_spacing(
    pred_nii: nib.Nifti1Image, resample: float, original_input_nii: nib.Nifti1Image
) -> nib.Nifti1Image:
    log.info("Resampling prediction to input spacing")
    # NOTE: force_affine, otherwise output affine is sometimes slightly off
    #       (which then is worsened by undo_canonical)
    pred_nii = change_spacing(
        pred_nii,
        [resample, resample, resample],
        original_input_nii.shape,
        order=0,
        dtype=np.uint8,
        force_affine=original_input_nii.affine,
    )
    log.debug(f"Shape original: {original_input_nii.shape}")
    log.debug(f"Shape resampled prediction: {pred_nii.shape}")
    return pred_nii


def get_prediction_np(pred_nii: nib.Nifti1Image, save_binary: bool) -> np.ndarray:
    pred_np = pred_nii.get_fdata().astype(np.uint8)
    if save_binary:
        pred_np = (pred_np > 0).astype(np.uint8)

    return pred_np


def maybe_recombine_image_parts(input_resampled_nii: nib.Nifti1Image, temp_dir: Path, do_triple_split: bool) -> Path:
    combined_output_path = temp_dir / consts.NN_UNET_PRED_FILENAME_PART_1
    if not do_triple_split:
        return combined_output_path

    log.info("Recombining image parts...")
    third = input_resampled_nii.shape[2] // 3
    margin = 20
    first_part_pred_path = temp_dir / consts.NN_UNET_PRED_FILENAME_PART_1
    second_part_pred_path = temp_dir / consts.NN_UNET_PRED_FILENAME_PART_2
    third_part_pred_path = temp_dir / consts.NN_UNET_PRED_FILENAME_PART_3

    first_part_pred_np = nib.load(first_part_pred_path).get_fdata()
    second_part_pred_np = nib.load(second_part_pred_path).get_fdata()
    third_part_pred_np = nib.load(third_part_pred_path).get_fdata()

    combined_nii = np.zeros(input_resampled_nii.shape, dtype=np.uint8)
    combined_nii[:, :, :third] = first_part_pred_np[:, :, :-margin]
    combined_nii[:, :, third : third * 2] = second_part_pred_np[:, :, margin - 1 : -margin]
    combined_nii[:, :, third * 2 :] = third_part_pred_np[:, :, margin - 1 :]
    combined_nib = nib.Nifti1Image(combined_nii, input_resampled_nii.affine)

    nib.save(combined_nib, combined_output_path)
    second_part_pred_path.unlink()
    third_part_pred_path.unlink()

    log.debug(f"Recombined image parts saved to {combined_output_path}")
    log.debug(f"Shape input resampled: {input_resampled_nii.shape}")
    log.debug(f"Shape first part pred: {first_part_pred_np.shape}")
    log.debug(f"Shape second part pred: {second_part_pred_np.shape}")
    log.debug(f"Shape third part pred: {third_part_pred_np.shape}")
    log.debug(f"Shape combined output: {combined_nib.shape}")

    return combined_output_path
