from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# from src.dicom_io import dcm_to_nifti, save_mask_as_rtstruct
from totalsegmenter import consts
from totalsegmenter.data.cropping import crop_to_mask
from totalsegmenter.data.nifti import combine_masks
from totalsegmenter.data.resampling import change_spacing
from totalsegmenter.enums import ImageType, MaskType
from totalsegmenter.logger import logger
from totalsegmenter.utils import log_duration, log_metrics

log = logger.getChild(__name__)


def check_input_path(input_path: Path) -> None:
    if not input_path.exists():
        sys.exit("ERROR: The input file or directory does not exist.")


def check_and_convert_image_type(input_path: Path, output_type: ImageType, temp_dir: Path) -> Path:
    image_type = (
        ImageType.NIFTI if str(input_path).endswith(".nii") or str(input_path).endswith(".nii.gz") else ImageType.DICOM
    )
    if image_type == ImageType.NIFTI and output_type == ImageType.DICOM:
        raise ValueError("To use output type dicom you also have to use a Dicom image as input.")

    if image_type == ImageType.DICOM:
        raise NotImplementedError("TODO: re-enable dicom to nifti conversion.")
        # log.info("Converting dicom to nifti...")
        # (temp_dir / "dcm").mkdir()
        # dcm_to_nifti(input_path, temp_dir / "dcm" / "converted_dcm.nii.gz")
        # input_path = temp_dir / "dcm" / "converted_dcm.nii.gz"
        # log.info(f"  found image with shape {nib.load(input_path).shape}")

    return input_path


def check_and_load_image(input_path: Path) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    input_nii = nib.load(input_path)

    if len(input_nii.shape) == 2:
        raise ValueError("TotalSegmentator does not work for 2D images. Use a 3D image.")

    if len(input_nii.shape) > 3:
        log.warning(f"Input image has {len(input_nii.shape)} dimensions, only using first three dimensions")
        input_nii = nib.Nifti1Image(input_nii.get_fdata()[:, :, :, 0], input_nii.affine)

    log.info(f"Loaded image with shape {input_nii.shape}")
    log_metrics({"input_shape": input_nii.shape}, buffer=True)
    original_input_nii = nib.Nifti1Image(input_nii.get_fdata(), input_nii.affine)
    return input_nii, original_input_nii


@log_duration
def resample_input(input_nii: nib.Nifti1Image, resample: float | None) -> nib.Nifti1Image:
    if resample is not None:
        log.info(f"Resampling input image to {resample}mm...")
        input_resampled_nii = change_spacing(
            input_nii,
            [resample, resample, resample],
            order=3,
            dtype=np.int32,
        )
        log.info(f"Resampled image to shape {input_resampled_nii.shape}")
        log_metrics({"resampled_shape": input_resampled_nii.shape}, buffer=True)
        return input_resampled_nii

    return input_nii


def maybe_crop_image(
    input_nii: nib.Nifti1Image,
    mask_type: MaskType,
    crop_mask_image: nib.Nifti1Image,
    crop_path: Path,
    crop_addon: list[int],
    output_dir: Path,
) -> tuple[nib.Nifti1Image, list[list[int]] | None]:
    if mask_type is not None:
        log.info(f"Cropping input image to {mask_type} mask...")
        crop_output_path = crop_path / f"{mask_type}.nii.gz"
        if mask_type == mask_type.LUNG or mask_type == mask_type.PELVIS or mask_type == mask_type.HEART:
            combine_masks(mask_dir=crop_path, output_path=crop_output_path, mask_type=mask_type)

    mask_nii = None
    if crop_mask_image is not None:
        mask_nii = crop_mask_image
    elif crop_path is not None and crop_path != output_dir:
        mask_nii = nib.load(str(crop_path))

    bbox = None
    if mask_nii is not None:
        log.info(f"Cropping input image (shape: {input_nii.shape}) to mask (shape: {mask_nii.shape})...")
        input_nii, bbox = crop_to_mask(input_nii=input_nii, mask_nii=mask_nii, addon=crop_addon, dtype=np.int32)

    return input_nii, bbox


def save_and_maybe_split_image(
    input_resampled_nii: nib.Nifti1Image, temp_dir: Path, multi_task: bool, force_split: bool
) -> bool:
    nib.save(input_resampled_nii, temp_dir / consts.NN_UNET_INPUT_FILENAME_PART_1)
    num_voxels_three = 256 * 256 * 900
    resampled_shape = input_resampled_nii.shape
    do_triple_split = np.prod(resampled_shape) > num_voxels_three and resampled_shape[2] > 200 and multi_task
    if not do_triple_split and force_split:
        log.info("Splitting image into subparts because --force-split is set")
        do_triple_split = True
    elif do_triple_split:
        log.info("Splitting image into subparts because it is too large")
        log_metrics({"do_triple_split": True}, buffer=True)

    if do_triple_split:
        split_into_three_parts(input_resampled_nii=input_resampled_nii, temp_dir=temp_dir)

    return do_triple_split


def split_into_three_parts(input_resampled_nii: nib.Nifti1Image, temp_dir: Path) -> None:
    third = input_resampled_nii.shape[2] // 3
    margin = 20
    input_resampled_np = input_resampled_nii.get_fdata()

    first_part_np = input_resampled_np[:, :, : third + margin]
    second_part_np = input_resampled_np[:, :, third + 1 - margin : third * 2 + margin]
    third_part_np = input_resampled_np[:, :, third * 2 + 1 - margin :]

    log.debug(f"Input shape: {input_resampled_nii.shape}")
    log.debug(f"Shape part 1: {first_part_np.shape}")
    log.debug(f"Shape part 2: {second_part_np.shape}")
    log.debug(f"Shape part 3: {third_part_np.shape}")

    first_part_nii = nib.Nifti1Image(first_part_np, affine=input_resampled_nii.affine)
    second_part_nii = nib.Nifti1Image(second_part_np, affine=input_resampled_nii.affine)
    third_part_nii = nib.Nifti1Image(third_part_np, affine=input_resampled_nii.affine)

    first_part_output_path = temp_dir / consts.NN_UNET_INPUT_FILENAME_PART_1
    second_part_output_path = temp_dir / consts.NN_UNET_INPUT_FILENAME_PART_2
    third_part_output_path = temp_dir / consts.NN_UNET_INPUT_FILENAME_PART_3

    nib.save(first_part_nii, first_part_output_path)
    nib.save(second_part_nii, second_part_output_path)
    nib.save(third_part_nii, third_part_output_path)
