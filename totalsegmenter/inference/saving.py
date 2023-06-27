from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
from p_tqdm import p_map

from totalsegmenter.data.nifti import add_label_map_to_nifti, combine_masks
from totalsegmenter.data.post_process import extract_skin, remove_outside_of_mask
from totalsegmenter.enums import ImageType, MaskType
from totalsegmenter.logger import logger
from totalsegmenter.settings import settings
from totalsegmenter.tasks import Task
from totalsegmenter.utils import log_duration, log_nora_tag

log = logger.getChild(__name__)


@log_duration
def save_prediction(
    task: Task,
    pred_nii: nib.Nifti1Image,
    pred_np: np.ndarray,
    original_input_nii: nib.Nifti1Image,
    temp_dir: Path,
    output_dir: Path,
    output_type: ImageType,
    roi_subset: list[str] | None = None,
    nora_tag: str | None = None,
):
    log.info("Saving segmentations...")

    if output_type == ImageType.DICOM:
        raise NotImplementedError("TODO: Saving as DICOM")
        # output_dir.mkdir(exist_ok=True, parents=True)
        # save_mask_as_rtstruct(pred_np, class_map, input_path_dcm, output_dir / "segmentations.dcm")

    output_header = set_output_dtype_header(original_input_nii)
    if not np.array_equal(original_input_nii.affine, pred_nii.affine):
        log.warning("Original input and prediction have different affine matrices!")

    log.debug(f"Shape original: {original_input_nii.shape}")
    log.debug(f"Shape pred nii: {pred_nii.shape}")
    log.debug(f"Shape pred np: {pred_np.shape}")

    multi_label_path = save_multilabel_segmentation(
        task=task,
        pred_nii=pred_nii,
        pred_np=pred_np,
        output_dir=output_dir,
        output_header=output_header,
        nora_tag=nora_tag,
    )

    save_single_label_segmentations(
        task=task,
        pred_nii=pred_nii,
        pred_np=pred_np,
        output_header=output_header,
        roi_subset=roi_subset,
        multi_label_path=multi_label_path,
        output_dir=output_dir,
        nora_tag=nora_tag,
    )

    save_task_specific_segmentations(task=task, original_input_nii=original_input_nii, output_dir=output_dir)


def save_multilabel_segmentation(
    task: Task,
    pred_nii: nib.Nifti1Image,
    pred_np: np.ndarray,
    output_dir: Path,
    output_header: nib.nifti1.Nifti1Header,
    nora_tag: str | None,
) -> Path:
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / task.name / settings.MULTI_LABEL_SEGMENTATION_FILENAME
    output_path.parent.mkdir(exist_ok=True, parents=True)
    log.info(f"Saving multi-label image to {output_path}")

    class_map = task.get_class_map()
    output_nii = nib.Nifti1Image(pred_np, pred_nii.affine, output_header)
    output_nii = add_label_map_to_nifti(output_nii, class_map)
    nib.save(output_nii, output_path)
    log_nora_tag(tag=nora_tag, output_path=output_path)
    log.debug(f"Saved multi-label image with shape {output_nii.shape}")
    return output_path


def save_single_label_segmentations(
    task: Task,
    pred_nii: nib.Nifti1Image,
    pred_np: np.ndarray,
    output_header: nib.nifti1.Nifti1Header,
    roi_subset: list[str] | None,
    multi_label_path: Path,
    output_dir: Path,
    nora_tag: str | None,
) -> None:
    class_map = task.get_class_map()
    if roi_subset:
        class_map = {k: v for k, v in class_map.items() if v in roi_subset}

    log.info("Saving single label images...")
    if np.prod(pred_np.shape) > 512 * 512 * 1000:
        log.warning("Shape of output image is very big. Saving each class in a single thread to avoid memory issues.")
        for class_index, class_name in class_map.items():
            binary_image_np = pred_np == class_index
            binary_image_np = binary_image_np.astype(np.uint8)
            output_path = output_dir / settings.SINGLE_LABEL_SEGMENTATIONS_SUBDIR / f"{class_name}.nii.gz"
            output_nii = nib.Nifti1Image(binary_image_np, pred_nii.affine, output_header)
            nib.save(output_nii, output_path)
            log_nora_tag(tag=nora_tag, output_path=output_path)
    else:
        log.info(f"Saving all classes in parallel using {settings.NUM_CORES_SAVING} cores...")
        p_map(
            partial(
                save_single_segmentation_nifti_parallelized,
                multi_label_path=multi_label_path,
                output_dir=output_dir,
                header=output_header,
                nora_tag=nora_tag,
            ),
            class_map.items(),
            num_cpus=settings.NUM_CORES_SAVING,
            disable=logger.getEffectiveLevel() != logging.INFO,
        )


def save_single_segmentation_nifti_parallelized(
    class_map_item: tuple[int, str],
    multi_label_path: Path,
    output_dir: Path,
    header: nib.nifti1.Nifti1Header,
    nora_tag: str | None = None,
) -> None:
    # NOTE: its faster to load the image inside of each thread
    pred_nii = nib.load(multi_label_path)
    image_np = pred_nii.get_fdata()
    class_index, class_name = class_map_item
    binary_image_np = image_np == class_index
    binary_image_np = binary_image_np.astype(np.uint8)
    log.debug(f"Loaded image from {multi_label_path} with shape {image_np.shape}")

    output_path = output_dir / settings.SINGLE_LABEL_SEGMENTATIONS_SUBDIR / f"{class_name}.nii.gz"
    log.debug(f"Saving single-label image to {output_path} with shape {binary_image_np.shape}")
    nib.save(nib.Nifti1Image(binary_image_np, pred_nii.affine, header), str(output_path))
    log_nora_tag(tag=nora_tag, output_path=output_path)


def save_task_specific_segmentations(task: Task, original_input_nii: nib.Nifti1Image, output_dir: Path) -> None:
    if task.name == "lung_vessels":
        remove_outside_of_mask(
            seg_path=output_dir / settings.SINGLE_LABEL_SEGMENTATIONS_SUBDIR / "lung_vessels.nii.gz",
            mask_path=output_dir / settings.SINGLE_LABEL_SEGMENTATIONS_SUBDIR / "lung.nii.gz",
        )

    if task.name == "heartchambers_test":
        for part in task.get_class_map().values():
            remove_outside_of_mask(
                seg_path=output_dir / settings.SINGLE_LABEL_SEGMENTATIONS_SUBDIR / f"{part}.nii.gz",
                mask_path=output_dir / settings.SINGLE_LABEL_SEGMENTATIONS_SUBDIR / "heart.nii.gz",
                dilation_iterations=5,
            )

    if task.name == "body":
        single_segmentations_output_dir = output_dir / settings.SINGLE_LABEL_SEGMENTATIONS_SUBDIR
        log.info("Creating body.nii.gz...")
        body_segmentation_path = output_dir / "body.nii.gz"
        combine_masks(
            mask_dir=single_segmentations_output_dir,
            output_path=body_segmentation_path,
            mask_type=MaskType.BODY,
        )
        log.info("Creating skin.nii.gz...")
        skin = extract_skin(original_input_nii, nib.load(str(body_segmentation_path)))
        skin_output_path = single_segmentations_output_dir / "skin.nii.gz"
        nib.save(skin, skin_output_path)


def set_output_dtype_header(original_input_nii: nib.Nifti1Image) -> nib.nifti1.Nifti1Header:
    # NOTE: we copy the header to make output header is exactly the same as the input
    #       we change the dtype, otherwise it will be float or int and the masks will use a lot more space
    #       (infos on header: https://nipy.org/nibabel/nifti_images.html)
    output_header = original_input_nii.header.copy()
    output_header.set_data_dtype(np.uint8)
    return output_header
