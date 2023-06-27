from pathlib import Path

import nibabel as nib
import numpy as np

from totalsegmenter.data.cropping import crop_to_mask
from totalsegmenter.enums import ImageType
from totalsegmenter.inference.download import download_pretrained_weights
from totalsegmenter.inference.nn_unet import nn_unet_predict
from totalsegmenter.logger import logger
from totalsegmenter.tasks import task_manager
from totalsegmenter.utils import log_duration

log = logger.getChild(__name__)


@log_duration
def generate_rough_body_segmentation(input_path: Path, output_type: ImageType = ImageType.NIFTI) -> nib.Nifti1Image:
    log.info("Generating rough body segmentation...")
    # NOTE: speedup for big images; not useful in combination with --fast option
    task = task_manager.get_task_by_name("body")
    download_pretrained_weights(task=task)
    log.info("Generating rough body segmentation...")
    body_segmentation_nii = nn_unet_predict(
        input_path=input_path,
        output_dir=None,
        task=task,
        save_binary=True,
        output_type=output_type,
        skip_saving=True,
    )
    return body_segmentation_nii


def crop_to_body(input_path: Path, output_path: Path, only_trunk: bool = False):
    log.info("Cropping to body...")
    body_segmentation_nii = generate_rough_body_segmentation(input_path=input_path)
    input_nii = nib.load(input_path)

    crop_to_mask(input_nii=input_nii, mask_nii=body_segmentation_nii, addon=[3, 3, 3])
    body_segmentation_np = body_segmentation_nii.get_fdata()
    if only_trunk:
        body_segmentation_np = body_segmentation_np == 1
    else:
        body_segmentation_np = body_segmentation_np > 0.5

    cropped_to_body_nii = nib.Nifti1Image(body_segmentation_np.astype(np.uint8), body_segmentation_nii.affine)
    nib.save(cropped_to_body_nii, output_path)
