import sys
from pathlib import Path

import torch

from totalsegmenter.data.nifti import get_file_stem_from_nifti_path
from totalsegmenter.enums import ImageType
from totalsegmenter.inference.download import download_pretrained_weights
from totalsegmenter.inference.nn_unet import nn_unet_predict
from totalsegmenter.inference.pre_segment_body import generate_rough_body_segmentation
from totalsegmenter.logger import logger
from totalsegmenter.settings import settings
from totalsegmenter.statistics.calculate import calculate_statistics
from totalsegmenter.tasks import Task, task_manager

log = logger.getChild(__name__)


def total_segmenter_predict(
    input_path: Path,
    task_name: str = "total",
    fast: bool = False,
    preview: bool = False,
    statistics: bool = False,
    radiomics: bool = False,
    radiomics_feature_set: dict[str, list[str]] = None,
    multi_label: bool = False,
    roi_subset: list[str] = None,
    crop_path: Path = None,
    body_seg: bool = False,
    force_split: bool = False,
    output_type: ImageType = ImageType.NIFTI,
    test_mode: int = 0,
    skip_saving: bool = False,
    nora_tag=None,
):
    task = task_manager.get_task_by_name(task_name)
    validate_and_warn(task=task, fast=fast, multi_label=multi_label)
    output_dir = set_output_dir(input_path=input_path)
    log.info(f"Saving results to {output_dir}")

    # TODO: unclear why this is needed
    crop_path = output_dir if crop_path is None else crop_path

    if task.subtasks is not None:
        for subtask in task.subtasks:
            download_pretrained_weights(subtask)
    else:
        download_pretrained_weights(task)

    crop_nii = None
    if task.mask_type is None and body_seg:
        crop_nii = generate_rough_body_segmentation(input_path=input_path, output_type=output_type)

    log.info(f"Predicting {input_path} with task {task_name}...")
    pred_nii = nn_unet_predict(
        input_path=input_path,
        output_dir=output_dir,
        task=task,
        crop_mask_image=crop_nii,
        crop_path=crop_path,
        preview=preview,
        force_split=force_split,
        roi_subset=roi_subset,
        output_type=output_type,
        test_mode=test_mode,
        skip_saving=skip_saving,
        nora_tag=nora_tag,
    )

    if statistics or radiomics:
        calculate_statistics(
            image_path=input_path,
            output_dir=output_dir,
            statistics=statistics,
            radiomics=radiomics,
            radiomics_feature_set=radiomics_feature_set,
            pred_nii=pred_nii,
        )

    return pred_nii


def validate_and_warn(task: Task, fast: bool, multi_label: bool):
    log.info("If you use this tool please cite: https://doi.org/10.48550/arXiv.2208.05868")

    if not torch.cuda.is_available():
        log.warning(
            "No GPU detected. Running on CPU. This can be very slow. The '--fast' option can help to some extent."
        )

    if not task.is_available():
        log.critical(
            "\nThis model is only available upon purchase of a license (free licenses available for "
            + "academic projects). \nContact jakob.wasserthal@usb.ch if you are interested.\n"
        )
        sys.exit()

    if task.name in ["total", "body"] and fast:
        log.info(f"Using 'fast' option: resampling to lower resolution ({task.resample}mm)")
    if (
        task.name
        in [
            "lung_vessels",
            "covid",
            "cerebral_bleed",
            "hip_implant",
            "coronary_arteries",
            "pleural_pericard_effusion",
            "liver_vessels",
            "heartchambers_test",
            "bones_tissue_test",
            "aortic_branches_test",
        ]
        and fast
    ):
        raise ValueError(f"task {task.name} does not work with option --fast")

    # TODO: test why this wouldn't work
    if task.name in ["lung_vessels", "body"] and multi_label:
        raise ValueError(f"task {task.name} does not work with option --multi_label, because of postprocessing.")

    if task.name in ["covid", "coronary_arteries"]:
        log.warning("The {task.name} model finds many types of lung opacity, not only COVID. Use with care!")


def set_output_dir(input_path: Path) -> Path:
    try:
        relative_path = input_path.relative_to(settings.DATA_DIR).parent
        output_dir = settings.OUTPUTS_DIR / relative_path / get_file_stem_from_nifti_path(input_path)
    except ValueError:
        output_dir = settings.OUTPUTS_DIR / get_file_stem_from_nifti_path(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    single_segmentations_dir = output_dir / settings.SINGLE_LABEL_SEGMENTATIONS_SUBDIR
    single_segmentations_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
