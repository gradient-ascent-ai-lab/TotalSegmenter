from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from totalsegmenter import consts
from totalsegmenter.enums import ImageType
from totalsegmenter.inference.post_process import maybe_recombine_image_parts, post_process_prediction
from totalsegmenter.inference.pre_process import (
    check_and_convert_image_type,
    check_and_load_image,
    check_input_path,
    maybe_crop_image,
    resample_input,
    save_and_maybe_split_image,
)
from totalsegmenter.inference.preview import generate_preview
from totalsegmenter.inference.saving import save_prediction, save_task_specific_segmentations
from totalsegmenter.logger import logger
from totalsegmenter.settings import settings
from totalsegmenter.tasks import Task
from totalsegmenter.utils import log_duration, suppress_stdout

with suppress_stdout():
    from nnunet.inference.predict import predict_from_folder


log = logger.getChild(__name__)


def nn_unet_predict(
    input_path: Path,
    output_dir: Path,
    task: Task,
    crop_mask_image: nib.Nifti1Image | None = None,
    crop_path: Path | None = None,
    preview: bool = False,
    save_binary: bool = False,
    force_split: bool = False,
    roi_subset: list[str] | None = None,
    output_type: ImageType = ImageType.NIFTI,
    test_mode: int | None = None,
    skip_saving: bool = False,
    nora_tag: str | None = None,
) -> nib.Nifti1Image | None:
    check_input_path(input_path=input_path)

    with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_dir:
        temp_dir = Path(tmp_dir)
        log.debug(f"temp_dir: {temp_dir}")

        multi_task = task.subtasks is not None
        input_path = check_and_convert_image_type(input_path=input_path, output_type=output_type, temp_dir=temp_dir)
        input_nii, original_input_nii = check_and_load_image(input_path=input_path)
        input_resampled_nii = resample_input(input_nii=input_nii, resample=task.resample)
        do_triple_split = save_and_maybe_split_image(
            input_resampled_nii=input_resampled_nii, temp_dir=temp_dir, multi_task=multi_task, force_split=force_split
        )
        input_nii, bbox = maybe_crop_image(
            input_nii=input_nii,
            mask_type=task.mask_type,
            crop_mask_image=crop_mask_image,
            crop_path=crop_path,
            crop_addon=task.crop_addon,
            output_dir=output_dir,
        )
        if multi_task:
            _predict_multi_task(
                task=task,
                roi_subset=roi_subset,
                temp_dir=temp_dir,
                input_resampled_nii=input_resampled_nii,
                do_triple_split=do_triple_split,
                test_mode=test_mode,
            )
        else:
            _predict_single_task(
                task=task,
                temp_dir=temp_dir,
                test_mode=test_mode,
            )

        pred_path = maybe_recombine_image_parts(
            input_resampled_nii=input_resampled_nii, temp_dir=temp_dir, do_triple_split=do_triple_split
        )
        pred_nii = nib.load(pred_path)

        if preview:
            # NOTE: we generate a preview before upsampling, as it's faster
            #       and still in canonical space for better orientation
            generate_preview(
                image_nii=input_resampled_nii,
                pred_nii=pred_nii,
                output_dir=output_dir,
                task=task,
            )

        pred_nii, pred_np = post_process_prediction(
            task=task,
            pred_nii=pred_nii,
            original_input_nii=original_input_nii,
            save_binary=save_binary,
            bbox=bbox,
        )

        if skip_saving:
            log.info("Skipping saving...")
            return nib.Nifti1Image(pred_np, pred_nii.affine)

        save_prediction(
            task=task,
            pred_nii=pred_nii,
            pred_np=pred_np,
            original_input_nii=original_input_nii,
            temp_dir=temp_dir,
            output_dir=output_dir,
            output_type=output_type,
            roi_subset=roi_subset,
            nora_tag=nora_tag,
        )

        save_task_specific_segmentations(task=task, original_input_nii=original_input_nii, output_dir=output_dir)

        return nib.Nifti1Image(pred_np, pred_nii.affine)


def _predict_single_task(
    task: Task,
    temp_dir: Path,
    test_mode: int,
):
    log.info(f"Predicting single task {task.name}")
    if test_mode and test_mode == 3:
        log.warning("Using reference segmentation instead of prediction for testing")
        shutil.copy(
            Path("tests") / "reference_files" / "example_seg_lung_vessels.nii.gz",
            temp_dir / consts.NN_UNET_INPUT_FILENAME_PART_1,
        )
        return

    _nn_unet_predict(
        input_dir=temp_dir,
        output_dir=temp_dir,
        task=task,
    )


@log_duration
def _predict_multi_task(
    task: Task,
    roi_subset: list[str],
    test_mode: int,
    temp_dir: Path,
    input_resampled_nii: nib.Nifti1Image,
    do_triple_split: bool,
):
    log.info(f"Predicting multi task '{task.name}'")
    required_subtasks = _determine_required_subtasks(task=task, roi_subset=roi_subset)

    if test_mode and test_mode == 1:
        log.warning("Using reference segmentation instead of prediction for testing")
        shutil.copy(
            Path("tests") / "reference_files" / "example_seg.nii.gz",
            temp_dir / settings.MULTI_LABEL_SEGMENTATION_FILENAME,
        )
        return

    class_map = task.get_class_map()
    class_map_inv = {v: k for k, v in class_map.items()}
    image_part_preds = {}

    # iterate over subparts of image
    num_image_parts = 1
    image_part_input_filenames = [consts.NN_UNET_INPUT_FILENAME_PART_1]
    image_part_pred_filenames = [consts.NN_UNET_PRED_FILENAME_PART_1]
    if do_triple_split:
        num_image_parts = 3
        image_part_input_filenames += [
            consts.NN_UNET_INPUT_FILENAME_PART_2,
            consts.NN_UNET_INPUT_FILENAME_PART_3,
        ]
        image_part_pred_filenames += [
            consts.NN_UNET_PRED_FILENAME_PART_2,
            consts.NN_UNET_PRED_FILENAME_PART_3,
        ]

    for part_index in range(num_image_parts):
        image_shape = nib.load(temp_dir / image_part_input_filenames[part_index]).shape
        image_part_preds[part_index] = np.zeros(image_shape, dtype=np.uint8)
        log.debug(f"Shape pred part {part_index}: {image_part_preds[part_index].shape}")

    # Run several tasks and combine results into one segmentation
    for subtask_index, subtask in enumerate(required_subtasks):
        log.info(f"Predicting part {subtask_index+1} of {len(required_subtasks)}: {subtask.name}")
        _nn_unet_predict(
            input_dir=temp_dir,
            output_dir=temp_dir,
            task=subtask,
        )

        # iterate over models (different sets of classes)
        for part_index in range(num_image_parts):
            image_part_pred_path = temp_dir / image_part_pred_filenames[part_index]
            subtask_image_part_pred_path = temp_dir / subtask.name / image_part_pred_filenames[part_index]
            subtask_image_part_pred_path.parent.mkdir(exist_ok=True, parents=True)
            (image_part_pred_path).rename(subtask_image_part_pred_path)

            pred_np = nib.load(subtask_image_part_pred_path).get_fdata()
            log.debug(f"Shape pred part {part_index}: {pred_np.shape}")
            for class_index, class_name in subtask.get_class_map().items():
                image_part_preds[part_index][pred_np == class_index] = class_map_inv[class_name]

    # iterate over subparts of image
    for part_index in range(num_image_parts):
        output_path = temp_dir / image_part_pred_filenames[part_index]
        nib.save(nib.Nifti1Image(image_part_preds[part_index], input_resampled_nii.affine), output_path)


def _determine_required_subtasks(task: Task, roi_subset: list[str]) -> list[Task]:
    required_subtasks = task.subtasks
    if roi_subset:
        log.debug(f"Using roi_subset: {', '.join(roi_subset)}")
        class_map = task.get_class_map()
        if any(organ not in class_map.values() for organ in roi_subset):
            not_found = [organ for organ in roi_subset if organ not in class_map.values()]
            log.debug(f"Could not find the following organs in the class map: {not_found}")
            log.info(f"Available organs are: {list(class_map.values())}")
            exit(1)

        required_subtasks = []
        for subtask in task.subtasks:
            if any(organ in roi_subset for organ in subtask.get_class_map().values()):
                required_subtasks.append(subtask)

    log.info(f"Using models: {','.join([t.name for t in required_subtasks])} based on the provided roi_subset")
    return required_subtasks


@log_duration
def _nn_unet_predict(
    input_dir: Path,
    output_dir: Path,
    task: Task,
):
    model_weights_dir = task.get_model_weights_dir()
    log.debug(f"Using model stored in {model_weights_dir}")
    with suppress_stdout():
        predict_from_folder(
            model=str(model_weights_dir),
            input_folder=str(input_dir),
            output_folder=str(output_dir),
            folds=task.folds,
            mode=settings.NN_UNET_PREDICTION_MODE.value,
            save_npz=settings.NN_UNET_SAVE_NPZ,
            num_threads_preprocessing=settings.NUM_CORES_PREPROCESSING,
            num_threads_nifti_save=settings.NUM_CORES_SAVING,
            lowres_segmentations=None,
            part_id=settings.NN_UNET_PART_ID,
            num_parts=settings.NN_UNET_NUM_PARTS,
            tta=settings.NN_UNET_TEST_TIME_AUGMENTATION,
            overwrite_existing=settings.NN_UNET_OVERWRITE_EXISTING,
            overwrite_all_in_gpu=settings.NN_UNET_ALL_IN_GPU,
            mixed_precision=settings.NN_UNET_MIXED_PRECISION,
            step_size=settings.NN_UNET_STEP_SIZE,
            checkpoint_name=settings.NN_UNET_CHECKPOINT_NAME,
        )
