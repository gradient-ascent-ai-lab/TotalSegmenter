import shutil
import traceback
from pathlib import Path
from time import time

import nibabel as nib

from totalsegmenter.data.nifti import get_file_stem_from_nifti_path
from totalsegmenter.data.verify import verify_images
from totalsegmenter.inference.predict import set_output_dir, total_segmenter_predict
from totalsegmenter.inference.preview import generate_preview
from totalsegmenter.logger import logger
from totalsegmenter.settings import settings
from totalsegmenter.statistics.calculate import calculate_statistics
from totalsegmenter.statistics.utils import (
    get_radiomics_statistics_to_calculate_per_segmentation,
    read_radiomics_from_path,
)
from totalsegmenter.tasks import task_manager
from totalsegmenter.utils import init_experiment_tracking, log_all_durations, log_metrics, notify

log = logger.getChild(__name__)


def batch_predict(
    input_dir: Path,
    task_names: list[str] = ["total"],
    multi_label: bool = False,
    roi_subset: list[str] = None,
    statistics: bool = False,
    radiomics: bool = False,
    radiomics_feature_set: dict[str, list[str]] = None,
    preview: bool = False,
    num_splits: int = 1,
    split_index: int = 0,
):
    log.info(f"Running batch prediction for tasks: {task_names}")
    image_paths = get_image_paths(input_dir)
    image_paths = filter_image_paths(
        image_paths=image_paths,
        task_names=task_names,
        roi_subset=roi_subset,
        statistics=statistics,
        radiomics=radiomics,
        radiomics_feature_set=radiomics_feature_set,
        preview=preview,
    )
    image_paths = maybe_split_image_paths(image_paths, num_splits, split_index)
    verify_images(image_paths=image_paths)
    image_paths = sorted(image_paths)

    if settings.EXPERIMENT_TRACKING_ENABLED:
        init_experiment_tracking()

    total_seconds = 0.0
    total_errors = 0
    consecutive_errors = 0
    for task_name in task_names:
        for idx, image_path in enumerate(image_paths):
            start_time = time()
            try:
                log.info(f"*** Processing {image_path} ({idx + 1}/{len(image_paths)}) ***")
                consecutive_errors = predict_single(
                    task_name=task_name,
                    image_path=image_path,
                    multi_label=multi_label,
                    roi_subset=roi_subset,
                    statistics=statistics,
                    radiomics=radiomics,
                    radiomics_feature_set=radiomics_feature_set,
                    preview=preview,
                )
                consecutive_errors = 0
                end_time = time()
                total_seconds += end_time - start_time
                average_seconds = total_seconds / (image_paths.index(image_path) + 1)
                estimated_remaining_minutes = average_seconds * (len(image_paths) - image_paths.index(image_path) - 1) / 60
                log.info(f"Processed {image_path} in {end_time - start_time:.2f} seconds")
                log.info(f"Average time per file: {average_seconds:.2f} seconds")
                msg = f"Estimated time remaining: {estimated_remaining_minutes:.2f} minutes"
                log.info(msg)
                notify(msg)
                log_metrics(
                    {
                        "task_name": task_name,
                        "image_idx": idx,
                        "total_seconds": total_seconds,
                        "statistics": statistics,
                        "radiomics": radiomics,
                        "multi_label": multi_label,
                        "roi_subset": roi_subset,
                        "preview": preview,
                    }
                )

            except Exception as e:
                end_time = time()
                total_seconds += end_time - start_time
                msg = f"Error processing {image_path} after {total_seconds:.2f} seconds: {e}"
                log.error(msg)
                tb = traceback.format_exc()
                log.debug(tb)
                notify(msg)
                if settings.BATCH_PREDICT_MOVE_ERROR_FILES:
                    relative_path = image_path.relative_to(settings.DATA_DIR)
                    error_path = settings.BATCH_PREDICT_ERROR_FILES_DIR / relative_path
                    error_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(image_path, error_path)
                    with open(error_path.parent / "error.log", "w") as f:
                        f.write(msg)
                        f.write("\n\n")
                        f.write(tb)

                total_errors += 1
                consecutive_errors += 1
                if consecutive_errors >= settings.BATCH_PREDICT_MAX_CONSECUTIVE_ERRORS:
                    msg = "Too many consecutive errors, aborting batch processing"
                    log.critical(msg)
                    notify(msg)
                    exit(1)

    total_minutes = total_seconds / 60
    avg_seconds = total_seconds / len(image_paths)
    avg_minutes = avg_seconds / 60
    msg = f"Processed {len(image_paths)} files in {total_seconds:.2f} seconds ({total_minutes:.2f} minutes)"
    msg += f"\nAverage time per file: {avg_seconds:.2f} seconds ({avg_minutes:.2f} minutes)"
    msg += f"\nTotal errors: {total_errors}"
    log.info(msg)
    notify(msg)
    log_all_durations(to_file=True)


def predict_single(
    task_name: str,
    image_path: Path,
    multi_label: bool = False,
    roi_subset: list[str] = None,
    statistics: bool = False,
    radiomics: bool = False,
    radiomics_feature_set: dict[str, list[str]] = None,
    preview: bool = False,
):
    task = task_manager.get_task_by_name(task_name=task_name)
    output_dir = set_output_dir(input_path=image_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_output_path = output_dir / task.name / "preview.png"
    if preview_output_path.exists() and preview:
        log.info(f"Preview already computed for {image_path}")
        preview = False

    predict = True
    single_segmentations_output_dir = output_dir / settings.SINGLE_LABEL_SEGMENTATIONS_SUBDIR
    multi_segmentation_output_path = output_dir / task.name / settings.MULTI_LABEL_SEGMENTATION_FILENAME
    if single_segmentations_output_dir.exists():
        rois_predicted = [
            get_file_stem_from_nifti_path(roi_path) for roi_path in single_segmentations_output_dir.iterdir()
        ]
        rois_to_predict = roi_subset or list(task.class_map.values())
        roi_subset = list(set(rois_to_predict) - set(rois_predicted))
        if not roi_subset:
            log.info(f"All segmentations already predicted for {image_path}")
            predict = False
            # TODO: fix combine_masks_to_multilabel_file
            # if not multi_segmentation_output_path.exists():
            #     combine_masks_to_multilabel_file(
            #         task_name=task.name,
            #         original_image_path=image_path,
            #         masks_dir=single_segmentations_output_dir,
            #     )
        else:
            if rois_predicted:
                log.info(f"Segmentations already predicted for {len(rois_predicted)} ROIs")
            log.info(f"Segmentations to predict: {len(roi_subset)}")

    if predict:
        total_segmenter_predict(
            input_path=image_path,
            task_name=task.name,
            multi_label=multi_label,
            roi_subset=roi_subset,
            preview=preview,
        )

    if preview:
        input_nii = nib.load(image_path)
        pred_nii = nib.load(multi_segmentation_output_path)
        generate_preview(
            image_nii=input_nii,
            pred_nii=pred_nii,
            output_dir=output_dir,
            task=task,
        )

    if statistics or radiomics:
        calculate_statistics(
            image_path=image_path,
            output_dir=output_dir,
            pred_path=multi_segmentation_output_path,
            statistics=statistics,
            radiomics=radiomics,
            radiomics_feature_set=radiomics_feature_set,
        )

    output_image_path = output_dir / image_path.name
    if not output_image_path.exists() and settings.BATCH_PREDICT_COPY_INPUT_IMAGE_TO_OUTPUT_DIR:
        log.info(f"Copying {image_path} to {output_image_path}")
        shutil.copy(image_path, output_image_path)


def get_image_paths(input_dir: Path) -> list[Path]:
    dicom_files = list(input_dir.glob("**/*.dcm"))
    if len(dicom_files) > 0:
        log.warning(f"Found {len(dicom_files)} files with .dcm extension, support for DICOM is not re-implemented yet")
        log.info("These files will be ignored")

    image_paths = list(input_dir.glob("**/*.nii.gz")) + list(input_dir.glob("**/*.nii"))
    if len(image_paths) == 0:
        log.info(f"Found no files to process in {input_dir}")
        exit(0)

    log.info(f"Found {len(image_paths)} image files in {input_dir}")
    return image_paths


def filter_image_paths(
    image_paths: list[Path],
    task_names: list[str],
    roi_subset: list[str],
    statistics: bool,
    radiomics: bool,
    radiomics_feature_set: dict[str, list[str]],
    preview: bool,
) -> list[Path]:
    filtered_image_paths = []
    for task_name in task_names:
        task = task_manager.get_task_by_name(task_name=task_name)
        for image_path in image_paths:
            needs_preview = preview
            output_dir = set_output_dir(input_path=image_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            preview_output_path = output_dir / task_name / "preview.png"
            if preview_output_path.exists() and preview:
                needs_preview = False

            needs_predict = True
            single_segmentations_output_dir = output_dir / settings.SINGLE_LABEL_SEGMENTATIONS_SUBDIR
            if single_segmentations_output_dir.exists():
                rois_predicted = [
                    get_file_stem_from_nifti_path(roi_path) for roi_path in single_segmentations_output_dir.iterdir()
                ]
                rois_to_predict = roi_subset or list(task.class_map.values())
                roi_subset = list(set(rois_to_predict) - set(rois_predicted))
                if not roi_subset:
                    needs_predict = False

            needs_statistics = statistics
            statistics_output_path = output_dir / settings.STATISTICS_FILENAME
            if statistics_output_path.exists() and statistics:
                needs_statistics = False

            needs_radiomics = radiomics
            radiomics_output_path = output_dir / settings.RADIOMICS_FILENAME
            if radiomics_output_path.exists() and radiomics:
                existing_stats = read_radiomics_from_path(stats_path=radiomics_output_path)
                seg_paths = list(single_segmentations_output_dir.glob("*.nii.gz"))
                needs_radiomics = (
                    get_radiomics_statistics_to_calculate_per_segmentation(
                        seg_paths=seg_paths,
                        radiomics_feature_set=radiomics_feature_set,
                        existing_statistics=existing_stats,
                    )
                    != []
                )

            if needs_predict or needs_preview or needs_statistics or needs_radiomics:
                filtered_image_paths.append(image_path)

    if len(filtered_image_paths) == 0:
        log.info("No files to process after filtering")
        exit(0)

    log.info(f"Found {len(filtered_image_paths)} files to process after filtering")
    for path in filtered_image_paths:
        log.debug(f" - {path}")
    return filtered_image_paths


def maybe_split_image_paths(image_paths: list[Path], num_splits: int, split_index: int) -> list[Path]:
    if num_splits > 1:
        log.info(f"Splitting {len(image_paths)} files into {num_splits} batches and processing batch {split_index}")
        image_paths = image_paths[split_index::num_splits]

    return image_paths
