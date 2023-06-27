#!/usr/bin/env python
import zipfile
from pathlib import Path
from time import sleep

from typer import Option, Typer, confirm, secho

from totalsegmenter.data.nifti import combine_masks_to_multilabel_file, gzip_nifti_files
from totalsegmenter.data.verify import verify_images
from totalsegmenter.enums import ImageType, NnUnetModelType
from totalsegmenter.inference.batch import batch_predict
from totalsegmenter.inference.download import download_pretrained_weights
from totalsegmenter.inference.pre_segment_body import crop_to_body
from totalsegmenter.inference.predict import total_segmenter_predict
from totalsegmenter.settings import settings
from totalsegmenter.statistics.utils import get_radiomics_feature_set
from totalsegmenter.tasks import task_manager
from totalsegmenter.utils import log_all_durations, log_nora_tag

cli = Typer(
    name="totalsegmenter",
    epilog="\n\nWritten by Jakob Wasserthal. If you use this tool please cite https://arxiv.org/abs/2208.05868\n\n",
    no_args_is_help=True,
)


@cli.command("predict")
def predict_command(
    input_path: Path = Option(..., "-i", "--input", help="Path to the input image"),
    task_name: str = "total",
    fast: bool = False,
    preview: bool = False,
    statistics: bool = False,
    radiomics: bool = False,
    radiomics_features_name: str = "py_radiomics",
    multi_label: bool = False,
    roi_subset: list[str] = None,
    crop_path: Path = None,
    body_seg: bool = False,
    force_split: bool = False,
    output_type: ImageType = ImageType.NIFTI.value,  # type: ignore
    test_mode: int = 0,
    skip_saving: bool = False,
    nora_tag: str = None,
):
    verify_images(image_paths=[input_path])
    radiomics_feature_set = get_radiomics_feature_set(radiomics_features_name)
    total_segmenter_predict(
        input_path=input_path,
        task_name=task_name,
        fast=fast,
        preview=preview,
        statistics=statistics,
        radiomics=radiomics,
        radiomics_feature_set=radiomics_feature_set,
        multi_label=multi_label,
        roi_subset=roi_subset,
        crop_path=crop_path,
        body_seg=body_seg,
        force_split=force_split,
        output_type=output_type,
        test_mode=test_mode,
        skip_saving=skip_saving,
        nora_tag=nora_tag,
    )
    log_all_durations(to_file=True)


@cli.command("predict-batch")
def batch_predict_command(
    input_dir: Path = Option(None, "-i", "--input-dir", help="Directory containing input images"),
    task_names: list[str] = Option(["total"], "-t", "--task-name", help="Names of the task to run"),
    multi_label: bool = False,
    roi_subset: list[str] = None,
    statistics: bool = False,
    radiomics: bool = False,
    radiomics_features_name: str = "py_radiomics",
    preview: bool = False,
    num_splits: int = 1,
    split_index: int = 0,
):
    if input_dir is None:
        input_dir = settings.DATA_DIR

    raw_nifti_paths = list(input_dir.glob("**/*.nii"))
    if len(raw_nifti_paths) > 0:
        secho(f"Found {len(raw_nifti_paths)} files with .nii extension", fg="yellow")
        secho(f"Files: {[str(p) for p in raw_nifti_paths]}")
        if confirm("Convert to .nii.gz?"):
            gzip_nifti_files(input_paths=raw_nifti_paths)

    radiomics_feature_set = get_radiomics_feature_set(radiomics_features_name)

    batch_predict(
        input_dir=input_dir,
        task_names=task_names,
        multi_label=multi_label,
        roi_subset=roi_subset,
        statistics=statistics,
        radiomics=radiomics,
        radiomics_feature_set=radiomics_feature_set,
        preview=preview,
        num_splits=num_splits,
        split_index=split_index,
    )


@cli.command("gzip")
def gzip_command(input_dir: Path = Option(None, "-i", "--input-dir", help="Directory containing input images")):
    raw_nifti_paths = list(input_dir.glob("**/*.nii"))
    if len(raw_nifti_paths) > 0:
        secho(f"Found {len(raw_nifti_paths)} files with .nii extension", fg="yellow")
        secho(f"Files: {[str(p) for p in raw_nifti_paths]}")
        if confirm("Convert to .nii.gz?"):
            gzip_nifti_files(input_paths=raw_nifti_paths)


@cli.command("combine-masks")
def combine_masks_command(
    task_name: str = Option("total", "-t", "--task", help="Name of the task with which the masks were created"),
    original_image_path: Path = Option(..., "-i", "--image", help="Path to the original image"),
    masks_dir: Path = Option(
        ..., "-m", "--masks-dir", help="Directory containing masks (e.g. output directory of predict command)"
    ),
    nora_tag: str = Option(None, help="tag in nora as mask. Pass nora project id as argument."),
):
    output_path = combine_masks_to_multilabel_file(
        task_name=task_name, original_image_path=original_image_path, masks_dir=masks_dir
    )
    log_nora_tag(tag=nora_tag, output_path=output_path)


@cli.command("crop-to-body")
def crop_to_body_command(
    input_path: Path = Option(..., "-i", "--input", help="Path to the input image"),
    output_path: Path = Option(..., "-o", "--output", help="Output path for cropped image"),
    only_trunk: bool = Option(False, help="Crop to trunk only"),
):
    crop_to_body(input_path=input_path, output_path=output_path, only_trunk=only_trunk)


@cli.command("download-weights")
def download_weights_command(task_name: str = Option(None, help="Name of the task")):
    if task_name is None:
        task_name = "total"

    task = task_manager.get_task_by_name(task_name)
    subtasks = [t for t in task.subtasks] or [task]
    for subtask in subtasks:
        secho(f"Processing {subtask.name}...", fg="blue")
        download_pretrained_weights(task=subtask)
        sleep(5)


@cli.command("import-weights")
def import_weights_command(weights_path: Path, model_type: NnUnetModelType = NnUnetModelType.FULLRES_3D):
    settings.WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
    secho(f"Extracting file {weights_path} to {settings.WEIGHTS_DIR}", fg="blue")
    with zipfile.ZipFile(weights_path, "r") as zip_f:
        zip_f.extractall(settings.WEIGHTS_DIR.absolute())


@cli.command("list-tasks")
def list_tasks_command():
    tasks = task_manager.list_tasks()
    available_tasks = [t for t in tasks if t.is_available()]
    unavailable_tasks = [t for t in tasks if not t.is_available()]
    secho("Available tasks:", fg="green")
    for task in available_tasks:
        secho(f"  - {task.name}")
    secho("Unavailable tasks:", fg="red")
    for task in unavailable_tasks:
        secho(f"  - {task.name}")


if __name__ == "__main__":
    cli()
