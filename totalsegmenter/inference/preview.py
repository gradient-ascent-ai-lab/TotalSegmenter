from pathlib import Path

import nibabel as nib
import numpy as np
from fury import actor, window
from xvfbwrapper import Xvfb

from totalsegmenter.inference.plotting import plot_mask
from totalsegmenter.logger import logger
from totalsegmenter.settings import settings
from totalsegmenter.tasks import Task
from totalsegmenter.utils import log_duration

log = logger.getChild(__name__)
np.random.seed(1234)
random_colors = np.random.rand(100, 4)


@log_duration
def generate_preview(image_nii: nib.Nifti1Image, pred_nii: nib.Nifti1Image, output_dir: Path, task: Task) -> None:
    log.info("Generating preview...")
    preview_output_path = output_dir / task.name / "preview.png"
    preview_output_path.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: do not set random seed, otherwise we cannot call xvfb in parallel, because all generate same tmp dir
    with Xvfb():
        window_size = (settings.PREVIEW_WINDOW_WIDTH, settings.PREVIEW_WINDOW_HEIGHT)
        scene = window.Scene()
        showm = window.ShowManager(scene, size=window_size, reset_camera=False)
        showm.initialize()

        image_np = image_nii.get_fdata()
        image_np = image_np.transpose(1, 2, 0)  # Show sagittal view
        image_np = image_np[::-1, :, :]
        pred_np = pred_nii.get_fdata()
        log.debug(f"Shape image: {image_np.shape}, pred: {pred_np.shape}")

        value_range = (-115, 225)  # soft tissue window
        slice_actor = actor.slicer(image_np, image_nii.affine, value_range)
        slice_actor.SetPosition(0, 0, 0)
        scene.add(slice_actor)

        for idx, preview_roi_group in enumerate(task.preview_roi_groups):
            # NOTE: increase by 1 because 0 is the ct image
            idx += 1
            x = (idx % settings.PREVIEW_NUM_COLUMNS) * settings.PREVIEW_SUBJECT_WIDTH
            if not settings.PREVIEW_USE_SUBJECT_HEIGHT:
                y = 0
            else:
                y = (idx // settings.PREVIEW_NUM_COLUMNS) * settings.PREVIEW_SUBJECT_HEIGHT

            plot_roi_group(
                scene=scene, roi_group=preview_roi_group, x=x, y=y, pred_np=pred_np, affine=image_nii.affine, task=task
            )

        scene.projection(proj_type="parallel")
        scene.reset_camera_tight(margin_factor=1.02)
        preview_output_path.parent.mkdir(parents=True, exist_ok=True)
        window.record(scene, size=window_size, out_path=preview_output_path, reset_camera=False)
        scene.clear()


def plot_roi_group(
    scene: window.Scene,
    roi_group: list[str],
    x: int,
    y: int,
    pred_np: np.ndarray,
    affine: np.ndarray,
    task: Task,
) -> None:
    roi_actors = []
    classname_2_idx = {v: k for k, v in task.get_class_map().items()}
    for roi_index, roi_name in enumerate(roi_group):
        color = random_colors[roi_index]
        if roi_name not in classname_2_idx:
            log.warning(f"ROI {roi_name} not found in class map")
            continue

        mask_data = pred_np == classname_2_idx[roi_name]
        # NOTE: empty mask
        if mask_data.max() > 0:
            # NOTE: make offset the same for all subjects
            affine[:3, 3] = 0
            cont_actor = plot_mask(
                mask_data=mask_data,
                affine=affine,
                x_current=x,
                y_current=y,
                color=color,
                opacity=1,
            )
            scene.add(cont_actor)
            roi_actors.append(cont_actor)
