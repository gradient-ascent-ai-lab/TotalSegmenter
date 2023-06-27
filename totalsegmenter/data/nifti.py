import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import xmltodict

from totalsegmenter.enums import MaskType
from totalsegmenter.logger import logger
from totalsegmenter.settings import settings
from totalsegmenter.tasks import task_manager

log = logger.getChild(__name__)


def compress_nifti(file_in: str, file_out: str, dtype: type = np.int32, force_3d: bool = True) -> None:
    log.info(f"Compressing {file_in} to {file_out}...")
    image = nib.load(file_in)
    data = image.get_fdata()
    if force_3d and len(data.shape) > 3:
        log.info("Input image contains more than 3 dimensions. Only keeping first 3 dimensions.")
        data = data[:, :, :, 0]
    new_image = nib.Nifti1Image(data.astype(dtype), image.affine)
    nib.save(new_image, file_out)


def gzip_nifti_files(input_paths: list[Path]) -> None:
    log.info(f"Gzipping {len(input_paths)} nifti files...")
    for input_path in input_paths:
        subprocess.call(f"gzip {input_path}", shell=True)


def check_if_shape_and_affine_identical(image_1_nii: nib.Nifti1Image, image_2_nii: nib.Nifti1Image) -> None:
    if not np.array_equal(image_1_nii.affine, image_2_nii.affine):
        log.warning("Output affine not equal to input affine. This should not happen.")
        log.warning(
            f"Input affine: {image_1_nii.affine}, output affine: {image_2_nii.affine}, "
            f"diff: {np.abs(image_1_nii.affine - image_2_nii.affine)}"
        )

    if image_1_nii.shape != image_2_nii.shape:
        log.warning("Output shape not equal to input shape. This should not happen.")
        log.warning(f"Input shape: {image_1_nii.shape}, output shape: {image_2_nii.shape}")


def add_label_map_to_nifti(image_in: nib.Nifti1Image, label_map: dict | list) -> nib.Nifti1Image:
    """
    This will save the information which label in a segmentation mask has which name to the extended header.

    image: nifti image
    label_map: a dictionary with label ids and names | a list of names and a running id will be generated starting at 1

    returns: nifti image
    """
    data = image_in.get_fdata()

    if label_map is None:
        label_map = {idx + 1: f"L{val}" for idx, val in enumerate(np.unique(data)[1:])}

    if type(label_map) is not dict:  # can be list or dict_values list
        label_map = {idx + 1: val for idx, val in enumerate(label_map)}

    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 128, 0],
        [255, 0, 128],
        [128, 255, 128],
        [0, 128, 255],
        [128, 128, 128],
        [185, 170, 155],
    ]
    xmlpre = '<?xml version="1.0" encoding="UTF-8"?> <CaretExtension>  <Date><![CDATA[2013-07-14T05:45:09]]></Date>   '
    xmlpre += '<VolumeInformation Index="0">   <LabelTable>'

    body = ""
    for label_id, label_name in label_map.items():
        rgb = colors[label_id % len(colors)]
        body += f'<Label Key="{label_id}" Red="{rgb[0]/255}" Green="{rgb[1]/255}" Blue="{rgb[2]/255}" Alpha="1">'
        body += f"<![CDATA[{label_name}]]></Label>\n"

    xmlpost = "  </LabelTable>  <StudyMetaDataLinkSet>  </StudyMetaDataLinkSet>  "
    xmlpost += "<VolumeType><![CDATA[Label]]></VolumeType>   "
    xmlpost += "</VolumeInformation></CaretExtension>"
    xml = xmlpre + "\n" + body + "\n" + xmlpost + "\n              "

    image_in.header.extensions.append(nib.nifti1.Nifti1Extension(0, bytes(xml, "utf-8")))

    return image_in


def load_multilabel_nifti(image_path: str) -> tuple[nib.Nifti1Image, dict[int, str]]:
    """
    image_path: path to the image
    returns:
        image: nifti image
        label_map: a dictionary with label ids and names
    """
    image = nib.load(image_path)
    ext_header = image.header.extensions[0].get_content()
    ext_header = xmltodict.parse(ext_header)
    ext_header = ext_header["CaretExtension"]["VolumeInformation"]["LabelTable"]["Label"]
    label_map = {int(e["@Key"]): e["#text"] for e in ext_header}
    return image, label_map


def combine_masks(mask_dir: Path, output_path: Path, mask_type: MaskType):
    """
    Combine classes to masks

    mask_dir: directory of totalsegmetator masks
    output: output path
    class_type: ribs | vertebrae | vertebrae_ribs | lung | heart
    """
    total_seg_task = task_manager.get_task_by_name("totalsegmentator")
    ribs_task = next(t for t in total_seg_task.subtasks if t.name == "ribs")
    vertebrae_task = next(t for t in total_seg_task.subtasks if t.name == "vertebrae")

    # TODO: move to config
    if mask_type == MaskType.RIBS:
        masks = list(ribs_task.class_map.values())
    elif mask_type == MaskType.VERTEBRAE:
        masks = list(vertebrae_task.class_map.values())
    elif mask_type == MaskType.VERTEBRAE_RIBS:
        masks = list(vertebrae_task.class_map.values()) + list(ribs_task.class_map.values())
    elif mask_type == MaskType.LUNG:
        masks = [
            "lung_upper_lobe_left",
            "lung_lower_lobe_left",
            "lung_upper_lobe_right",
            "lung_middle_lobe_right",
            "lung_lower_lobe_right",
        ]
    elif mask_type == MaskType.LUNG_LEFT:
        masks = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
    elif mask_type == MaskType.LUNG_RIGHT:
        masks = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
    elif mask_type == MaskType.HEART:
        masks = [
            "heart_myocardium",
            "heart_atrium_left",
            "heart_ventricle_left",
            "heart_atrium_right",
            "heart_ventricle_right",
        ]
    elif mask_type == MaskType.PELVIS:
        masks = ["femur_left", "femur_right", "hip_left", "hip_right"]
    elif mask_type == MaskType.BODY:
        masks = ["body_trunc", "body_extremities"]

    reference_image_nib = None
    for mask in masks:
        if (mask_dir / f"{mask}.nii.gz").exists():
            reference_image_nib = nib.load(mask_dir / f"{masks[0]}.nii.gz")
        else:
            raise ValueError(f"Could not find {mask_dir / mask}.nii.gz. Did you run TotalSegmentator successfully?")

    combined = np.zeros(reference_image_nib.shape, dtype=np.uint8)
    for idx, mask in enumerate(masks):
        if (mask_dir / f"{mask}.nii.gz").exists():
            image = nib.load(mask_dir / f"{mask}.nii.gz").get_fdata()
            combined[image > 0.5] = 1

    nib.save(nib.Nifti1Image(combined, reference_image_nib.affine), str(output_path))


def combine_masks_to_multilabel_file(task_name: str, original_image_path: Path, masks_dir: Path) -> Path:
    """
    Generate one multilabel nifti file from a directory of single binary masks of each class.
    This multilabel file is needed to train a nnU-Net.
    """
    output_path = masks_dir.parent / task_name / settings.MULTI_LABEL_SEGMENTATION_FILENAME
    log.info(f"Generating multilabel file '{output_path}' from all masks in '{masks_dir}'")
    reference_image_nib = nib.load(original_image_path)
    masks = task_manager.get_task_by_name(task_name).class_map.values()
    image_out_np = np.zeros(reference_image_nib.shape).astype(np.uint8)

    for idx, mask in enumerate(masks):
        if os.path.exists(f"{masks_dir}/{mask}.nii.gz"):
            image = nib.load(f"{masks_dir}/{mask}.nii.gz").get_fdata()
        else:
            log.warning(f"Mask {mask} is missing. Filling with zeros.")
            image = np.zeros(reference_image_nib.shape)
        image_out_np[image > 0.5] = idx + 1

    nib.save(nib.Nifti1Image(image_out_np, reference_image_nib.affine), output_path)
    return output_path


def get_file_stem_from_nifti_path(nifti_path: Path) -> str:
    """
    Get the file stem from a nifti path, e.g. 'case_001' from '/path/to/case_001.nii.gz'
    Used because Path(nifti_path).stem does not work with .nii.gz files (returns 'case_001.nii')
    and Path(nifti_path).stem.split(".")[0] or Path(nifti_path.stem).stem is not
    very clear.
    """
    return Path(nifti_path.stem).stem
