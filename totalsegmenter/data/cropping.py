from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def get_bbox_from_mask(mask: np.ndarray, outside_value: int = -900, addon: list[int] = [0, 0, 0]) -> list[list[int]]:
    if (mask > outside_value).sum() == 0:
        print("WARNING: Could not crop because no foreground detected")
        minzidx, maxzidx = 0, mask.shape[0]
        minxidx, maxxidx = 0, mask.shape[1]
        minyidx, maxyidx = 0, mask.shape[2]
    else:
        mask_voxel_coords: tuple[np.ndarray, ...] = np.where(mask > outside_value)
        minzidx = int(np.min(mask_voxel_coords[0])) - addon[0]
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1 + addon[0]
        minxidx = int(np.min(mask_voxel_coords[1])) - addon[1]
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1 + addon[1]
        minyidx = int(np.min(mask_voxel_coords[2])) - addon[2]
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1 + addon[2]
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1 + addon[2]

    # Avoid bbox to get out of image size
    s = mask.shape
    minzidx = max(0, minzidx)
    maxzidx = min(s[0], maxzidx)
    minxidx = max(0, minxidx)
    maxxidx = min(s[1], maxxidx)
    minyidx = max(0, minyidx)
    maxyidx = min(s[2], maxyidx)

    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image: np.ndarray, bbox: list[list[int]]) -> np.ndarray:
    """
    image: 3d nd.array
    bbox: list of lists [[minx_idx, maxx_idx], [miny_idx, maxy_idx], [minz_idx, maxz_idx]]
          Indices of bbox must be in voxel coordinates  (not in world space)
    """
    assert len(image.shape) == 3, f"crop_to_bbox only supports 3d images, got {len(image.shape)}"
    return image[bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]]


def crop_to_bbox_nifti(image_nii: nib.Nifti1Image, bbox: list[list[int]], dtype: str = None) -> nib.Nifti1Image:
    """
    Crop nifti image to bounding box and adapt affine accordingly

    image: nib.Nifti1Image
    bbox: list of lists [[minx_idx, maxx_idx], [miny_idx, maxy_idx], [minz_idx, maxz_idx]]
          Indices of bbox must be in voxel coordinates  (not in world space)
    dtype: dtype of the output image

    returns: nib.Nifti1Image
    """
    assert len(image_nii.shape) == 3, f"crop_to_bbox_nifti only supports 3d images, got {len(image_nii.shape)}"
    data = image_nii.get_fdata()
    data_cropped = crop_to_bbox(data, bbox)
    affine = np.copy(image_nii.affine)
    affine[:3, 3] = np.dot(affine, np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1]))[:3]
    data_type = image_nii.dataobj.dtype if dtype is None else dtype
    return nib.Nifti1Image(data_cropped.astype(data_type), affine)


def crop_to_mask(
    input_nii: nib.Nifti1Image,
    mask_nii: nib.Nifti1Image,
    addon: list[int] = [0, 0, 0],
    dtype: np.dtype | None = None,
) -> tuple[nib.Nifti1Image, list[list[int]]]:
    """
    Crops a nifti image to a mask and adapts the affine accordingly.

    input_nii: nifti image
    mask_nii: nifti image
    addon = addon in mm along each axis
    dtype: output dtype

    Returns a nifti image.
    """
    mask = mask_nii.get_fdata()
    addon = (np.array(addon) / input_nii.header.get_zooms()).astype(int)  # mm to voxels
    bbox = get_bbox_from_mask(mask=mask, outside_value=0, addon=addon)
    output_nii = crop_to_bbox_nifti(image_nii=input_nii, bbox=bbox, dtype=dtype)
    return output_nii, bbox


def crop_to_mask_nifti(image_path: Path, mask_path: Path, output_path: Path, addon=[0, 0, 0], dtype=None):
    """
    Crops a nifti image to a mask and adapts the affine accordingly.

    image_path: nifti image path
    mask_path: nifti image path
    output_path: output path
    addon = addon in mm along each axis
    dtype: output dtype

    Returns bbox coordinates.
    """
    input_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))
    image_out, bbox = crop_to_mask(input_nii, mask_nii, addon, dtype)
    nib.save(image_out, str(output_path))
    return bbox


def undo_crop(input_nii: nib.Nifti1Image, reference_nii: nib.Nifti1Image, bbox: list[list[int]]) -> nib.Nifti1Image:
    """
    Fit the image which was cropped by bbox back into the shape of reference_nii.
    """
    output_np = np.zeros(reference_nii.shape)
    output_np[bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]] = input_nii.get_fdata()
    return nib.Nifti1Image(output_np, reference_nii.affine)


def undo_crop_nifti(image_path: Path, reference_image_path: Path, output_path: Path, bbox: list[list[int]]) -> None:
    """
    Fit the image which was cropped by bbox back into the shape of ref_image.
    """
    image = nib.load(image_path)
    ref_image = nib.load(reference_image_path)
    image_out = undo_crop(image, ref_image, bbox)
    nib.save(image_out, output_path)
