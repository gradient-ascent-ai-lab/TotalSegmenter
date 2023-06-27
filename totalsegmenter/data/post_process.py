from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion


def keep_largest_blob(data, debug=False):
    blob_map, _ = ndimage.label(data)
    # NOTE: number of pixels in each blob
    counts = list(np.bincount(blob_map.flatten()))
    if len(counts) <= 1:
        # NOTE: no foreground
        return data
    if debug:
        print(f"size of second largest blob: {sorted(counts)[-2]}")
    key_second = counts.index(sorted(counts)[-2])
    return (blob_map == key_second).astype(np.uint8)


def remove_small_blobs(image: np.ndarray, interval: list[int] = [10, 30], debug: bool = False) -> np.ndarray:
    """
    Find blobs/clusters of same label. Remove all blobs which have a size which is outside of the interval.

    Args:
        image: Binary image.
        interval: Boundaries of the sizes to remove.
        debug: Show debug information.
    Returns:
        Detected blobs.
    """
    mask, number_of_blobs = ndimage.label(image)
    if debug:
        print("Number of blobs before: " + str(number_of_blobs))
    counts = np.bincount(mask.flatten())  # number of pixels in each blob

    # NOTE: if only one blob (only background) abort because nothing to remove
    if len(counts) <= 1:
        return image

    remove = np.where((counts <= interval[0]) | (counts > interval[1]), True, False)
    remove_idx = np.nonzero(remove)[0]
    mask[np.isin(mask, remove_idx)] = 0
    mask[mask > 0] = 1  # set everything else to 1

    if debug:
        print(f"counts: {sorted(counts)[::-1]}")
        _, number_of_blobs_after = ndimage.label(mask)
        print("Number of blobs after: " + str(number_of_blobs_after))

    return mask


def remove_outside_of_mask(seg_path: Path, mask_path: Path, dilation_iterations: int = 1):
    """
    Remove all segmentations outside of mask.

    seg_path: path to nifti file
    mask_path: path to nifti file
    """
    seg_image = nib.load(seg_path)
    seg = seg_image.get_fdata()
    mask = nib.load(mask_path).get_fdata()
    mask = binary_dilation(mask, iterations=dilation_iterations)
    seg[mask == 0] = 0
    nib.save(nib.Nifti1Image(seg.astype(np.uint8), seg_image.affine), seg_path)


def extract_skin(image_nii, body_image):
    """
    Extract the skin from a segmentation of the body.

    image_nii: nifti image
    body_image: nifti image

    returns: nifti image
    """
    ct = image_nii.get_fdata()
    body = body_image.get_fdata()

    # Select skin region
    body = binary_dilation(body, iterations=1).astype(np.uint8)  # add 1 voxel margin at the outside
    body_inner = binary_erosion(body, iterations=3).astype(np.uint8)
    skin = body - body_inner

    # Segment by density
    # Roughly the skin density range. Made large to make segmentation not have holes
    # (0 to 250 would have many small holes in skin)
    density_mask = (ct > -200) & (ct < 250)
    skin[~density_mask] = 0

    # Fill holes
    # skin = binary_closing(skin, iterations=1)  # no real difference
    # skin = binary_dilation(skin, iterations=1)  # not good

    # Removing blobs
    skin = remove_small_blobs(skin > 0.5, interval=[5, 1e10])
    return nib.Nifti1Image(skin.astype(np.uint8), image_nii.affine)
