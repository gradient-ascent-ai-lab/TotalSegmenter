import importlib

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage

from totalsegmenter.settings import settings

cupy_available = importlib.util.find_spec("cupy") is not None  # type: ignore
cucim_available = importlib.util.find_spec("cucim") is not None  # type: ignore


def change_spacing(
    image_in: np.ndarray,
    new_spacing=1.25,
    target_shape=None,
    order=0,
    dtype=None,
    remove_negative=False,
    force_affine=None,
):
    """
    Resample nifti image to the new spacing (uses resample_image() internally).

    image_in: nifti image
    new_spacing: float or sequence of float
    target_shape: sequence of int (optional)
    order: resample order (optional)
    dtype: output datatype
    remove_negative: set all negative values to 0. Useful if resampling introduced negative values.
    force_affine: if you pass an affine then this will be used for the output image (useful if you have to make sure
                  that the resampled has identical affine to some other image. In this case also set target_shape.)

    Works for 2D and 3D and 4D images.

    If downsampling an image and then upsampling again to original resolution the resulting image can have
    a shape which is +-1 compared to original shape, because of rounding of the shape to int.
    To avoid this the exact output shape can be provided. Then new_spacing will be ignored and the exact
    spacing will be calculated which is needed to get to target_shape.
    In this case however the calculated spacing can be slighlty different from the desired new_spacing. This will
    result in a slightly different affine. To avoid this the desired affine can be writen by force with "force_affine".

    Note: Only works properly if affine is all 0 except for diagonal and offset (=no rotation and sheering)
    """
    data = image_in.get_fdata()
    old_shape = np.array(data.shape)
    image_spacing = np.array(image_in.header.get_zooms())

    if len(image_spacing) == 4:
        # NOTE: for 4D images only use spacing of first 3 dims
        image_spacing = image_spacing[:3]

    if type(new_spacing) is float:
        # NOTE: for 3D and 4D
        new_spacing = [new_spacing] * 3
    new_spacing = np.array(new_spacing)

    if len(old_shape) == 2:
        image_spacing = np.array(list(image_spacing) + [new_spacing[2]])

    if target_shape is not None:
        zoom = np.array(target_shape) / old_shape
        new_spacing = image_spacing / zoom
    else:
        zoom = image_spacing / new_spacing

    new_affine = np.copy(image_in.affine)
    new_affine[:3, 0] = new_affine[:3, 0] / zoom[0]
    new_affine[:3, 1] = new_affine[:3, 1] / zoom[1]
    new_affine[:3, 2] = new_affine[:3, 2] / zoom[2]
    # NOTE: how to get spacing from affine with rotation by calculating the length of each column vector:
    #       vecs = affine[:3, :3]
    #       spacing = tuple(np.sqrt(np.sum(vecs ** 2, axis=0)))

    if cupy_available and cucim_available:
        new_data = resample_image_cucim(data, zoom=zoom, order=order)  # gpu resampling
    else:
        new_data = resample_image(data, zoom=zoom, order=order)  # cpu resampling

    if remove_negative:
        new_data[new_data < 1e-4] = 0

    if dtype is not None:
        new_data = new_data.astype(dtype)

    if force_affine is not None:
        new_affine = force_affine

    return nib.Nifti1Image(new_data, new_affine)


def change_spacing_of_affine(affine: np.ndarray, zoom: float = 0.5) -> np.ndarray:
    new_affine = np.copy(affine)
    for i in range(3):
        new_affine[i, i] /= zoom
    return new_affine


def resample_image(image_np: np.ndarray, zoom: float = 0.5, order: int = 0):
    """
    image_np: [x,y,z,(t)]
    zoom: 0.5 will reduce the image size by half

    Resize numpy image array to new size
    Faster than resample_image_nnunet
    Resample_image_nnunet maybe slighlty better quality on CT (but not sure)

    Works for 2D, 3D and 4D images
    """

    def _process_gradient(grad_idx):
        return ndimage.zoom(image_np[:, :, :, grad_idx], zoom, order=order)

    # NOTE: add dimensions to make each input 4D
    dimensions = len(image_np.shape)
    if dimensions == 2:
        image_np = image_np[..., None, None]
    if dimensions == 3:
        image_np = image_np[..., None]

    image_resampled_np = Parallel(n_jobs=settings.NUM_CORES_RESAMPLING)(
        delayed(_process_gradient)(grad_idx) for grad_idx in range(image_np.shape[3])
    )
    # NOTE: restore channel order
    image_resampled_np = np.array(image_resampled_np).transpose(1, 2, 3, 0)

    # NOTE: Remove added dimensions
    if dimensions == 3:
        image_resampled_np = image_resampled_np[:, :, :, 0]
    if dimensions == 2:
        image_resampled_np = image_resampled_np[:, :, 0, 0]
    return image_resampled_np


def resample_image_cucim(image_np: np.ndarray, zoom: float = 0.5, order: int = 0) -> np.ndarray:
    """
    Speed up versus CPU depends on image size due to the overhead of copying to the GPU
    For large images may reduce resampling time by over 50%
    """
    import cupy as cp
    from cucim.skimage.transform import resize

    image_cp = cp.asarray(image_np)
    new_shape = (np.array(image_cp.shape) * zoom).round().astype(np.int32)
    resampled_image = resize(image_cp, output_shape=new_shape, order=order, mode="edge", anti_aliasing=False)
    resampled_image = cp.asnumpy(resampled_image)
    return resampled_image
