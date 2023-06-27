from pathlib import Path

import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

from totalsegmenter.logger import logger

log = logger.getChild(__name__)


def as_closest_canonical(input_nii: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Convert the given nifti file to the closest canonical nifti file.
    """
    log.debug("Converting to closest canonical...")
    return nib.as_closest_canonical(input_nii)


def undo_canonical(canonical_nii: nib.Nifti1Image, original_nii: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Inverts nib.to_closest_canonical()

    canonical_nii: the image we want to move back
    original_nii: the original image because transforming to canonical

    returns image in original space

    https://github.com/nipy/nibabel/issues/1063
    """
    log.debug("Undoing canonical...")
    image_orientation = io_orientation(original_nii.affine)
    ras_orientation = axcodes2ornt("RAS")
    from_canonical = ornt_transform(ras_orientation, image_orientation)
    return canonical_nii.as_reoriented(from_canonical)


def as_closest_canonical_nifti(input_path: Path, output_path: Path) -> None:
    """
    Convert the given nifti file to the closest canonical nifti file.
    """
    input_nii = nib.load(input_path)
    output_nii = nib.as_closest_canonical(input_nii)
    nib.save(output_nii, output_path)


def undo_canonical_nifti(canonical_nii_path: Path, original_nii_path: Path, output_path: Path) -> None:
    canonical_nii = nib.load(canonical_nii_path)
    original_nii = nib.load(original_nii_path)
    output_nii = undo_canonical(canonical_nii, original_nii)
    nib.save(output_nii, output_path)
