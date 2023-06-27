from enum import Enum


class ImageType(Enum):
    DICOM = "dicom"
    NIFTI = "nifti"


class MaskType(Enum):
    BODY = "body"
    BRAIN = "brain"
    HEART = "heart"
    LIVER = "liver"
    LUNG = "lung"
    LUNG_LEFT = "lung_left"
    LUNG_RIGHT = "lung_right"
    PELVIS = "pelvis"
    RIBS = "ribs"
    VERTEBRAE = "vertebrae"
    VERTEBRAE_RIBS = "vertebrae_ribs"


class NnUnetModelType(Enum):
    TWO_D = "2d"
    LOWRES_3D = "3d_lowres"
    FULLRES_3D = "3d_fullres"
    CASCADE = "3d_cascade_fullres"


class NnUnetPredictionMode(Enum):
    NORMAL = "normal"
    FAST = "fast"
    FASTEST = "fastest"
