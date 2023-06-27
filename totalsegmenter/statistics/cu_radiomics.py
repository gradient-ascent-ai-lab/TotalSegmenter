from pathlib import Path

import nibabel as nib
import numpy as np
import tensorflow as tf

from totalsegmenter.logger import logger

log = logger.getChild(__name__)


def cu_radiomics_is_available() -> bool:
    try:
        tf.load_op_library("./build/libRadiomics.so").radiomics
        return True
    except Exception:
        return False


def calculate_cu_radiomics_features(
    image_path: Path, seg_path: Path, radiomics_features_dict: dict[str, list[str]]
) -> dict:
    calculate_first_order = 1
    calculate_glcm = 1
    label_value = 1

    image_np = nib.load(str(image_path)).get_fdata()
    seg_np = nib.load(str(seg_path)).get_fdata()

    value_range = [np.min(image_np).astype("int"), np.max(image_np).astype("int")]
    settings = np.array([value_range[0], value_range[1], calculate_first_order, calculate_glcm, label_value])
    image_np[np.where(seg_np != label_value)] = -1

    radiomics_module = tf.load_op_library("./build/libRadiomics.so").radiomics
    features = radiomics_module(image_np, settings)
    feature_names_to_calculate: list[str] = []
    for _feature_class, feature_names in radiomics_features_dict.values():
        feature_names_to_calculate += feature_names

    features = np.reshape(features, newshape=[len(feature_names_to_calculate), image_np.shape[0]])

    # TODO: find out why the value is np.ndarray
    stats: dict[str, np.ndarray] = {}
    for feature_index, feature_name in enumerate(feature_names_to_calculate):
        stats[feature_name] = features[feature_index, :]

    return stats
