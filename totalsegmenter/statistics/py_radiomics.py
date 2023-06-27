from pathlib import Path

import nibabel as nib
import numpy as np
from radiomics import featureextractor

from totalsegmenter.data.nifti import get_file_stem_from_nifti_path
from totalsegmenter.logger import logger
from totalsegmenter.settings import settings

log = logger.getChild(__name__)


def calculate_py_radiomics_features(
    seg_path_and_features: tuple[Path, dict[str, list[str]]],
    image_path: Path,
) -> tuple[str, dict[str, dict[str, float]]]:
    seg_path, radiomics_features_dict = seg_path_and_features
    try:
        if len(np.unique(nib.load(seg_path).get_fdata())) > 1:
            radiomics_config = {}
            radiomics_config["resampledPixelSpacing"] = settings.RADIOMICS_RESAMPLED_PIXEL_SPACING
            radiomics_config["geometryTolerance"] = settings.RADIOMICS_GEOMETRY_TOLERANCE  # type: ignore
            radiomics_config["featureClass"] = list(radiomics_features_dict.keys())  # type: ignore
            extractor = featureextractor.RadiomicsFeatureExtractor(**radiomics_config)
            extractor.disableAllFeatures()
            extractor.enableFeaturesByName(**radiomics_features_dict)
            compound_name_features = extractor.execute(str(image_path), str(seg_path))
            features: dict[str, dict[str, float]] = {}
            for compound_feature_name, feature_value in compound_name_features.items():
                if compound_feature_name.startswith("original_"):
                    compound_feature_name = compound_feature_name.replace("original_", "")
                    feature_class, feature_name = compound_feature_name.split("_")
                    if feature_class not in radiomics_features_dict:
                        continue

                    if feature_class not in features:
                        features[feature_class] = {}

                    features[feature_class][feature_name] = feature_value
        else:
            log.debug(f"Entire mask is 0 or 1 for {seg_path}. Setting all features to 0")
            features = {}
            for feature_class, feature_names in radiomics_features_dict.items():
                features[feature_class] = {feature_name: 0 for feature_name in feature_names}

    except Exception as e:
        log.warning(f"Radiomics raised an exception processing {seg_path} (settings all features to 0): {e}")
        features = {}
        for feature_class, feature_names in radiomics_features_dict.items():
            features[feature_class] = {feature_name: 0 for feature_name in feature_names}

    for feature_class, features_and_values in features.items():
        for feature_name, feature_value in features_and_values.items():
            features[feature_class][feature_name] = round(
                float(feature_value), settings.RADIOMICS_VALUES_ROUNDING_NDIGITS
            )

    return get_file_stem_from_nifti_path(seg_path), features
