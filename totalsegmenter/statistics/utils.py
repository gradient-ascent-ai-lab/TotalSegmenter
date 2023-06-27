import json
from collections import defaultdict
from pathlib import Path

import yaml

from totalsegmenter.data.nifti import get_file_stem_from_nifti_path
from totalsegmenter.logger import logger
from totalsegmenter.settings import settings

log = logger.getChild(__name__)


RadiomicsFeatureList = list[str]
RadiomicsFeatureSet = dict[str, RadiomicsFeatureList]
SegmentationName = str
RadiomicsFeatureClass = str
RadiomicsFeatureName = str
RadiomicsFeatureAndValue = dict[RadiomicsFeatureName, float]
RadiomicsStatistics = dict[SegmentationName, dict[RadiomicsFeatureClass, RadiomicsFeatureAndValue]]
RadiomicsStatisticsWithCompoundKeys = dict[SegmentationName, RadiomicsFeatureAndValue]


def get_radiomics_feature_set(radiomics_feature_set_name: str) -> RadiomicsFeatureSet | None:
    if radiomics_feature_set_name is None:
        return None

    radiomics_features_path = settings.RADIOMICS_FEATURES_DIR / f"{radiomics_feature_set_name}.yaml"
    if not radiomics_features_path.exists():
        raise ValueError(
            f"Radiomics feature set file '{radiomics_features_path}.yaml' "
            f"not found in {settings.RADIOMICS_FEATURES_DIR}"
        )

    with open(radiomics_features_path) as f:
        radiomics_feature_set = yaml.safe_load(f)

    validate_radiomics_feature_set(radiomics_feature_set)
    return radiomics_feature_set


def get_radiomics_statistics_to_calculate_per_segmentation(
    seg_paths: list[Path],
    radiomics_feature_set: RadiomicsFeatureSet,
    existing_statistics: RadiomicsStatistics,
) -> list[tuple[Path, RadiomicsFeatureSet]]:
    radiomics_statistics_to_calculate: list[tuple[Path, RadiomicsFeatureSet]] = []
    for seg_path in seg_paths:
        seg_name = get_file_stem_from_nifti_path(seg_path)
        feature_set_copy = radiomics_feature_set.copy()
        if seg_name not in existing_statistics:
            radiomics_statistics_to_calculate.append((seg_path, feature_set_copy))
            continue

        seg_existing_statistics = existing_statistics[seg_name]
        for feature_class_name, features_and_values_dict in seg_existing_statistics.items():
            for feature_name in features_and_values_dict:
                if feature_name in radiomics_feature_set[feature_class_name]:
                    feature_set_copy[feature_class_name].remove(feature_name)

        feature_classes = list(feature_set_copy.keys())
        for feature_class in feature_classes:
            if len(feature_set_copy[feature_class]) == 0:
                del feature_set_copy[feature_class]

        if len(feature_set_copy) > 0:
            radiomics_statistics_to_calculate.append((seg_path, feature_set_copy))

    log.debug(f"Found {len(radiomics_statistics_to_calculate)} segmentations to process")
    return radiomics_statistics_to_calculate


def split_radiomics_features_by_implementation(
    radiomics_features: RadiomicsFeatureSet, cu_radiomics_available: bool = False
) -> tuple[RadiomicsFeatureSet, RadiomicsFeatureSet]:
    if not cu_radiomics_available:
        return radiomics_features, {}

    py_radiomics_features_path = settings.RADIOMICS_FEATURES_DIR / "py_radiomics.yaml"
    with open(py_radiomics_features_path) as f:
        py_radiomics_feature_set = yaml.safe_load(f)

    cu_radiomics_features_path = settings.RADIOMICS_FEATURES_DIR / "cu_radiomics.yaml"
    with open(cu_radiomics_features_path) as f:
        cu_radiomics_feature_set = yaml.safe_load(f)

    py_radiomics_features = defaultdict(list)
    cu_radiomics_features = defaultdict(list)
    for feature_class, feature_list in radiomics_features.items():
        if feature_class in cu_radiomics_feature_set:
            for feature_name in feature_list:
                if feature_name in cu_radiomics_feature_set[feature_class]:
                    cu_radiomics_features[feature_class].append(feature_name)
                else:
                    if feature_name not in py_radiomics_feature_set[feature_class]:
                        raise ValueError(f"Feature {feature_name} not found in either pyradiomics or curadiomics")
                    py_radiomics_features[feature_class].append(feature_name)
        else:
            if feature_class not in py_radiomics_feature_set:
                raise ValueError(f"Feature class {feature_class} not found in either pyradiomics or curadiomics")
            for feature_name in feature_list:
                if feature_name not in py_radiomics_feature_set[feature_class]:
                    raise ValueError(f"Feature {feature_name} not found in either pyradiomics or curadiomics")
                py_radiomics_features[feature_class].append(feature_name)

    num_cu_features = sum([len(feature_list) for feature_list in cu_radiomics_features.values()])
    num_py_features = sum([len(feature_list) for feature_list in py_radiomics_features.values()])
    num_total_features = num_cu_features + num_py_features
    log.info(
        f"Found {num_cu_features} curadiomics features and {num_py_features} pyradiomics features "
        f"({num_total_features} total features)"
    )
    return dict(py_radiomics_features), dict(cu_radiomics_features)


def validate_radiomics_feature_set(radiomics_feature_set: RadiomicsFeatureSet) -> None:
    try:
        split_radiomics_features_by_implementation(radiomics_feature_set)
    except ValueError as e:
        raise ValueError(f"Radiomics feature set {radiomics_feature_set} is invalid: {e}")


def merge_radiomics_stats(
    stats: RadiomicsStatistics,
    py_radiomics_stats: RadiomicsStatistics,
    cu_radiomics_stats: RadiomicsStatistics,
) -> RadiomicsStatistics:
    merged_stats = stats.copy()
    merged_stats = _merge_radiomics_stats(merged_stats, py_radiomics_stats)
    merged_stats = _merge_radiomics_stats(merged_stats, cu_radiomics_stats)
    return merged_stats


def _merge_radiomics_stats(first_stats: RadiomicsStatistics, second_stats: RadiomicsStatistics) -> RadiomicsStatistics:
    if not first_stats:
        return second_stats
    if not second_stats:
        return first_stats

    for seg_name, feature_classes_and_features in second_stats.items():
        if seg_name not in first_stats:
            first_stats[seg_name] = feature_classes_and_features
            continue

        for feature_class, features_and_values_dict in feature_classes_and_features.items():
            if feature_class not in first_stats[seg_name]:
                first_stats[seg_name][feature_class] = features_and_values_dict
                continue

            for feature_name, feature_value in features_and_values_dict.items():
                first_stats[seg_name][feature_class][feature_name] = feature_value

    return first_stats


def convert_from_compound_keys(stats: RadiomicsStatisticsWithCompoundKeys) -> RadiomicsStatistics:
    stats_without_compound_keys: RadiomicsStatistics = {}
    for seg_name, features_and_values in stats.items():
        stats_without_compound_keys[seg_name] = {}
        for feature_name, feature_value in features_and_values.items():
            try:
                feature_class, feature_name = feature_name.split("_")
            except ValueError:
                raise ValueError(f"Invalid feature name {feature_name}")

            if feature_class not in stats_without_compound_keys[seg_name]:
                stats_without_compound_keys[seg_name][feature_class] = {}
            stats_without_compound_keys[seg_name][feature_class][feature_name] = feature_value
    return stats_without_compound_keys


def convert_to_compound_keys(stats: RadiomicsStatistics) -> RadiomicsStatisticsWithCompoundKeys:
    stats_with_compound_keys: RadiomicsStatisticsWithCompoundKeys = {}
    for seg_name, feature_classes_and_features in stats.items():
        stats_with_compound_keys[seg_name] = {}
        for feature_class, features_and_values in feature_classes_and_features.items():
            for feature_name, feature_value in features_and_values.items():
                stats_with_compound_keys[seg_name][f"{feature_class}_{feature_name}"] = feature_value
    return stats_with_compound_keys


def read_radiomics_from_path(stats_path: Path) -> RadiomicsStatistics:
    log.debug(f"Reading radiomics statistics from {stats_path}")
    with open(stats_path) as f:
        stats = json.load(f)

    stats = convert_from_compound_keys(stats)
    return stats


def write_radiomics_to_path(output_path: Path, stats: RadiomicsStatistics) -> None:
    log.debug(f"Writing radiomics statistics to {output_path}")
    stats_out = convert_to_compound_keys(stats)
    with open(output_path, "w") as f:
        json.dump(stats_out, f, indent=2)
