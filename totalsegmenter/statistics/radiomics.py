import logging
from functools import partial
from pathlib import Path

from p_tqdm import p_map

from totalsegmenter.logger import logger
from totalsegmenter.settings import settings
from totalsegmenter.statistics.cu_radiomics import calculate_cu_radiomics_features, cu_radiomics_is_available
from totalsegmenter.statistics.py_radiomics import calculate_py_radiomics_features
from totalsegmenter.statistics.utils import (
    get_radiomics_statistics_to_calculate_per_segmentation,
    merge_radiomics_stats,
    read_radiomics_from_path,
    split_radiomics_features_by_implementation,
    write_radiomics_to_path,
)
from totalsegmenter.utils import log_duration

log = logger.getChild(__name__)


@log_duration
def calculate_radiomics_features_for_all_segmentations(
    image_path: Path, output_dir: Path, radiomics_feature_set: dict[str, list[str]]
):
    output_path = output_dir / settings.RADIOMICS_FILENAME
    stats: dict[str, dict[str, dict[str, float]]] = {}
    if output_path.exists():
        stats = read_radiomics_from_path(stats_path=output_path)

    segmentations_dir = output_dir / settings.SINGLE_LABEL_SEGMENTATIONS_SUBDIR
    seg_paths = sorted(list(segmentations_dir.glob("*.nii.gz")))
    log.debug(f"Found {len(seg_paths)} segmentations in {segmentations_dir}")
    if len(seg_paths) == 0:
        log.error(f"No segmentations found in {segmentations_dir}! Cannot calculate radiomics features")
        return

    seg_paths_and_features = get_radiomics_statistics_to_calculate_per_segmentation(
        seg_paths=seg_paths,
        radiomics_feature_set=radiomics_feature_set,
        existing_statistics=stats,
    )

    py_radiomics_segs_and_features = []
    cu_radiomics_segs_and_features = []
    for seg_path, features_to_calculate in seg_paths_and_features:
        py_radiomics_features, cu_radiomics_features = split_radiomics_features_by_implementation(
            radiomics_features=features_to_calculate,
            cu_radiomics_available=cu_radiomics_is_available(),
        )
        py_radiomics_segs_and_features.append((seg_path, py_radiomics_features))
        if len(cu_radiomics_features) > 0:
            cu_radiomics_segs_and_features.append((seg_path, cu_radiomics_features))

    py_radiomics_stats = {}
    if len(py_radiomics_segs_and_features) > 0:
        log.info(f"Calculating pyRadiomics features for {len(py_radiomics_segs_and_features)} segmentations")
        log.info(f"Parallelizing over {settings.NUM_CORES_RADIOMICS} cores")
        py_radiomics_stats_tuples = p_map(
            partial(calculate_py_radiomics_features, image_path=image_path),
            py_radiomics_segs_and_features,
            num_cpus=settings.NUM_CORES_RADIOMICS,
            disable=logger.getEffectiveLevel() != logging.INFO,
        )
        py_radiomics_stats = {seg_name: stats for seg_name, stats in py_radiomics_stats_tuples}

    cu_radiomics_stats = {}
    if len(cu_radiomics_segs_and_features) > 0:
        log.info(f"Calculating cuRadiomics features for {len(cu_radiomics_segs_and_features)} segmentations")
        for seg_path, cu_radiomics_features in cu_radiomics_segs_and_features:
            log.debug(f"Calculating cuRadiomics features for {seg_path}")
            # TODO: check behaviour of cuRadiomics when parallelized
            cu_radiomics_stats = calculate_cu_radiomics_features(
                image_path=image_path,
                seg_path=seg_path,
                radiomics_features_dict=cu_radiomics_features,
            )

    stats = merge_radiomics_stats(
        stats=stats,
        py_radiomics_stats=py_radiomics_stats,
        cu_radiomics_stats=cu_radiomics_stats,
    )

    write_radiomics_to_path(output_path=output_path, stats=stats)
