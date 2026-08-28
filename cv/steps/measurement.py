"""
Step 8 — Broken Grain Classification

Classifies grains as "broken" if their length falls below 80% of the
median grain length.

Grain measurement itself (contour -> length/width/ratio in mm) now lives in
`cv.steps.wheat_measurement`, which owns segmentation and coin calibration as
well; the old `measure_grains` here was left unreferenced by that move and has
been removed.
"""

import numpy as np

from cv.config import BROKEN_THRESHOLD_PCT


def classify_broken(measurements: list[dict]) -> tuple[float, float]:
    """
    Classify grains as broken based on median length.

    A grain is "broken" if its length is less than
    BROKEN_THRESHOLD_PCT (80%) of the median grain length.

    Args:
        measurements: list of per-grain dicts carrying at least "length_mm"
            (the pipeline adapts the measurement module's grains to this shape).

    Returns:
        (broken_pct, median_length_mm):
            broken_pct — percentage of grains classified as broken.
            median_length_mm — the median grain length used as reference.
    """
    if not measurements:
        return 0.0, 0.0

    lengths = [m["length_mm"] for m in measurements]
    median_length = float(np.median(lengths))

    threshold = median_length * BROKEN_THRESHOLD_PCT
    broken_count = sum(1 for length in lengths if length < threshold)
    broken_pct = (broken_count / len(lengths)) * 100.0

    return round(broken_pct, 2), round(median_length, 3)
