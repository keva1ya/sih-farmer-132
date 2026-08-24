"""
Steps 7 & 8 — Grain Measurement & Broken Classification

For each grain contour:
  1. Fit a minimum-area rotated rectangle (minAreaRect)
  2. Convert pixel dimensions to mm using the coin calibration factor
  3. Compute length, width, and L/W ratio

Then classify grains as "broken" if their length falls below 80%
of the median grain length.
"""

import cv2
import numpy as np

from cv.config import BROKEN_THRESHOLD_PCT


def measure_grains(
    grain_contours: list[np.ndarray],
    pixels_per_mm: float,
) -> list[dict]:
    """
    Measure each grain's physical dimensions in millimeters.

    Args:
        grain_contours: list of grain contours from segmentation.
        pixels_per_mm: calibration factor from coin detection.

    Returns:
        List of dicts, each containing:
            length_mm, width_mm, lw_ratio, area_px
    """
    measurements = []

    for contour in grain_contours:
        if len(contour) < 5:
            continue

        rect = cv2.minAreaRect(contour)
        (_, (w_px, h_px), _) = rect

        # length = longer side, width = shorter side
        length_px = max(w_px, h_px)
        width_px = min(w_px, h_px)

        length_mm = length_px / pixels_per_mm
        width_mm = width_px / pixels_per_mm

        lw_ratio = length_mm / width_mm if width_mm > 0 else 0.0

        measurements.append(
            {
                "length_mm": round(length_mm, 3),
                "width_mm": round(width_mm, 3),
                "lw_ratio": round(lw_ratio, 3),
                "area_px": cv2.contourArea(contour),
            }
        )

    return measurements


def classify_broken(measurements: list[dict]) -> tuple[float, float]:
    """
    Classify grains as broken based on median length.

    A grain is "broken" if its length is less than
    BROKEN_THRESHOLD_PCT (80%) of the median grain length.

    Args:
        measurements: list of grain measurement dicts from measure_grains().

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
