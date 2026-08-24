"""
Steps 9 & 10 — Defect Detection & Foreign Matter

Two-pass defect detection:
  1. HSV pass — catches obvious discoloration (dark, green, white spots)
  2. LAB pass — catches subtle fungal/moisture damage in the a/b channels

A pixel is "defective" if flagged by EITHER pass (union).

Foreign matter detection:
  - Small contours from segmentation are already classified as chaff
  - Additionally, any grain-sized contour whose mean HSV is entirely
    outside the wheat hue range is reclassified as a foreign object
    (e.g., gray stones, green stems)
"""

import cv2
import numpy as np

from cv.config import (
    HSV_HEALTHY_H_MIN,
    HSV_HEALTHY_H_MAX,
    HSV_HEALTHY_S_MIN,
    HSV_HEALTHY_S_MAX,
    HSV_HEALTHY_V_MIN,
    HSV_HEALTHY_V_MAX,
    LAB_HEALTHY_A_MIN,
    LAB_HEALTHY_A_MAX,
    LAB_HEALTHY_B_MIN,
    LAB_HEALTHY_B_MAX,
)


def _create_contour_mask(
    contour: np.ndarray, img_shape: tuple[int, int]
) -> np.ndarray:
    """Create a binary mask for a single contour."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    return mask


def detect_discoloration(
    image_wb: np.ndarray,
    grain_contours: list[np.ndarray],
) -> float:
    """
    Detect discolored / damaged grain pixels using dual HSV + LAB pass.

    Args:
        image_wb: White-balanced BGR image.
        grain_contours: list of grain contours.

    Returns:
        discolored_pct — percentage of total grain pixel area that is defective.
    """
    if not grain_contours:
        return 0.0

    hsv = cv2.cvtColor(image_wb, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_wb, cv2.COLOR_BGR2LAB)

    total_grain_pixels = 0
    total_defect_pixels = 0

    # HSV healthy bounds
    hsv_lower = np.array([HSV_HEALTHY_H_MIN, HSV_HEALTHY_S_MIN, HSV_HEALTHY_V_MIN])
    hsv_upper = np.array([HSV_HEALTHY_H_MAX, HSV_HEALTHY_S_MAX, HSV_HEALTHY_V_MAX])

    for contour in grain_contours:
        mask = _create_contour_mask(contour, image_wb.shape)
        grain_pixel_count = int(np.sum(mask > 0))

        if grain_pixel_count == 0:
            continue

        total_grain_pixels += grain_pixel_count

        # --- HSV pass ---
        hsv_in_range = cv2.inRange(hsv, hsv_lower, hsv_upper)
        hsv_healthy = cv2.bitwise_and(hsv_in_range, mask)
        hsv_defect_count = grain_pixel_count - int(np.sum(hsv_healthy > 0))

        # --- LAB pass ---
        # LAB values: L is 0-255, a and b are 0-255 (centered at 128)
        lab_pixels = lab[mask > 0]
        a_channel = lab_pixels[:, 1].astype(float) - 128.0  # Center at 0
        b_channel = lab_pixels[:, 2].astype(float) - 128.0

        lab_healthy_mask = (
            (a_channel >= LAB_HEALTHY_A_MIN)
            & (a_channel <= LAB_HEALTHY_A_MAX)
            & (b_channel >= LAB_HEALTHY_B_MIN)
            & (b_channel <= LAB_HEALTHY_B_MAX)
        )
        lab_defect_count = int(np.sum(~lab_healthy_mask))

        # --- Union: defective if flagged by EITHER pass ---
        # For simplicity, take the max of both defect counts
        # (a pixel flagged by both is still one defect pixel)
        defect_count = max(hsv_defect_count, lab_defect_count)
        total_defect_pixels += defect_count

    if total_grain_pixels == 0:
        return 0.0

    discolored_pct = (total_defect_pixels / total_grain_pixels) * 100.0
    return round(discolored_pct, 2)


def detect_foreign_matter(
    image_wb: np.ndarray,
    grain_contours: list[np.ndarray],
    chaff_contours: list[np.ndarray],
) -> tuple[float, list[np.ndarray]]:
    """
    Detect foreign matter among all detected contours.

    Foreign matter comes from two sources:
      1. chaff_contours — small contours already classified by segmentation
      2. grain-sized contours whose mean HSV hue is entirely outside
         the wheat range (e.g., gray stones, green leaves)

    Args:
        image_wb: White-balanced BGR image.
        grain_contours: list of grain-sized contours.
        chaff_contours: list of small contours (chaff).

    Returns:
        (foreign_pct, updated_grain_contours):
            foreign_pct — percentage of foreign matter by area.
            updated_grain_contours — grain contours with foreign objects removed.
    """
    hsv = cv2.cvtColor(image_wb, cv2.COLOR_BGR2HSV)

    foreign_area = 0.0
    total_area = 0.0

    # All chaff contours are foreign matter by definition
    for contour in chaff_contours:
        foreign_area += cv2.contourArea(contour)
        total_area += cv2.contourArea(contour)

    # Check grain-sized contours: reclassify if hue is outside wheat range
    valid_grains: list[np.ndarray] = []

    for contour in grain_contours:
        area = cv2.contourArea(contour)
        total_area += area

        mask = _create_contour_mask(contour, image_wb.shape)
        hsv_pixels = hsv[mask > 0]

        if len(hsv_pixels) == 0:
            valid_grains.append(contour)
            continue

        mean_hue = hsv_pixels[:, 0].mean()

        # If the mean hue is entirely outside wheat hue range → foreign object
        if mean_hue < HSV_HEALTHY_H_MIN - 10 or mean_hue > HSV_HEALTHY_H_MAX + 10:
            foreign_area += area  # Reclassify as foreign
        else:
            valid_grains.append(contour)

    if total_area == 0:
        return 0.0, grain_contours

    foreign_pct = (foreign_area / total_area) * 100.0
    return round(foreign_pct, 2), valid_grains
