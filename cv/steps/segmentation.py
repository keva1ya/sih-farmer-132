"""
Step 5 — Grain Segmentation

Isolates grain contours from the white-sheet background using
Otsu thresholding + morphological opening. Splits detected contours
into two categories:
  - grain_contours: large enough to be actual grains
  - chaff_contours: too small → classified as foreign matter / chaff

Also filters out shadows (via solidity check) and partial grains
touching the image border.

[FUTURE] This module is designed to be swappable. To upgrade to
Watershed or U-Net segmentation, replace the `segment_grains`
function while keeping the same return signature.
"""

import cv2
import numpy as np

from cv.config import (
    MORPH_KERNEL_SIZE,
    SOLIDITY_THRESHOLD,
    MIN_GRAIN_AREA_PX,
)


def _is_touching_border(contour: np.ndarray, img_shape: tuple[int, int]) -> bool:
    """Check if any point of the contour touches the image border."""
    h, w = img_shape[:2]
    for point in contour:
        x, y = point[0]
        if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
            return True
    return False


def _compute_solidity(contour: np.ndarray) -> float:
    """Compute solidity = contour area / convex hull area."""
    area = cv2.contourArea(contour)
    if area == 0:
        return 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return area / hull_area


def segment_grains(
    image: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Segment the image into grain contours and chaff contours.

    Args:
        image: BGR image with the coin already masked out (zeroed).

    Returns:
        (grain_contours, chaff_contours):
            grain_contours — list of contours large enough to be grains,
                             with shadows and border-touching contours removed.
            chaff_contours — list of small contours classified as
                             foreign matter / chaff.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding (works well with bimodal histogram: white bg + dark grains)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening to break thin connections between grains
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find all contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    grain_contours: list[np.ndarray] = []
    chaff_contours: list[np.ndarray] = []

    img_shape = image.shape

    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip negligible noise (< 10 pixels)
        if area < 10:
            continue

        # Border check — discard partial grains at image edges
        if _is_touching_border(contour, img_shape):
            continue

        # Solidity check — discard shadows and thin artifacts
        solidity = _compute_solidity(contour)
        if solidity < SOLIDITY_THRESHOLD:
            continue

        # Size classification
        if area >= MIN_GRAIN_AREA_PX:
            grain_contours.append(contour)
        else:
            chaff_contours.append(contour)

    return grain_contours, chaff_contours
