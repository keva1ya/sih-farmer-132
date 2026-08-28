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
    DISCOLORATION_EDGE_EROSION_PX,
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

_EROSION_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _create_contour_mask(
    contour: np.ndarray, img_shape: tuple[int, int]
) -> np.ndarray:
    """Create a full-frame binary mask for a single contour.

    Retained for callers that genuinely need a full-frame mask. The hot loops
    in this module use :func:`_contour_roi` instead, which is equivalent but
    allocates only the contour's bounding rect.
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    return mask


def _contour_roi(
    contour: np.ndarray, img_shape: tuple[int, int]
) -> tuple[int, int, int, int, np.ndarray] | None:
    """Rasterize one contour into a mask the size of its bounding rect.

    Returns ``(x0, y0, x1, y1, mask)`` where ``mask`` has shape
    ``(y1 - y0, x1 - x0)`` and holds exactly the pixels that
    :func:`_create_contour_mask` would set inside that window, or ``None``
    when the window is empty.

    Equivalence to the full-frame mask: a filled ``LINE_8`` polygon never
    paints outside the extent of its own points, so the bounding rect
    contains every set pixel. The rect is then intersected with the image, so
    wherever the ROI edge is not the image edge no clipping can occur (the
    contour does not reach there), and wherever it is the image edge the
    clipping is the same clipping the full-frame draw would apply.
    """
    img_h, img_w = img_shape[:2]
    x, y, w, h = cv2.boundingRect(contour)

    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, img_w)
    y1 = min(y + h, img_h)

    if x1 <= x0 or y1 <= y0:
        return None

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1, offset=(-x0, -y0))
    return x0, y0, x1, y1, mask


def _interval_to_uint8_bounds(lo: float, hi: float) -> tuple[int, int] | None:
    """Map a centered-at-128 LAB interval onto inclusive uint8 bounds.

    Reproduces ``(float(v) - 128.0 >= lo) and (float(v) - 128.0 <= hi)``
    exactly for every possible uint8 value ``v``, so the resulting
    ``cv2.inRange`` bounds select precisely the pixels the original
    per-pixel float comparison accepted. Returns ``None`` if no uint8 value
    satisfies the interval.
    """
    centered = np.arange(256, dtype=np.float64) - 128.0
    accepted = np.flatnonzero((centered >= lo) & (centered <= hi))
    if accepted.size == 0:
        return None
    return int(accepted[0]), int(accepted[-1])


def detect_discoloration(
    image_wb: np.ndarray,
    grain_contours: list[np.ndarray],
    return_mask: bool = False,
) -> float | tuple[float, np.ndarray]:
    """
    Detect discolored / damaged grain pixels using dual HSV + LAB pass.

    Args:
        image_wb: White-balanced BGR image.
        grain_contours: list of grain contours.
        return_mask: if True, also return a full-frame binary mask of every
            pixel flagged defective by either pass (for debug visualization
            only). Does not change `discolored_pct`, which stays the
            max(hsv_defect_count, lab_defect_count) approximation below —
            the true per-pixel union can exceed that max, so the mask can
            show marginally more red than the percentage alone implies.

    Returns:
        discolored_pct — percentage of total grain pixel area that is
        defective. If `return_mask` is True, returns
        `(discolored_pct, defect_mask)` instead.
    """
    if not grain_contours:
        empty = np.zeros(image_wb.shape[:2], dtype=np.uint8)
        return (0.0, empty) if return_mask else 0.0

    hsv = cv2.cvtColor(image_wb, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_wb, cv2.COLOR_BGR2LAB)

    total_grain_pixels = 0
    total_defect_pixels = 0
    defect_mask = np.zeros(image_wb.shape[:2], dtype=np.uint8) if return_mask else None

    # HSV healthy bounds
    hsv_lower = np.array([HSV_HEALTHY_H_MIN, HSV_HEALTHY_S_MIN, HSV_HEALTHY_V_MIN])
    hsv_upper = np.array([HSV_HEALTHY_H_MAX, HSV_HEALTHY_S_MAX, HSV_HEALTHY_V_MAX])

    # --- Full-frame healthy masks, computed ONCE and then indexed per contour.
    # Both passes are strictly per-pixel, so evaluating them over the whole
    # frame and slicing is identical to evaluating them per contour.
    hsv_healthy_full = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # LAB values: L is 0-255, a and b are 0-255 (centered at 128)
    a_bounds = _interval_to_uint8_bounds(LAB_HEALTHY_A_MIN, LAB_HEALTHY_A_MAX)
    b_bounds = _interval_to_uint8_bounds(LAB_HEALTHY_B_MIN, LAB_HEALTHY_B_MAX)
    if a_bounds is None or b_bounds is None:
        lab_healthy_full = np.zeros(lab.shape[:2], dtype=np.uint8)
    else:
        lab_healthy_full = cv2.inRange(
            lab,
            np.array([0, a_bounds[0], b_bounds[0]]),
            np.array([255, a_bounds[1], b_bounds[1]]),
        )

    for contour in grain_contours:
        roi = _contour_roi(contour, image_wb.shape)
        if roi is None:
            continue

        x0, y0, x1, y1, mask = roi

        # Excludes the outer rim (anti-aliased edge blending, specular rim
        # highlights) from measurement -- see DISCOLORATION_EDGE_EROSION_PX
        # in config.py for why. Falls back to the un-eroded mask if erosion
        # empties it, so a grain too small to have any interior left still
        # gets measured on its full area rather than being silently skipped.
        eroded = cv2.erode(mask, _EROSION_KERNEL, iterations=DISCOLORATION_EDGE_EROSION_PX)
        if cv2.countNonZero(eroded) > 0:
            mask = eroded

        grain_pixel_count = cv2.countNonZero(mask)

        if grain_pixel_count == 0:
            continue

        total_grain_pixels += grain_pixel_count

        # --- HSV pass ---
        hsv_healthy = cv2.bitwise_and(hsv_healthy_full[y0:y1, x0:x1], mask)
        hsv_defect_count = grain_pixel_count - cv2.countNonZero(hsv_healthy)

        # --- LAB pass ---
        lab_healthy = cv2.bitwise_and(lab_healthy_full[y0:y1, x0:x1], mask)
        lab_defect_count = grain_pixel_count - cv2.countNonZero(lab_healthy)

        # --- Union: defective if flagged by EITHER pass ---
        # For simplicity, take the max of both defect counts
        # (a pixel flagged by both is still one defect pixel)
        defect_count = max(hsv_defect_count, lab_defect_count)
        total_defect_pixels += defect_count

        if defect_mask is not None:
            # True per-pixel union (not the max approximation above), used
            # only to draw the debug overlay. NOT the full-frame slice (not
            # the already-masked `hsv_healthy`/`lab_healthy`) then re-AND
            # with `mask`, so pixels outside the grain stay excluded.
            hsv_defect_px = cv2.bitwise_and(
                cv2.bitwise_not(hsv_healthy_full[y0:y1, x0:x1]), mask
            )
            lab_defect_px = cv2.bitwise_and(
                cv2.bitwise_not(lab_healthy_full[y0:y1, x0:x1]), mask
            )
            defect_px = cv2.bitwise_or(hsv_defect_px, lab_defect_px)
            defect_mask[y0:y1, x0:x1] = cv2.bitwise_or(defect_mask[y0:y1, x0:x1], defect_px)

    if total_grain_pixels == 0:
        empty = np.zeros(image_wb.shape[:2], dtype=np.uint8)
        return (0.0, empty) if return_mask else 0.0

    discolored_pct = round((total_defect_pixels / total_grain_pixels) * 100.0, 2)
    return (discolored_pct, defect_mask) if return_mask else discolored_pct
