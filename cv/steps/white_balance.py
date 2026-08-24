"""
Step 4 — White Balance Calibration

Uses the white sheet visible in the background as a "true white"
reference point. Samples background pixels from an annular region
around the coin (the cleanest, flattest part of the sheet) and
scales all channels so that the background becomes [255, 255, 255].

This neutralizes color tints from warm bulbs, fluorescent lights,
or shade, ensuring the HSV/LAB defect detection downstream sees
consistent, accurate grain colors.
"""

import cv2
import numpy as np


def calibrate_white_balance(
    image: np.ndarray,
    coin_center: tuple[int, int],
    coin_radius: int,
    sample_ring_width: int = 40,
) -> np.ndarray:
    """
    Apply white balance using the white sheet background.

    Samples an annular ring just outside the coin (where the sheet
    should be cleanest) and scales the image so that region becomes
    pure white.

    Args:
        image: BGR image (numpy array).
        coin_center: (x, y) center of the detected coin.
        coin_radius: pixel radius of the outer coin ring.
        sample_ring_width: width of the sampling annulus in pixels.

    Returns:
        White-balanced BGR image.
    """
    h, w = image.shape[:2]
    cx, cy = int(coin_center[0]), int(coin_center[1])

    # Create an annular mask: ring from (coin_radius + 20) to
    # (coin_radius + 20 + sample_ring_width)
    inner_r = coin_radius + 20  # Small gap to avoid coin edge artifacts
    outer_r = inner_r + sample_ring_width

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), outer_r, 255, thickness=-1)
    cv2.circle(mask, (cx, cy), inner_r, 0, thickness=-1)

    # Extract the background pixels from the annular region
    bg_pixels = image[mask == 255]

    if len(bg_pixels) < 100:
        # Not enough background pixels — fall back to no correction
        return image.copy()

    # Compute the mean BGR of the background
    mean_bgr = bg_pixels.mean(axis=0)

    # Avoid division by zero
    mean_bgr = np.maximum(mean_bgr, 1.0)

    # Scale factors to make the background white (255, 255, 255)
    scale = 255.0 / mean_bgr

    # Cap the scale to avoid extreme amplification in very dark images
    scale = np.minimum(scale, 3.0)

    # Apply the scaling
    corrected = image.astype(np.float64) * scale
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    return corrected
