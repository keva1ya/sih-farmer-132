"""
Step 1 — Blur Check

Uses Laplacian variance to determine if an image is too blurry for
reliable grain measurement. This is the first gate in the pipeline:
if the image is blurry, we stop immediately and ask for a retake.
"""

import cv2
import numpy as np

from cv.config import BLUR_THRESHOLD


def check_blur(image: np.ndarray) -> tuple[bool, float]:
    """
    Check if the image is too blurry for analysis.

    Args:
        image: BGR image (numpy array).

    Returns:
        (is_acceptable, variance):
            is_acceptable — True if the image is sharp enough.
            variance — the Laplacian variance score (higher = sharper).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= BLUR_THRESHOLD, float(variance)
