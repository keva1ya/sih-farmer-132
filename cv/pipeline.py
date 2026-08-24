"""
KrishiDrishti CV Pipeline — Orchestrator

Calls each pipeline step in sequence and handles early returns
for error conditions (blur, tilt, missing coin, too few grains).

This is the single function the FastAPI endpoint calls.
"""

import hashlib
import json

import cv2
import numpy as np

from cv.config import IMAGE_RESIZE_WIDTH, MIN_GRAIN_COUNT
from cv.steps.blur_check import check_blur
from cv.steps.coin_calibration import detect_and_calibrate, mask_coin
from cv.steps.white_balance import calibrate_white_balance
from cv.steps.segmentation import segment_grains
from cv.steps.measurement import measure_grains, classify_broken
from cv.steps.defect_detection import detect_discoloration, detect_foreign_matter
from cv.steps.grading import compute_grade


def _resize_image(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resize image to target width while maintaining aspect ratio."""
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)


def _build_result_hash(result: dict) -> str:
    """Generate SHA-256 hash of the result JSON (excluding the hash field itself)."""
    # Canonical JSON for deterministic hashing
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_pipeline(
    image_bytes: bytes,
    detailed: bool = False,
    commodity: str = "wheat",
) -> dict:
    """
    Execute the full CV grading pipeline.

    Args:
        image_bytes: raw image bytes (JPEG/PNG).
        detailed: if True, include per_metric_grades in the response.
        commodity: commodity name for AGMARK threshold lookup.

    Returns:
        A dict matching the AnalyzeSuccessResponse or AnalyzeErrorResponse schema.
    """
    # --- Step 0: Decode & Resize ---
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if image is None:
        return {"status": "invalid_format"}

    image = _resize_image(image, IMAGE_RESIZE_WIDTH)

    # --- Step 1: Blur Check ---
    is_sharp, _variance = check_blur(image)
    if not is_sharp:
        return {"status": "too_blurry"}

    # --- Step 2: Coin Detection & Calibration ---
    error, pixels_per_mm, coin_center, coin_radius = detect_and_calibrate(image)
    if error is not None:
        return {"status": error}

    # --- Step 3: Mask Coin ---
    image_no_coin = mask_coin(image, coin_center, coin_radius)

    # --- Step 4: White Balance ---
    image_wb = calibrate_white_balance(image_no_coin, coin_center, coin_radius)

    # --- Step 5: Grain Segmentation ---
    grain_contours, chaff_contours = segment_grains(image_wb)

    # --- Step 6: Minimum Grain Count ---
    if len(grain_contours) < MIN_GRAIN_COUNT:
        return {"status": "too_few_grains"}

    # --- Step 10: Foreign Matter Detection (before measurement, to reclassify) ---
    foreign_pct, grain_contours = detect_foreign_matter(
        image_wb, grain_contours, chaff_contours
    )

    # Re-check grain count after foreign matter reclassification
    if len(grain_contours) < MIN_GRAIN_COUNT:
        return {"status": "too_few_grains"}

    # --- Step 7: Grain Measurement ---
    measurements = measure_grains(grain_contours, pixels_per_mm)

    if not measurements:
        return {"status": "too_few_grains"}

    # --- Step 8: Broken Grain Classification ---
    broken_pct, median_length = classify_broken(measurements)

    # --- Step 9: Defect Detection ---
    discolored_pct = detect_discoloration(image_wb, grain_contours)

    # --- Aggregate Stats ---
    lengths = [m["length_mm"] for m in measurements]
    widths = [m["width_mm"] for m in measurements]
    ratios = [m["lw_ratio"] for m in measurements]

    avg_length = round(float(np.mean(lengths)), 2)
    avg_width = round(float(np.mean(widths)), 2)
    avg_ratio = round(float(np.mean(ratios)), 2)

    # --- Step 11: AGMARK Grading ---
    overall_grade, per_metric_grades = compute_grade(
        avg_length_mm=avg_length,
        avg_lw_ratio=avg_ratio,
        broken_pct=broken_pct,
        discolored_pct=discolored_pct,
        foreign_matter_pct=foreign_pct,
        commodity=commodity,
    )

    # --- Step 12: Build Result & Hash ---
    result = {
        "status": "success",
        "grain_count": len(measurements),
        "measurements": {
            "avg_length_mm": avg_length,
            "avg_width_mm": avg_width,
            "avg_lw_ratio": avg_ratio,
            "median_length_mm": median_length,
            "broken_pct": broken_pct,
            "discolored_pct": discolored_pct,
            "foreign_matter_pct": foreign_pct,
        },
        "grade": overall_grade,
    }

    # Hash is computed BEFORE adding it to the result (and before adding optional fields)
    result["hash"] = _build_result_hash(result)

    # Optional detailed breakdown
    if detailed:
        result["per_metric_grades"] = per_metric_grades

    return result
