"""
KrishiDrishti CV Pipeline — Orchestrator

Decodes the uploaded bytes, runs the sharpness gate, hands the decoded frame to
the wheat measurement module (which owns paper detection, coin calibration,
illumination flat-fielding and grain segmentation), white-balances the accepted
crop against the paper ring around the coin, then runs the defect and AGMARK
grading steps over the contours that module accepted.

This is the single function the FastAPI endpoint calls.
"""

import hashlib
import json

import cv2
import numpy as np

from cv.config import BLUR_THRESHOLD, IMAGE_RESIZE_WIDTH
from cv.steps.blur_check import check_blur
from cv.steps.measurement import classify_broken
from cv.steps.defect_detection import detect_discoloration
from cv.steps.grading import compute_grade
from cv.steps.white_balance import calibrate_white_balance
from cv.steps.wheat_measurement import (
    CoinNotDetectedError,
    CoinTiltedError,
    ImageLoadError,
    PaperNotDetectedError,
    TooFewGrainsError,
    _measure_wheat_grains_from_image,
)


# The measurement module raises typed exceptions and never swallows them. The
# mapping to the /analyze error contract lives here, in the adapter, so the
# module stays free of any knowledge of the HTTP layer. There is no
# "too_many_grains" status: the module has no upper bound on grain count.
_MEASUREMENT_ERROR_STATUS: tuple[tuple[type[Exception], str], ...] = (
    (ImageLoadError, "invalid_format"),
    (PaperNotDetectedError, "paper_not_detected"),
    (CoinNotDetectedError, "coin_not_found"),
    (CoinTiltedError, "too_tilted"),
    (TooFewGrainsError, "too_few_grains"),
)


def _failure(status: str, reason: str) -> dict[str, str]:
    """Build a deterministic, user-facing pipeline failure response."""
    return {"status": status, "reason": reason}


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


def _measurement_failure(error: Exception) -> dict[str, str] | None:
    """Map a measurement-module exception onto the pipeline error contract."""
    for exception_type, status in _MEASUREMENT_ERROR_STATUS:
        if isinstance(error, exception_type):
            # The module's message is already user-facing and names the actual
            # numbers (axis ratio, grain breakdown), so it is carried through
            # verbatim. `reason` is never empty: every one of these exceptions is
            # raised with a message, and the frontend's formatError() prefers
            # `reason` over its own status->message table, which has no entry
            # for paper_not_detected.
            return _failure(status, str(error))
    return None


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
    # --- Step 0: Decode ---
    if not image_bytes:
        return _failure("invalid_format", "The uploaded image is empty.")

    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if image is None:
        return _failure("invalid_format", "The uploaded file is not a readable JPEG or PNG image.")

    # Downscale only, never upscale. The measurement module deliberately works at
    # native resolution and scales every kernel as a fraction of the shortest
    # side, so it is resolution independent, and its mm scale comes from the coin,
    # which makes a uniform resize scale-neutral either way. What is left is
    # detail loss vs. runtime: upscaling a small photo to 1920 only interpolates
    # blur into the grain edges the module fits ellipses to and costs runtime for
    # nothing, so the old unconditional resize is gone. The cap on oversized
    # phone uploads stays, because segmentation cost grows with pixel count and a
    # 4000px frame is several times the module's measured budget. Images already
    # at or below IMAGE_RESIZE_WIDTH now reach the module untouched.
    if image.shape[1] > IMAGE_RESIZE_WIDTH:
        image = _resize_image(image, IMAGE_RESIZE_WIDTH)

    # `_crop_to_sheet` is deliberately gone: the measurement module runs its own
    # HSV paper detection and crops to the filled paper contour, and the legacy
    # brightness/aspect-ratio bounding box could clip the coin or part of the
    # sheet before the module ever saw the frame.

    # --- Step 1: Blur Check ---
    # Kept. Flat-fielding fixes illumination, not focus: a defocused photo still
    # yields soft contours whose fitted ellipses read wrong in mm, and this is the
    # only source of the `too_blurry` status the frontend handles.
    is_sharp, variance = check_blur(image)
    if not is_sharp:
        return _failure(
            "too_blurry",
            f"Laplacian variance {variance:.2f} is below the required threshold of {BLUR_THRESHOLD:.2f}.",
        )

    # --- Steps 2-7: Paper crop, coin calibration, segmentation, measurement ---
    # The decoded array goes straight in; no temp-file round trip, and none of the
    # module's segmentation or calibration rules are duplicated here.
    try:
        measurement = _measure_wheat_grains_from_image(image)
    except Exception as error:  # noqa: BLE001 - mapped to a JSON response below
        failure = _measurement_failure(error)
        if failure is not None:
            return failure
        # An exception type the measurement module doesn't document (an
        # unanticipated cv2/numpy edge case deep in coin/paper/segmentation
        # logic) must still return the deterministic JSON contract every other
        # failure path uses. main.py registers no exception handler, so
        # re-raising here would otherwise escape /analyze as a raw HTTP 500.
        return _failure(
            "processing_error",
            "The image could not be processed. Please retake the photo with "
            "the grains spread out, the coin fully visible, and good lighting.",
        )

    # Internal hand-offs: the accepted contours and the exact BGR pixels they
    # index into, so the defect passes run on what the module actually accepted.
    grain_contours = measurement["_grain_contours"]
    cropped_image = measurement["_cropped_image"]
    grains = measurement["grains"]
    summary = measurement["summary"]

    # --- White balance (for the colour passes only) ---
    # The module's flat-fielding normalises *brightness* so segmentation survives
    # uneven light; it does not normalise the illuminant's *colour*. The HSV/LAB
    # bands in config.py were tuned assuming a white-balance pass, so without one
    # discolored_pct measures the bulb rather than the grain: with the contours
    # held fixed, an ordinary warm cast moved it by tens of points and the
    # observed range across normal lighting spanned essentially 0-100 %.
    #
    # This is safe to run here, and only here, because it happens strictly AFTER
    # the module has finished: paper detection, coin calibration and segmentation
    # have already run on the untouched pixels, so none of the geometry the
    # module was tuned against is disturbed. `_coin_center` / `_coin_radius` are
    # in the same cropped-image frame as `_cropped_image` and `_grain_contours`,
    # which is exactly the frame `calibrate_white_balance` samples its paper
    # annulus in. If that annulus lands outside the frame or on too few pixels
    # the step returns the image unchanged, so it degrades to today's behaviour
    # rather than failing.
    balanced_image = calibrate_white_balance(
        cropped_image,
        measurement["_coin_center"],
        measurement["_coin_radius"],
    )

    # --- Step 8: Broken Grain Classification ---
    # `classify_broken` expects the legacy per-grain dict shape, so the module's
    # grains are adapted to it rather than the step being changed.
    broken_inputs = [
        {
            "length_mm": float(grain["length_mm"]),
            "width_mm": float(grain["width_mm"]),
            "lw_ratio": float(grain["aspect_ratio"]),
        }
        for grain in grains
    ]
    broken_pct, median_length = classify_broken(broken_inputs)

    # --- Step 9: Discoloration ---
    # Runs on the white-balanced pixels, which is what `detect_discoloration`'s
    # `image_wb` parameter has always meant and what the config bands assume.
    discolored_pct = detect_discoloration(balanced_image, grain_contours)

    # --- Step 10: Foreign Matter ---
    # Taken from the module, not from `detect_foreign_matter`. The module already
    # classifies foreign matter (the 2-8 mm2 band plus grain-sized off-hue blobs)
    # and accumulates its area, and the legacy call would re-scan a contour set it
    # no longer owns and double-count what the module already removed.
    #
    # The denominator is the sample, not the sheet. Paper area is mostly empty
    # white space, so dividing by it reads ~34x lower than the legacy figure and
    # would push every sample under the 0.5% grade-A threshold in config.py,
    # silently grading contaminated lots as A.
    #
    # Touching-cluster area is included alongside individually measured grains:
    # those blobs are real wheat that the module couldn't split into individual
    # kernels, not artifacts, so excluding them from the denominator understates
    # the true sample area and inflates foreign_pct on an ordinary photo where
    # grains happen to be poured close together. Ring/low-solidity/border
    # exclusions are deliberately NOT added here -- those are segmentation
    # artifacts (shadows, clipped edges) with no physical grain behind them.
    foreign_area_mm2 = float(measurement["_foreign_matter_area_mm2"])
    grain_area_mm2 = sum(float(grain["area_mm2"]) for grain in grains)
    touching_cluster_area_mm2 = float(measurement["_touching_cluster_area_mm2"])
    sample_area_mm2 = foreign_area_mm2 + grain_area_mm2 + touching_cluster_area_mm2
    foreign_pct = (
        round((foreign_area_mm2 / sample_area_mm2) * 100.0, 2) if sample_area_mm2 > 0 else 0.0
    )

    # --- Aggregate Stats ---
    # Rounded to 2 decimals exactly as before: these values are hashed and the
    # frontend prints them unformatted, so the rounding is part of the contract.
    avg_length = round(float(summary["length_mm"]["mean"]), 2)
    avg_width = round(float(summary["width_mm"]["mean"]), 2)
    avg_ratio = round(float(np.mean([float(g["aspect_ratio"]) for g in grains])), 2)
    avg_area = round(float(np.mean([float(g["area_mm2"]) for g in grains])), 2)

    # --- Step 11: AGMARK Grading ---
    overall_grade, per_metric_grades = compute_grade(
        avg_length_mm=avg_length,
        avg_width_mm=avg_width,
        avg_area_mm2=avg_area,
        avg_lw_ratio=avg_ratio,
        broken_pct=broken_pct,
        discolored_pct=discolored_pct,
        foreign_matter_pct=foreign_pct,
        commodity=commodity,
    )

    # --- Step 12: Build Result & Hash ---
    result = {
        "status": "success",
        "grain_count": int(summary["count"]),
        "measurements": {
            "avg_length_mm": avg_length,
            "avg_width_mm": avg_width,
            "avg_area_mm2": avg_area,
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
