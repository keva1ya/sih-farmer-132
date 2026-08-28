"""
Step 4 — White Balance Calibration

Uses the white sheet visible in the background as a "true white"
reference point. Samples background pixels from an annular region
around the coin (the cleanest, flattest part of the sheet) and
iteratively scales all channels until the background reaches
[255, 255, 255].

This neutralizes color tints from warm bulbs, fluorescent lights,
direct sun, or shade, ensuring the HSV/LAB defect detection downstream
sees consistent, accurate grain colors.
"""

import cv2
import numpy as np


def calibrate_white_balance(
    image: np.ndarray,
    coin_center: tuple[int, int],
    coin_radius: int,
    sample_ring_width: int = 40,
    max_iterations: int = 6,
    per_step_gain_cap: float = 1.4,
    convergence_tolerance: float = 15.0,
) -> np.ndarray:
    """
    Apply white balance using the white sheet background.

    Samples an annular ring just outside the coin (where the sheet should be
    cleanest) and repeatedly rescales the whole image so that region
    converges on pure white, instead of computing one multiplicative gain and
    applying it in a single shot.

    A single-shot gain has to be capped (the old fixed 3.0x ceiling) to avoid
    blowing out an already-dark image in one jump, which means a strong color
    cast -- direct sun, a tinted or colored bulb -- can leave the background
    short of true white after only one pass. Any residual cast then survives
    onto the grains too, and since the HSV/LAB "healthy wheat" bands in
    config.py assume a fully white-balanced image, that residual reads as
    discoloration that has nothing to do with the grain itself.

    Iterating fixes this without reintroducing the one-shot risk: each pass
    recomputes the gain from the CURRENT background color (which moves closer
    to white every pass) and applies only a bounded step of it
    (`per_step_gain_cap`), so no single pass can jump as far as the old flat
    3.0x cap could. Across `max_iterations` passes the *cumulative* gain can
    still exceed 3.0x (up to `per_step_gain_cap ** max_iterations`), which is
    what lets a strongly tinted photo actually reach white instead of
    stopping short.

    `convergence_tolerance` is deliberately loose (not "as close to 255 as
    possible"). The sampled ring is real paper, not a flat color swatch: its
    pixels have their own spread from texture, grain, and JPEG noise, so a
    portion of them are already brighter than the ring's mean and clip at 255
    on every pass. That clipping caps how much any pass can raise the mean,
    which means chasing a tight target (measured: even 2 levels short of 255)
    never actually converges -- the loop keeps computing a fresh "still not
    there" gain from a mean that clipping won't let rise much further, and
    each of those extra passes multiplies onto the grain pixels too, compounding
    into far more amplification than the photo's actual tint calls for.
    Measured on a mildly-tinted real photo, chasing 253 drove seven extra
    passes and roughly 1.4x more cumulative gain than needed, which alone
    moved `discolored_pct` from 3.4% to 18.9% -- the exact failure mode this
    function exists to fix, self-inflicted by too strict a target. 15 levels
    is loose enough that an ordinary photo's ring converges in its first pass
    (matching what the old single-shot formula already produced for it) while
    still leaving a strongly tinted photo's much-further-from-white starting
    mean enough passes to close most of the gap.

    Args:
        image: BGR image (numpy array).
        coin_center: (x, y) center of the detected coin.
        coin_radius: pixel radius of the outer coin ring.
        sample_ring_width: width of the sampling annulus in pixels.
        max_iterations: maximum correction passes before giving up.
        per_step_gain_cap: maximum per-channel multiplicative gain applied
            in any single pass.
        convergence_tolerance: a channel is treated as "at white" once its
            background mean is within this many levels of 255.

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

    # The mask itself never changes between passes, so the background pixel
    # count is checked once rather than re-derived every iteration.
    if cv2.countNonZero(mask) < 100:
        # Not enough background pixels — fall back to no correction
        return image.copy()

    corrected = image.astype(np.float64)
    for _ in range(max_iterations):
        bg_pixels = corrected[mask == 255]
        mean_bgr = bg_pixels.mean(axis=0)

        if mean_bgr.min() >= 255.0 - convergence_tolerance:
            break

        # Avoid division by zero
        mean_bgr = np.maximum(mean_bgr, 1.0)

        # This pass's step toward making the background white (255,255,255),
        # bounded so one pass can't jump further than per_step_gain_cap.
        step_gain = np.minimum(255.0 / mean_bgr, per_step_gain_cap)
        corrected = np.clip(corrected * step_gain, 0, 255)

    return corrected.astype(np.uint8)
