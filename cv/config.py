"""
KrishiDrishti CV Module — Central Configuration

ALL tunable constants live here. To adjust thresholds, HSV ranges,
or add a new commodity, edit ONLY this file.
"""

# ---------------------------------------------------------------------------
# Image Pre-processing
# ---------------------------------------------------------------------------
IMAGE_RESIZE_WIDTH: int = 1920  # Normalize all inputs to this width (px)
MAX_FILE_SIZE_MB: int = 10
ALLOWED_EXTENSIONS: set[str] = {"image/jpeg", "image/png"}

# ---------------------------------------------------------------------------
# Blur Detection (Step 1)
# ---------------------------------------------------------------------------
BLUR_THRESHOLD: float = 16.0  # Laplacian variance; below = too blurry

# ---------------------------------------------------------------------------
# Coin Calibration & Segmentation (Steps 2 and 5)
# ---------------------------------------------------------------------------
# These stages now live inside `cv.steps.wheat_measurement`, which owns its own
# constants (COIN_DIAMETER_MM, COIN_TILT_MIN_AXIS_RATIO, GRAIN_MIN_SOLIDITY,
# MIN_VALID_GRAIN_COUNT, ...) at module scope. The old config entries here
# (COIN_DIAMETER_MM, COIN_INNER_OUTER_RATIO, COIN_RATIO_TOLERANCE,
# TILT_THRESHOLD, the six HOUGH_* values, MORPH_KERNEL_SIZE, SOLIDITY_THRESHOLD,
# MIN_GRAIN_AREA_PX, MAX_GRAIN_AREA_PX and MIN_GRAIN_COUNT) were reachable only
# through the retired coin_calibration.py / segmentation.py modules and have
# been removed. Do NOT reintroduce them as a "source of truth": several
# disagreed with the live values (TILT_THRESHOLD 0.60 vs the live 0.95,
# MIN_GRAIN_COUNT 15 vs the live 30), so restoring them would silently change
# behaviour. Tune the module's own constants instead.

# ---------------------------------------------------------------------------
# Broken Grain Classification (Step 8)
# ---------------------------------------------------------------------------
BROKEN_THRESHOLD_PCT: float = 0.80  # Grain is "broken" if length < 80% of median

# ---------------------------------------------------------------------------
# Defect Detection — HSV Pass (Step 9)
# ---------------------------------------------------------------------------
# Healthy wheat HSV range (golden-yellow)
# NOTE: These are initial estimates — MUST be validated against real photos
HSV_HEALTHY_H_MIN: int = 10
HSV_HEALTHY_H_MAX: int = 40
HSV_HEALTHY_S_MIN: int = 30
HSV_HEALTHY_S_MAX: int = 255
HSV_HEALTHY_V_MIN: int = 70
HSV_HEALTHY_V_MAX: int = 255

# ---------------------------------------------------------------------------
# Defect Detection — LAB Pass (Step 9)
# ---------------------------------------------------------------------------
# Healthy wheat LAB range (warm yellow tones)
# NOTE: These are initial estimates — MUST be validated against real photos
LAB_HEALTHY_A_MIN: float = -10.0
LAB_HEALTHY_A_MAX: float = 25.0
LAB_HEALTHY_B_MIN: float = 5.0
LAB_HEALTHY_B_MAX: float = 55.0

# Discoloration measurement excludes the outermost ring of each grain contour
# before comparing pixels against the HSV/LAB healthy bands above. Measured on
# a real photo (wheatsheetplain.jpeg): 100% of flagged "defect" pixels sat
# within 2px of the contour boundary and 0% were in the interior -- these are
# anti-aliased edge blending against the white background and specular rim
# highlights, not grain damage, and they alone read as 3.41% discoloration on
# a visually clean sample. Excluding this rim from measurement (not loosening
# the HSV/LAB bands themselves) fixes exactly this artifact without reducing
# sensitivity to a real pale/faded defect that happens to sit in a grain's
# interior.
DISCOLORATION_EDGE_EROSION_PX: int = 2

# ---------------------------------------------------------------------------
# AGMARK Grading Thresholds (Step 11)
# ---------------------------------------------------------------------------
# Keyed by commodity name. Adding a new crop = adding a new dict entry.
# Grade is determined by the WORST metric across all parameters.
#
# Structure per commodity:
#   "parameter_name": [(upper_bound, "grade"), ...] sorted best → worst
#   For "higher is better" params (length, ratio): thresholds are minimums
#   For "lower is better" params (broken, discolored, foreign): thresholds are maximums

AGMARK_THRESHOLDS: dict = {
    "wheat": {
        # --- Higher is better (major/minor axis, area) ---
        # Only 3 tiers: the geometric parameters bottom out at "C" rather than
        # "Reject" -- a small, dense, or short grain is substandard, not the
        # kind of contamination/damage "Reject" is reserved for elsewhere in
        # this table.
        "avg_length_mm": [
            (6.5, "A"),   # ≥ 6.5 → A
            (5.5, "B"),   # 5.5–6.4 → B
            (0.0, "C"),   # < 5.5 → C (Stunted)
        ],
        "avg_width_mm": [
            (3.4, "A"),   # ≥ 3.4 → A
            (2.8, "B"),   # 2.8–3.3 → B
            (0.0, "C"),   # < 2.8 → C (Shriveled)
        ],
        "avg_area_mm2": [
            (18.0, "A"),  # ≥ 18.0 → A
            (14.0, "B"),  # 14.0–17.9 → B
            (0.0, "C"),   # < 14.0 → C (Underweight)
        ],
        # --- Band: aspect ratio must fall in a valid range, not just clear a
        # floor. A ratio that is too low means the fitted ellipse is closer to
        # round than grain-shaped -- two overlapping grains segmented as one
        # blob -- and too high means a fragment, not a whole kernel. Both ends
        # of the band land on the same "in range vs out of range" verdict, so
        # this metric never distinguishes A from B on its own; it only ever
        # vetoes down to "C" (Overlap/Broken) when the ratio falls outside
        # [min, max]. See `_grade_single_metric`'s band branch in grading.py.
        "avg_lw_ratio": {
            "type": "band",
            "min": 1.6,
            "max": 2.4,
            "in_band_grade": "A",
            "out_of_band_grade": "C",
        },
        # --- Lower is better ---
        # Broken/discoloured kernels and foreign matter now all bottom out at
        # "Reject" past a fixed ceiling (damaged/discoloration > 3.5%, foreign
        # matter > 1.5%) rather than a "C" floor -- past that point the lot is
        # rejected outright, not merely downgraded.
        "broken_pct": [
            (1.5, "A"),    # ≤ 1.5% → A
            (2.5, "B"),    # 1.5–2.5% → B
            (3.5, "C"),    # 2.5–3.5% → C
            (100.0, "Reject"),  # > 3.5% → Reject
        ],
        "discolored_pct": [
            (1.0, "A"),    # ≤ 1.0% → A
            (2.0, "B"),    # 1.0–2.0% → B
            (3.5, "C"),    # 2.0–3.5% → C
            (100.0, "Reject"),  # > 3.5% → Reject
        ],
        "foreign_matter_pct": [
            (0.5, "A"),    # ≤ 0.5% → A
            (1.5, "B"),    # 0.5–1.5% → B
            (100.0, "Reject"),  # > 1.5% → Reject (no C tier)
        ],
    },
    # Future: add "rice", "maize", etc. here with their own thresholds
}

# Grade ranking for worst-grade logic
GRADE_RANK: dict[str, int] = {"A": 0, "B": 1, "C": 2, "Reject": 3}

# ---------------------------------------------------------------------------
# Video Upload (buyer visual record)
# ---------------------------------------------------------------------------
MAX_VIDEO_SIZE_MB: int = 50
ALLOWED_VIDEO_EXTENSIONS: set[str] = {"video/mp4", "video/quicktime"}
MAX_VIDEO_DURATION_SECONDS: int = 30
