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
BLUR_THRESHOLD: float = 30.0  # Laplacian variance; below = too blurry

# ---------------------------------------------------------------------------
# Coin Calibration (Step 2)
# ---------------------------------------------------------------------------
COIN_DIAMETER_MM: float = 27.0  # ₹10 bimetallic outer ring (RBI verified)
COIN_INNER_OUTER_RATIO: float = 1.35  # outer / inner ≈ 27mm / 20mm
COIN_RATIO_TOLERANCE: float = 0.10  # ±10% tolerance on the ratio check
TILT_THRESHOLD: float = 0.85  # min aspect ratio of fitted ellipse (1.0 = perfect circle)

# HoughCircles parameters (tuned for 1920px-wide images)
HOUGH_DP: float = 1.2
HOUGH_MIN_DIST: int = 100
HOUGH_PARAM1: int = 100  # Canny upper threshold
HOUGH_PARAM2: int = 40   # Accumulator threshold (lower = more sensitive)
HOUGH_MIN_RADIUS: int = 40
HOUGH_MAX_RADIUS: int = 300

# ---------------------------------------------------------------------------
# Segmentation (Step 5)
# ---------------------------------------------------------------------------
MORPH_KERNEL_SIZE: int = 3  # Kernel for morphological opening
SOLIDITY_THRESHOLD: float = 0.80  # Contours below this are shadows/artifacts
MIN_GRAIN_AREA_PX: int = 150  # Contours smaller than this → chaff/foreign matter
MIN_GRAIN_COUNT: int = 30  # Minimum grains needed for statistical validity

# ---------------------------------------------------------------------------
# Broken Grain Classification (Step 8)
# ---------------------------------------------------------------------------
BROKEN_THRESHOLD_PCT: float = 0.80  # Grain is "broken" if length < 80% of median

# ---------------------------------------------------------------------------
# Defect Detection — HSV Pass (Step 9)
# ---------------------------------------------------------------------------
# Healthy wheat HSV range (golden-yellow)
# NOTE: These are initial estimates — MUST be validated against real photos
HSV_HEALTHY_H_MIN: int = 15
HSV_HEALTHY_H_MAX: int = 35
HSV_HEALTHY_S_MIN: int = 80
HSV_HEALTHY_S_MAX: int = 255
HSV_HEALTHY_V_MIN: int = 100
HSV_HEALTHY_V_MAX: int = 255

# ---------------------------------------------------------------------------
# Defect Detection — LAB Pass (Step 9)
# ---------------------------------------------------------------------------
# Healthy wheat LAB range (warm yellow tones)
# NOTE: These are initial estimates — MUST be validated against real photos
LAB_HEALTHY_A_MIN: float = -5.0
LAB_HEALTHY_A_MAX: float = 15.0
LAB_HEALTHY_B_MIN: float = 15.0
LAB_HEALTHY_B_MAX: float = 45.0

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
        # --- Higher is better ---
        "avg_length_mm": [
            (6.8, "A"),   # ≥ 6.8 → A
            (6.0, "B"),   # 6.0–6.7 → B
            (5.5, "C"),   # 5.5–5.9 → C
            (0.0, "Reject"),  # < 5.5 → Reject
        ],
        "avg_lw_ratio": [
            (3.0, "A"),
            (2.5, "B"),
            (2.0, "C"),
            (0.0, "Reject"),
        ],
        # --- Lower is better ---
        "broken_pct": [
            (2.0, "A"),   # ≤ 2.0% → A
            (5.0, "B"),   # ≤ 5.0% → B
            (8.0, "C"),   # ≤ 8.0% → C
            (100.0, "Reject"),  # > 8.0% → Reject
        ],
        "discolored_pct": [
            (1.0, "A"),
            (3.0, "B"),
            (6.0, "C"),
            (100.0, "Reject"),
        ],
        "foreign_matter_pct": [
            (0.5, "A"),
            (1.5, "B"),
            (3.0, "C"),
            (100.0, "Reject"),
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
