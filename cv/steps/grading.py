"""
Step 11 — AGMARK Grading

Maps aggregate measurements to AGMARK grades using the threshold
table from config.py. The overall grade is the WORST grade across
all seven parameters (length, width, area, aspect ratio, broken,
discolored, foreign matter).

Two threshold shapes are supported. Most parameters use a cascading list of
(boundary, grade) pairs: "higher is better" params (length, width, area) or
"lower is better" params (broken, discolored, foreign). Aspect ratio instead
uses a band -- a valid [min, max] range -- because a ratio can fail by being
either too low (overlapping grains) or too high (a broken fragment); see its
entry in config.py for why a single monotonic scale can't express that.
"""

from typing import Any

from cv.config import AGMARK_THRESHOLDS, GRADE_RANK

# Parameters where a HIGHER value is better
_HIGHER_IS_BETTER = {"avg_length_mm", "avg_width_mm", "avg_area_mm2"}
# Parameters where a LOWER value is better
_LOWER_IS_BETTER = {"broken_pct", "discolored_pct", "foreign_matter_pct"}


def _grade_single_metric(
    value: float, thresholds: list[tuple[float, str]] | dict[str, Any], higher_is_better: bool
) -> str:
    """
    Determine the grade for a single metric.

    Args:
        value: the measured value.
        thresholds: either a list of (boundary, grade) pairs sorted best →
            worst, or a band dict ({"type": "band", "min", "max",
            "in_band_grade", "out_of_band_grade"}) for a parameter that must
            fall inside a range rather than clear a single floor/ceiling.
        higher_is_better: if True, value must be >= boundary to earn the grade.
            Ignored for a band threshold, which judges range membership instead.

    Returns:
        The grade string ("A", "B", "C", or "Reject").
    """
    if isinstance(thresholds, dict) and thresholds.get("type") == "band":
        if thresholds["min"] <= value <= thresholds["max"]:
            return thresholds["in_band_grade"]
        return thresholds["out_of_band_grade"]

    if higher_is_better:
        # Thresholds are minimums, sorted descending
        for boundary, grade in thresholds:
            if value >= boundary:
                return grade
        return "Reject"
    else:
        # Thresholds are maximums, sorted ascending
        for boundary, grade in thresholds:
            if value <= boundary:
                return grade
        return "Reject"


def compute_grade(
    avg_length_mm: float,
    avg_width_mm: float,
    avg_area_mm2: float,
    avg_lw_ratio: float,
    broken_pct: float,
    discolored_pct: float,
    foreign_matter_pct: float,
    commodity: str = "wheat",
) -> tuple[str, dict[str, str]]:
    """
    Compute the overall AGMARK grade and per-metric breakdown.

    Args:
        avg_length_mm: average grain length (major axis) in mm.
        avg_width_mm: average grain width (minor axis) in mm.
        avg_area_mm2: average grain area in mm^2.
        avg_lw_ratio: average length/width ratio.
        broken_pct: percentage of broken grains.
        discolored_pct: percentage of discolored grain area.
        foreign_matter_pct: percentage of foreign matter by area.
        commodity: commodity name to look up thresholds (default: "wheat").

    Returns:
        (overall_grade, per_metric_grades):
            overall_grade — the worst grade across all metrics.
            per_metric_grades — dict mapping metric name to its individual grade.
    """
    if commodity not in AGMARK_THRESHOLDS:
        raise ValueError(
            f"No AGMARK thresholds defined for commodity '{commodity}'. "
            f"Available: {list(AGMARK_THRESHOLDS.keys())}"
        )

    thresholds = AGMARK_THRESHOLDS[commodity]

    metrics = {
        "avg_length_mm": avg_length_mm,
        "avg_width_mm": avg_width_mm,
        "avg_area_mm2": avg_area_mm2,
        "avg_lw_ratio": avg_lw_ratio,
        "broken_pct": broken_pct,
        "discolored_pct": discolored_pct,
        "foreign_matter_pct": foreign_matter_pct,
    }

    per_metric_grades: dict[str, str] = {}
    worst_grade = "A"

    for metric_name, value in metrics.items():
        higher_is_better = metric_name in _HIGHER_IS_BETTER
        grade = _grade_single_metric(
            value, thresholds[metric_name], higher_is_better
        )

        # Map to response-friendly key names
        response_key = {
            "avg_length_mm": "length",
            "avg_width_mm": "width",
            "avg_area_mm2": "area",
            "avg_lw_ratio": "lw_ratio",
            "broken_pct": "broken",
            "discolored_pct": "discolored",
            "foreign_matter_pct": "foreign_matter",
        }[metric_name]

        per_metric_grades[response_key] = grade

        # Track worst grade
        if GRADE_RANK.get(grade, 99) > GRADE_RANK.get(worst_grade, 0):
            worst_grade = grade

    return worst_grade, per_metric_grades
