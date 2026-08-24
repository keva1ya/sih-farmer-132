"""
Step 11 — AGMARK Grading

Maps aggregate measurements to AGMARK grades using the threshold
table from config.py. The overall grade is the WORST grade across
all five parameters (length, ratio, broken, discolored, foreign matter).

The threshold structure supports "higher is better" params (length, ratio)
and "lower is better" params (broken, discolored, foreign).
"""

from cv.config import AGMARK_THRESHOLDS, GRADE_RANK

# Parameters where a HIGHER value is better
_HIGHER_IS_BETTER = {"avg_length_mm", "avg_lw_ratio"}
# Parameters where a LOWER value is better
_LOWER_IS_BETTER = {"broken_pct", "discolored_pct", "foreign_matter_pct"}


def _grade_single_metric(
    value: float, thresholds: list[tuple[float, str]], higher_is_better: bool
) -> str:
    """
    Determine the grade for a single metric.

    Args:
        value: the measured value.
        thresholds: list of (boundary, grade) pairs, sorted best → worst.
        higher_is_better: if True, value must be >= boundary to earn the grade.

    Returns:
        The grade string ("A", "B", "C", or "Reject").
    """
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
    avg_lw_ratio: float,
    broken_pct: float,
    discolored_pct: float,
    foreign_matter_pct: float,
    commodity: str = "wheat",
) -> tuple[str, dict[str, str]]:
    """
    Compute the overall AGMARK grade and per-metric breakdown.

    Args:
        avg_length_mm: average grain length in mm.
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
