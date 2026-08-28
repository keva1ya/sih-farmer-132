import cv2
import numpy as np
import pytest

from cv.pipeline import run_pipeline
from cv.steps.grading import compute_grade
from cv.steps.wheat_measurement import (
    TooFewGrainsError,
    CoinNotDetectedError,
    PaperNotDetectedError,
    CoinTiltedError,
    ImageLoadError,
    _measure_wheat_grains_from_image,
)

DECLARED_ERRORS = (
    ImageLoadError,
    PaperNotDetectedError,
    CoinNotDetectedError,
    CoinTiltedError,
    TooFewGrainsError,
)

WHEAT_BGR = tuple(
    int(v) for v in cv2.cvtColor(np.uint8([[[25, 150, 190]]]), cv2.COLOR_HSV2BGR)[0, 0]
)


def build_scene(n_grains=30, paper=(255, 255, 255), clusters=0, ring=False, crescent=False):
    """Synthetic top-down sheet: white paper, a Rs 10 coin, and elliptical grains."""
    image = np.full((1100, 1600, 3), paper, dtype=np.uint8)
    cv2.circle(image, (1300, 250), 135, (130, 130, 130), -1)
    cv2.circle(image, (1300, 250), 80, (170, 170, 170), -1)

    positions = [(130 + col * 155, 150 + row * 165) for row in range(5) for col in range(6)]
    for idx, center in enumerate(positions[:n_grains]):
        cv2.ellipse(image, center, (35, 15), (idx * 23) % 180, 0, 360, WHEAT_BGR, -1)

    # Fused triples: one contour holding three grains.
    for k in range(clusters):
        cx, cy = 200 + (k % 5) * 260, 700 + (k // 5) * 150
        for dx in (-26, 0, 26):
            cv2.ellipse(image, (cx + dx, cy), (35, 15), 90, 0, 360, WHEAT_BGR, -1)

    if ring:  # donut shadow: convex outline, hollow interior
        cv2.circle(image, (900, 950), 24, WHEAT_BGR, -1)
        cv2.circle(image, (900, 950), 14, (255, 255, 255), -1)
    if crescent:  # concave shadow artifact
        cv2.circle(image, (1100, 950), 25, WHEAT_BGR, -1)
        cv2.circle(image, (1113, 950), 20, (255, 255, 255), -1)
    return image


def test_ratio_is_part_of_worst_metric_grade():
    # Every other metric graded A; only the ratio sits outside its [1.6, 2.4]
    # band, so it alone should drag the overall grade down. Band metrics cap
    # at "C", never "Reject" -- see the band entry in config.py.
    grade, metric_grades = compute_grade(
        avg_length_mm=7.0,
        avg_width_mm=3.6,
        avg_area_mm2=20.0,
        avg_lw_ratio=1.0,
        broken_pct=0.0,
        discolored_pct=0.0,
        foreign_matter_pct=0.0,
    )

    assert metric_grades["lw_ratio"] == "C"
    assert grade == "C"


def test_public_result_matches_the_specified_contract():
    result = _measure_wheat_grains_from_image(build_scene())

    assert result["summary"]["count"] == 30
    assert result["excluded_touching"] == 0
    assert {"length_mm", "width_mm", "area_mm2", "aspect_ratio", "id"} <= set(result["grains"][0])


@pytest.mark.parametrize("value", [255, 240, 231, 220, 200, 180])
def test_flatfield_survives_dim_paper(value):
    """Regression: a fixed `V < 235` cutoff collapsed the sheet into one blob at V=231."""
    result = _measure_wheat_grains_from_image(build_scene(paper=(value,) * 3))

    assert result["summary"]["count"] == 30


@pytest.mark.parametrize("saturation", [5, 20, 25, 27, 30, 40, 55])
def test_colour_cast_does_not_turn_the_sheet_into_foreground(saturation):
    """Regression: an absolute `S > 25` rule sat inside the paper band (S <= 60),
    so a mild warm cast marked the whole sheet as grain (hundreds of contours)."""
    paper = cv2.cvtColor(np.uint8([[[20, saturation, 250]]]), cv2.COLOR_HSV2BGR)[0, 0]
    result = _measure_wheat_grains_from_image(build_scene(paper=tuple(int(v) for v in paper)))

    assert result["summary"]["count"] == 30


def test_shadow_artifacts_are_rejected():
    """Solidity catches concave crescents; the hole ratio catches convex rings."""
    result = _measure_wheat_grains_from_image(build_scene(ring=True, crescent=True))

    assert result["summary"]["count"] == 30
    assert result["_excluded_ring"] == 1
    assert result["_excluded_low_solidity"] == 1


def test_touching_clusters_are_excluded_without_gating_the_count():
    """`excluded_touching` counts clusters, per the output contract, and a
    cluster's estimated grain content is a diagnostic only -- it does not rescue
    a sheet that measured fewer than MIN_VALID_GRAIN_COUNT individually."""
    result = _measure_wheat_grains_from_image(build_scene(n_grains=30, clusters=1))

    assert result["summary"]["count"] == 30
    assert result["excluded_touching"] == 1  # clusters, per spec
    assert result["_estimated_touching_grains"] == 3


def test_touching_clusters_can_no_longer_be_estimated_into_a_passing_count():
    """Regression: an earlier version validated on measured + estimated grains,
    so a sheet with too few individually measured grains could still pass if
    enough of them were hidden in touching clusters. Excluded blobs cannot be
    measured reliably, so they no longer count toward the minimum at all."""
    with pytest.raises(TooFewGrainsError):
        _measure_wheat_grains_from_image(build_scene(n_grains=27, clusters=1))


def test_border_clipped_contours_are_excluded_not_measured():
    """Regression: a real photo produced a false contour running along the crop
    edge (paper boundary / background bleed), which was measured as a grain.
    Any contour whose bounding box touches the crop edge is only partially in
    frame and must be dropped, not counted."""
    image = build_scene(n_grains=30)
    cv2.ellipse(image, (0, 550), (35, 15), 0, 0, 360, WHEAT_BGR, -1)

    result = _measure_wheat_grains_from_image(image)

    assert result["summary"]["count"] == 30
    assert result["_excluded_border"] >= 1


@pytest.mark.parametrize("clusters", [9, 10, 12, 15])
def test_all_clustered_sheet_raises_instead_of_crashing(clusters):
    """Regression: an earlier version validated an effective total that could
    reach the minimum with `len(grains) == 0`, and `_summary_stats([])` then
    raised a ValueError that escaped as an HTTP 500. Validating on the measured
    count alone makes an empty `grains` list structurally unable to pass."""
    with pytest.raises(TooFewGrainsError):
        _measure_wheat_grains_from_image(build_scene(n_grains=0, clusters=clusters))


def test_grain_count_has_no_upper_bound():
    """Regression: a fixed maximum (formerly 50) rejected sheets with more grain
    than usual with `TooManyGrainsError`. There is no cap -- a sheet with more
    individually-measured grains than the old ceiling now succeeds."""
    image = np.full((1600, 2200, 3), 255, dtype=np.uint8)
    cv2.circle(image, (1900, 300), 135, (130, 130, 130), -1)
    cv2.circle(image, (1900, 300), 80, (170, 170, 170), -1)

    # 8x10 grid, well clear of the coin and spaced wide enough that none touch.
    positions = [(120 + col * 130, 130 + row * 140) for row in range(8) for col in range(10)]
    for idx, center in enumerate(positions):
        cv2.ellipse(image, center, (35, 15), (idx * 23) % 180, 0, 360, WHEAT_BGR, -1)

    result = _measure_wheat_grains_from_image(image)

    assert result["status"] == "ok"
    assert result["summary"]["count"] == len(positions) > 50


@pytest.mark.parametrize(
    "image",
    [
        np.full((800, 1000, 3), 255, np.uint8),
        np.zeros((800, 1000, 3), np.uint8),
        np.full((1, 1, 3), 255, np.uint8),
        np.random.default_rng(0).integers(0, 255, (800, 1000, 3), dtype=np.uint8),
    ],
    ids=["blank", "black", "one-pixel", "noise"],
)
def test_degenerate_inputs_raise_only_declared_exceptions(image):
    """Nothing may escape as an unmapped exception: the pipeline turns those into HTTP 500."""
    try:
        _measure_wheat_grains_from_image(image)
    except DECLARED_ERRORS:
        pass


def test_pipeline_response_keeps_the_analyze_contract():
    payload = cv2.imencode(".png", build_scene())[1].tobytes()

    result = run_pipeline(payload, detailed=True)

    assert result["status"] == "success"
    assert result["grain_count"] == 30
    assert set(result["measurements"]) == {
        "avg_length_mm",
        "avg_width_mm",
        "avg_area_mm2",
        "avg_lw_ratio",
        "median_length_mm",
        "broken_pct",
        "discolored_pct",
        "foreign_matter_pct",
    }
    # Internal hand-off keys must never reach the API surface.
    assert not [key for key in result if key.startswith("_")]


def test_pipeline_errors_always_carry_a_reason():
    """frontend/app.js prefers `reason` over its own status table, which has no
    entry for paper_not_detected."""
    for payload in (b"", b"not-an-image"):
        result = run_pipeline(payload)

        assert result["status"] != "success"
        assert result["reason"]
