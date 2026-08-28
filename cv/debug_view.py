"""
Debug Visualization — a human-readable view into the CV pipeline.

Not used by the normal /analyze flow. `build_debug_report` re-runs the same
stages `pipeline.run_pipeline` does, but instead of returning only the final
JSON grade, it renders each stage's binary mask / contour overlay as an
image in one HTML page: paper detection, the foreground binary mask, coin
detection, grain contours, and the discoloration overlay.

If the image fails at some stage (blurry, no coin, too few grains, ...), the
report still renders every stage that succeeded up to that point, plus the
error message, so you can see exactly what the pipeline saw before it gave up.

Visual design ("Assay Trace"): a numbered stage-by-stage trace with a verdict
banner up top when grading completes, using the same token/typography system
as the standalone report this was built to match -- Fraunces for headings,
Public Sans for body copy, IBM Plex Mono for every number and caption.
"""

import base64
import html

import cv2
import numpy as np

from cv.config import BLUR_THRESHOLD, IMAGE_RESIZE_WIDTH
from cv.steps.blur_check import check_blur
from cv.steps.defect_detection import detect_discoloration
from cv.steps.grading import compute_grade
from cv.steps.measurement import classify_broken
from cv.steps.white_balance import calibrate_white_balance
from cv.steps.wheat_measurement import (
    MIN_VALID_GRAIN_COUNT,
    calculate_geometry,
    calibrate_image,
    extract_grain_masks,
)

_ENCODE_WIDTH = 640

_METRIC_LABELS = {
    "length": "length",
    "width": "width",
    "area": "area",
    "lw_ratio": "aspect ratio",
    "broken": "broken kernels",
    "discolored": "discoloration",
    "foreign_matter": "foreign matter",
}
_GRADE_CHIP_CLASS = {"A": "chip-good", "B": "chip-good", "C": "chip-warn", "Reject": "chip-bad"}
_GRADE_BADGE_CLASS = {"A": "badge-good", "B": "badge-good", "C": "badge-warn", "Reject": "badge-bad"}

_PAGE_STYLE = """
:root {
  --paper: #eef0e7;
  --surface: #f8f9f3;
  --surface-raised: #ffffff;
  --ink: #1a2018;
  --muted: #62705d;
  --line: #d7dbc7;
  --gold: #8f6220;
  --gold-strong: #6b480f;
  --moss: #2f6b45;
  --brick: #a4342d;
  --good-bg: #dcecdf; --good-fg: #1f5335;
  --warn-bg: #f2e2be; --warn-fg: #5f420d;
  --bad-bg: #f4d7d3;  --bad-fg: #74241d;
  --neutral-bg: #e4e6da; --neutral-fg: #4b5245;
  --shadow: 0 14px 34px rgba(30, 38, 24, .10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #14170f; --surface: #1a1e14; --surface-raised: #21261a;
    --ink: #edf0e4; --muted: #99a38b; --line: #333a27;
    --gold: #d9a13f; --gold-strong: #f0bb66; --moss: #6cb984; --brick: #e2837b;
    --good-bg: #1e3226; --good-fg: #8fd6a6;
    --warn-bg: #3a2f14; --warn-fg: #f0c069;
    --bad-bg: #3a201d;  --bad-fg: #f0958e;
    --neutral-bg: #262c1c; --neutral-fg: #b7c0a8;
    --shadow: 0 14px 34px rgba(0, 0, 0, .45);
  }
}
:root[data-theme="dark"] {
  --paper: #14170f; --surface: #1a1e14; --surface-raised: #21261a;
  --ink: #edf0e4; --muted: #99a38b; --line: #333a27;
  --gold: #d9a13f; --gold-strong: #f0bb66; --moss: #6cb984; --brick: #e2837b;
  --good-bg: #1e3226; --good-fg: #8fd6a6;
  --warn-bg: #3a2f14; --warn-fg: #f0c069;
  --bad-bg: #3a201d;  --bad-fg: #f0958e;
  --neutral-bg: #262c1c; --neutral-fg: #b7c0a8;
  --shadow: 0 14px 34px rgba(0, 0, 0, .45);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--paper); color: var(--ink); font-family: "Public Sans", system-ui, sans-serif; -webkit-font-smoothing: antialiased; }
.page { max-width: 780px; margin: 0 auto; padding: 40px 24px 72px; }
.eyebrow { margin: 0 0 10px; color: var(--muted); font: 600 11px "IBM Plex Mono", monospace; letter-spacing: .14em; text-transform: uppercase; }
h1 { margin: 0 0 10px; font: 600 clamp(24px, 5vw, 34px)/1.1 "Fraunces", Georgia, serif; letter-spacing: -.01em; text-wrap: balance; }
.hero-sub { max-width: 56ch; margin: 0 0 24px; color: var(--muted); font-size: 14px; line-height: 1.6; }
.error-banner { padding: 16px 20px; margin-bottom: 24px; background: var(--bad-bg); color: var(--bad-fg); border: 1px solid var(--brick); border-radius: 10px; font-size: 13.5px; line-height: 1.6; }
.error-banner strong { font-weight: 600; }
.verdict-row { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; padding: 20px 22px; background: var(--surface-raised); border: 1px solid var(--line); border-radius: 10px; box-shadow: var(--shadow); }
.grade-badge { display: grid; place-items: center; flex: 0 0 auto; width: 68px; height: 68px; border-radius: 50%; font: 700 28px "Fraunces", serif; }
.grade-badge.badge-good { background: var(--good-bg); color: var(--good-fg); border: 2px solid var(--moss); }
.grade-badge.badge-warn { background: var(--warn-bg); color: var(--warn-fg); border: 2px solid var(--gold-strong); }
.grade-badge.badge-bad  { background: var(--bad-bg);  color: var(--bad-fg);  border: 2px solid var(--brick); }
.verdict-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 4px 20px; margin: 0; flex: 1 1 300px; }
.verdict-stats dt { color: var(--muted); font: 500 10px "IBM Plex Mono", monospace; letter-spacing: .06em; text-transform: uppercase; }
.verdict-stats dd { margin: 3px 0 0; font: 600 18px "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
.verdict-note { max-width: 62ch; margin: 16px 2px 0; color: var(--muted); font-size: 13px; line-height: 1.6; }
.verdict-note strong { color: var(--ink); font-weight: 600; }
.trace { list-style: none; margin: 32px 0 0; padding: 0; display: grid; gap: 12px; }
.stage { background: var(--surface-raised); border: 1px solid var(--line); border-radius: 10px; padding: 18px 20px; }
.stage-head { display: flex; align-items: flex-start; gap: 14px; }
.stage-num { flex: 0 0 auto; width: 32px; padding-top: 2px; color: var(--gold); font: 600 12px "IBM Plex Mono", monospace; }
.stage-head > div:nth-child(2) { flex: 1 1 auto; min-width: 0; }
.stage h2 { margin: 0 0 4px; font: 600 15px "Fraunces", serif; letter-spacing: -.01em; }
.stage-caption { margin: 0; color: var(--muted); font-size: 12px; font-family: "IBM Plex Mono", monospace; line-height: 1.55; white-space: pre-line; }
.chip { flex: 0 0 auto; padding: 4px 9px; border-radius: 100px; font: 600 10px "IBM Plex Mono", monospace; letter-spacing: .05em; text-transform: uppercase; white-space: nowrap; }
.chip-good { background: var(--good-bg); color: var(--good-fg); }
.chip-warn { background: var(--warn-bg); color: var(--warn-fg); }
.chip-bad  { background: var(--bad-bg);  color: var(--bad-fg); }
.chip-neutral { background: var(--neutral-bg); color: var(--neutral-fg); }
.stage img { display: block; width: 100%; max-width: 100%; height: auto; margin-top: 14px; border-radius: 6px; border: 1px solid var(--line); background: #111; }
.result-table { width: 100%; margin-top: 14px; border-collapse: collapse; }
.result-table td { padding: 7px 0; border-top: 1px solid var(--line); font-size: 13px; }
.result-table td:first-child { color: var(--muted); }
.result-table td:last-child { text-align: right; font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
.result-table tr:first-child td { border-top: none; }
.mgrade { display: inline-block; margin-left: 8px; padding: 1px 7px; border-radius: 100px; font: 600 10px "IBM Plex Mono", monospace; }
@media (max-width: 560px) {
  .page { padding: 32px 16px 56px; }
  .verdict-stats { grid-template-columns: repeat(2, minmax(0,1fr)); }
}
"""

_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;'
    "9..144,600;9..144,700&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:"
    'wght@400;500;600&display=swap" rel="stylesheet">'
)


def _resize_image(image: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= target_width:
        return image
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)


def _encode_jpeg(image: np.ndarray, quality: int = 82) -> str:
    """Downscale for fast loading in the iframe and encode as base64 JPEG."""
    small = _resize_image(image, _ENCODE_WIDTH)
    success, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def _encode_png_mask(mask: np.ndarray) -> str:
    """Downscale a single-channel binary mask and encode as base64 PNG."""
    small = _resize_image(mask, _ENCODE_WIDTH)
    bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    success, buffer = cv2.imencode(".png", bgr)
    if not success:
        return ""
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def _stage(
    num: str,
    title: str,
    caption: str,
    chip_text: str,
    chip_class: str,
    img_uri: str | None = None,
    img_alt: str = "",
) -> str:
    img_html = (
        f'<img src="{img_uri}" alt="{html.escape(img_alt)}" />' if img_uri else ""
    )
    return (
        '<li class="stage">'
        '<div class="stage-head">'
        f'<span class="stage-num">{html.escape(num)}</span>'
        f"<div><h2>{html.escape(title)}</h2>"
        f'<p class="stage-caption">{html.escape(caption)}</p></div>'
        f'<span class="chip {chip_class}">{html.escape(chip_text)}</span>'
        "</div>"
        f"{img_html}"
        "</li>"
    )


def _result_stage(
    num: str,
    overall_grade: str,
    grain_count: int,
    avg_length: float,
    avg_width: float,
    avg_area: float,
    avg_ratio: float,
    median_length: float,
    broken_pct: float,
    discolored_pct: float,
    foreign_pct: float,
    per_metric: dict[str, str],
) -> str:
    def row(label: str, value: str, metric_key: str | None = None) -> str:
        grade_html = ""
        if metric_key is not None:
            grade = per_metric[metric_key]
            grade_html = f' <span class="mgrade {_GRADE_CHIP_CLASS.get(grade, "chip-warn")}">{html.escape(grade)}</span>'
        return f"<tr><td>{html.escape(label)}</td><td>{html.escape(value)}{grade_html}</td></tr>"

    rows = "".join(
        [
            row("Avg length", f"{avg_length} mm", "length"),
            row("Avg width", f"{avg_width} mm", "width"),
            row("Avg area", f"{avg_area} mm²", "area"),
            row("Avg L/W ratio", f"{avg_ratio}", "lw_ratio"),
            row("Median length", f"{median_length} mm"),
            row("Broken", f"{broken_pct}%", "broken"),
            row("Discolored", f"{discolored_pct}%", "discolored"),
            row("Foreign matter", f"{foreign_pct}%", "foreign_matter"),
        ]
    )
    chip_class = _GRADE_CHIP_CLASS.get(overall_grade, "chip-warn")
    return (
        '<li class="stage">'
        '<div class="stage-head">'
        f'<span class="stage-num">{html.escape(num)}</span>'
        "<div><h2>Final result</h2>"
        '<p class="stage-caption">All seven AGMARK metrics, graded individually. '
        "The overall grade is the worst of the seven.</p></div>"
        f'<span class="chip {chip_class}">Grade {html.escape(overall_grade)}</span>'
        "</div>"
        f'<table class="result-table">{rows}</table>'
        "</li>"
    )


def _verdict_note(per_metric: dict[str, str], overall_grade: str, values: dict[str, str]) -> str:
    non_a = [key for key, grade in per_metric.items() if grade != "A"]
    if not non_a:
        return "Every one of the seven AGMARK metrics graded A."
    parts = [
        f"{_METRIC_LABELS[key]} ({values[key]}, grade {per_metric[key]})" for key in non_a
    ]
    return (
        "Every other metric graded <strong>A</strong>. "
        + " and ".join(parts)
        + f' alone pulled the overall grade to <strong>{html.escape(overall_grade)}</strong>.'
    )


def _page(source_name: str, hero_html: str, stages_html: str) -> str:
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Assay Trace &middot; {html.escape(source_name)}</title>
{_FONT_LINKS}
<style>{_PAGE_STYLE}</style>
</head>
<body>
<main class="page">
<header>
<p class="eyebrow">CV pipeline &middot; assay trace</p>
<h1>{html.escape(source_name)}</h1>
<p class="hero-sub">Every stage the grading pipeline ran on this photo, in the order it ran them.</p>
{hero_html}
</header>
<ol class="trace">
{stages_html}
</ol>
</main>
</body>
</html>"""


def _error_banner(message: str) -> str:
    return f'<div class="error-banner"><strong>Stopped:</strong> {html.escape(message)}</div>'


def _error_page(source_name: str, message: str) -> str:
    return _page(source_name, _error_banner(message), "")


def build_debug_report(
    image_bytes: bytes, commodity: str = "wheat", source_name: str | None = None
) -> str:
    """
    Run the pipeline stage by stage and render an HTML page showing the
    binary mask / contour overlay produced at each stage.
    """
    source_name = source_name or "uploaded photo"
    stages: list[str] = []

    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        return _error_page(source_name, "The uploaded file is not a readable JPEG or PNG image.")

    orig_h, orig_w = image.shape[:2]
    if image.shape[1] > IMAGE_RESIZE_WIDTH:
        image = _resize_image(image, IMAGE_RESIZE_WIDTH)
    stages.append(
        _stage(
            "00",
            "Resized input",
            f"{orig_w}x{orig_h} px source, downscaled to {image.shape[1]}x{image.shape[0]} px "
            "before processing.",
            "Source",
            "chip-neutral",
            _encode_jpeg(image),
            "Resized input photograph",
        )
    )

    is_sharp, variance = check_blur(image)
    stages.append(
        _stage(
            "01",
            "Blur check",
            f"Laplacian variance {variance:.2f} -- needs >= {BLUR_THRESHOLD:.2f} to pass.",
            "Pass" if is_sharp else "Fail",
            "chip-good" if is_sharp else "chip-bad",
        )
    )
    if not is_sharp:
        return _page(source_name, _error_banner("Image too blurry to continue."), "".join(stages))

    try:
        calibration = calibrate_image(image)
    except Exception as error:  # noqa: BLE001 - shown to the user, not re-raised
        return _page(
            source_name,
            _error_banner(f"Paper/coin detection failed: {error}"),
            "".join(stages),
        )

    cropped_image = calibration["cropped_image"]
    paper_mask = calibration["paper_mask"]
    foreground = calibration["foreground"]
    coin = calibration["coin"]
    coin_mask = calibration["coin_mask"]

    # --- 02/03. Paper detection ---
    paper_overlay = cropped_image.copy()
    dimmed = (paper_overlay.astype(np.uint16) // 3).astype(np.uint8)
    paper_overlay = np.where(paper_mask[:, :, None] > 0, paper_overlay, dimmed)
    stages.append(
        _stage(
            "02",
            "Paper detection",
            "Cropped to the detected sheet's bounding box. Dimmed area is outside "
            "the paper and excluded from analysis.",
            "Crop",
            "chip-neutral",
            _encode_jpeg(paper_overlay),
            "Cropped photo with non-paper area dimmed",
        )
    )
    stages.append(
        _stage(
            "03",
            "Paper binary mask",
            "White = pixels the HSV threshold accepted as paper.",
            "Mask",
            "chip-neutral",
            _encode_png_mask(paper_mask),
            "Binary mask of the detected paper region",
        )
    )

    # --- 04. Foreground binary mask ---
    stages.append(
        _stage(
            "04",
            "Foreground mask",
            "White = anything on the sheet that is not paper -- coin, grains, "
            "debris -- after illumination flat-fielding.",
            "Mask",
            "chip-neutral",
            _encode_png_mask(foreground),
            "Binary foreground mask",
        )
    )

    # --- 05. Coin detection ---
    coin_overlay = cropped_image.copy()
    cv2.ellipse(coin_overlay, coin["ellipse"], (255, 128, 0), 3)
    cv2.circle(coin_overlay, coin["center"], 4, (0, 0, 255), -1)
    stages.append(
        _stage(
            "05",
            "Coin detection",
            f"px_per_mm {coin['px_per_mm']:.3f}  ·  tilt ratio {coin['tilt_ratio']:.3f} "
            f"(>= 0.95 required)  ·  circularity {coin['circularity']:.3f}",
            "Pass",
            "chip-good",
            _encode_jpeg(coin_overlay),
            "Detected coin outlined with its center marked",
        )
    )

    try:
        extraction = extract_grain_masks(cropped_image, foreground, coin_mask, coin)
        geometry = calculate_geometry(extraction["grain_contours"], coin["px_per_mm"])
    except Exception as error:  # noqa: BLE001 - shown to the user, not re-raised
        return _page(
            source_name,
            _error_banner(f"Grain segmentation failed: {error}"),
            "".join(stages),
        )

    grain_contours = geometry["measured_contours"]
    grain_count = geometry["summary"]["count"]

    # --- 06. Grain contours ---
    contour_overlay = cropped_image.copy()
    cv2.drawContours(contour_overlay, grain_contours, -1, (0, 255, 0), 2)
    stages.append(
        _stage(
            "06",
            "Grain contours",
            f"{grain_count} grains measured individually. Excluded: "
            f"{extraction['excluded_touching']} touching cluster(s), "
            f"{extraction['excluded_low_solidity']} low-solidity, "
            f"{extraction['excluded_ring']} ring artifact(s), "
            f"{extraction['excluded_border']} border-clipped, "
            f"{extraction['foreign_matter_count']} foreign-matter object(s).",
            f"{grain_count} / {MIN_VALID_GRAIN_COUNT} min",
            "chip-good" if grain_count >= MIN_VALID_GRAIN_COUNT else "chip-bad",
            _encode_jpeg(contour_overlay),
            "Photo with every measured grain outlined in green",
        )
    )

    # --- 07. Discoloration overlay ---
    balanced_image = calibrate_white_balance(cropped_image, coin["center"], coin["radius"])
    discolored_pct, defect_mask = detect_discoloration(
        balanced_image, grain_contours, return_mask=True
    )
    defect_overlay = balanced_image.copy()
    defect_overlay[defect_mask > 0] = (0, 0, 255)
    cv2.drawContours(defect_overlay, grain_contours, -1, (0, 255, 0), 1)
    stages.append(
        _stage(
            "07",
            "Discoloration overlay",
            "Red = pixels flagged by the HSV or LAB defect pass, on the "
            "white-balanced image. Green = grain outline.",
            f"{discolored_pct:.2f}%",
            "chip-neutral",
            _encode_jpeg(defect_overlay),
            "White-balanced photo with defective pixels highlighted red",
        )
    )

    if grain_count < MIN_VALID_GRAIN_COUNT:
        return _page(
            source_name,
            _error_banner(
                f"Only {grain_count} grains were measured; {MIN_VALID_GRAIN_COUNT} are "
                "required, so no grade can be computed."
            ),
            "".join(stages),
        )

    # --- 08. Final result ---
    broken_inputs = [
        {
            "length_mm": grain["length_mm"],
            "width_mm": grain["width_mm"],
            "lw_ratio": grain["aspect_ratio"],
        }
        for grain in geometry["grains"]
    ]
    broken_pct, median_length = classify_broken(broken_inputs)

    foreign_area_mm2 = float(extraction["foreign_matter_area_mm2"])
    grain_area_mm2 = sum(float(g["area_mm2"]) for g in geometry["grains"])
    touching_cluster_area_mm2 = float(extraction["touching_cluster_area_mm2"])
    sample_area_mm2 = foreign_area_mm2 + grain_area_mm2 + touching_cluster_area_mm2
    foreign_pct = (
        round((foreign_area_mm2 / sample_area_mm2) * 100.0, 2) if sample_area_mm2 > 0 else 0.0
    )

    avg_length = round(float(geometry["summary"]["length_mm"]["mean"]), 2)
    avg_width = round(float(geometry["summary"]["width_mm"]["mean"]), 2)
    avg_ratio = round(float(np.mean([g["aspect_ratio"] for g in geometry["grains"]])), 2)
    avg_area = round(float(np.mean([g["area_mm2"] for g in geometry["grains"]])), 2)

    try:
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
    except ValueError as error:
        return _page(source_name, _error_banner(str(error)), "".join(stages))

    stages.append(
        _result_stage(
            "08",
            overall_grade,
            grain_count,
            avg_length,
            avg_width,
            avg_area,
            avg_ratio,
            median_length,
            broken_pct,
            discolored_pct,
            foreign_pct,
            per_metric_grades,
        )
    )

    metric_values = {
        "length": f"{avg_length} mm",
        "width": f"{avg_width} mm",
        "area": f"{avg_area} mm2",
        "lw_ratio": f"{avg_ratio}",
        "broken": f"{broken_pct}%",
        "discolored": f"{discolored_pct}%",
        "foreign_matter": f"{foreign_pct}%",
    }
    hero_html = (
        '<div class="verdict-row">'
        f'<div class="grade-badge {_GRADE_BADGE_CLASS.get(overall_grade, "badge-warn")}">'
        f"{html.escape(overall_grade)}</div>"
        '<dl class="verdict-stats">'
        f"<div><dt>Grains</dt><dd>{grain_count}</dd></div>"
        f"<div><dt>Discolored</dt><dd>{discolored_pct}%</dd></div>"
        f"<div><dt>Broken</dt><dd>{broken_pct}%</dd></div>"
        f"<div><dt>Foreign matter</dt><dd>{foreign_pct}%</dd></div>"
        "</dl>"
        "</div>"
        f'<p class="verdict-note">{_verdict_note(per_metric_grades, overall_grade, metric_values)}</p>'
    )

    return _page(source_name, hero_html, "".join(stages))
