"""
Path-based wheat grain measurement using a 10 Indian Rupee coin reference.

The public entry point loads a JPEG/PNG image from disk, crops to the white
paper region, detects the reference coin, segments foreground objects on the
paper, excludes touching grain clusters, and returns raw JSON-serializable
measurements and summary statistics.

Three public functions expose that pipeline as isolated layers so the
extraction stage can be replaced (e.g. with a YOLOv8-Seg backend) without
touching calibration or geometry:

* `calibrate_image`     -- paper crop, shared foreground mask, coin geometry.
* `extract_grain_masks` -- segments the foreground into per-grain contours and
                            masks; the only layer a trained segmenter replaces.
* `calculate_geometry`  -- ellipse-fits contours into length/width/area/ratio;
                            reads only contour points and scale, so it is
                            indifferent to how those contours were produced.

`_measure_wheat_grains_from_image` composes the three for in-process callers
such as the pipeline adapter, additionally returning underscore-prefixed
diagnostics; `measure_wheat_grains` strips every underscore key, so its result
always has exactly the five contracted top-level keys.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from cv.config import HSV_HEALTHY_H_MAX, HSV_HEALTHY_H_MIN


COIN_DIAMETER_MM = 27.0
COIN_TILT_MIN_AXIS_RATIO = 0.95
NORMAL_GRAIN_AREA_MM2 = 18.0
TOUCHING_GRAIN_AREA_MULTIPLIER = 1.5
MIN_VALID_GRAIN_COUNT = 30

FOREIGN_MATTER_MIN_AREA_MM2 = 2.0
FOREIGN_MATTER_MAX_AREA_MM2 = 8.0
PAPER_SATURATION_MAX = 60
PAPER_VALUE_MIN = 180
PAPER_MIN_AREA_RATIO = 0.15

# --- Sheet recovery on a creased sheet -------------------------------------
# Keeping only the largest bright component loses a creased sheet: crease
# shadows fall below PAPER_VALUE_MIN, so the surviving bright region snakes
# around them and its filled external contour leaves unfilled pockets in the
# middle of the sheet. Measured on `wheatsheetcrumbled.jpeg`, that filled
# contour covered only 47.8% of its own bounding box (80.1% on the flat sheet)
# and the reference coin sat inside one of the pockets, outside the searchable
# region entirely.
#
# A relaxed brightness rule cannot fix this. Measured on both photos, the
# shadowed sheet and the surrounding tablecloth occupy the same brightness band
# -- at V >= 150 a relaxed gate recovers 63.7% of the lost sheet but also admits
# 52.1% of the tablecloth ring. Brightness is not separable here; geometry is. A
# sheet of paper is convex, so the convex hull of the bright evidence is the
# sheet, and it is by construction a subset of the bounding rectangle this
# function already crops to.
#
# Fragments are merged before the hull is taken so that a sheet a deep crease
# has split into two large bright pieces is rebuilt rather than half-discarded.
# A fragment joins only if it is substantial and essentially inside the seed's
# reach, which is what stops a bright patch of tablecloth from dragging the hull
# off the sheet.
PAPER_FRAGMENT_MIN_AREA_RATIO = 0.02
PAPER_FRAGMENT_MARGIN_FRACTION = 0.05
PAPER_FRAGMENT_MIN_CONTAINMENT = 0.5
# The hull is an extrapolation, so it must stay mostly supported by pixels that
# actually looked like paper. Measured support (bright pixels inside the hull,
# over hull area): 85.7% on the flat sheet, 61.0% on the crumbled one. At 0.45
# the crumbled sheet clears the bar by 16 points while a hull that is more than
# half invention -- what tablecloth leakage would produce -- is refused and the
# old filled-contour region is used instead.
PAPER_HULL_MIN_BRIGHT_SUPPORT = 0.45

# --- Coin identification ---------------------------------------------------
# Raised from 0.001. That floor was 393 px on the real crop while single wheat
# grains measure 400-900 px, so it admitted individual grains as coin
# candidates. 0.003 is derived from geometry rather than guessed: a 27 mm coin
# covers 573 mm2, an A4 sheet 62370 mm2, so a genuine coin is 0.92% of the
# sheet -- 3.1x above this floor. A single grain is 18 mm2 (0.029% of A4) and
# even a fused triple only 0.087%, so nothing grain-shaped can reach 0.3%.
# Measured headroom on real photos: coin/paper is 2.14% (plain) and 2.73%
# (crumbled), 7.1x and 9.1x above the floor.
COIN_MIN_AREA_RATIO = 0.003
COIN_MAX_AREA_RATIO = 0.25
# Raised from 0.60. Measured on the search mask this module actually builds,
# the coin scores 0.872 (plain), 0.863 (crumbled -- the adverse, shadowed case)
# and 0.899 (synthetic), so 0.80 sits 7.3% below the worst real coin: a dirty or
# partly shadowed coin still qualifies, a 0.60 gate no longer waves through
# anything vaguely round.
COIN_MIN_CIRCULARITY = 0.80
# The coin's size relative to the objects around it, derived from the image
# rather than assumed. Everything on a wheat sheet except the coin is a grain or
# a clump of grains, so the median blob area IS the grain scale.
#
# Measured, on the same photographs with the coin digitally removed and the
# grains left untouched:
#   single grain          <= ~2x the median (it IS roughly the median)
#   largest grain clump    6.3x (plain), 10.0x (crumbled)
#   the real coin         27.7x (plain), 32.8x (crumbled), 34.4x (synthetic)
#   coin on a pathological all-clustered sheet, where the median is a whole
#   fused triple                                                  11.9x
#
# 8.0 is the largest multiple that still admits that last case, which the
# all-clustered regression requires to reach `TooFewGrainsError` rather than
# dying at coin detection. It is 4x above the largest possible SINGLE grain,
# which is what "a grain cannot impersonate the coin" means here and is met with
# room to spare.
#
# What this rule deliberately does NOT claim: it cannot by itself exclude a
# large fused CLUMP of grains. The worst measured clump is 10.0x and the
# all-clustered ceiling is 11.9x, so no threshold separates both -- an area rule
# alone cannot do this job on real photographs, and setting one high enough to
# try would break a legitimate case. Clumps are excluded by the conjunction
# instead, and the separation there is clean: among blobs that pass these area
# gates, the best coin-free clump scores 0.740 circularity against the coin's
# 0.863, none is corroborated by Hough, and none reaches a 0.95 axis ratio.
COIN_MIN_POPULATION_AREA_MULTIPLE = 8.0
# A median is only a population statistic when there is a population. Below this
# many blobs the rule is skipped and the geometric floors above stand alone --
# otherwise a sheet holding nothing but the coin would compare the coin against
# itself and reject it. Any scene with fewer than 8 objects is far short of the
# 30 grains a measurement needs, so it fails downstream regardless.
COIN_POPULATION_MIN_BLOBS = 8
# The coin search opens the mask with a disc this fraction of the largest disc
# that fits inside the foreground. That largest inscribed disc is the coin, so
# the element is always about half the coin's radius: too wide for any grain or
# grain-bridge to contain (killing them outright), and small enough that opening
# a convex blob this much larger returns it unchanged.
COIN_SEARCH_OPEN_RADIUS_FRACTION = 0.4
# Cross-validation against cv2.HoughCircles, which locates the true coin
# independently in both real photos from image gradients alone. A contour
# candidate is corroborated only if a Hough circle agrees on the centre.
# Measured centre disagreement for the true coin: 0.008 (plain), 0.063
# (crumbled), 0.010 (synthetic) of the candidate radius, so 0.25 leaves a 4x
# margin over the worst real case.
COIN_HOUGH_CENTER_TOLERANCE = 0.25
# Radius agreement is deliberately looser than centre agreement, because a coin
# has two concentric rims and Hough legitimately locks onto either. A bimetallic
# Rs 10 coin's inner disc is 19 mm inside a 27 mm rim (0.70); the synthetic
# coin's inner disc is 0.59 of its outer rim, and Hough returns exactly that.
# The window admits a concentric inner rim without admitting an unrelated
# circle. It is not a loophole: the Hough radius search window is itself derived
# from the coin area gates above, so Hough cannot propose a grain-sized circle
# at all.
COIN_HOUGH_MIN_RADIUS_RATIO = 0.50
COIN_HOUGH_MAX_RADIUS_RATIO = 1.30
COIN_HOUGH_DP = 1.2
COIN_HOUGH_PARAM1 = 100
COIN_HOUGH_PARAM2 = 30

GRAIN_MIN_SOLIDITY = 0.80
# How far above the sheet's OWN saturation a pixel must sit before colour alone
# marks it as foreground. This replaces the former absolute
# `GRAIN_FOREGROUND_SATURATION_MIN = 25`, which contradicted
# PAPER_SATURATION_MAX = 60: every pixel in the 35-unit window 25 < S <= 60 was
# simultaneously "confidently paper" (accepted by `_detect_and_crop_paper`) and
# "confidently grain" (the segmentation OR term). Any global colour cast that
# lifted the sheet past S=25 therefore turned the whole sheet into foreground.
# Measured relative to the observed paper level, the term now moves with the
# cast instead of being crossed by it. 25 is the same number as before and
# preserves the old behaviour on neutral paper, where S is ~3.
GRAIN_SATURATION_MARGIN_OVER_PAPER = 25

# Illumination flat-fielding. The background field is a heavy grayscale
# morphological close: closing removes dark structures narrower than the kernel,
# so a kernel several grain-widths across erases the grains and leaves only the
# paper's own lighting (vignetting, the photographer's shadow). Sized as a
# fraction of the shortest side so it holds at any capture resolution.
GRAIN_BACKGROUND_KERNEL_FRACTION = 0.06
GRAIN_BACKGROUND_KERNEL_CAP = 101
# The closed field is blocky at kernel scale; a smaller blur makes it a smooth
# surface before it is divided out.
GRAIN_BACKGROUND_SMOOTH_FRACTION = 0.02
GRAIN_BACKGROUND_SMOOTH_CAP = 41
# Otsu on the flattened image always splits *something*, so this floor keeps a
# grain-free sheet (sensor noise only) from being split down the middle: a
# foreground pixel must sit at least this many levels below the flattened paper
# level, which after flat-fielding is ~255 everywhere.
GRAIN_FLATFIELD_MIN_DROP = 12
# Solidity is a concavity test and scores a ring/donut shadow ~0.98, so hollow
# artifacts need their own rule: reject a candidate whose interior holes cover
# more than this fraction of its outer contour area. Well above the pinhole a
# specular highlight punches in a real grain, well below a shadow ring.
GRAIN_MAX_HOLE_AREA_RATIO = 0.15
# A thin shadow or fold line near the paper edge can survive segmentation as a
# sliver just inside the crop boundary rather than touching it, so a strict
# edge-touch test alone does not catch it. Measured on two real photos: every
# grain that passed every other check had a bounding-box aspect ratio of at
# most 2.3, while paper-edge shadow slivers found within a few pixels of the
# crop boundary measured 7.0-26.7 -- not remotely grain-shaped. A candidate
# this close to the edge and this elongated is treated the same as one that
# touches the edge outright: possibly clipped, and not trustworthy at any
# area. Expressed in millimetres so it holds at any photo resolution.
GRAIN_BORDER_ARTIFACT_MARGIN_MM = 5.0
GRAIN_MAX_PLAUSIBLE_ASPECT_RATIO = 4.0

Ellipse = tuple[tuple[float, float], tuple[float, float], float]


class ImageLoadError(Exception):
    """Raised when OpenCV cannot load the supplied image path."""


class PaperNotDetectedError(Exception):
    """Raised when the white paper region cannot be confidently isolated."""


class CoinNotDetectedError(Exception):
    """Raised when no plausible 10 Rupee coin contour is found."""


class CoinTiltedError(Exception):
    """Raised when the fitted coin ellipse is too elliptical for calibration."""


class TooFewGrainsError(Exception):
    """Raised when too few valid grain contours remain after filtering."""


def _load_image(image_path: str) -> np.ndarray:
    """Load a JPEG/PNG image from disk without resizing."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"Could not load a readable JPEG/PNG image from '{image_path}'.")
    return image


def _scaled_odd_size(shape: tuple[int, int], fraction: float, cap: int) -> int:
    """Scale an odd kernel size to the image so behaviour is resolution independent."""
    shortest_side = max(1, min(shape[:2]))
    size = int(round(shortest_side * fraction))
    size = max(3, min(cap, size))
    if size % 2 == 0:
        size += 1
    return size


def _scaled_odd_kernel(shape: tuple[int, int], fraction: float, cap: int) -> np.ndarray:
    """Build a small odd-sized elliptical morphology kernel for the image size."""
    size = _scaled_odd_size(shape, fraction, cap)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _sheet_region_contour(
    bright_mask: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    seed_label: int,
    seed_contour: np.ndarray,
) -> np.ndarray:
    """
    Recover the whole sheet from a bright mask that creases have broken up.

    Two failures are repaired here, both measured on `wheatsheetcrumbled.jpeg`:

    1. Fragmentation. Crease shadows drop below `PAPER_VALUE_MIN` and split the
       sheet into separate bright components. Components that are substantial
       (at least `PAPER_FRAGMENT_MIN_AREA_RATIO` of the seed) and essentially
       inside the seed's reach are merged back in. The containment test is what
       keeps the tablecloth out: a bright patch of background sits outside the
       seed hull and is refused however large it is.

    2. Interior pockets. Even when the sheet stays connected, the surviving
       bright region snakes around the crease shadows, and filling only its
       external contour leaves pockets unfilled -- the reference coin was inside
       one of them. Taking the convex hull closes every pocket. A sheet of paper
       is convex, so this is the sheet's own shape, not a licence: the hull is
       always inside the axis-aligned bounding rectangle that the caller already
       crops to, so it can never reach anywhere the crop does not already go.

    The hull is still an extrapolation over the shadowed part, so it is accepted
    only when `PAPER_HULL_MIN_BRIGHT_SUPPORT` of it is backed by pixels that
    actually looked like paper. If it is not, the sheet-shaped assumption is
    wrong for this image and the filled seed contour is returned unchanged.
    """
    shape = bright_mask.shape[:2]
    seed_area = float(stats[seed_label, cv2.CC_STAT_AREA])

    seed_hull = cv2.convexHull(seed_contour)
    seed_hull_mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(seed_hull_mask, [seed_hull], -1, 255, thickness=-1)

    # Reach is scaled to the sheet, not to the frame, so the rule holds at any
    # capture resolution: a fragment may sit up to 5% of the sheet's own width
    # outside the seed hull and still be recognised as part of the same sheet.
    # Built on first need: on an intact sheet nothing is large enough to test
    # against it, and this is the second most expensive operation in the module.
    reach: np.ndarray | None = None

    hull_points = [seed_contour]
    for label in range(1, int(stats.shape[0])):
        if label == seed_label:
            continue
        fragment_area = float(stats[label, cv2.CC_STAT_AREA])
        # Cheap test first: only a handful of components are ever large enough
        # to be worth allocating a full-frame mask for.
        if fragment_area < seed_area * PAPER_FRAGMENT_MIN_AREA_RATIO:
            continue
        if reach is None:
            margin = max(
                3,
                int(round(math.sqrt(max(seed_area, 1.0)) * PAPER_FRAGMENT_MARGIN_FRACTION)),
            )
            reach = _dilate_by_radius(seed_hull_mask, margin)
        fragment = np.uint8(labels == label) * 255
        inside = float(np.count_nonzero(cv2.bitwise_and(fragment, reach)))
        if inside < fragment_area * PAPER_FRAGMENT_MIN_CONTAINMENT:
            continue
        fragment_contours, _ = cv2.findContours(
            fragment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if fragment_contours:
            hull_points.append(max(fragment_contours, key=cv2.contourArea))

    hull = cv2.convexHull(np.vstack(hull_points))
    hull_mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
    hull_area = float(np.count_nonzero(hull_mask))
    if hull_area <= 0:
        return seed_contour

    support = float(np.count_nonzero(cv2.bitwise_and(bright_mask, hull_mask))) / hull_area
    if support < PAPER_HULL_MIN_BRIGHT_SUPPORT:
        return seed_contour
    return hull


def _dilate_by_radius(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Dilate a binary mask by a disc of `radius`, in time independent of `radius`.

    Both places that need a disc this large -- the sheet's fragment reach and the
    coin search's opening -- want elements around 100 px across, and
    `cv2.morphologyEx` with an elliptical element that size costs hundreds of
    milliseconds per call (measured 421 ms for a single 107x107 opening on a
    1600x1100 frame, twice the entire rest of the module).

    A dilation by a disc of radius r is exactly "every pixel within Euclidean
    distance r of a set pixel", which is one distance transform and one compare,
    and costs the same whatever r is: measured 22 ms for the same 107x107
    opening, agreeing with `cv2.morphologyEx` on 99.991% of pixels (150 px of
    1.76M, all on the rasterised disc boundary). Same operation, one twentieth
    of the time.
    """
    if radius <= 0 or not np.count_nonzero(mask):
        return mask.copy() if radius <= 0 else np.zeros_like(mask)
    distance_to_mask = cv2.distanceTransform(np.uint8(255) - mask, cv2.DIST_L2, 5)
    return np.uint8(distance_to_mask <= radius) * 255


def _erode_by_radius(distance: np.ndarray, radius: int) -> np.ndarray:
    """
    Erode by a disc of `radius`, read straight off a precomputed distance map.

    The erosion of a set by a disc of radius r is precisely the pixels whose
    distance to the outside is at least r, so where the distance transform has
    already been computed the erosion is free.
    """
    return np.uint8(distance >= radius) * 255


def _detect_and_crop_paper(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect the white paper, recover the whole sheet, and crop to it.

    The returned mask is the filled paper region inside the crop. This keeps the
    grains and coin inside the usable paper region even though they are not
    white pixels themselves -- including when crease shadows have carved the
    bright evidence into fragments and pockets, which is what
    `_sheet_region_contour` repairs.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    paper_pixels = cv2.inRange(
        hsv,
        np.array([0, 0, PAPER_VALUE_MIN], dtype=np.uint8),
        np.array([179, PAPER_SATURATION_MAX, 255], dtype=np.uint8),
    )

    close_kernel = _scaled_odd_kernel(image.shape[:2], 0.01, 31)
    open_kernel = _scaled_odd_kernel(image.shape[:2], 0.003, 9)
    cleaned = cv2.morphologyEx(paper_pixels, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    if label_count <= 1:
        raise PaperNotDetectedError("White paper could not be detected in the image.")

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(component_areas)) + 1
    best_area = int(stats[best_label, cv2.CC_STAT_AREA])
    image_area = int(image.shape[0] * image.shape[1])
    # Assumption: a confident sheet occupies at least 15% of the input frame.
    if best_area < image_area * PAPER_MIN_AREA_RATIO:
        raise PaperNotDetectedError(
            "White paper region is too small or weak to measure grains reliably."
        )

    component_mask = np.uint8(labels == best_label) * 255
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise PaperNotDetectedError("White paper contour could not be recovered.")

    seed_contour = max(contours, key=cv2.contourArea)
    paper_contour = _sheet_region_contour(cleaned, labels, stats, best_label, seed_contour)
    x, y, width, height = cv2.boundingRect(paper_contour)
    if width <= 0 or height <= 0:
        raise PaperNotDetectedError("White paper crop has invalid dimensions.")

    full_paper_region = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(full_paper_region, [paper_contour], -1, 255, thickness=-1)

    cropped_image = image[y : y + height, x : x + width].copy()
    cropped_mask = full_paper_region[y : y + height, x : x + width].copy()
    return cropped_image, cropped_mask


def _contour_circularity(contour: np.ndarray) -> float:
    """Compute contour circularity, where 1.0 is a perfect circle."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0 or perimeter <= 0:
        return 0.0
    return float((4.0 * math.pi * area) / (perimeter * perimeter))


def _contour_solidity(contour: np.ndarray) -> float:
    """
    Compute contour solidity: contour area divided by its convex hull area.

    A wheat grain is a compact, near-convex blob and scores close to 1.0. Cast
    shadows and merged edge artifacts are concave or ragged and score lower, so
    solidity separates them without depending on color or absolute size.
    """
    area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    if area <= 0 or hull_area <= 0:
        return 0.0
    return area / hull_area


def _fill_contour_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill every enclosed hole in a binary mask by redrawing its outer contours.

    A coin is not a solid blob in the foreground mask: the Rs 10 coin is
    bimetallic and its brighter inner disc, plus the relief of the design, drop
    out, leaving an annulus. That matters twice over. The distance transform
    below reads the annulus's *wall thickness* instead of the coin's radius
    (measured 26 px against a true 53 px radius on the flat sheet), and at high
    resolution the opening that follows then broke the ring apart entirely.
    Filling first makes the coin the solid disc it physically is, so the
    distance transform recovers its true radius.

    This never joins two separate objects -- only regions already enclosed by a
    single outer contour are filled -- and it is confined to the coin search.
    Grain segmentation keeps the unfilled mask, because it needs those holes:
    `GRAIN_MAX_HOLE_AREA_RATIO` is how ring-shaped shadow artifacts are
    recognised.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return filled


def _hough_circles(
    gray: np.ndarray, min_area: float, max_area: float
) -> list[tuple[float, float, float]]:
    """
    Find circles by gradient voting, over the radius window the coin gates imply.

    This is an independent second opinion on the contour search: it works from
    image gradients and never looks at the foreground mask, so the two methods
    fail in unrelated ways. The radius window is derived from the same area
    gates the contour candidates must satisfy, which means Hough is structurally
    incapable of proposing a grain-sized circle -- it is a corroborator, never a
    source of coin candidates.
    """
    if min(gray.shape[:2]) < 5:
        return []

    min_radius = max(3, int(round(math.sqrt(max(min_area, 1.0) / math.pi))))
    max_radius = max(min_radius + 1, int(round(math.sqrt(max(max_area, 1.0) / math.pi))))
    circles = cv2.HoughCircles(
        cv2.medianBlur(gray, 5),
        cv2.HOUGH_GRADIENT,
        dp=COIN_HOUGH_DP,
        minDist=max(8, min_radius),
        param1=COIN_HOUGH_PARAM1,
        param2=COIN_HOUGH_PARAM2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    return [(float(c[0]), float(c[1]), float(c[2])) for c in circles[0]]


def _matching_hough_circle(
    circles: list[tuple[float, float, float]],
    center_x: float,
    center_y: float,
    radius: float,
) -> tuple[float, float, float] | None:
    """Return the first Hough circle that agrees with a contour candidate."""
    if radius <= 0:
        return None
    for hough_x, hough_y, hough_radius in circles:
        if math.hypot(hough_x - center_x, hough_y - center_y) > radius * COIN_HOUGH_CENTER_TOLERANCE:
            continue
        radius_ratio = hough_radius / radius
        if COIN_HOUGH_MIN_RADIUS_RATIO <= radius_ratio <= COIN_HOUGH_MAX_RADIUS_RATIO:
            return (hough_x, hough_y, hough_radius)
    return None


def _find_best_coin_candidate(
    image: np.ndarray, paper_mask: np.ndarray, foreground: np.ndarray
) -> dict[str, Any]:
    """
    Locate the reference coin on the cropped paper and calibrate the pixel scale.

    The search runs on a mask that morphology has NOT bridged. The previous
    version searched a mask that had already been through `MORPH_CLOSE` with two
    iterations, which welded the coin to the grains lying beside it: measured on
    `wheatsheetplain.jpeg`, the coin's bounding box grew from 111x109 to 141x140
    and its circularity collapsed from 0.878 to 0.494. The coin was then thrown
    out, a wheat grain won the ranking, and the grain's 0.59 axis ratio was
    reported as a tilted coin. Closing is gone from this path entirely; the only
    morphology here is an opening, which removes thin bridges instead of
    creating them.

    That opening is also the scale search. Its element is a disc sized to a
    fraction of the largest disc that fits inside the foreground, and the
    largest disc that fits inside anything on a sheet of wheat is the coin. So
    the element is always about half the coin's radius: no grain and no bridge
    between grains can contain it and they are erased, while the coin, being
    convex and far larger than the element, survives an opening unchanged.
    Nothing needs to know the coin's size in advance.

    Four independent conditions must then hold before a contour is accepted as
    the coin:

    * area against the sheet (`COIN_MIN_AREA_RATIO`, geometric);
    * area against the other objects in this image
      (`COIN_MIN_POPULATION_AREA_MULTIPLE`, measured 27.7-34.4x for a real coin
      against at most 2x for a single grain);
    * shape (`COIN_MIN_CIRCULARITY`);
    * agreement with an independent Hough circle detector.

    A single grain fails the second by 4x on its own. A large fused clump of
    grains can clear the area gates -- measured 10.0x on the crumbled sheet --
    and is stopped by the other three: verified on both photographs with the
    coin digitally removed, no blob passes both area and circularity, none is
    Hough-corroborated, and none reaches the 0.95 axis ratio. No single gate
    carries this; the conjunction does, which is why all four are kept.

    Only contours that pass all four are considered coin-like, and only then is
    tilt assessed. That ordering is deliberate: it is what makes
    `CoinTiltedError` mean "your coin is tilted" instead of "the best of several
    wheat grains happened to be elongated".
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    solid = _fill_contour_holes(foreground)

    # The population of everything on the sheet, measured before the opening
    # removes the grains -- the grains ARE the population this coin is sized
    # against.
    population, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    population_areas = [
        area for area in (float(cv2.contourArea(c)) for c in population) if area > 0
    ]

    paper_area = max(float(np.count_nonzero(paper_mask)), 1.0)
    min_coin_area = max(100.0, paper_area * COIN_MIN_AREA_RATIO)
    max_coin_area = paper_area * COIN_MAX_AREA_RATIO
    if len(population_areas) >= COIN_POPULATION_MIN_BLOBS:
        min_coin_area = max(
            min_coin_area,
            float(np.median(population_areas)) * COIN_MIN_POPULATION_AREA_MULTIPLE,
        )

    distance = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    largest_inscribed_radius = float(distance.max()) if distance.size else 0.0
    open_radius = max(1, int(round(largest_inscribed_radius * COIN_SEARCH_OPEN_RADIUS_FRACTION)))
    # Opening by a disc, spelled out as its definition -- erode then dilate --
    # so it can reuse the distance transform already computed above instead of
    # paying for a ~100 px elliptical structuring element.
    unbridged = _dilate_by_radius(_erode_by_radius(distance, open_radius), open_radius)

    contours, _ = cv2.findContours(unbridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, float, np.ndarray, Ellipse]] = []
    for contour in contours:
        if len(contour) < 5:
            continue

        area = float(cv2.contourArea(contour))
        if area < min_coin_area or area > max_coin_area:
            continue

        circularity = _contour_circularity(contour)
        if circularity < COIN_MIN_CIRCULARITY:
            continue

        ellipse = cv2.fitEllipse(contour)
        _, axes, _ = ellipse
        minor_axis, major_axis = sorted(float(axis) for axis in axes)
        if minor_axis <= 0 or major_axis <= 0:
            continue

        candidates.append((area, circularity, contour, ellipse))

    if not candidates:
        raise CoinNotDetectedError("No coin-like circular contour was detected on the paper.")

    # Largest first, circularity breaking ties. Area, not circularity, leads:
    # after the opening the coin outweighs the next surviving blob by at least
    # 5x on every image measured, whereas its circularity lead can be as thin as
    # 0.014 -- ranking on the 5x margin is the safer of the two.
    candidates.sort(key=lambda candidate: (-candidate[0], -candidate[1]))

    # Everything in `candidates` is already the right size and the right shape
    # to be a coin, so the top-ranked one (same ordering the acceptance loop
    # below uses) decides which failure to report if none is finally accepted.
    # This must be the TOP-RANKED candidate's own axis ratio, not the best axis
    # ratio across every candidate: an unrelated, lower-ranked blob that
    # happens to be rounder than the real (tilted) coin would otherwise mask
    # the coin's own tilt and misreport CoinNotDetectedError instead of
    # CoinTiltedError. Verified on both photographs with the coin removed: not
    # one blob gets this far, so a wheat grain cannot reach the tilt verdict.
    _, _, _, top_ellipse = candidates[0]
    top_minor_axis, top_major_axis = sorted(float(axis) for axis in top_ellipse[1])
    best_tilt_ratio = top_minor_axis / top_major_axis if top_major_axis > 0 else 0.0

    # Computed once, lazily: Hough is the expensive step and there is nothing to
    # corroborate unless a contour candidate survived the gates above.
    hough = _hough_circles(gray, min_coin_area, max_coin_area)

    accepted = None
    for area, circularity, contour, ellipse in candidates:
        (center_x, center_y), axes, _ = ellipse
        minor_axis, major_axis = sorted(float(axis) for axis in axes)
        # Deliberate: reject rather than guess. Every downstream measurement is
        # divided by this coin's diameter, so accepting an uncorroborated
        # candidate does not risk one bad number, it silently rescales the whole
        # report. Hough locates the true coin unaided in both real photographs,
        # so its silence here is evidence, not an outage. Note also what is NOT
        # done: Hough is never allowed to *create* a coin the contour search did
        # not find, because a Hough circle carries no boundary to measure -- its
        # radius is a vote peak, and on a bimetallic coin it lands on the inner
        # rim as often as the outer one.
        radius = (minor_axis + major_axis) / 4.0
        if _matching_hough_circle(hough, float(center_x), float(center_y), radius) is None:
            continue
        tilt_ratio = minor_axis / major_axis if major_axis > 0 else 0.0
        if tilt_ratio < COIN_TILT_MIN_AXIS_RATIO:
            continue
        accepted = (circularity, area, contour, ellipse, minor_axis, major_axis, tilt_ratio)
        break

    if accepted is None:
        # Nothing was accepted; the two failures say genuinely different things
        # and the user needs to be told which one happened.
        #
        # Tilt is judged on shape and size alone, without requiring Hough to
        # agree. That is not a loophole, it is the only way the message can be
        # right: `cv2.HoughCircles` votes for circles, so a coin tilted far
        # enough to fail this gate is precisely the coin Hough cannot see, and
        # demanding its corroboration would report every tilted coin as a
        # missing one. Leniency here costs nothing, because it decides only
        # which message is shown -- a candidate still cannot become the scale
        # reference without Hough agreeing, and this branch never returns a
        # measurement. Measured on both photographs with the coin removed, no
        # blob passes the size and shape gates at all, so a wheat grain cannot
        # reach this verdict.
        if best_tilt_ratio < COIN_TILT_MIN_AXIS_RATIO:
            raise CoinTiltedError(
                f"Reference coin is too tilted: axis ratio {best_tilt_ratio:.3f} is below "
                f"{COIN_TILT_MIN_AXIS_RATIO:.3f}."
            )
        raise CoinNotDetectedError(
            "A coin-like contour was found but no independent circle detection "
            "agreed with it, so the scale reference could not be trusted. "
            "Photograph the sheet straight down with the coin fully visible."
        )

    circularity, area, contour, ellipse, minor_axis, major_axis, tilt_ratio = accepted

    # Mean axes are less noisy than either single fitted axis when the coin edge
    # has small segmentation chips, while the tilt check preserves top-down use.
    diameter_px = (major_axis + minor_axis) / 2.0
    px_per_mm = diameter_px / COIN_DIAMETER_MM

    # Circular hand-off of the already-accepted ellipse, in the SAME cropped-image
    # coordinates as `_cropped_image` and `_grain_contours`. Derived, never
    # re-detected: the centre is the fitted ellipse centre and the radius is half
    # the same mean diameter `px_per_mm` is computed from, so neither detection
    # nor calibration is affected. A circle is the right shape to hand on because
    # the tilt check above has already guaranteed the axis ratio is >= 0.95.
    (center_x, center_y), _, _ = ellipse
    center = (int(round(float(center_x))), int(round(float(center_y))))
    radius = max(1, int(round(diameter_px / 2.0)))

    return {
        "contour": contour,
        "ellipse": ellipse,
        "center": center,
        "radius": radius,
        "diameter_px": float(diameter_px),
        "px_per_mm": float(px_per_mm),
        "tilt_ratio": float(tilt_ratio),
        "circularity": float(circularity),
        "area_px": float(area),
    }


def _create_coin_mask(
    shape: tuple[int, int],
    ellipse: Ellipse,
    diameter_px: float,
) -> np.ndarray:
    """Create a filled mask around the accepted coin ellipse."""
    (center_x, center_y), axes, angle = ellipse
    padding = max(4.0, diameter_px * 0.04)
    axis_x = max(1, int(round((float(axes[0]) / 2.0) + padding)))
    axis_y = max(1, int(round((float(axes[1]) / 2.0) + padding)))
    center = (int(round(center_x)), int(round(center_y)))

    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, (axis_x, axis_y), float(angle), 0, 360, 255, thickness=-1)
    return mask


def _flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """
    Divide the paper's illumination field out of the grayscale image.

    A heavy grayscale morphological close erases every structure darker and
    narrower than its kernel, so with a kernel several grain-widths across the
    result is the lighting on the sheet with the grains removed. Dividing the
    original by that field renormalizes the paper to ~255 everywhere, which is
    what lets a single global threshold survive vignetting and cast shadows.
    """
    background_size = _scaled_odd_size(
        gray.shape[:2], GRAIN_BACKGROUND_KERNEL_FRACTION, GRAIN_BACKGROUND_KERNEL_CAP
    )
    # A rectangular element, not the elliptical one used elsewhere: OpenCV runs
    # rectangular morphology as two separable 1-D passes, which at this kernel
    # size is an order of magnitude cheaper than the equivalent ellipse. What
    # matters here is only that no grain can contain the element, and a square
    # this wide cannot fit inside a grain either.
    background_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (background_size, background_size)
    )
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, background_kernel, iterations=1)

    smooth_size = _scaled_odd_size(
        gray.shape[:2], GRAIN_BACKGROUND_SMOOTH_FRACTION, GRAIN_BACKGROUND_SMOOTH_CAP
    )
    background = cv2.GaussianBlur(background, (smooth_size, smooth_size), 0)
    # Guard the division; a zero here would saturate the pixel to foreground.
    background = np.maximum(background, np.uint8(1))
    return cv2.divide(gray, background, scale=255.0)


def _flatfield_foreground(gray: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    """
    Threshold the flat-fielded image into a foreground mask.

    Otsu, not `cv2.adaptiveThreshold`: flat-fielding has already removed the
    shading an adaptive block exists to absorb, and any block small enough to
    sit between two grains would also carve the interior out of every grain.
    After flattening, grain darkness is an absolute quantity again, so one
    global split is both valid and stable. The split point is computed over
    paper pixels only, so objects lying off the sheet cannot drag it.

    This replaces the former fixed `value < 235` cutoff, which had a hard cliff:
    a few levels of extra shading turned the paper itself into one giant blob.
    """
    flattened = _flatten_illumination(gray)

    paper_pixels = flattened[paper_mask > 0]
    if paper_pixels.size == 0:
        return np.zeros(gray.shape[:2], dtype=np.uint8)

    otsu_threshold, _ = cv2.threshold(
        paper_pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Otsu always splits something, even a grain-free sheet full of sensor noise.
    # Requiring a minimum drop below the flattened paper level keeps that case
    # empty instead of half-white.
    threshold = min(float(otsu_threshold), 255.0 - GRAIN_FLATFIELD_MIN_DROP)

    _, foreground = cv2.threshold(flattened, threshold, 255, cv2.THRESH_BINARY_INV)
    return foreground


def _saturation_foreground(hsv: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    """
    Threshold saturation against the sheet's own saturation, not an absolute level.

    Colour still matters after flat-fielding: an object can be as bright as paper
    and still not be paper, so a saturation term stays ORed into the foreground.
    But the old form, `S > 25`, was an absolute cutoff sitting *inside* the range
    `_detect_and_crop_paper` accepts as paper (`S <= PAPER_SATURATION_MAX`, 60).
    A warm bulb that lifted the sheet from S=22 to S=27 -- a mild, ordinary cast --
    therefore satisfied both rules at once and marked the entire sheet foreground.
    Flat-fielding could not rescue it because the term is ORed in afterwards.

    Two things make that impossible now:

    1. The cutoff is measured relative to the paper actually in this image. The
       median saturation over the paper mask is the sheet's own level: grains and
       the coin lie inside that mask but are a small minority of its area, so the
       median tracks the paper and ignores them (a mean would be dragged by them).
       A colour cast moves the paper and the cutoff together instead of the cast
       sliding the paper across a fixed line.
    2. The cutoff can never drop below `PAPER_SATURATION_MAX`. That is the
       structural half: paper detection accepts a pixel only when `S <= 60`, and
       this rule calls a pixel foreground only when `S > cutoff >= 60`, so no
       pixel can satisfy both rules however the image is lit. The two constants
       can no longer disagree about what white paper is.

    Real wheat is far outside the window either way -- the synthetic grains here
    sit near S=174, and dull real grain well above 60 -- so the term keeps doing
    its job for genuinely coloured objects while losing the ability to swallow
    the sheet.
    """
    saturation = hsv[:, :, 1]
    paper_saturation = saturation[paper_mask > 0]
    if paper_saturation.size == 0:
        return np.zeros(saturation.shape[:2], dtype=np.uint8)

    observed_paper_level = float(np.median(paper_saturation))
    cutoff = max(
        float(PAPER_SATURATION_MAX),
        observed_paper_level + GRAIN_SATURATION_MARGIN_OVER_PAPER,
    )
    return np.uint8(saturation > cutoff) * 255


def _foreground_mask(image: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    """
    Everything on the sheet that is not the sheet: the coin, the grains, debris.

    One definition, two consumers. The coin search and the grain segmentation
    used to derive their own foregrounds by different rules, and the coin's rule
    -- "brighter-than-paper saturation OR darker than `PAPER_VALUE_MIN`" -- was
    what put the crumbled sheet beyond reach: crease shadows are darker than
    `PAPER_VALUE_MIN` too, so the coin fused into a single 190,000 px shadow
    network spanning most of the sheet. Flat-fielding removes exactly that
    low-frequency shading, which is why the grain path already used it, so both
    stages now share it.

    What the two stages do NOT share is the morphology applied afterwards, and
    that separation is the point. The coin search needs holes filled and thin
    bridges opened away, at coin scale. Grain segmentation needs its holes
    intact (they are how ring artifacts are spotted) and a gentle 3x3 clean-up,
    at grain scale. Same evidence, different questions asked of it.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flat_foreground = _flatfield_foreground(gray, paper_mask)
    # Colour still matters: a saturated object is not paper even when it is as
    # bright as paper, so a saturation term stays ORed in -- but measured against
    # this sheet's own saturation, so a colour cast cannot flip the whole sheet.
    saturation_foreground = _saturation_foreground(hsv, paper_mask)

    foreground = cv2.bitwise_or(flat_foreground, saturation_foreground)
    return cv2.bitwise_and(foreground, paper_mask)


def _segment_foreground_contours(
    foreground: np.ndarray,
    coin_mask: np.ndarray,
) -> list[tuple[np.ndarray, float]]:
    """
    Segment foreground contours on the white paper after excluding the coin.

    Takes the shared `_foreground_mask` result and applies the grain-scale
    morphology to it. The pixel rule is unchanged from before the coin search
    was split out; only the point at which it is computed moved, so the same
    image is no longer flat-fielded twice.

    Returns one `(outer_contour, interior_hole_area_px)` pair per candidate.
    Contours are retrieved with `RETR_CCOMP` so hollow artifacts (ring shadows,
    grains with an unfilled specular highlight) can be recognised by their holes;
    the hole contours themselves are never returned as candidates.

    This intentionally uses connected foreground contours only; touching or
    overlapping grains are excluded later by area and are not watershed-split.
    """
    # Copied, not modified in place: the caller shares this mask with the coin
    # search and must not have the coin punched out of its copy.
    foreground = foreground.copy()
    foreground[coin_mask > 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        foreground, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None or not contours:
        return []

    # RETR_CCOMP gives a two-level hierarchy of [next, previous, child, parent]:
    # outer boundaries have parent == -1, holes carry their parent's index.
    hierarchy = hierarchy[0]
    candidates: list[tuple[np.ndarray, float]] = []
    for index, contour in enumerate(contours):
        if hierarchy[index][3] != -1:
            continue

        hole_area_px = 0.0
        child = int(hierarchy[index][2])
        while child != -1:
            hole_area_px += float(cv2.contourArea(contours[child]))
            child = int(hierarchy[child][0])

        candidates.append((contour, hole_area_px))
    return candidates


def _create_contour_mask(contour: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Create a filled binary mask for one contour."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    return mask


def _passes_wheat_color_check(hsv_image: np.ndarray, contour: np.ndarray) -> bool:
    """
    Reuse the existing HSV foreign-matter rule for grain-sized candidates.

    The current defect module classifies grain-sized foreign matter by mean hue
    outside the healthy wheat hue range with a +/-10 tolerance.

    The caller converts the cropped image to HSV once and passes it in, and the
    mask is confined to the contour's bounding rect: converting and allocating a
    full frame per contour dominated the pipeline's runtime. The pixel set tested
    and the test applied to it are unchanged.
    """
    x, y, width, height = cv2.boundingRect(contour)
    if width <= 0 or height <= 0:
        return False

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1, offset=(-x, -y))
    hue_pixels = hsv_image[y : y + height, x : x + width, 0][mask > 0]
    if hue_pixels.size == 0:
        return False

    mean_hue = float(hue_pixels.mean())
    return (HSV_HEALTHY_H_MIN - 10) <= mean_hue <= (HSV_HEALTHY_H_MAX + 10)


def _measure_contour(contour: np.ndarray, px_per_mm: float) -> dict[str, float] | None:
    """Fit an ellipse to one contour and convert dimensions into millimeters."""
    if len(contour) < 5 or px_per_mm <= 0:
        return None

    ellipse = cv2.fitEllipse(contour)
    _, axes, _ = ellipse
    minor_axis, major_axis = sorted(float(axis) for axis in axes)
    if minor_axis <= 0 or major_axis <= 0:
        return None

    length_mm = major_axis / px_per_mm
    width_mm = minor_axis / px_per_mm
    area_mm2 = float(cv2.contourArea(contour)) / (px_per_mm * px_per_mm)
    aspect_ratio = length_mm / width_mm if width_mm > 0 else 0.0
    return {
        "length_mm": float(length_mm),
        "width_mm": float(width_mm),
        "area_mm2": float(area_mm2),
        "aspect_ratio": float(aspect_ratio),
    }


def _summary_stats(values: list[float]) -> dict[str, float]:
    """Compute population summary statistics for JSON output."""
    if not values:
        # Unreachable by construction: `_validate_grain_count` requires
        # `measured_count >= MIN_VALID_GRAIN_COUNT` (30) before this is called,
        # and these are the only two call sites. Kept as a backstop with teeth,
        # because the failure it replaces was a `ValueError` from `np.min` on an
        # empty array -- an exception the pipeline does not map, which escaped to
        # the API as an HTTP 500. Raising a declared exception instead means even
        # a future refactor that breaks the invariant degrades to a clean 422
        # rather than a stack trace.
        raise TooFewGrainsError(
            "No grains could be measured individually, so no grain statistics "
            "can be reported. Spread the grains out on the sheet so none of them "
            "touch, then take the photo again."
        )
    array = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _validate_grain_count(
    measured_count: int,
    excluded_touching: int,
    estimated_touching_grains: int,
) -> None:
    """
    Validate the number of individually measured grains.

    There is no upper bound: a sheet with more grain than usual is not an error
    and reports every grain that could be measured reliably.

    Only `measured_count` gates this, not any estimate of what touching clusters
    might have contained. Clusters, ring/hole artifacts, low-solidity shadows,
    border-clipped contours and off-hue blobs cannot be measured reliably and
    are excluded outright rather than counted or estimated into the total; this
    validates exactly the population that `grains` and `summary` report. Because
    the threshold applies to `measured_count` directly, `_summary_stats` can
    never be called with fewer than `MIN_VALID_GRAIN_COUNT` values -- the
    previous failure mode, an unmapped `ValueError` from an empty list reaching
    the pipeline as an HTTP 500, is structurally impossible here.

    When many grains were excluded as touching clusters, the message says so:
    the sheet likely holds enough grain but poured in clumps, which needs a
    different fix (spread it out) than the generic "not enough grain" case.
    """
    if measured_count >= MIN_VALID_GRAIN_COUNT:
        return

    if excluded_touching > 0:
        raise TooFewGrainsError(
            f"Only {measured_count} grains could be measured individually; "
            f"expected at least {MIN_VALID_GRAIN_COUNT}. {excluded_touching} "
            f"touching cluster(s), an estimated {estimated_touching_grains} "
            f"grains, were excluded because they could not be measured "
            f"reliably. Spread the grains out into a single layer with a gap "
            f"between them and take the photo again."
        )
    raise TooFewGrainsError(
        f"Only {measured_count} grains were detected on the sheet; expected at "
        f"least {MIN_VALID_GRAIN_COUNT}."
    )


def _classify_grain_candidates(
    candidates: list[tuple[np.ndarray, float]],
    cropped_hsv: np.ndarray,
    coin: dict[str, Any],
    crop_width: int,
    crop_height: int,
) -> dict[str, Any]:
    """
    Sort segmentation candidates into measurable grains, foreign matter, and
    the exclusion buckets that keep the grain count trustworthy.

    This is the classification loop `extract_grain_masks` runs after
    `_segment_foreground_contours`; pulled out on its own so that function's
    contract is just "candidates in, filtered contours out".
    """
    px_per_mm2 = coin["px_per_mm"] * coin["px_per_mm"]
    touching_threshold_mm2 = NORMAL_GRAIN_AREA_MM2 * TOUCHING_GRAIN_AREA_MULTIPLIER
    border_margin_px = GRAIN_BORDER_ARTIFACT_MARGIN_MM * coin["px_per_mm"]

    grain_contours: list[np.ndarray] = []
    excluded_touching = 0
    estimated_touching_grains = 0
    touching_cluster_area_mm2 = 0.0
    excluded_low_solidity = 0
    excluded_ring = 0
    excluded_border = 0
    foreign_matter_area_mm2 = 0.0
    foreign_matter_count = 0

    for contour, hole_area_px in candidates:
        # The crop is tight to the paper contour, so a candidate can be
        # uncountable near the border in two distinct ways. First, literally
        # clipped: its bounding box touches the crop edge, so it is only
        # partially in view and its true area and shape outside the frame are
        # unknown -- dropped at any shape or area. Second, a thin shadow or
        # fold line can survive segmentation just inside the boundary without
        # touching it; a strict touch test alone does not catch that, so a
        # candidate within GRAIN_BORDER_ARTIFACT_MARGIN_MM of the edge is also
        # dropped if its bounding-box aspect ratio is far beyond anything a
        # grain measured on it. Either way, the exclusion is the same: not
        # counted, not measured, not estimated into any cluster total.
        x, y, w, h = cv2.boundingRect(contour)
        touches_edge = x <= 0 or y <= 0 or x + w >= crop_width or y + h >= crop_height
        edge_distance_px = min(x, y, crop_width - (x + w), crop_height - (y + h))
        aspect_ratio_bbox = max(w, h) / max(1, min(w, h))
        near_edge_implausible_shape = (
            edge_distance_px < border_margin_px
            and aspect_ratio_bbox > GRAIN_MAX_PLAUSIBLE_ASPECT_RATIO
        )
        if touches_edge or near_edge_implausible_shape:
            excluded_border += 1
            continue

        area_px = float(cv2.contourArea(contour))
        if area_px <= 0:
            continue

        area_mm2 = area_px / px_per_mm2
        # Two genuinely distinct bands. Below 2 mm2 is sensor noise and dust: it
        # is discarded and deliberately accounted nowhere. Between 2 and 8 mm2 is
        # foreign matter, which the pipeline needs as an area so it can express
        # foreign_matter_pct against the paper area.
        if area_mm2 < FOREIGN_MATTER_MIN_AREA_MM2:
            continue
        if area_mm2 < FOREIGN_MATTER_MAX_AREA_MM2:
            foreign_matter_area_mm2 += area_mm2
            foreign_matter_count += 1
            continue
        if area_mm2 > touching_threshold_mm2:
            # Oversized does not automatically mean "grain cluster" -- a clump
            # of debris or a stone can just as easily clear this area alone.
            # Colour is what tells the two apart: a wheat-coloured oversized
            # blob is grains touching each other, an off-hue one is foreign
            # matter that happens to be bigger than the 2-8 mm2 band below.
            if not _passes_wheat_color_check(cropped_hsv, contour):
                foreign_matter_area_mm2 += area_mm2
                foreign_matter_count += 1
                continue
            # excluded_touching counts CLUSTERS, per the output contract. The
            # grains hidden inside them are estimated separately, for count
            # validation only, and never enter `grains` or `summary.count`.
            # The area itself IS real wheat, though -- unlike the ring/
            # low-solidity/border buckets below, which are segmentation
            # artifacts with no physical grain behind them -- so it is tracked
            # separately for callers computing a sample-area denominator (e.g.
            # foreign_matter_pct), which would otherwise undercount the sample
            # by exactly the clustered grain and read artificially high.
            excluded_touching += 1
            estimated_touching_grains += max(
                2, int(round(area_mm2 / NORMAL_GRAIN_AREA_MM2))
            )
            touching_cluster_area_mm2 += area_mm2
            continue
        # Hollow candidates are ring shadows or highlight rings, not grains.
        # Solidity cannot see them: their outer boundary is convex.
        if hole_area_px > area_px * GRAIN_MAX_HOLE_AREA_RATIO:
            excluded_ring += 1
            continue
        # Grain-sized but non-convex blobs are cast shadows or edge artifacts;
        # counting them inflates the grain count and skews the grade.
        if _contour_solidity(contour) < GRAIN_MIN_SOLIDITY:
            excluded_low_solidity += 1
            continue
        if not _passes_wheat_color_check(cropped_hsv, contour):
            # Grain-sized but off-hue: this is foreign matter, not a grain.
            foreign_matter_area_mm2 += area_mm2
            foreign_matter_count += 1
            continue

        grain_contours.append(contour)

    return {
        "grain_contours": grain_contours,
        "excluded_touching": int(excluded_touching),
        "estimated_touching_grains": int(estimated_touching_grains),
        "touching_cluster_area_mm2": float(touching_cluster_area_mm2),
        "excluded_low_solidity": int(excluded_low_solidity),
        "excluded_ring": int(excluded_ring),
        "excluded_border": int(excluded_border),
        "foreign_matter_area_mm2": float(foreign_matter_area_mm2),
        "foreign_matter_count": int(foreign_matter_count),
    }


def calibrate_image(image: np.ndarray) -> dict[str, Any]:
    """
    Calibration layer: detect the paper sheet, build the shared foreground
    evidence, and locate + validate the reference coin.

    Kept as one function rather than split further, because the coin search
    and grain segmentation both consume the same expensive foreground mask --
    see `_foreground_mask`'s docstring -- so splitting it apart would either
    duplicate that computation or leak an internal hand-off across a public
    boundary for no benefit.

    Returns the cropped image, the paper and foreground masks, the accepted
    coin's full geometry dict, a filled mask over the coin, and the paper area
    in mm2 (needed downstream for foreign-matter percentage).
    """
    cropped_image, paper_mask = _detect_and_crop_paper(image)
    # Computed once and shared: the coin search and the grain segmentation ask
    # different questions of the same foreground evidence, and flat-fielding it
    # is the most expensive step in the module.
    foreground = _foreground_mask(cropped_image, paper_mask)
    coin = _find_best_coin_candidate(cropped_image, paper_mask, foreground)
    coin_mask = _create_coin_mask(
        cropped_image.shape[:2],
        coin["ellipse"],
        coin["diameter_px"],
    )
    px_per_mm2 = coin["px_per_mm"] * coin["px_per_mm"]
    paper_area_mm2 = float(np.count_nonzero(paper_mask)) / px_per_mm2

    return {
        "cropped_image": cropped_image,
        "paper_mask": paper_mask,
        "foreground": foreground,
        "coin": coin,
        "coin_mask": coin_mask,
        "paper_area_mm2": float(paper_area_mm2),
    }


def extract_grain_masks(
    cropped_image: np.ndarray,
    foreground: np.ndarray,
    coin_mask: np.ndarray,
    coin: dict[str, Any],
) -> dict[str, Any]:
    """
    Extraction layer: segment the shared foreground into individual grain
    candidates and filter out everything that cannot be measured reliably.

    This is the function a future YOLOv8-Seg backend would replace outright --
    it is the only place in the module that decides "which pixels are one
    grain". `calculate_geometry` and the defect/grading steps downstream only
    ever consume its output (contours and masks), so swapping the classical
    thresholding/solidity/hole/border heuristics here for a trained
    instance-segmentation model requires no change anywhere else.

    Returns both contours and one binary mask per accepted grain -- contours
    are what this module and `detect_discoloration` already consume; masks are
    what a future per-grain colour or texture pass would want instead of
    redrawing the contour itself -- plus the same exclusion/foreign-matter
    counters `_measure_wheat_grains_from_image` used to compute inline.
    """
    candidates = _segment_foreground_contours(foreground, coin_mask)
    # Converted once for every colour check in the classification pass.
    cropped_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    shape = cropped_image.shape[:2]
    crop_height, crop_width = shape

    result = _classify_grain_candidates(candidates, cropped_hsv, coin, crop_width, crop_height)
    result["grain_masks"] = [
        _create_contour_mask(contour, shape) for contour in result["grain_contours"]
    ]
    return result


def calculate_geometry(grain_contours: list[np.ndarray], px_per_mm: float) -> dict[str, Any]:
    """
    Geometry layer: fit an ellipse to each accepted grain contour and convert
    its dimensions to millimetres.

    Reads only contour points and the calibrated scale, so it is indifferent
    to whether the contours came from classical segmentation or any future
    instance-segmentation backend.

    A contour can still fail here (too few points to fit an ellipse); when
    that happens it is dropped from `measured_contours` too, so that list
    always stays index-aligned with `grains` for callers -- such as the defect
    passes -- that need contour and measurement to match 1:1.
    """
    grains: list[dict[str, float | int]] = []
    measured_contours: list[np.ndarray] = []
    excluded_unmeasurable = 0

    for contour in grain_contours:
        measurement = _measure_contour(contour, px_per_mm)
        if measurement is None:
            # Too few points to fit an ellipse. Counted so the drop is at least
            # observable instead of an invisible under-count.
            excluded_unmeasurable += 1
            continue
        grains.append(
            {
                "id": len(grains) + 1,
                "length_mm": measurement["length_mm"],
                "width_mm": measurement["width_mm"],
                "area_mm2": measurement["area_mm2"],
                "aspect_ratio": measurement["aspect_ratio"],
            }
        )
        measured_contours.append(contour)

    lengths = [float(grain["length_mm"]) for grain in grains]
    widths = [float(grain["width_mm"]) for grain in grains]
    # `_summary_stats` raises on an empty list (see its docstring): here that
    # would fire before the caller's own, more specific `_validate_grain_count`
    # message, so an empty result reports zeroed stats instead and lets the
    # caller decide whether zero grains is an error.
    zero_stats = {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return {
        "grains": grains,
        "measured_contours": measured_contours,
        "excluded_unmeasurable": int(excluded_unmeasurable),
        "summary": {
            "count": int(len(grains)),
            "length_mm": _summary_stats(lengths) if grains else dict(zero_stats),
            "width_mm": _summary_stats(widths) if grains else dict(zero_stats),
        },
    }


def _measure_wheat_grains_from_image(image: np.ndarray) -> dict:
    """
    Measure wheat grains from an already decoded BGR image.

    Thin orchestrator over the three public layers above
    (`calibrate_image` -> `extract_grain_masks` -> `calculate_geometry`): it
    exists so the pipeline adapter gets one call and one dict with the
    underscore-prefixed hand-offs it needs, without every caller having to
    know the composition order.
    """
    calibration = calibrate_image(image)
    cropped_image = calibration["cropped_image"]
    coin = calibration["coin"]

    extraction = extract_grain_masks(
        cropped_image, calibration["foreground"], calibration["coin_mask"], coin
    )
    geometry = calculate_geometry(extraction["grain_contours"], coin["px_per_mm"])

    _validate_grain_count(
        geometry["summary"]["count"],
        extraction["excluded_touching"],
        extraction["estimated_touching_grains"],
    )

    return {
        # Underscore-prefixed diagnostics and hand-offs for in-process callers
        # (the pipeline adapter needs the accepted contours, the image they index
        # into, and the foreign-matter totals). The public entry point strips
        # every underscore key so the returned contract stays exactly as
        # specified.
        #
        # `measured_contours`, not `extraction["grain_contours"]`: geometry can
        # drop a contour that fails ellipse-fitting, and this keeps `grains`
        # and `_grain_contours` the same length and order for the defect passes.
        "_grain_contours": geometry["measured_contours"],
        "_cropped_image": cropped_image,
        # Coin geometry in CROPPED-image coordinates, the same frame as
        # `_cropped_image` and `_grain_contours`. A restored white-balance step
        # needs a centre and radius to sample the paper ring around the coin, and
        # the module previously exposed neither.
        "_coin_center": coin["center"],
        "_coin_radius": int(coin["radius"]),
        "_paper_area_mm2": calibration["paper_area_mm2"],
        "_foreign_matter_area_mm2": extraction["foreign_matter_area_mm2"],
        "_foreign_matter_count": extraction["foreign_matter_count"],
        "_touching_cluster_area_mm2": extraction["touching_cluster_area_mm2"],
        "_estimated_touching_grains": extraction["estimated_touching_grains"],
        "_excluded_low_solidity": extraction["excluded_low_solidity"],
        "_excluded_ring": extraction["excluded_ring"],
        "_excluded_unmeasurable": geometry["excluded_unmeasurable"],
        "_excluded_border": extraction["excluded_border"],
        "status": "ok",
        "coin": {
            "diameter_px": float(coin["diameter_px"]),
            "px_per_mm": float(coin["px_per_mm"]),
            "tilt_ratio": float(coin["tilt_ratio"]),
        },
        "grains": geometry["grains"],
        "excluded_touching": extraction["excluded_touching"],
        "summary": geometry["summary"],
    }


def measure_wheat_grains(image_path: str) -> dict:
    """
    Measure wheat grain dimensions from a top-down photograph.

    Steps:
    1. Load the image from disk at original resolution.
    2. Crop to the detected white paper sheet.
    3. Detect and validate the 10 Rupee coin, then compute pixel scale.
    4. Flat-field the illumination and segment grain-like foreground contours
       while excluding the coin.
    5. Exclude blobs that cannot be measured reliably -- touching clusters,
       foreign matter, hollow ring artifacts, low-solidity shadow artifacts, and
       contours clipped by the crop border -- by the requested area, color,
       hole, solidity, and border rules. There is no upper bound on grain count.
    6. Validate the individually measured count, then return per-grain plus
       summary measurements.

    Returns exactly `status`, `coin`, `grains`, `excluded_touching` and
    `summary`; every internal diagnostic the helper exposes is underscore
    prefixed and stripped here.

    Custom exceptions are intentionally allowed to propagate to the caller.
    """
    image = _load_image(image_path)
    result = _measure_wheat_grains_from_image(image)
    return {key: value for key, value in result.items() if not key.startswith("_")}
