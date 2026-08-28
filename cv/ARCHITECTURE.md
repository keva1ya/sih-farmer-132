# KrishiDrishti CV Module — Architecture

This document explains how every file in `cv/` works and how they fit together.
Scope: `cv/config.py`, `cv/main.py`, `cv/pipeline.py`, `cv/schemas.py`,
`cv/debug_view.py`, `cv/requirements.txt`, and everything in `cv/steps/`
except tests. §17 also covers the small `frontend/` hookup that calls into
the debug endpoint, since it can't be understood without a `cv/`-side view.

## 1. What this module does

Given a photo of wheat grains spread on a white sheet next to a ₹10 coin (used
as a size reference), the module:

1. Validates the photo is usable (right format, not too blurry, coin visible
   and not tilted, enough grain present).
2. Detects the coin to establish a pixel-to-millimetre scale.
3. Segments the grains from the background and measures each one
   (length, width, area, aspect ratio) in millimetres.
4. Classifies grains as broken, discolored, or foreign matter.
5. Grades the sample against AGMARK thresholds (A / B / C / Reject).
6. Returns a deterministic, hashed JSON result.

Everything is classical OpenCV (thresholding, contours, ellipse fitting,
morphology) — there is no ML model in the loop today. The code is written so a
trained segmentation model could later replace exactly one layer
(`extract_grain_masks`) without touching anything else.

## 2. Request flow (top to bottom)

```
Client
  │  POST /analyze (image bytes, ?detailed, ?commodity)
  ▼
main.py            — FastAPI endpoint: validates content-type & file size
  │
  ▼
pipeline.py         run_pipeline()
  │  Step 0: decode bytes → BGR image (cv2.imdecode), downscale if > 1920px wide
  │  Step 1: check_blur()                              (steps/blur_check.py)
  │  Steps 2-7: _measure_wheat_grains_from_image()      (steps/wheat_measurement.py)
  │      → calibrate_image()      paper crop, foreground mask, coin scale
  │      → extract_grain_masks()  segment + classify contours into grains
  │      → calculate_geometry()   ellipse-fit each grain → mm measurements
  │  White balance: calibrate_white_balance()           (steps/white_balance.py)
  │  Step 8: classify_broken()                          (steps/measurement.py)
  │  Step 9: detect_discoloration()                     (steps/defect_detection.py)
  │  Step 10: foreign matter %  (read from measurement module's own tally)
  │  Step 11: compute_grade()                            (steps/grading.py)
  │  Step 12: build result dict, SHA-256 hash it
  ▼
JSON response (schemas.py defines the shape, not enforced at runtime here)
```

Every numbered "Step" comment in the code refers to this same sequence, which
is why step numbers in file docstrings look non-contiguous (paper detection,
coin calibration and segmentation — old steps 2–7 — were consolidated into
`wheat_measurement.py` and no longer exist as separate files; see §6).

## 3. `main.py` — FastAPI application

Four endpoints:

- **`GET /health`** — liveness check, returns `{"status": "ok", ...}`.
- **`POST /analyze`** — the core endpoint.
  - Validates `file.content_type` is JPEG or PNG (`ALLOWED_EXTENSIONS`).
  - Validates size ≤ `MAX_FILE_SIZE_MB` (10 MB).
  - Calls `run_pipeline(image_bytes, detailed, commodity)`.
  - Maps the pipeline's `status` field to an HTTP status:
    - `success` → 200
    - `invalid_format` / `file_too_large` → 400
    - anything else (`too_blurry`, `coin_not_found`, `too_tilted`,
      `paper_not_detected`, `too_few_grains`, `processing_error`) → 422
      (user-actionable, not a server error).
- **`POST /upload-video`** — no CV processing. Stores a buyer's grain video
  as a blob under `VIDEO_STORAGE_DIR` (env var, default `./uploaded_videos`)
  with a random UUID filename, and returns a local `/videos/{filename}` URL.
  Validates content-type (`video/mp4`, `video/quicktime`) and size
  (≤ `MAX_VIDEO_SIZE_MB` = 50 MB). This is purely file storage, not analysis.
- **`POST /analyze/debug`** — same input as `/analyze` (photo, optional
  `?commodity`), but returns an HTML page instead of JSON: every intermediate
  OpenCV image (paper mask, foreground mask, coin overlay, grain contours,
  defect overlay) rendered stage by stage, for a human to actually look at.
  See §14 (`debug_view.py`). Not used by the frontend — it's a developer tool
  for seeing what the pipeline is doing rather than only its final JSON grade.

CORS is added at the app level (`app.add_middleware(CORSMiddleware, ...)`),
so it covers every route above, `/analyze/debug` included: allowed origins
are `http://localhost:4173` / `http://127.0.0.1:4173`, methods `GET`/`POST`
only. This is what lets `frontend/app.js` call `/analyze/debug` directly
(see §17) as long as the frontend itself is served from port 4173 — opening
`frontend/index.html` straight from disk (`file://`) sends no matching
Origin header and CORS will block both `/analyze` and `/analyze/debug`.

## 4. `pipeline.py` — orchestrator

Single public function: `run_pipeline(image_bytes, detailed, commodity) -> dict`.

Walks through:

1. **Decode.** `cv2.imdecode` the bytes. Empty bytes or an undecodable file
   returns `invalid_format` immediately.
2. **Downscale only, never upscale.** If width > `IMAGE_RESIZE_WIDTH` (1920px),
   resize down (`cv2.INTER_AREA`), preserving aspect ratio. Images already at
   or below that width reach the measurement module untouched — a deliberate
   choice, since the module scales all its internal kernels to image
   resolution and derives its mm scale from the coin, so it doesn't need a
   fixed input size. Downscaling only exists to cap segmentation cost on
   oversized phone photos.
3. **Blur gate.** `check_blur()` — if too blurry, return `too_blurry` with the
   measured Laplacian variance in the reason string.
4. **Delegate to `wheat_measurement._measure_wheat_grains_from_image()`.**
   This single call does paper detection, coin calibration, grain
   segmentation, and geometry (see §6). Its typed exceptions
   (`ImageLoadError`, `PaperNotDetectedError`, `CoinNotDetectedError`,
   `CoinTiltedError`, `TooFewGrainsError`) are caught and mapped to pipeline
   status codes via the `_MEASUREMENT_ERROR_STATUS` table; any other
   exception becomes a generic `processing_error` (this is what stops an
   unanticipated cv2/numpy edge case from escaping as a raw HTTP 500).
5. **White balance.** Runs `calibrate_white_balance()` on the cropped image,
   using the coin's center/radius (in the same cropped-image coordinate
   frame) to sample the paper ring. Runs *after* the measurement module has
   already finished, so it never disturbs the geometry the module was tuned
   against — it only affects the colour-based passes downstream.
6. **Broken classification.** Adapts each grain's `length_mm`/`width_mm`/
   `aspect_ratio` into the shape `classify_broken()` expects, and computes
   `broken_pct` + `median_length_mm`.
7. **Discoloration.** `detect_discoloration()` runs on the white-balanced
   image and the accepted grain contours.
8. **Foreign matter %.** Computed from figures the measurement module already
   accumulated during segmentation — *not* a separate detection pass. The
   denominator is `foreign_area + measured_grain_area + touching_cluster_area`
   (the sample itself, not the whole sheet — dividing by sheet area would
   read ~34x too low and silently grade contaminated lots as A). Touching
   clusters count as real grain area even though they weren't measured
   individually.
9. **Aggregate stats.** Mean length/width/ratio/area across all measured
   grains, rounded to 2 decimals (rounding is part of the response contract —
   these values are hashed and displayed unformatted by the frontend).
10. **AGMARK grading.** `compute_grade()` against `config.AGMARK_THRESHOLDS`.
11. **Build & hash the result.** The result dict (status, grain_count,
    measurements, grade) is canonically JSON-serialized
    (`sort_keys=True`, no extra whitespace) and SHA-256 hashed *before* the
    hash itself or the optional `per_metric_grades` are added, so the hash
    only ever covers the stable, always-present fields. Used for QR-code
    verification per `schemas.py`.

## 5. `config.py` — the single source of tunable constants

Everything you'd want to tune without touching logic lives here. Notable
sections:

- **Image preprocessing:** `IMAGE_RESIZE_WIDTH=1920`, `MAX_FILE_SIZE_MB=10`,
  `ALLOWED_EXTENSIONS`.
- **Blur:** `BLUR_THRESHOLD=16.0` (Laplacian variance floor).
- **Coin calibration / segmentation:** deliberately *not* here anymore. A
  comment explains that these constants moved into
  `wheat_measurement.py` as module-scope constants when that module was
  built, and the old config values were stale/wrong (e.g. `TILT_THRESHOLD`
  0.60 vs. the live 0.95) — so they were deleted rather than left as a
  misleading second source of truth. **If you need to tune coin or
  segmentation behaviour, edit the constants at the top of
  `steps/wheat_measurement.py`, not this file.**
- **Broken grain:** `BROKEN_THRESHOLD_PCT=0.80` (length < 80% of median → broken).
- **Defect detection (HSV + LAB healthy-wheat colour bands):** these are
  flagged in comments as initial estimates that must be validated against
  real photos.
- **`DISCOLORATION_EDGE_EROSION_PX=2`:** how many pixels of each grain
  contour's outer rim are excluded before the HSV/LAB healthy-band comparison
  runs (§10). Measured on a real photo: 100% of flagged "defect" pixels sat
  within 2px of the contour boundary (anti-aliased edge blending, specular
  rim highlights), 0% in the interior — this constant fixes that without
  loosening the HSV/LAB bands themselves.
- **`AGMARK_THRESHOLDS`:** a dict keyed by commodity (`"wheat"` is the only
  one defined). Per commodity, per parameter:
  - **Cascading list** `[(boundary, grade), ...]` for "higher is better"
    params (length, width, area — thresholds are minimums) or "lower is
    better" params (broken/discolored/foreign — thresholds are maximums).
  - **Band dict** `{"type": "band", "min", "max", "in_band_grade",
    "out_of_band_grade"}` for `avg_lw_ratio` only — a ratio can fail by being
    *either* too low (two grains segmented as one round blob) *or* too high
    (a broken fragment), so a single monotonic scale can't express it. This
    metric can only ever push the grade down to `"C"`, never up past `"A"`.
  - Geometric metrics (length/width/area/ratio) bottom out at grade `"C"`,
    never `"Reject"` — a small/dense/short grain is substandard, not
    contamination. `broken_pct`, `discolored_pct`, and `foreign_matter_pct`
    can all produce `"Reject"`: damaged/discoloration above 3.5% or foreign
    matter above 1.5% means the lot is rejected outright, not merely
    downgraded.
  - Adding a new commodity = adding a new dict entry with the same 7 keys.
- **`GRADE_RANK`:** `{"A": 0, "B": 1, "C": 2, "Reject": 3}` — used to find the
  worst grade across all metrics.
- **Video upload:** `MAX_VIDEO_SIZE_MB=50`, `ALLOWED_VIDEO_EXTENSIONS`,
  `MAX_VIDEO_DURATION_SECONDS=30` (defined but not currently enforced in
  `main.py` — no duration check is actually performed on upload).

## 6. `steps/wheat_measurement.py` — the core CV engine (~1,400 lines)

This is where nearly all of the actual computer vision happens. It is
structured as three composable public layers plus one orchestrator, so an
extraction backend (e.g. a trained YOLOv8-Seg model) could later replace the
middle layer without touching calibration or geometry:

```
calibrate_image()        → paper crop, shared foreground mask, coin geometry
extract_grain_masks()    → foreground → per-grain contours (the only layer
                            a trained segmenter would replace)
calculate_geometry()     → contours → length/width/area/ratio in mm
_measure_wheat_grains_from_image()  → composes all three + validation,
                                        returns underscore-prefixed diagnostics
measure_wheat_grains(path)          → public/disk entrypoint, strips
                                        underscore keys, returns the 5-key
                                        contract: status, coin, grains,
                                        excluded_touching, summary
```

`pipeline.py` calls `_measure_wheat_grains_from_image()` directly (in-process,
already-decoded image) rather than `measure_wheat_grains()` (which loads from
a file path) — it needs the extra diagnostic fields (contours, cropped image,
coin center/radius, foreign-matter tallies) to run the downstream defect and
grading steps.

### 6.1 Custom exceptions

- `ImageLoadError` — file on disk unreadable (only relevant to
  `measure_wheat_grains`, not the in-process path).
- `PaperNotDetectedError` — white sheet not confidently found.
- `CoinNotDetectedError` — no plausible coin contour, or one was found but not
  corroborated by an independent circle detector.
- `CoinTiltedError` — a coin-shaped, correctly-sized, Hough-corroborated
  contour was found but its fitted-ellipse axis ratio is below 0.95 (i.e. the
  photo wasn't taken top-down enough).
- `TooFewGrainsError` — fewer than 30 individually measurable grains.

`pipeline.py` maps each of these to a distinct HTTP-facing status code and
carries the exception's own message through verbatim as `reason`.

### 6.2 Paper detection & crop (`_detect_and_crop_paper`)

1. **Strict-first, adaptive-fallback threshold.** Threshold the image in HSV
   for "paper-like" pixels: low saturation (`S <= PAPER_SATURATION_MAX = 60`),
   high value (`V >= PAPER_VALUE_MIN = 180`). This fixed band is tried first
   and handles the overwhelming majority of photos on its own. If it fails to
   find a component covering at least 15% of the frame, `_detect_and_crop_paper`
   steps through `PAPER_THRESHOLD_RELAXATION_STEPS` (fractions 0.0 → 1.0),
   relaxing `V`/`S` toward a per-photo bound from `_adaptive_paper_thresholds`
   (Otsu's method on V and on S, keeping whichever side of each split has the
   larger pixel population — paper is the dominant class on any accepted
   sheet), and stops at the first, least-permissive step that finds a valid
   region. The function returns the saturation ceiling it actually used
   alongside the crop and mask (see §6.3).

   This ladder — rather than jumping straight to the Otsu bound on every photo
   — exists because that jump regressed a real photo the strict band alone
   already handled: Otsu on V separates "coin/shadow" from "everything else",
   with no notion of "background" as a third class, so a background merely
   brighter than the coin (not necessarily brighter than the paper) lands on
   paper's side of the split and gets swallowed into the crop wholesale. The
   inflated `paper_area` then widens the coin search's Hough radius window
   (§6.4) enough that the real coin is never returned among the results,
   raising `CoinNotDetectedError` on a photo where the coin was plainly
   visible. Stepping from strict and stopping at the first success means a
   photo the strict band already handles is never touched by that failure
   mode, while a photo the strict band rejects outright (dim lighting, a
   uniform colour cast) still gets the full benefit of the adaptive fallback.
2. Morphological close (bridge small gaps) then open (remove speckle), sizes
   scaled to image resolution via `_scaled_odd_kernel`.
3. Connected-components; take the largest bright blob. Reject if it's under
   15% of the frame (`PAPER_MIN_AREA_RATIO`) — likely no sheet in view.
4. **Crease recovery (`_sheet_region_contour`).** A creased/crumbled sheet
   gets fragmented by shadow lines and its filled contour can leave interior
   pockets uncovered (the code notes this measured as low as 47.8% bounding-box
   coverage on a crumbled test photo, with the coin landing inside a lost
   pocket). Fix: merge any other bright fragment that's both substantial
   (≥2% of the seed's area) and mostly contained within a small dilation of
   the seed's convex hull (this is what stops unrelated bright background,
   e.g. a light tablecloth, from being pulled in), then take the convex hull
   of everything merged — a sheet of paper is convex by construction, so this
   is a legitimate reconstruction, not a guess. The hull is only accepted if
   at least 45% of its area (`PAPER_HULL_MIN_BRIGHT_SUPPORT`) is backed by
   actual bright pixels; otherwise it falls back to the plain filled seed
   contour.
5. Crop the image and a paper mask to the resulting contour's bounding box.

### 6.3 Shared foreground mask (`_foreground_mask`)

One foreground definition, shared by both the coin search and grain
segmentation (previously they used different, inconsistent rules — this
unification is what let the module handle a crumbled/shadowed sheet at all).

- **Illumination flat-fielding (`_flatten_illumination`):** grayscale image
  divided by a heavily-closed version of itself (closing with a kernel sized
  to ~6% of the shortest side removes anything grain-sized, leaving pure
  lighting — vignetting, photographer's shadow). Result renormalizes the
  paper to ~255 everywhere so a single global threshold survives uneven
  lighting.
- **Flat-fielded threshold (`_flatfield_foreground`):** Otsu's method
  computed *only over paper pixels*, capped so it never sits closer than 12
  levels (`GRAIN_FLATFIELD_MIN_DROP`) below the flattened paper level — this
  stops Otsu from carving a grain-free, noise-only sheet in half.
- **Saturation term (`_saturation_foreground`):** a pixel is foreground if its
  saturation exceeds `max(paper_saturation_ceiling, this image's own median
  paper saturation + 25)`, where `paper_saturation_ceiling` is the *actual* `S`
  ceiling `_detect_and_crop_paper` used for this photo (§6.2) — no longer
  always the `PAPER_SATURATION_MAX` constant, now that paper detection can
  relax its own ceiling per photo. Measuring relative to the sheet's own
  observed saturation (rather than a fixed absolute cutoff) is what stops a
  mild warm light cast from turning the whole sheet into "foreground," and
  threading through the actual ceiling (rather than assuming the constant) is
  what keeps this term structurally unable to disagree with paper detection
  about what counts as paper, however far the fallback relaxed.
- Foreground = flat-field term **OR** saturation term, masked to the paper
  region only.

### 6.4 Coin detection (`_find_best_coin_candidate`)

Runs on the shared foreground mask *before* any bridging morphology (bridging
was previously fusing the coin to adjacent grains and destroying its
circularity). Key idea: fill mask holes (the coin is bimetallic and its
brighter inner disc drops out, leaving a ring — filling makes it solid so the
distance transform reads its true radius), then open the mask with a disc
sized to roughly half the largest inscribed disc in the foreground — that
largest disc is geometrically the coin itself, so the opening element is
automatically about half the coin's radius: too big for any grain or
grain-bridge to survive, too small to shrink the coin.

A contour is accepted as the coin only if **all four** independent checks
pass:

1. **Area vs. sheet:** between 0.3% (`COIN_MIN_AREA_RATIO`) and 25%
   (`COIN_MAX_AREA_RATIO`) of paper area.
2. **Area vs. population:** if ≥8 objects are on the sheet, must be at least
   8x (`COIN_MIN_POPULATION_AREA_MULTIPLE`) the *median* blob area — derived
   from measured real-photo ratios (real coin: 27.7–34.4x; worst grain
   clump: up to 10.0x; single grain: ≤2x).
3. **Circularity:** ≥ 0.80 (`COIN_MIN_CIRCULARITY`).
4. **Hough corroboration:** an independent `cv2.HoughCircles` pass (gradient
   voting on image edges, blind to the foreground mask) must find a circle
   whose center is within 25% of the candidate's radius and whose own radius
   is 0.5–1.3x the candidate's (the wide radius tolerance deliberately
   accommodates Hough locking onto the coin's *inner* rim instead of the
   outer one).

Candidates are ranked by area (not circularity — area gives a ≥5x margin,
circularity's margin can be as thin as 0.014). The top-ranked candidate is
then tilt-checked: axis ratio (minor/major of the fitted ellipse) must be
≥ 0.95 (`COIN_TILT_MIN_AXIS_RATIO`), *and* must have Hough corroboration, to
be accepted as the calibration reference. If nothing is accepted, the code
distinguishes `CoinTiltedError` (best candidate found but under-ratio) from
`CoinNotDetectedError` (nothing plausible at all) using the top candidate's
own tilt ratio.

Output includes `px_per_mm` = mean coin diameter in pixels ÷ 27.0mm
(`COIN_DIAMETER_MM`, the real ₹10 coin diameter) — this is the scale every
downstream mm measurement depends on.

### 6.5 Grain segmentation & classification

`_segment_foreground_contours`: takes the shared foreground, zeroes out the
coin's own mask region, applies a gentle 3×3 open+close, then finds contours
with `RETR_CCOMP` (so each contour also reports its own interior hole area —
needed to catch ring-shaped shadow artifacts that pass every other check).

`_classify_grain_candidates` then sorts every candidate contour into exactly
one bucket, in this order:

1. **Border-excluded** — bounding box touches the crop edge outright, *or*
   sits within 5mm of the edge with an implausible bounding-box aspect ratio
   (> 4.0) — catches thin shadow/fold slivers that don't literally touch the
   boundary.
2. **Discarded (unaccounted)** — area < 2 mm² (`FOREIGN_MATTER_MIN_AREA_MM2`):
   sensor noise / dust, not counted anywhere.
3. **Foreign matter** — area 2–8 mm² (`FOREIGN_MATTER_MAX_AREA_MM2`): small
   debris/chaff by size alone.
4. **Oversized (> 27 mm², `NORMAL_GRAIN_AREA_MM2 × 1.5`)** — split by colour:
   - wheat-coloured → **touching cluster** (real grain that couldn't be
     split into individual kernels; its area counts toward the sample-area
     denominator downstream but not toward `grains`/`summary.count`; the
     estimated grain count inside is tracked separately for the
     `TooFewGrainsError` message only, never for actual counting).
   - off-hue → **foreign matter** (e.g. a stone or clump of debris large
     enough to clear the size band above).
5. **Ring-excluded** — interior hole area > 15% of contour area
   (`GRAIN_MAX_HOLE_AREA_RATIO`): shadow rings / specular-highlight donuts
   that solidity alone can't catch (their outer boundary is still convex).
6. **Low-solidity-excluded** — contour area / convex-hull area < 0.80
   (`GRAIN_MIN_SOLIDITY`): cast shadows, ragged merged edges.
7. **Off-hue foreign matter** — grain-sized but mean hue outside the wheat
   band (±10° tolerance around the config HSV healthy range).
8. **Accepted as a grain.**

### 6.6 Geometry (`calculate_geometry` / `_measure_contour`)

For each accepted grain contour: `cv2.fitEllipse`, sort axes into
minor/major, convert to mm via `px_per_mm`. `length_mm` = major axis,
`width_mm` = minor axis, `area_mm2` = contour area ÷ `px_per_mm²`,
`aspect_ratio` = length/width. A contour with too few points to fit an
ellipse is dropped and counted in `excluded_unmeasurable` (kept index-aligned
with `_grain_contours` so downstream defect passes see exactly the grains
that were actually measured).

### 6.7 Grain-count validation (`_validate_grain_count`)

Requires ≥30 measured grains (`MIN_VALID_GRAIN_COUNT`). No upper bound — a
sheet with a lot of grain is not an error. If the shortfall is because many
grains were excluded as touching clusters, the error message says so
explicitly (different fix: spread the grains out) rather than giving a
generic "not enough grain" message.

## 7. `steps/blur_check.py` — Step 1

`check_blur(image) -> (is_acceptable, variance)`. Converts to grayscale,
computes the variance of the Laplacian (a standard focus-measure — sharp
edges produce high-frequency response, blur suppresses it). Acceptable if
`variance >= BLUR_THRESHOLD` (16.0).

## 8. `steps/white_balance.py` — Step 4 (run after measurement, before defect passes)

`calibrate_white_balance(image, coin_center, coin_radius, sample_ring_width=40,
max_iterations=6, per_step_gain_cap=1.4, convergence_tolerance=15.0)`.
Samples an annular ring of background pixels just outside the coin (a small
20px gap avoids coin-edge artifacts, then a 40px-wide ring) — this is the
cleanest, flattest part of the visible sheet. Falls back to returning the
image unchanged if fewer than 100 background pixels are sampled (e.g. ring
falls outside the frame). This exists because the HSV/LAB defect-colour bands
in `config.py` assume a white-balanced image; without it, `discolored_pct`
was found to measure the light bulb's colour rather than the grain (swinging
across nearly the full 0–100% range under ordinary lighting variation).

**Iterative, not single-shot.** Each pass computes the mean BGR of the
*current* background, applies a per-channel gain toward white bounded by
`per_step_gain_cap` (1.4x), and repeats until every channel's mean is within
`convergence_tolerance` of 255 or `max_iterations` is reached. This replaced
an earlier single multiply capped at a flat 3.0x, which could leave a
strongly tinted photo (direct sun, a coloured bulb) short of true white in
one step — any residual cast then survived onto the grains and read as
discoloration that had nothing to do with the grain itself. Iterating lets
the *cumulative* gain exceed 3.0x (up to `per_step_gain_cap ** max_iterations`
≈ 7.5x) so a strong tint can actually be fully corrected, while no single
pass jumps as far as the old flat cap could.

**Why the tolerance is loose (15, not tight).** A real paper ring isn't a
flat colour swatch — its pixels have their own spread from texture and JPEG
noise, so some of them are already brighter than the ring's mean and clip at
255 on every pass. That clipping caps how much any pass can raise the mean,
so chasing a tight target never actually converges: the loop keeps computing
a fresh "still short" gain from a mean that clipping won't let rise further,
and each extra pass multiplies onto the *grain* pixels too. Measured directly:
a tolerance of 2 (not 15) drove seven extra passes on a mildly-tinted real
photo and moved `discolored_pct` from 3.4% to 18.9% — the exact failure mode
this function exists to prevent, self-inflicted by too strict a target. At
15, an ordinary photo's ring converges in its first pass (matching what the
old single-shot formula already produced for it), while a strongly tinted
photo's much-further-from-white starting mean still gets several passes to
close most of the gap.

## 9. `steps/measurement.py` — Step 8: broken-grain classification

`classify_broken(measurements) -> (broken_pct, median_length_mm)`. Takes a
list of `{"length_mm": ...}` dicts, computes the median length, and marks any
grain whose length is below `median × BROKEN_THRESHOLD_PCT` (80%) as broken.
Returns the percentage broken and the median (both rounded). Note: this file
used to also contain grain measurement itself (contour → mm); that
responsibility moved to `wheat_measurement.py` and was removed from here.

## 10. `steps/defect_detection.py` — Steps 9 & 10 (only Step 9 is actually used from here; Step 10/foreign-matter is computed in `pipeline.py` from the measurement module's own tally, not from this file)

`detect_discoloration(image_wb, grain_contours, return_mask=False) ->
discolored_pct` (or `(discolored_pct, defect_mask)` if `return_mask=True`).
Dual-pass, per-pixel, union logic — a pixel is "defective" if flagged by
*either* pass:

- **HSV pass:** pixel is healthy if within the config's HSV healthy-wheat
  band; unhealthy otherwise.
- **LAB pass:** pixel is healthy if within the config's LAB `a`/`b` healthy
  band (catches subtler fungal/moisture damage the HSV pass misses). LAB `a`
  and `b` are OpenCV uint8 channels centered at 128 — `_interval_to_uint8_bounds`
  converts the float-centered config bounds into the exact uint8 range that
  reproduces the equivalent per-pixel float comparison.

Both healthy masks are computed **once, full-frame**, then indexed per
contour via `_contour_roi` (rasterizes just the contour's bounding-rect
window rather than a full-frame mask per contour — a performance
optimization; produces bit-identical results to the full-frame approach).
For each grain contour: **erode the contour mask inward by
`DISCOLORATION_EDGE_EROSION_PX`** (2px, config.py) before doing anything
else, falling back to the un-eroded mask if erosion empties it (so a grain
too small to have any interior left still gets measured on its full area
rather than being silently skipped). Then count grain pixels, count
HSV-defective and LAB-defective pixels within the eroded mask, take the max
of the two defect counts (a pixel flagged by both isn't double-counted),
accumulate. Final `discolored_pct` = total defective pixels / total grain
pixels across all grains, as a percentage.

The erosion step exists because the outermost ring of a segmented grain
contour is anti-aliased edge blending against the white background plus
specular rim highlights, not grain colour — measured on a real photo, 100%
of flagged defect pixels sat within that 2px rim and 0% were in the
interior, reading as 3.41% discoloration (and a C grade) on a visually clean
sample. Eroding the measurement mask fixes exactly this without loosening
the HSV/LAB healthy bands themselves, so a real pale/faded defect sitting in
a grain's interior is still caught (verified with a synthetic grain carrying
an actual interior dark spot: still flags ~19% discolored after this
change).

This module also defines `_create_contour_mask` (full-frame, kept for other
callers) and `_passes_wheat_color_check`-style helpers are actually in
`wheat_measurement.py` (that module's foreign-matter colour check is
independent of this file, despite similar HSV logic — see §6.5, bucket 4/7).

**`return_mask=True`** (used only by `debug_view.py`, §14) additionally
builds and returns a full-frame binary mask of every pixel flagged by
*either* pass — the true per-pixel union, drawn straight from the same
full-frame `hsv_healthy_full`/`lab_healthy_full` arrays used for counting.
This is a separate computation from `discolored_pct` itself, which keeps
using `max(hsv_defect_count, lab_defect_count)` per contour as before (see
docstring) — that max is an approximation of the true union *count*, so on
some images the true-union mask can show marginally more red pixels than the
reported percentage alone would suggest. `discolored_pct`'s value is
byte-for-byte unchanged by this parameter; it is purely an additional return
value for visualization.

## 11. `steps/grading.py` — Step 11: AGMARK grading

`compute_grade(avg_length_mm, avg_width_mm, avg_area_mm2, avg_lw_ratio,
broken_pct, discolored_pct, foreign_matter_pct, commodity="wheat")`
`-> (overall_grade, per_metric_grades)`.

For each of the 7 metrics, looks up the commodity's threshold entry in
`config.AGMARK_THRESHOLDS` and grades it via `_grade_single_metric`:

- **Band thresholds** (currently only `avg_lw_ratio`): in-range →
  `in_band_grade`, out-of-range → `out_of_band_grade`.
- **Cascading thresholds**, higher-is-better (length/width/area): walk the
  list in order, first boundary the value meets-or-exceeds wins; falls
  through to `"Reject"` if none match (though in practice every commodity's
  cascade bottoms out at a `0.0` boundary so this never actually triggers for
  those three metrics).
- **Cascading thresholds**, lower-is-better (broken/discolored/foreign): same
  idea, first boundary the value is at-or-under wins.

`overall_grade` = the single worst grade across all 7 metrics, using
`GRADE_RANK` for ordering. `per_metric_grades` maps each metric to its own
grade under response-friendly keys (`length`, `width`, `area`, `lw_ratio`,
`broken`, `discolored`, `foreign_matter`) — only returned by the API when
`?detailed=true`. Raises `ValueError` if `commodity` isn't in
`AGMARK_THRESHOLDS` (currently only `"wheat"` is defined).

## 12. `schemas.py` — response contract

Pydantic models documenting the JSON shape (not currently wired into FastAPI
as `response_model=...`, so they're documentation/contract, not enforced at
the endpoint):

- `Measurements` — the 8 aggregate numbers (`avg_length_mm` … `foreign_matter_pct`).
- `PerMetricGrades` — the 7 per-parameter grade letters.
- `AnalyzeSuccessResponse` — `status`, `grain_count`, `measurements`, `grade`,
  `hash`, optional `per_metric_grades`.
- `AnalyzeErrorResponse` — `status` (one of the documented error codes),
  `reason`.
- `VideoUploadResponse` — `status`, `video_url`, optional `duration_seconds`
  (not currently populated by `main.py`'s `/upload-video` handler).

## 13. `requirements.txt` — dependencies

```
fastapi==0.115.0
uvicorn==0.30.0
python-multipart==0.0.9      (multipart form parsing, needed for file uploads)
opencv-python-headless==4.10.0.84   (no GUI bindings — server-safe)
numpy                          (unpinned)
```

`cv/venv/` is a local virtual environment holding these packages plus their
transitive dependencies (e.g. `pydantic`, `pytest` is present too, used by
`cv/tests/`) — it is not part of the module's own code, just an installed
environment.

## 14. `debug_view.py` — human-readable pipeline visualization ("Assay Trace")

Not part of the `/analyze` flow. `build_debug_report(image_bytes, commodity="wheat",
source_name=None) -> str` returns a complete, self-contained HTML page served
by the `POST /analyze/debug` endpoint (§3). `main.py` passes the uploaded
file's own name through as `source_name` so the page heading shows the real
photo name rather than a generic label.

It exists because the FastAPI-generated `/docs`/`/redoc` pages only describe
the JSON *shape* of `/analyze` — they can't show what the pipeline actually
saw in a given photo. This module re-runs the same stages `pipeline.py` does,
but calls `wheat_measurement.py`'s three public layers
(`calibrate_image` → `extract_grain_masks` → `calculate_geometry`, §6)
directly instead of the underscore-prefixed orchestrator, so it can grab and
render each layer's intermediate images rather than only the final numbers.

**Design ("Assay Trace"): a verdict banner over a numbered stage trace.**
Light/dark-aware CSS custom-property tokens (`:root`, redefined under
`@media (prefers-color-scheme: dark)` and `:root[data-theme="dark"]`, per the
standard three-state artifact pattern), three Google Fonts loaded via
`<link>` — Fraunces for headings, Public Sans for body copy, IBM Plex Mono
for every number/caption/data value (`font-variant-numeric: tabular-nums`
where digits line up). On a successful run, a `.verdict-row` sits above the
stage list: a circular grade badge (`badge-good`/`badge-warn`/`badge-bad`,
mapped from the grade via `_GRADE_BADGE_CLASS`), a stat grid (grain count,
discolored/broken/foreign-matter %), and a plain-language note built by
`_verdict_note()` that names exactly which metrics graded below `"A"` and by
how much. Below that, each stage is a numbered card (`_stage()` /
`_result_stage()`) with a title, a mono caption, a status chip
(good/warn/bad/neutral), and — for image stages — the image itself:

| Stage | What it shows | Source |
|---|---|---|
| 00. Resized input | The image after the same downscale `pipeline.py` applies | — |
| 01. Blur check | Pass/fail chip + the measured Laplacian variance | `blur_check.check_blur` |
| 02. Paper detection | Cropped photo, non-paper pixels dimmed | `calibrate_image`'s `paper_mask` |
| 03. Paper binary mask | Raw black/white mask | same |
| 04. Foreground mask | Black/white: everything that isn't paper | `calibrate_image`'s `foreground` |
| 05. Coin detection | Orange fitted ellipse + red center dot, with `px_per_mm`/`tilt_ratio`/`circularity` | `calibrate_image`'s `coin` dict |
| 06. Grain contours | Green contour outlines, exclusion counts, chip flags below `MIN_VALID_GRAIN_COUNT` | `extract_grain_masks` + `calculate_geometry` |
| 07. Discoloration overlay | Red = defective pixels, on the white-balanced image | `defect_detection.detect_discoloration(..., return_mask=True)` (§10) |
| 08. Final result | Grade + all 7 measurements, each with its per-metric grade chip | `measurement.classify_broken` + `grading.compute_grade`, replicating `pipeline.py`'s foreign-matter-% formula (§4, step 8) inline |

**Images are downscaled and JPEG-encoded for the report, not full
resolution.** `_encode_jpeg()` resizes to 640px wide (quality 82) before
base64-encoding; binary masks use `_encode_png_mask()` (PNG, same resize —
lossless matters more than size for a mostly-solid black/white image). This
dropped a real report's payload from 6.6MB (full-resolution PNGs, the
original plain-dev-page version) to ~440KB, which matters because the
frontend loads this into a new tab on every click (§17).

**Failure handling is graceful, not exceptional.** If the photo would fail
at some stage in `/analyze` (blurry, no paper, no coin, coin tilted, too few
grains), the report still renders every stage that succeeded before the
failure, plus a styled error banner (`_error_banner()`, `--bad` tokens) with
the same exception message `/analyze` would return — so a developer can see
exactly what the pipeline saw right up to the point it gave up. No verdict
banner is shown in this case (there's no grade to summarize yet). If grain
count comes in under `MIN_VALID_GRAIN_COUNT` (30, imported from
`wheat_measurement.py`), stage 06's chip turns `chip-bad` and the report
stops before stage 08 (no grade is computed on too few grains, matching
`/analyze`'s own behaviour).

This module intentionally duplicates a small amount of `pipeline.py`'s logic
(the resize helper, the foreign-matter-percentage formula) rather than
importing `pipeline.py`'s private (underscore-prefixed) helpers — it needs
to insert visualization after each stage, which the pipeline's single
top-to-bottom function doesn't expose seams for.

## 15. Error contract (status codes returned by `/analyze`)

| `status`             | HTTP | Meaning |
|-----------------------|------|---------|
| `success`              | 200  | Grading completed. |
| `invalid_format`       | 400  | Not JPEG/PNG, or unreadable/empty bytes. |
| `file_too_large`       | 400  | Exceeds `MAX_FILE_SIZE_MB`. |
| `too_blurry`           | 422  | Laplacian variance below `BLUR_THRESHOLD`. |
| `paper_not_detected`   | 422  | White sheet not confidently found. |
| `coin_not_found`       | 422  | No coin-shaped, size/circularity/Hough-corroborated contour. |
| `too_tilted`           | 422  | Coin found but axis ratio < 0.95 (photo not top-down enough). |
| `too_few_grains`       | 422  | Fewer than 30 individually measurable grains. |
| `processing_error`     | 422  | Unanticipated exception inside the measurement module. |

## 16. Design principles worth knowing before changing this code

- **Determinism.** No randomness anywhere; the same photo always produces the
  same result, which is what makes the SHA-256 result hash meaningful for
  downstream QR verification.
- **Single source of truth per concern.** Coin/segmentation tuning constants
  live only in `wheat_measurement.py`, not duplicated in `config.py` — the
  file itself documents why a duplicate copy was actively dangerous (stale
  values silently disagreeing with the live ones).
- **Fail loud, fail typed.** The measurement module never silently returns a
  wrong answer when it's unsure — it raises one of five specific exceptions,
  and `pipeline.py` is the only place that translates those into the HTTP
  contract, keeping the measurement module itself ignorant of HTTP.
- **Resolution independence.** Nearly every pixel-based threshold in
  `wheat_measurement.py` is derived either from the image's own dimensions
  (`_scaled_odd_kernel`), from the coin's own detected size, or from the
  sheet's own measured colour statistics — not hardcoded pixel counts — so
  behaviour doesn't quietly change with a different phone camera or capture
  resolution.
- **Replaceable extraction layer.** The three-layer split in
  `wheat_measurement.py` (`calibrate_image` / `extract_grain_masks` /
  `calculate_geometry`) exists specifically so a future trained
  instance-segmentation model could replace only the middle layer.

## 17. Frontend hookup for the debug view (`frontend/app.js`, `frontend/index.html`)

The debug page (§14) is a `POST`-only endpoint expecting a multipart file —
a browser can't open its URL directly (typing it into the address bar sends
a `GET` and gets `405 Method Not Allowed`). Instead, `frontend/index.html`'s
quality-desk UI has a second button, **"View CV breakdown"**, next to the
existing **"Run quality check"** button in the assay panel. Clicking it opens
the report in a **new browser tab** (an earlier version rendered it inline
in an `<iframe>` panel on the same page; that markup/CSS has been removed).

- Clicking it requires a photo to already be selected (same `selectedFile`
  the main "Run quality check" flow uses) — otherwise it just shows a toast.
- `window.open('', '_blank')` is called **synchronously, first, inside the
  click handler** — before any `await` — specifically so browsers still
  treat it as a trusted user gesture and don't block it as an unsolicited
  pop-up. Everything after that point is async (the fetch), so opening the
  tab any later would get blocked in most browsers. If `window.open` itself
  returns `null` (pop-ups disabled), a toast tells the user to allow them.
- The new tab shows a plain "Loading breakdown..." message immediately, then
  the button `POST`s the same file as multipart form data to
  `${apiBase}/analyze/debug?commodity=...`, reads the response as text
  (`response.text()` — the response is an HTML document, not JSON), wraps it
  in a `Blob`, and navigates the already-open tab to
  `URL.createObjectURL(blob)`.
- This is entirely independent of the "Run quality check" flow — clicking
  "View CV breakdown" does not call `/analyze` and does not affect
  `resultPanel`.

**Same-origin requirement.** `CORSMiddleware` (§3) only allows requests from
`http://localhost:4173` / `http://127.0.0.1:4173`. If `frontend/index.html`
is opened directly from disk (`file://...`) rather than served from port
4173, the browser sends no matching `Origin` header and every fetch to the
backend — `/health`, `/analyze`, and `/analyze/debug` alike — is blocked by
CORS, not just the new button. Serve the `frontend/` folder from port 4173
(e.g. `python -m http.server 4173` from inside `frontend/`) for any of the
three to work.

**No hot-reload.** `uvicorn cv.main:app` without `--reload` does not pick up
edits to `cv/` while running — a code change here needs the backend process
restarted before it shows up through `/analyze` or `/analyze/debug`, in the
frontend or otherwise. This has been the source of more than one "it doesn't
work" report during development that turned out to be a stale running
process, not a code bug.
