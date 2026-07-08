# Airport Baggage Belt Monitor

Computer-vision pipeline that watches a fixed airport ramp camera feed of a
conveyor belt loader and automatically determines, frame by frame, whether
the belt is **LOADING**, **UNLOADING**, or **IDLE** — with no manual
calibration required for the common case.

It does this in three cooperating stages:

1. **Auto ROI Detection** (`belt_detection.py`) — finds the conveyor belt
   in the frame and fits a rotated bounding band around it.
2. **Belt Type Classification** (`belt_type_classifier.py`) — decides
   whether that belt is **roofed** (canopy-covered) or **open**, because
   roofed belts hide the cargo from the camera and can't be reliably
   monitored for bag movement.
3. **Status Detection** (`status_detector.py`) — runs background
   subtraction + object tracking inside the ROI, classifies each frame's
   status, and writes out a clean, de-noised event timeline (CSV +
   snapshots).

> **Note on pipeline order:** the module docstrings describe the *logical*
> stage order (1 → 2 → 3), but at runtime `status_detector.py` actually
> runs ROI detection first (it needs a locked-on belt before anything
> else is possible), and triggers the belt-type classification once the
> belt is first detected as "connected." See [How it fits
> together](#how-it-fits-together) below.

---

## Table of contents

- [How it fits together](#how-it-fits-together)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Local setup](#local-setup)
- [Quick start](#quick-start)
- [Module deep-dive](#module-deep-dive)
  - [`belt_detection.py`](#belt_detectionpy--auto-roi-detection)
  - [`belt_type_classifier.py`](#belt_type_classifierpy--roofed-vs-open)
  - [`status_detector.py`](#status_detectorpy--tracking--status--events)
- [CLI reference](#cli-reference)
- [Outputs](#outputs)
- [Tuning guide](#tuning-guide)
- [Known limitations](#known-limitations--gotchas)
- [License](#license)

---

## How it fits together

```
                         ┌─────────────────────────────┐
                         │   status_detector.py (main)  │
                         └───────────────┬───────────────┘
                                          │
                     ┌────────────────────┴─────────────────────┐
                     │ 1. Locate belt every frame until connected │
                     │    (or use --roi / manual click-selection) │
                     └────────────────────┬─────────────────────┘
                                          │  BeltDetector.detect()
                                          ▼
                         ┌─────────────────────────────┐
                         │      belt_detection.py       │
                         │  yellow rails + Hough lines   │
                         │  → robust Huber axis fit      │
                         │  → BeltROI (rotated band)     │
                         └───────────────┬───────────────┘
                                          │  ROI locked
                                          ▼
                         ┌─────────────────────────────┐
                         │   belt_type_classifier.py     │
                         │  canopy color % + railing %   │
                         │  inside the ROI on a median    │
                         │  background frame              │
                         └───────────────┬───────────────┘
                             ROOFED ◄─────┴─────► OPEN
                        (stop, no tracking)   (continue)
                                                  │
                                                  ▼
                         ┌─────────────────────────────┐
                         │  status_detector.py (cont.)   │
                         │  MOG2 background subtraction  │
                         │  → contour filtering            │
                         │  → centroid tracking             │
                         │  → LOADING / UNLOADING / IDLE     │
                         │  → event CSV + snapshots           │
                         └─────────────────────────────┘
```

If the belt turns out to be **ROOFED**, `status_detector.py` prints a
message and **stops immediately** — a roof means the camera can't see
bags moving on the belt, so tracking would just produce noise.

---

## Repository layout

```
.
├── belt_detection.py          # Stage: locates the belt (rotated ROI)
├── belt_type_classifier.py    # Stage: roofed vs. open classification
├── status_detector.py         # Main entry point: tracking + status + events
├── requirements.txt
└── README.md
```

All three `.py` files must stay in the same directory — `status_detector.py`
imports from the other two, and `belt_type_classifier.py` imports
`AutoROIConfig` back from `status_detector.py` (a local, function-scoped
import used specifically to avoid a circular-import crash at module load
time).

---

## Requirements

- Python 3.9+ (uses `from __future__ import annotations` plus modern type
  hints like `list[np.ndarray] | None`)
- [OpenCV](https://opencv.org/) with GUI support (`opencv-python`) —
  needed for `cv2.imshow`/mouse callbacks if you use `--show_video` or
  manual ROI selection. If you're running headless (e.g. on a server /
  in Docker / over SSH without X11), use `opencv-python-headless`
  instead and always pass `--roi` so the manual-selection window is
  never invoked.
- NumPy
- A video file of a conveyor-belt loader (airport ramp camera footage,
  `.mp4`/`.avi`/etc. — anything OpenCV's `VideoCapture` can open)

No GPU is required; everything runs on CPU with classical OpenCV
operations (background subtraction, Hough transform, contour analysis).

---

## Local setup

```bash
# 1. Clone your repo (or just place the 3 files in a folder)
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Sanity-check OpenCV has GUI support (skip if running headless)
python -c "import cv2; print(cv2.__version__)"
```

`requirements.txt`:

```
opencv-python>=4.8.0
numpy>=1.24.0
```

If you're deploying headless (no display), swap the first line for
`opencv-python-headless` and always pass `--roi` (see [CLI
reference](#cli-reference)) so the script never tries to open a GUI
window for manual ROI selection.

---

## Quick start

Fully automatic — the script will auto-detect the belt, classify it, and
start tracking:

```bash
python status_detector.py --video path/to/ramp_footage.mp4
```

With a live preview window (runs at real-time playback speed):

```bash
python status_detector.py --video path/to/ramp_footage.mp4 --show_video
```

If auto-detection struggles on a difficult scene, click out the belt
polygon yourself on the first frame (a window pops up automatically when
no `--roi` is passed and auto-detection needs a hint), or pin down the
exact ROI from a previous run's console output:

```bash
python status_detector.py --video path/to/ramp_footage.mp4 \
  --roi "120,340 610,300 640,420 140,470"
```

Classify a belt's roof type on its own, without running full tracking:

```bash
python belt_type_classifier.py --videos clip1.mp4 clip2.mp4
```

---

## Module deep-dive

### `belt_detection.py` — Auto ROI detection

Defines `BeltROI` (a rotated band described by two axis endpoints —
`p_ground` and `p_hold` — and a half-width) and `BeltDetector`, which
finds that band automatically in a raw frame.

**Why this approach:** the module's docstring walks through four
candidate cues and why the final design fuses two of them:

| Cue | Strength | Weakness | Used as |
|---|---|---|---|
| Yellow color segmentation | Belt rails are high-vis yellow, rare elsewhere, survives night lighting | Yellow tarmac paint / hi-vis vests | Verification (capped vote) |
| Hough line/edge detection | Rails are the longest inclined structures in frame | Fuselage/engine edges also produce strong lines | Proposal (only inclined lines may propose an axis) |
| Shape/geometry (incline direction) | Belts always ramp upward toward the aircraft hold | — | Disambiguates which end is "hold" vs "ground" |
| Motion | Very reliable, but belt must be detectable while *idle* too | Not usable as a primary cue | Extra verification when available |

Key implementation details:

- **`_rail_color_evidence`** — HSV color threshold for yellow, then
  connected-component filtering keeps only elongated, roughly
  horizontal-ish blobs (rejects hi-vis vests, which are blobby, and most
  ground paint).
- **`_line_segments`** — Canny edges + probabilistic Hough transform,
  keeping only segments inclined between 3° and 50°. The 3° floor is
  what rejects flat tarmac stripes/lane markings, since belts always
  ramp.
- **`_detect_single`** — hypothesis-driven fitting: each long Hough
  segment seeds a candidate axis; segments that are angle-aligned and
  co-linear with it "vote" for that axis (rail + canopy edges
  reinforce each other); yellow-pixel proximity and motion proximity add
  further, capped support so no single pollutant (like a large painted
  stand marking) can dominate the vote. The winning hypothesis is then
  refined with a **robust Huber line fit** (`cv2.fitLine` with
  `cv2.DIST_HUBER`), which shrugs off residual outliers without any
  hand-tuned masking.
- **`detect`** — runs `_detect_single` across several sampled frames and
  **median-fuses** the resulting ROIs for stability, then rejects any
  update that jumps implausibly far in angle or position from the
  currently locked ROI (`roi_max_angle_jump_deg`,
  `roi_max_center_jump_frac`) so the lock doesn't flicker between the
  belt rails and a parallel canopy edge.
- **`BeltROI.strip_warp`** — an affine transform available for warping
  the rotated band into an axis-aligned strip, with the ground end
  always at `x=0` and the hold end at `x=strip_w` regardless of camera
  orientation (not currently used by `status_detector.py`, but handy for
  building strip-based analyses).

### `belt_type_classifier.py` — Roofed vs. open

Once a belt ROI is locked, this module decides whether the belt has a
canopy/roof over it.

**Why the earlier "look above the bbox" idea failed:** the belt's ROI
bounding box *already includes* any attached canopy structure. Looking
above the box measures whatever happens to sit behind it in the
scene — an aircraft engine (busy, high-edge-density) for open belts, or
sky/tarmac (nearly featureless) for roofed ones — which is almost the
exact *opposite* signal from what you want.

**The fix — look *inside* the box:**

1. **Canopy coverage ratio** — inside the belt polygon, on a
   **median background frame** (median-stacking sampled frames removes
   moving bags/handlers), count pixels matching canopy colors: solid
   blue/fiberglass (`H` 90–135 in HSV) or white/grey polycarbonate
   (low saturation, high value). Roofed belts show 55–70% coverage;
   open belts show under 33%.
2. **Exposed railing ratio** — count yellow rail-colored pixels
   (`H` 15–35) inside the same polygon. Open belts expose 7–24% yellow
   railing along their sides; roofed belts hide it under the canopy
   (≈0–3%).

**Decision rule** (`classify_from_roi`):

```python
is_roofed = (canopy_ratio > 0.42) or (canopy_ratio > 0.30 and railing_ratio < 0.02)
```

i.e. either canopy coverage alone is high enough, or moderate canopy
coverage *combined with* almost no visible railing (the railing check
catches borderline cases where canopy color alone is ambiguous).
Confidence is a simple linear function of how far the ratios sit from
the threshold, clamped to `[0.5, 1.0]`.

**`classify(video_path, ...)`** is the full standalone entry point: it
samples frames at four time offsets (0s, 15s, 30s, 45s — because a
belt may not be docked yet at the very start of a clip), runs
`BeltDetector` at each offset, and prefers the first offset where the
detected ROI looks like a *stably docked* belt (width ≥150px, incline
between 3° and 35°). If no offset produces a "stable" ROI, it falls back
to the first successful detection found at any offset.

**`classify_from_roi(frames, belt_roi, ...)`** is the version
`status_detector.py` actually calls — it skips re-detection and
classifies directly against an already-locked ROI.

**Diagnostic overlays** (`--save_overlays` / `save_roof_overlays`):
writes a JPEG per video to `roof_detection_output/<video_name>/` with
the canopy pixels highlighted in cyan/green, exposed railings in
orange, the ROI polygon in blue, and the classification result +
ratios printed on the image — useful for eyeballing *why* a belt was
classified the way it was.

### `status_detector.py` — Tracking, status, and events

The main script. Once the ROI is locked and the belt is classified as
**open**, it runs the live monitoring loop.

**Per-frame pipeline:**

1. *(Optional)* gamma correction for night footage, then Gaussian blur
   to suppress sensor noise/grain.
2. *(Optional)* CLAHE contrast enhancement on the L-channel (helps
   detect bags through glass/transparent belt covers).
3. **MOG2 background subtraction** (`cv2.createBackgroundSubtractorMOG2`)
   produces a foreground mask; thresholded at 200 to drop shadow pixels
   (MOG2's `detectShadows=True` marks shadows as gray ~127).
4. **Morphological cleanup:** `MORPH_OPEN` (5×5) removes small noise
   without destroying bag-sized fragments, then an aggressive
   `MORPH_CLOSE` (25×25) reconnects a single bag that got split into
   pieces by the railings crossing over it, followed by two dilation
   passes.
5. **Contour filtering** — found on the *entire* frame (so a large
   human isn't chopped off at the ROI edge), then rejected unless the
   contour:
   - has area within `[min_area, max_area]`,
   - has its centroid inside the belt ROI mask/polygon,
   - has aspect ratio ≤ 4.0 (rejects thin artifacts like railings or
     wing edges),
   - is **not** tall + near-vertical (aspect ratio > 1.5 and angle
     60°–120°) — a heuristic reject for standing humans,
   - has convex-hull solidity ≥ 0.70 — suitcases are rigid blocks
     (high solidity); human limbs/irregular shapes are not,
   - *(optional)* matches an expected belt-relative angle
     (`--belt_angle` ± `--angle_threshold`),
   - *(optional)* matches a reference suitcase's area/aspect ratio if
     one was captured interactively via the `s` key during playback.
6. **Tracking** — simple greedy centroid tracker (`TrackedObject`):
   each existing track is matched to the nearest unclaimed detection
   within `max_distance` pixels; unmatched detections spawn new tracks;
   unmatched tracks accumulate `frames_missing` and are dropped after
   `--max_missing` frames.
7. **Confirmation & direction** — a track only "counts" once it has been
   visible for `sustained_secs` (`CONFIRM_FRAMES = fps * sustained_secs`)
   **and** has moved at least 15px **and** that movement is
   roughly parallel to the belt axis (within `angle_threshold + 10°`
   tolerance). This filters out jitter and objects crossing the belt
   at an angle (e.g. a handler walking past).
8. **Status decision** — for all currently-confirmed, moving objects,
   the average `dx`/`dy` is computed. Whichever axis (horizontal or
   vertical) dominates determines the read direction, and sign
   determines `LOADING` vs `UNLOADING` vs `IDLE`. `IDLE` is
   **debounced**: it only latches after 2 full seconds with no
   confirmed moving objects, so momentary tracking gaps don't cause
   status flicker.
9. **Event logging** — every `IDLE → LOADING/UNLOADING` transition
   writes a timestamped row to `events.csv` plus a JPEG snapshot to
   `pipeline_output/<video_name>/`.

**Post-processing — `filter_events`:** raw per-frame transition events
are noisy, so after the video finishes, a sliding-window pass smooths
them:

- Slide a `window_size_seconds`-wide window (default 30s) across each
  event type's timestamps in 1-second steps.
- A window "confirms" once it contains at least `min_detections_in_window`
  (default 2) raw events of that type.
- Consecutive confirmed windows within `merge_gap_seconds` (default 15s)
  of each other are merged into one continuous interval.

The result — one clean start/end interval per real loading/unloading
episode — is written to `filtered_events.csv`.

**Interactive keys** (only relevant with `--show_video`):
- `q` — quit early
- `s` — drag-select a reference suitcase on screen; its area/aspect
  ratio become an extra detection filter for the rest of the run

---

## CLI reference

### `status_detector.py`

| Flag | Default | Description |
|---|---|---|
| `--video` | *(required)* | Path to the input video |
| `--min_area` | `800` | Minimum contour area (px²) to consider as a candidate object |
| `--max_area` | `25000` | Maximum contour area (px²) |
| `--sustained_secs` | `1.5` | Seconds an object must be tracked before it can influence status |
| `--belt_angle` | `30.0` | Expected belt incline in degrees (0 = horizontal, 90 = vertical); used to filter detections/movement by orientation |
| `--angle_threshold` | `20` | Max allowed deviation from `--belt_angle` |
| `--var_threshold` | `16` | MOG2 sensitivity — lower = more sensitive; try 4–8 for glass/transparent belt covers |
| `--max_missing` | `15` | Frames a track can go undetected before being dropped; raise for transparent-cover belts |
| `--use_clahe` | off | Apply CLAHE contrast boost pre-detection (helps see through glass covers) |
| `--brightness_gamma` | `1.0` | Gamma correction; >1.0 brightens shadows — useful for night footage |
| `--roi` | `None` | Exact 4-point polygon `"x1,y1 x2,y2 x3,y3 x4,y4"` to skip auto-detection and manual selection entirely |
| `--save_roof_overlays` | `True` | Save the belt-type diagnostic overlay JPEG |
| `--show_video` | off | Show a live playback window (runs at real-time speed; needed for the `s` reference-selection key) |

### `belt_type_classifier.py`

| Flag | Default | Description |
|---|---|---|
| `--videos` | *(required)* | One or more video paths to classify |
| `--canopy_threshold` | `0.45` | Canopy coverage threshold (stored on the result; the actual decision boundary used internally is `0.42`/`0.30`, see [Module deep-dive](#belt_type_classifierpy--roofed-vs-open)) |
| `--railing_threshold` | `0.04` | Railing ratio reference threshold (see note above) |

---

## Outputs

```
pipeline_output/<video_name>/
├── events.csv              # raw IDLE→LOADING/UNLOADING transitions, 1 row per event
├── filtered_events.csv     # de-noised, merged start/end intervals
└── event_XXX_*.jpg         # snapshot at the moment each raw event fired

roof_detection_output/<video_name>/
└── classification_overlay.jpg   # canopy/railing visual diagnostic

output_frames/
└── output_XXXX.jpg         # full-frame HUD snapshot every 30 frames
```

`events.csv` columns: `event_id, status, video_timestamp, frame_number, snapshot`

`filtered_events.csv` columns: `event_id, event_type, start_time, end_time, duration_seconds, detection_count`

---

## Tuning guide

- **Belt has a transparent/glass cover:** add `--use_clahe`, lower
  `--var_threshold` to `4`–`8`, and raise `--max_missing` (bags can
  briefly disappear under glare/reflections).
- **Night footage / underlit ramp:** set `--brightness_gamma` to
  `1.5`–`2.0`.
- **Too many false-positive detections (humans, shadows, wing edges):**
  tighten `--min_area`/`--max_area` around your actual bag size in
  pixels, or lower `--angle_threshold` if you know the belt's exact
  incline.
- **Auto ROI keeps drifting or locking onto the wrong structure:**
  capture the console's printed `--roi "..."` string from a good run
  and pass it back in directly — this also disables periodic
  re-detection entirely (`manual_roi_locked = True`), which is more
  robust for a fixed camera on a long unattended run.
- **A roofed belt is being misclassified as open (or vice versa):** run
  `belt_type_classifier.py --videos <clip> ` with overlays enabled and
  inspect `roof_detection_output/`, then adjust the underlying
  `canopy_ratio > 0.42` / `0.30` decision boundary in
  `belt_type_classifier.py` if your fleet uses unusual canopy/railing
  colors.

---

## Known limitations & gotchas

- **Roofed belts are not monitored at all** — by design, once a belt is
  classified `ROOFED`, `status_detector.py` prints a message and
  **exits the whole run**, since bag movement can't be seen under a
  canopy.
- The three files have a **circular import**: `belt_type_classifier.py`
  imports `AutoROIConfig` from `status_detector.py` inside a method body
  (not at module top-level) specifically to avoid a hard circular
  import at load time. Keep all three files together in one directory.
- Manual ROI selection and `--show_video` both require an OpenCV build
  with GUI support (`opencv-python`, not `opencv-python-headless`) and
  an available display. On headless servers, always pass `--roi`
  explicitly.
- The tracker is a simple greedy nearest-centroid matcher, not a
  Kalman/Hungarian-algorithm tracker — it works well for the
  sparse, slow-moving bag scenario here but isn't a general-purpose
  multi-object tracker.
- Detection thresholds (color ranges, area bounds, solidity, etc.) were
  tuned against airport ramp footage specifically; a different camera
  height/angle or lighting setup may need re-tuning.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
