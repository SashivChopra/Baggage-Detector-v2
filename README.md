# Conveyor Belt Monitoring Pipeline

Computer-vision pipeline for airport baggage conveyor belts. Given a raw
video of a ramp scene, it locates the belt, figures out what kind of belt
it is, and then tracks bags to report **LOADING**, **UNLOADING**, or
**IDLE** status with timestamped events.

The pipeline runs in three sequential stages, in the order they actually
execute at runtime (renumbered here to match execution order rather than
each file's internal docstring labels):

```
Step 1  →  belt_detection.py        (BeltDetector)        — Auto ROI Detection
Step 2  →  belt_type_classifier.py  (BeltTypeClassifier)   — Belt Type Classification
Step 3  →  status_detector.py       (run_status_detector)  — Status Detection
                                     + orchestrates Steps 1 & 2 at runtime
```

You need a located belt (Step 1) before you can classify what kind of
belt it is (Step 2), and you need to know the belt type before tracking
sensitivity can be tuned correctly (Step 3). `status_detector.py` is the
entry point — it drives Steps 1 and 2 internally, then takes over with
tracking.

---

## Pipeline at a Glance

```
video frames
     │
     ▼
┌─────────────────────────┐
│ Step 1: Auto ROI         │  BeltDetector.detect()
│ (belt_detection.py)      │  → finds the belt's rails + axis, locks a
│                          │    rotated ROI band, re-detects periodically
└───────────┬──────────────┘
            │ BeltROI (ground end, hold end, half-width)
            ▼
┌─────────────────────────┐
│ Step 2: Belt Type        │  BeltTypeClassifier.classify_from_roi()
│ Classification           │  → ROOFED vs OPEN, using color evidence
│ (belt_type_classifier.py)│    *inside* the located ROI
└───────────┬──────────────┘
            │ BeltType (tunes downstream sensitivity)
            ▼
┌─────────────────────────┐
│ Step 3: Status Detector  │  run_status_detector()
│ (status_detector.py)     │  → background subtraction + bag tracking
│                          │    inside the ROI, emits LOADING/UNLOADING/
│                          │    IDLE + an events.csv log
└──────────────────────────┘
```

`status_detector.py` runs this whole loop per-frame: it keeps calling the
ROI detector until a belt is "connected," classifies it exactly once, then
switches into tracking mode and starts emitting status/events.

---

## Setup — Running This in Your Own Folder

### 1. Project layout

Put all three files in the **same directory** — they import from each
other by module name (`from belt_detection import ...`, etc.), not by
package path:

```
conveyor-belt-pipeline/
├── belt_detection.py
├── belt_type_classifier.py
├── status_detector.py
└── videos/                 # put your input .mp4 files here (optional)
```

### 2. Environment

A plain virtualenv is enough — there's no ML framework here, just OpenCV
+ NumPy:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install opencv-python numpy
```

If you're running headless (e.g. inside a Docker container / no display
server), swap in `opencv-python-headless` instead of `opencv-python` —
note this pipeline currently calls `cv2.imshow`/`cv2.waitKey` in
`status_detector.py`, so full `opencv-python` (with GUI support) is
required unless you strip those calls out for a server deployment.

### 3. Run it

```bash
cd conveyor-belt-pipeline
python status_detector.py --video videos/your_clip.mp4
```

No `--roi` flag needed for a fresh video — Step 1 (`belt_detection.py`)
will auto-detect the belt. Pass `--roi "X1,Y1 X2,Y2 X3,Y3 X4,Y4"` if you
want to lock a manual polygon instead (see the CLI section under Step 3
for the full flag list).

### 4. Output locations created at runtime

```
pipeline_output/<video_name>/events.csv        # LOADING/UNLOADING event log
pipeline_output/<video_name>/event_*.jpg       # snapshot per event
roof_detection_output/<video_name>/            # only with --save_roof_overlays
output_frames/                                 # periodic debug frame dumps
```

These directories are created automatically on first run — no manual
setup needed beyond the two steps above.

---

## Step 1 — Auto ROI Detection (`belt_detection.py`)

**Goal:** find the belt's rotated bounding region (the rail axis, from the
ground/feed end up to the aircraft/hold end) without any manual clicking,
and keep it locked and stable across frames.

### Why not the obvious cues?

The module's docstring walks through the reasoning:

| Cue | Verdict |
|---|---|
| Color (yellow hi-vis rails) | Good, but yellow tarmac paint / hi-vis vests are false positives |
| Straight-line/edge detection | Good, but fuselage and engine edges also produce long lines |
| Geometry (belts incline up toward the hold) | Used only to disambiguate which end is which |
| Motion | Belt must be detectable while **idle**, so motion can't be a primary cue |

**Chosen approach:** fuse color + line evidence, and use incline geometry to
resolve orientation.

### How detection works, step by step

1. **`_rail_color_evidence`** — threshold HSV for the yellow rail color,
   then filter at the *connected-component* level: keep only components
   that are elongated (`elong >= 2.0`) and wider than tall
   (`w >= 0.8*h`). This is what throws out hi-vis vests (blobby) and most
   painted ground markings.

2. **`_line_segments`** — run CLAHE (local contrast boost) → Canny edges →
   probabilistic Hough transform, then keep only segments with an incline
   between 3° and 50°. Painted tarmac markings are horizontal, so this
   floor alone rejects most paint.

3. **`_detect_single`** — the core per-frame hypothesis search:
   - Every long inclined Hough segment becomes a **candidate axis
     hypothesis**. Only these segments may *propose* an axis; ground
     paint is horizontal and structurally cannot compete.
   - Each hypothesis is scored by: (a) total length of other segments
     that share its angle and lie on the same line (rail + canopy edge
     reinforcement), (b) an amount of nearby yellow evidence, capped so
     it can *verify* but never *dominate* a proposal, (c) a small bonus
     for being lower in frame (prefers the belt over a canopy edge
     above it), (d) a continuity bonus if it agrees with the
     previously-locked ROI (damps flip-flopping between parallel
     structures across frames).
   - The winning hypothesis's supporting points (segment samples +
     nearby yellow pixels) are fit with `cv2.fitLine(..., DIST_HUBER, ...)`
     — a robust regression that shrugs off outliers (stray fuselage
     lines, paint) without any hand-tuned masking.
   - A final **extent refinement** re-derives the axis endpoints from
     *all* evidence in a wider corridor around the winning axis, since a
     single seed segment usually under-spans the true rail length.
   - Orientation is resolved geometrically: whichever endpoint has the
     smaller image `y` is the **hold end** (belts always ramp up toward
     the aircraft), the other is the **ground end**.

4. **`detect`** (multi-frame) — runs `_detect_single` over several sampled
   frames and **median-fuses** the resulting endpoints/half-widths, which
   smooths out single-frame noise. If a previous ROI exists, a sanity
   gate rejects any new result that jumps too far in angle or center
   (`roi_max_angle_jump_deg`, `roi_max_center_jump_frac`) — the old ROI is
   kept instead, so a single bad frame can't yank the lock around.

### `BeltROI` — the data structure everything downstream consumes

A `BeltROI` stores just `p_ground`, `p_hold`, and `halfwidth`, and derives
everything else:

- `angle_deg`, `length`, `center` — simple axis geometry.
- `box_points()` — the 4 corners of the rotated rectangle, for drawing/
  masking.
- `strip_warp()` — an affine transform that unrolls the rotated belt band
  into an axis-aligned strip, where **+x always means "toward the
  aircraft"** regardless of camera angle. (Defined here; consumed by
  anything that wants a canonical, camera-orientation-independent view
  of belt motion.)

---

## Step 2 — Belt Type Classification (`belt_type_classifier.py`)

**Goal:** decide whether the belt has a canopy/roof, because roofed vs.
open belts need different downstream detection sensitivity (a canopy adds
reflections/glare, dims contrast, etc.).

### Why not "look above the bbox"?

The original idea — inspect the region above the detected belt box for a
roofline — backfires:
- **Open belts** often have an aircraft engine/fuselage directly above
  them, which is highly textured → looks "busy," easily mistaken for a
  structure.
- **Roofed belts** may have plain sky/tarmac directly above → looks
  "empty."
That's the *opposite* of the correct signal.

### The fix: look *inside* the belt ROI itself

Since `BeltROI`'s box already encompasses any attached canopy structure,
the classifier analyzes pixels **inside** the polygon on a **median
background frame** (built by median-stacking several sampled frames,
which removes moving bags/handlers and leaves the static structure):

1. **Canopy Coverage Ratio** — fraction of the ROI matching canopy colors:
   blue fiberglass (HSV hue 90–135) or white/grey polycarbonate (low
   saturation, high value).
2. **Exposed Railing Ratio** — fraction of the ROI matching the yellow
   rail color. A canopy physically occludes the rails from camera view,
   so roofed belts show ≈0–3% yellow, while open belts expose 7–24%.

### Decision rule

```python
is_roofed = (canopy_ratio > 0.42) or (canopy_ratio > 0.30 and railing_ratio < 0.02)
```

i.e. either canopy coverage is high on its own, or it's moderately high
*and* corroborated by an absence of exposed rails. Confidence is derived
from how far the ratios sit from the threshold, clamped to `[0.5, 1.0]`.

### Two entry points

- `classify(video_path, ...)` — standalone use. Scans multiple time
  offsets (0s/15s/30s/45s, or a hardcoded per-video offset for known
  sample clips) looking for a frame window where a *stably docked* belt
  (wide enough box, plausible incline angle) is visible, runs
  `BeltDetector` itself to get an ROI, then classifies.
- `classify_from_roi(frames, belt_roi, ...)` — used by the live pipeline,
  which passes in the ROI that Step 1 already found, skipping re-detection.

`save_overlays=True` writes an annotated diagnostic JPEG (canopy pixels in
cyan/green, exposed rails in orange, ROI polygon in blue) to
`roof_detection_output/<video_name>/classification_overlay.jpg`.

---

## Step 3 — Status Detection (`status_detector.py`)

**Goal:** the actual runtime loop — orchestrates Steps 1 & 2, then tracks
bags on the belt and reports status + logs events.

### Orchestration inside `run_status_detector`

Per frame:

1. **ROI acquisition** — if no manual `--roi` was given, frames are
   buffered until `roi_sample_frames` (5) accumulate, then
   `BeltDetector.detect()` is called. Once a belt is found
   (`belt_connected = True`), auto-detection still re-runs every 10s to
   keep the lock current (unless a manual ROI was passed, which disables
   auto re-detection entirely). A few sample videos have hardcoded
   default ROI polygons/timestamps baked in for guaranteed results.
2. **Classification (once)** — the moment the belt is first connected,
   `belt_classifier.classify_from_roi()` runs exactly once. If the belt
   is `ROOFED`, the background-subtractor's `varThreshold` is floored at
   12 (i.e., made *more* sensitive if it wasn't already) to compensate
   for the canopy's contrast/reflection effects.
3. **Gating** — no bag tracking or status changes happen until a belt is
   connected; the frame just displays "WAITING FOR BELT..." until then.

### Bag detection & tracking (once the belt is connected)

- **Preprocessing:** optional gamma correction (night videos), Gaussian
  blur (sensor noise), optional CLAHE-in-LAB (helps see through
  glass/transparent covers).
- **Background subtraction:** `MOG2` → threshold → morphological open
  (5×5, denoise) → morphological close (25×25, reconnects a bag's
  fragments that got split by the rail bars in front of it).
- **Contour filtering** — a detected blob only becomes a candidate bag if
  it passes *all* of:
  - area within `[min_area, max_area]`
  - centroid inside the ROI mask/polygon
  - aspect ratio ≤ 4.0 (rejects thin rail/edge artifacts)
  - **not** shaped like a standing human (tall aspect ratio *and*
    oriented near-vertical)
  - solidity ≥ 0.70 (rigid suitcase blocks vs. irregular human/limb
    silhouettes)
  - orientation within `angle_threshold` of the expected belt angle, if
    one is configured
  - (optional) area/aspect close to a user-captured reference object,
    set live via the `s` hotkey
- **Tracking:** simple nearest-centroid association (`max_distance`)
  frame-to-frame, building up a `TrackedObject` history. An object only
  starts influencing status once it's been visible for
  `sustained_secs` (`CONFIRM_FRAMES`), and only counts as "moving" once
  its net displacement since first-seen exceeds 15px **and** that
  displacement is roughly parallel to the belt axis (rejects sideways/
  incidental motion).

### Status decision

Averaged `(dx, dy)` across all currently-confirmed moving objects decides
direction along whichever axis (x or y) dominates:
- moving toward the aircraft/hold end → `LOADING`
- moving toward the ground/feed end → `UNLOADING`
- no confirmed moving objects for 2 full seconds (debounced) → `IDLE`

### Event logging

A transition from `IDLE → LOADING/UNLOADING` creates a timestamped event:
a snapshot JPEG and a row in `pipeline_output/<video_name>/events.csv`
(`event_id, status, video_timestamp, frame_number, snapshot`). Repeated
transitions of the *same* type within `EVENT_MERGE_GAP_SECS` (10s) are
merged into the existing event rather than creating a new one, to avoid
spamming near-duplicate events during a single continuous loading session.

### CLI

```bash
python status_detector.py --video conv_full_D01.mp4 \
    [--min_area 800] [--max_area 25000] [--sustained_secs 1.5] \
    [--belt_angle 30.0] [--angle_threshold 20] \
    [--var_threshold 16] [--max_missing 15] \
    [--use_clahe] [--brightness_gamma 1.0] \
    [--roi "X1,Y1 X2,Y2 X3,Y3 X4,Y4"] \
    [--save_roof_overlays]
```

Pass `--roi` to lock a manual polygon (disables periodic auto-detection);
omit it to let Step 1 find and continuously re-verify the belt on its own.

---

## Key Design Ideas Worth Noting

- **Hypothesis-driven fusion, not pooled evidence.** `belt_detection.py`
  never fits a single line through all yellow+edge pixels pooled
  together; it lets *inclined structure* propose axis candidates and lets
  color only *verify* (with a capped vote), so a scene dominated by
  ground paint or fuselage clutter can't hijack the fit.
- **Classify inside the ROI, not around it.** `belt_type_classifier.py`'s
  key insight is that the belt's own bounding box already contains any
  attached canopy — looking *outside* it measures the wrong thing
  entirely (background clutter, not belt structure).
- **Canonical motion direction.** `BeltROI.strip_warp()` normalizes belt
  motion to a fixed "+x = toward the aircraft" convention, independent of
  how the camera happens to be oriented in a given install.
- **Everything is gated on a stable lock.** Both the ROI (`detect()`'s
  jump-rejection) and the belt type (`belt_classified` computed once) are
  designed to resist being perturbed by noisy individual frames — status
  detection only starts once there's a durable geometric and semantic
  understanding of the scene.
