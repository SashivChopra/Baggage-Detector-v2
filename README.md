# 🛄 Conveyor Belt Baggage Status Detector

> **A real-time computer vision pipeline that automatically detects baggage *loading* and *unloading* events on airport conveyor belts — even at night, in low contrast, or through metal railings.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat-square&logo=numpy&logoColor=white)

---

## 📖 Overview

Airport ground-ops teams need to know precisely **when** and **how long** a conveyor belt is loading or unloading baggage. Manual monitoring is expensive and error-prone.

This pipeline processes raw video footage of airport apron conveyor belts and produces a clean, timestamped CSV log of every loading and unloading session — fully automatically, with no manual annotation required.

**Key capabilities:**

- 🌙 Works in **day and night** footage (with gamma correction & CLAHE)
- 🏗️ Handles **metal railings** that fragment bag silhouettes (morphological reconstruction)
- 👷 Filters out **ground handlers** and human motion (solidity, aspect ratio, size filters)
- 🏠 Automatically **skips roofed/canopy belts** where detection is impossible (HSV classifier)
- 📐 **Auto-detects the belt ROI** — no manual region drawing needed
- ⚡ Runs in **headless mode** for fast bulk batch processing

---

## 🗂️ Project Structure

```
.
├── status_detector.py        # Main pipeline entrypoint — run this
├── belt_detection.py         # Auto-ROI belt surface detection
├── belt_type_classifier.py   # Stage 1: Open vs. Roofed belt classifier
├── pipeline_output/          # Per-video output (CSVs + event snapshots)
│   └── <video_name>/
│       ├── events.csv            # Raw per-detection log
│       └── filtered_events.csv   # Clean, merged session log
├── roof_detection_output/    # Diagnostic overlays from belt classifier
├── videos/                   # Input video files
└── sashiv_engineering_log.md # Engineering decisions & research log
```

---

## 🏗️ Pipeline Architecture

The system runs five sequential stages. Each stage is a prerequisite for the next.

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT VIDEO                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1 · Belt Type Classifier                             │
│  HSV Color-Ratio analysis on the first 1–2 seconds.        │
│  ► ROOFED → print status, halt immediately (saves CPU)      │
│  ► OPEN   → proceed to Stage 2                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2 · Auto-ROI Detection                               │
│  Locks a polygon tightly around the belt surface.           │
│  All subsequent detection is confined inside this mask.     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3 · Preprocessing & MOG2 Background Subtraction      │
│  Gaussian blur → Gamma correction → CLAHE (optional)        │
│  MOG2 produces a binary "motion mask" per frame.            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4 · Multi-Layer Blob Filtering & Tracking            │
│  Size → Aspect Ratio → Solidity → Angle → Reference Match   │
│  Centroid tracker requires N seconds of continuous motion   │
│  before a blob is confirmed as a bag.                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5 · Trajectory Analysis & Post-Processing Filter     │
│  Y-trajectory → LOADING / UNLOADING label                   │
│  Sliding-window noise filter → merged session CSV output    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   filtered_events.csv                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Stage-by-Stage Deep Dive

### Stage 1 — Belt Type Classifier (`belt_type_classifier.py`)

Before any expensive detection, the system performs a fast, one-shot classification of whether the belt has a **canopy/roof** structure.

**Why this matters:** Running MOG2, morphological ops, and contour tracking on a roofed belt wastes CPU and produces garbage detections (canopy shadows, reflections, and ground vehicles look like bags).

**Why GrabCut was rejected** (see [`sashiv_engineering_log.md`](./sashiv_engineering_log.md) for full analysis):

| Problem | Root Cause |
|---|---|
| Color bleed into walls | Airport canopies, walls, and tarmac share near-identical grey/blue palettes — GrabCut's GMM cannot distinguish them |
| Bounding-box instability | No clean rectangular boundary isolates the roof from belt/background |
| Iterative drift | GMM color model slowly consumes background pixels across iterations, causing the mask to expand or vanish |

**The working solution — HSV Color-Ratio Analysis:**

The classifier samples a median background frame, applies the Auto-ROI polygon mask, and measures two ratios inside the belt surface:

| Metric | OPEN Belt | ROOFED Belt |
|---|---|---|
| `canopy_ratio` (blue/white pixels in upper half) | < 33% | 55–70% |
| `railing_ratio` (yellow pixels across belt) | 7–24% | ≈ 0–3% |

`canopy_ratio > 42%` **or** `railing_ratio < 4%` → **ROOFED** → halt pipeline.

This approach achieves **100% accuracy** across all tested day and night videos.

---

### Stage 2 — Auto-ROI Detection (`belt_detection.py`)

The belt surface is automatically located using **colour thresholding + Hough Lines**. The result is a tight polygon (`BeltROI`) around the conveyor surface.

Benefits:
- Physically excludes the floor, standing zones, and aircraft body
- Any blob whose centroid falls outside the polygon is instantly rejected
- Can be overridden with a manual polygon via the interactive `--show_video` mode

---

### Stage 3 — Preprocessing & MOG2 Background Subtraction

Raw video feeds, especially at night, are too noisy to process directly.

| Step | Purpose |
|---|---|
| **Gaussian Blur** | Dissolves ISO sensor grain (night cameras) that appears as thousands of tiny moving pixels |
| **Gamma Correction** | Brightens dark/night frames before detection (`--brightness_gamma > 1.0`) |
| **CLAHE** | Boosts local contrast so dark bags on dark belts become visible (`--use_clahe`) |
| **MOG2** | Maintains a rolling background history and adapts to slow lighting changes; outputs a binary motion mask per frame |

**Why MOG2 over simple frame-differencing?**
Frame-differencing is fooled by any slow lighting change (shadows, clouds, aircraft movement). MOG2 *learns* the background over time and handles gradual changes gracefully.

---

### Stage 4 — Blob Filtering & Centroid Tracking

Once MOG2 produces a binary motion mask, morphological operations and a cascade of filters determine if each moving blob is a bag.

**Morphological Reconstruction (handles metal railings):**

As a bag slides behind railings, MOG2 "chops" it into disconnected fragments. Two passes fix this:
1. `MORPH_OPEN (5×5)` — deletes tiny floating noise pixels
2. `MORPH_CLOSE (25×25)` — bridges gaps from railings, reconnecting fragments into one solid block

**The Anti-Human Filter Cascade** (applied in sequence):

| Filter | Threshold | What It Rejects |
|---|---|---|
| **Size** | `min_area` – `max_area` | Hands (too small) and full human bodies (too large) |
| **ROI Mask** | Polygon boundary | Anything outside the belt surface |
| **Aspect Ratio** | > 4.0 | Long/thin reflections, rails |
| **Human Shape** | Aspect > 1.5 & angle 60–120° | Standing/leaning handlers |
| **Solidity** | < 0.70 | Bags are rigid rectangles (~0.90+ solidity); humans bending/reaching have jagged outlines (0.50–0.70) |
| **Angle** | ± `angle_threshold` from `belt_angle` | Blobs not physically rotated to match the belt's slant |
| **Reference Match** | Optional (press `S`) | Blobs that don't match a user-selected reference bag's size & shape |

**Centroid Tracker:**
Approved blobs are tracked frame-by-frame. A blob must be **continuously tracked for `sustained_secs`** (default: 1.5 s) before it can influence the belt status. This eliminates brief human-arm intrusions that momentarily pass the filters.

---

### Stage 5 — Trajectory Analysis & Post-Processing

**Direction detection:**
Once a bag is confirmed by the tracker, the system compares its starting Y-coordinate to its current Y-coordinate:
- `total_dy < 0` (moved **up** the screen) → belt pushing bags toward plane → **LOADING**
- `total_dy > 0` (moved **down** the screen) → belt pulling bags from plane → **UNLOADING**

**Sliding-Window Noise Filter:**
Raw frame-by-frame detections are sparse and noisy. After the video finishes, an automated filter:
1. Steps across the timeline in 1-second increments with a 30-second window
2. Marks windows with ≥ 2 detections as "confirmed"
3. Merges confirmed windows separated by < 15 seconds into a single continuous session
4. Outputs the result to `filtered_events.csv`

---

## ⚙️ Requirements

```bash
pip install opencv-python numpy
```

- Python 3.x
- OpenCV 4.x
- NumPy

---

## 🚀 Quick Start

### Headless (fast batch processing)
```bash
python3 status_detector.py --video videos/<your_video>.mp4
```

### Real-time (watch detections live)
```bash
python3 status_detector.py --video videos/<your_video>.mp4 --show_video
```

> `--show_video` opens a playback window with live bounding boxes, event banners, and belt status overlaid on the video. It forces 1× speed so you can watch exactly what the detector sees in real time. Press **Q** to quit or **S** to select a reference suitcase mid-playback.

---

## 📋 All Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--video` | str | **Required** | Path to the input video file |
| `--min_area` | int | `200` | Minimum contour area (px). Smaller objects are ignored (noise, hands, artifacts) |
| `--max_area` | int | `10000` | Maximum contour area (px). Larger objects are ignored (full human bodies) |
| `--sustained_secs` | float | `2.5` | Seconds an object must be **continuously tracked** before being counted. Higher = fewer false positives |
| `--belt_angle` | float | `0` | Expected physical rotation angle of bags on the belt (degrees). 0° = horizontal |
| `--angle_threshold` | float | `20` | Max allowed deviation from `belt_angle`. Set to `90` to disable angle filtering entirely |
| `--var_threshold` | int | `8` | MOG2 sensitivity. **Lower = more sensitive.** Use `4–8` for dark bags, night footage, or glass-covered belts |
| `--max_missing` | int | `15` | Max consecutive frames a tracked bag can disappear before the tracker drops it. Raise for railing-heavy or glass-covered belts |
| `--use_clahe` | flag | `false` | Apply CLAHE contrast enhancement before detection. Essential for dark bags on dark belts |
| `--brightness_gamma` | float | `1.0` | Gamma correction multiplier. Values > 1.0 brighten dark areas. Use `1.5`–`2.0` for night footage |
| `--show_video` | flag | `false` | Display real-time playback with bounding boxes (forces 1× speed). Omit for fast headless processing |
| `--save_roof_overlays` | flag | `true` | Save diagnostic overlay images from the roof/canopy classifier to `roof_detection_output/` |

---

## 🖱️ Interactive Controls

### ROI Selection (shown at startup with `--show_video`)

| Key / Action | Effect |
|---|---|
| **Left-click** | Add a polygon point |
| **Right-click** | Undo the last point |
| **Enter / Space** | Confirm the polygon (minimum 3 points) |
| **R** | Reset all points |
| **C** | Skip manual ROI — let auto-detection take over |

> **Tip:** Just press **C** and the system will automatically find the belt. Only draw manually if auto-detection fails on a particularly challenging video.

### During Playback

| Key | Effect |
|---|---|
| **Q** | Quit the detector |
| **S** | Pause and select a **reference suitcase** — the detector will lock onto objects matching that bag's exact size, shape, and angle |

---

## 🎯 Recommended Commands by Scenario

### Daytime — Standard
```bash
python3 status_detector.py \
  --video videos/conv_D02.mp4 \
  --belt_angle 0 \
  --angle_threshold 20 \
  --min_area 200 \
  --max_area 10000 \
  --sustained_secs 2.0 \
  --var_threshold 8
```

### Daytime — Dark-Colored Baggage
```bash
python3 status_detector.py \
  --video videos/conv_D03.mp4 \
  --belt_angle 0 \
  --angle_threshold 20 \
  --min_area 200 \
  --max_area 10000 \
  --sustained_secs 2.0 \
  --var_threshold 4 \
  --use_clahe
```

### Night Footage
```bash
python3 status_detector.py \
  --video videos/conv_N01.mp4 \
  --belt_angle 0 \
  --angle_threshold 20 \
  --min_area 200 \
  --max_area 10000 \
  --sustained_secs 2.0 \
  --var_threshold 4 \
  --use_clahe \
  --brightness_gamma 1.5 \
  --max_missing 25
```

> For extremely dark night videos, increase `--brightness_gamma` to `2.0`.

---

## 📂 Output

All output is saved to `pipeline_output/<video_name>/`.

| File | Description |
|---|---|
| `filtered_events.csv` | **Final clean output.** Merged loading/unloading sessions with timestamps and durations |
| `events.csv` | Raw per-detection log (ingested by the post-processing filter) |
| `event_XXX_STATUS_TIMESTAMP.jpg` | Snapshot of each raw detected event with an overlay banner |

Diagnostic overlays from the belt type classifier are saved to `roof_detection_output/<video_name>/classification_overlay.jpg`.

### Example `filtered_events.csv`

```csv
event_id,event_type,start_time,end_time,duration_seconds,detection_count
1,LOADING,00:08:40.73,00:08:51.18,10.45,3
2,UNLOADING,00:12:13.45,00:12:13.45,0.00,1
3,LOADING,00:12:41.09,00:12:41.09,0.00,1
4,UNLOADING,00:14:07.64,00:14:19.00,11.36,2
```

---

## 🛠️ Tuning Guide

| Symptom | Solution |
|---|---|
| Missing small bags | Lower `--min_area` (e.g., `200`) |
| Detecting humans / false positives | Lower `--max_area` (e.g., `8000`), raise `--sustained_secs` (e.g., `2.5`), draw a tighter ROI manually |
| Missing dark/black bags | Add `--use_clahe`, lower `--var_threshold` to `4` |
| Too much noise from empty-belt detections | Raise `--min_area`, raise `--var_threshold`, raise `--sustained_secs` |
| Bags flickering in/out (railings or glass cover) | Raise `--max_missing` (e.g., `25–40`) |
| Night video is too dark | Add `--brightness_gamma 1.5` or `2.0` |
| Angle filter rejecting all bags | Set `--angle_threshold 90` to disable it completely |
| LOADING/UNLOADING labels appear swapped | Camera is mounted on the opposite side — direction is relative to camera orientation |

---

## 🔍 Filter Pipeline Reference

All seven filters are applied sequentially on every contour, in this exact order:

1. **Size Filter** (`min_area` / `max_area`) — Rejects objects outside the expected suitcase size range
2. **ROI Mask** — Rejects blobs whose centroid falls outside the auto-detected belt polygon
3. **Aspect Ratio Filter** — Rejects extremely elongated shapes (ratio > 4.0), e.g., reflections and railings
4. **Human Shape Filter** — Rejects tall, vertically-oriented blobs (ratio > 1.5 & angle 60°–120°)
5. **Solidity Filter** — Rejects irregular shapes (solidity < 0.70). Suitcases score ~0.90+; bending/reaching humans score 0.50–0.70
6. **Angle Filter** (`belt_angle` / `angle_threshold`) — Rejects blobs not physically rotated to match the belt's slant
7. **Reference Match** *(optional, `S` key)* — Rejects blobs that don't match a user-selected reference bag's size and shape

---

## 📓 Engineering Notes

The full research and decision log — including a detailed comparison of GrabCut vs. HSV Color-Ratio for canopy classification and the exact failure modes of each approach — is in [`sashiv_engineering_log.md`](./sashiv_engineering_log.md).

**Author:** Sashiv Chopra — July 2026
