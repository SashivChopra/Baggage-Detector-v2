# Conveyor Belt Baggage Status Detector

A real-time computer vision pipeline that detects baggage **loading** and **unloading** events on airport conveyor belts using background subtraction (MOG2), contour analysis, and object tracking.

---

## Pipeline Architecture (In-Depth)

This system is designed to handle challenging, real-world airport environments involving camera grain, low light, physical obstructions (like railings), and human interference. Here is exactly how and why each stage works:

### 1. Preprocessing (Noise & Contrast Control)
Raw video feeds are often too dirty to process directly, especially at night.
- **Gaussian Blur**: Night cameras generate heavy ISO sensor noise (static/grain), which looks like thousands of tiny moving objects to a computer. We apply a Gaussian Blur to melt this grain away before any detection happens.
- **Gamma & CLAHE**: Bags are often dark grey/black, moving on dark grey/black belts. Gamma correction brightens the entire frame, while CLAHE (Contrast Limited Adaptive Histogram Equalization) aggressively forces contrast in low-light areas so the bag edges pop out.

### 2. ROI Isolation
By drawing a polygon strictly around the conveyor belt, we physically exclude the floor and standing zones. Any motion detected outside this blue box is instantly ignored.

### 3. Background Subtraction (MOG2)
We use `cv2.createBackgroundSubtractorMOG2`. 
- **Why MOG2?** Unlike simple frame-differencing, MOG2 maintains a rolling "history" of the background. It learns what the static belt looks like over time and adapts to slow changes in lighting or shadows, returning a binary mask where white pixels represent movement.

### 4. Morphological Reconstruction
Conveyor belts often have metal railings running across them. As a bag moves behind these railings, the background subtractor "chops" the bag into small disconnected pieces. 
- **MORPH_OPEN (5x5)**: A gentle pass to delete tiny floating noise pixels without destroying the bag fragments.
- **MORPH_CLOSE (25x25)**: An aggressive, large-kernel pass that bridges the gaps created by the railings, reconnecting the fragmented bag pieces back into one massive, solid block.

### 5. Multi-Stage Filtering (The "Anti-Human" Checks)
Once we have moving blobs (contours), we must determine if they are bags or humans (handlers loading the belt). We run these tests:
- **Size (`min_area` / `max_area`)**: A handler's body is massive; a hand is tiny. We only accept blobs the size of a standard suitcase.
- **Aspect Ratio**: We reject extremely long/thin shapes (aspect ratio > 4.0), which are usually reflections on wing edges or belt rails.
- **Solidity (> 0.70)**: This is our most powerful anti-human filter. Solidity measures how "blocky" an object is. Suitcases are rigid rectangles with >90% solidity. Humans bending, reaching, or walking have highly irregular, jagged outlines (50-70% solidity) and are instantly rejected.
- **Angle (Optional)**: Rejects bags that aren't physically rotated to match the belt's slant. 

### 6. Temporal Object Tracking
We don't trust single frames. We use **centroid-based tracking** to follow approved blobs frame-by-frame.
- **Why?** A handler's arm might briefly swing into the frame and pass the filters for a split second. By requiring the object to be tracked continuously for `sustained_secs` (e.g., 2.0 seconds), we guarantee that brief human interference is ignored, while real bags riding the belt are verified.

### 7. Trajectory & Status Logic
Once a bag is "confirmed" by the tracker, the script calculates its physical trajectory to determine the belt's status.
- It compares the bag's starting Y-coordinate to its current Y-coordinate.
- If it has moved **UP** the screen (`total_dy < 0`), the belt is pushing things into the plane: **LOADING**.
- If it has moved **DOWN** the screen (`total_dy > 0`), the belt is pulling things out of the plane: **UNLOADING**.

---

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

---

## Quick Start

```bash
python3 status_detector.py --video videos/<your_video>.mp4
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--video` | str | **Required** | Path to the input video file |
| `--min_area` | int | `800` | Minimum contour area (in pixels) to detect. Objects smaller than this are ignored (filters out noise, hands, small artifacts) |
| `--max_area` | int | `25000` | Maximum contour area (in pixels) to detect. Objects larger than this are ignored (filters out full human bodies) |
| `--sustained_secs` | float | `1.5` | How many seconds an object must be continuously tracked before it can influence the belt status. Higher values reduce false positives from brief human motion |
| `--belt_angle` | float | `30.0` | Expected physical rotation angle of objects on the belt in degrees (0° = horizontal, 90° = vertical). **Note**: This filters based on how the bag is *rotated*, not the direction it moves |
| `--angle_threshold` | float | `20` | Maximum allowed deviation from `belt_angle` in degrees. Set to `90` to effectively disable angle filtering |
| `--var_threshold` | int | `16` | Background subtractor sensitivity. **Lower = more sensitive**. Use `4–8` for low-contrast scenes (glass covers, dark bags, night footage) |
| `--max_missing` | int | `15` | Max consecutive frames a tracked object can disappear before the tracker drops it. Raise for belts with railings or transparent covers where bags flicker in/out |
| `--use_clahe` | flag | `false` | Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) before detection. Boosts local contrast to detect dark bags on dark belts or bags behind glass |
| `--brightness_gamma` | float | `1.0` | Gamma correction for dark/night videos. Values `> 1.0` brighten dark areas (e.g., `1.5` to `2.0`). Use `1.0` for daytime |

---

## ROI Selection Controls

When the video starts, a window titled **"Select Belt ROI"** will appear:

| Key / Action | Effect |
|---|---|
| **Left-click** | Add a polygon point |
| **Right-click** | Undo the last point |
| **ENTER / SPACE** | Confirm the polygon (minimum 3 points) |
| **R** | Reset all points |
| **C** | Skip ROI — use the full frame |

> **Tip**: Draw the polygon **tightly around the conveyor belt surface only**. Keeping floor areas and handler standing zones outside the polygon is the most effective way to eliminate human false positives.

---

## Runtime Keyboard Controls

While the video is playing:

| Key | Effect |
|---|---|
| **Q** | Quit the detector |
| **S** | Pause and select a **reference suitcase**. The detector will lock onto objects matching that bag's size, shape, and angle |

---

## Recommended Commands

### Daytime Videos

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

### Daytime Videos with Dark-Colored Baggage

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

### Night Videos

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

> If the night video is extremely dark, increase `--brightness_gamma` to `2.0`.

---

## Output

All output is saved to `pipeline_output/<video_name>/`:

| File | Description |
|---|---|
| `events.csv` | Log of all detected events with columns: `event_id`, `status`, `video_timestamp`, `frame_number`, `snapshot` |
| `event_XXX_STATUS_TIMESTAMP.jpg` | Snapshot of each detected event with a banner overlay showing event details |

### Example `events.csv`

```csv
event_id,status,video_timestamp,frame_number,snapshot
1,UNLOADING,00:01:51.00,1221,event_001_UNLOADING_00015100.jpg
2,UNLOADING,00:02:07.91,1407,event_002_UNLOADING_00020791.jpg
3,UNLOADING,00:02:36.91,1726,event_003_UNLOADING_00023691.jpg
```

---

## Tuning Tips

| Problem | Solution |
|---|---|
| Missing small bags | Lower `--min_area` (e.g., `200`) |
| Detecting humans | Lower `--max_area` (e.g., `10000`), increase `--sustained_secs` (e.g., `2.0`), draw tighter ROI |
| Missing dark bags | Add `--use_clahe`, lower `--var_threshold` to `4` |
| Too much noise / false detections | Raise `--min_area`, raise `--var_threshold`, raise `--sustained_secs` |
| Bags flickering in/out (railings, glass) | Raise `--max_missing` (e.g., `25`) |
| Night video too dark | Add `--brightness_gamma 1.5` or `2.0` |
| Angle filter rejecting all bags | Set `--angle_threshold 90` to disable it |

---

## Built-in Filters

The pipeline applies these filters sequentially on every detected contour:

1. **Size Filter** (`min_area` / `max_area`) — Rejects objects outside the expected bag size range
2. **ROI Mask** — Rejects objects whose centroid falls outside the drawn polygon
3. **Aspect Ratio Filter** — Rejects extremely long/thin shapes (e.g., railings) with aspect ratio > 4.0
4. **Human Filter** — Rejects tall, vertically-oriented objects (aspect ratio > 1.5 and angle between 60°–120°)
5. **Solidity Filter** — Rejects irregular shapes (solidity < 0.70). Bags are solid blocks (~0.90+), humans have irregular outlines
6. **Angle Filter** (`belt_angle` / `angle_threshold`) — Rejects objects whose physical rotation doesn't match the expected belt angle
7. **Reference Match** (optional, via `S` key) — Rejects objects that don't match the size/shape of a user-selected reference bag
