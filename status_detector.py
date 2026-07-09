import cv2
import numpy as np
import argparse
import math
import os
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict
import csv

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def filter_events(events: List[Dict], window_size_seconds: float = 30.0, min_detections_in_window: int = 2, merge_gap_seconds: float = 15.0) -> List[Dict]:
    """
    Takes raw event list and returns cleaned start/end times.
    
    The sliding window moves across the timeline in steps. At each position,
    if the window contains >= min_detections_in_window detections of the same
    type, that window is "confirmed". Consecutive confirmed windows separated
    by less than merge_gap_seconds are merged into a single event.
    """
    if not events:
        return []

    events_by_type = defaultdict(list)
    for ev in events:
        events_by_type[ev["event_type"]].append(float(ev["timestamp"]))

    all_filtered = []

    for event_type, timestamps in events_by_type.items():
        timestamps.sort()
        if not timestamps:
            continue

        confirmed_intervals = []
        t_start = timestamps[0]
        t_end = timestamps[-1]

        step = 1.0  # 1-second step
        pos = t_start

        while pos <= t_end:
            window_end = pos + window_size_seconds
            count = sum(1 for t in timestamps if pos <= t <= window_end)

            if count >= min_detections_in_window:
                detections_in_window = [t for t in timestamps if pos <= t <= window_end]
                interval_start = detections_in_window[0]
                interval_end = detections_in_window[-1]
                confirmed_intervals.append((interval_start, interval_end, count))

            pos += step

        if not confirmed_intervals:
            continue

        confirmed_intervals.sort(key=lambda x: x[0])
        merged = []
        current_start, current_end, current_count = confirmed_intervals[0]

        for i in range(1, len(confirmed_intervals)):
            next_start, next_end, next_count = confirmed_intervals[i]
            if next_start - current_end <= merge_gap_seconds:
                current_end = max(current_end, next_end)
                current_count = max(current_count, next_count)
            else:
                merged.append((current_start, current_end, current_count))
                current_start, current_end, current_count = next_start, next_end, next_count

        merged.append((current_start, current_end, current_count))

        for m_start, m_end, _ in merged:
            actual_count = sum(1 for t in timestamps if m_start <= t <= m_end)
            all_filtered.append({
                'start_time': m_start,
                'end_time': m_end,
                'event_type': event_type,
                'detection_count': actual_count,
            })

    all_filtered.sort(key=lambda x: x['start_time'])
    return all_filtered

from belt_detection import BeltROI, BeltDetector
from belt_type_classifier import BeltTypeClassifier, BeltType

@dataclass
class AutoROIConfig:
    rail_hsv_lo: np.ndarray = field(default_factory=lambda: np.array([10, 40, 40])) # broadened for night/dark video rails
    rail_hsv_hi: np.ndarray = field(default_factory=lambda: np.array([45, 255, 255]))
    roi_min_points: int = 15
    roi_halfwidth_frac: float = 0.15  # Tighter bounding box around the rails
    roi_hypo_band_frac: float = 0.05
    roi_max_angle_jump_deg: float = 15.0
    roi_max_center_jump_frac: float = 0.15
    roi_sample_frames: int = 5
    roi_redetect_every_s: float = 20.0

def polygon_to_belt_roi(poly: np.ndarray) -> BeltROI:
    rect = cv2.minAreaRect(poly.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    long_len, short_len = max(w, h), min(w, h)
    theta = np.radians(angle if w >= h else angle + 90.0)
    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    c = np.array([cx, cy], dtype=np.float32)
    e1, e2 = c - u * long_len / 2.0, c + u * long_len / 2.0
    p_hold, p_ground = (e1, e2) if e1[1] < e2[1] else (e2, e1)
    return BeltROI(p_ground=p_ground, p_hold=p_hold, halfwidth=short_len / 2.0)

def select_belt_roi(frame):
    """Returns polygon points (N,2) int32, or None if skipped/unavailable."""
    clone = frame.copy()
    h, w = frame.shape[:2]
    points: list[tuple[int, int]] = []
    mouse_pos = [w // 2, h // 2]

    def draw_state():
        out = clone.copy()
        for i, p in enumerate(points):
            cv2.circle(out, p, 5, (0, 220, 255), -1)
            if i > 0:
                cv2.line(out, points[i - 1], p, (0, 220, 255), 2)
        if len(points) >= 3:
            cv2.line(out, points[-1], points[0], (0, 220, 255), 1)
            overlay = out.copy()
            cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], (0, 180, 255))
            cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
        mx, my = mouse_pos
        if points:
            cv2.line(out, points[-1], (mx, my), (100, 220, 255), 1)
        cv2.line(out, (mx, 0), (mx, h), (0, 255, 255), 1)
        cv2.line(out, (0, my), (w, my), (0, 255, 255), 1)
        cv2.putText(out, f"({mx},{my})", (mx + 6, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(out, f"Points: {len(points)}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        cv2.putText(out, "L-click=add  R-click=undo  ENTER=confirm  R=reset  C/ESC=auto",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow("Select Belt ROI", out)

    def on_mouse(event, mx, my, flags, param):
        mouse_pos[0], mouse_pos[1] = mx, my
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((mx, my))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
        draw_state()

    try:
        cv2.namedWindow("Select Belt ROI", cv2.WINDOW_NORMAL)
        draw_state()
        cv2.setMouseCallback("Select Belt ROI", on_mouse)
    except cv2.error:
        print("[roi] no GUI available - manual selection skipped")
        return None

    confirmed = False
    while True:
        draw_state()
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and len(points) >= 3:
            confirmed = True
            break
        if key in (ord("r"), ord("R")):
            points.clear()
        elif key in (ord("c"), ord("C"), 27):
            break
    cv2.destroyWindow("Select Belt ROI")
    return np.array(points, dtype=np.int32) if confirmed else None

class TrackedObject:
    def __init__(self, obj_id, centroid, bbox, frame_idx):
        self.id = obj_id
        self.centroids = [centroid]
        self.bboxes = [bbox]
        self.frame_indices = [frame_idx]
        self.frames_visible = 1
        self.frames_missing = 0
        self.reported = False
        self.finished = False

def run_status_detector(video_path, min_area=200, max_area=7000, max_distance=60,
                        sustained_secs=2, belt_angle=None, angle_threshold=20,
                        var_threshold=16, use_clahe=False, brightness_gamma=1.0, max_missing=15,
                        roi_str=None, save_roof_overlays=False, show_video=False):
    # Belt type classifier initialized; classification deferred until belt connects
    belt_classifier = BeltTypeClassifier()
    belt_type = BeltType.UNKNOWN
    belt_classified = False
    belt_connected = False

    # ── STAGE 2 & 3: Auto ROI + Status Detection ─────────────────────────
    print(f"Running Status Detector on {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25.0
        
    # varThreshold: lower = more sensitive (good for glass/low contrast). Higher = less noise.
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=var_threshold, detectShadows=True)
    frame_delay = max(1, int(1000 / fps))  # ms to wait per frame to match real playback speed

    # CLAHE boosts local contrast — helps detect suitcases through transparent glass
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

    # Gamma correction table (for night videos)
    gamma_lut = None
    if brightness_gamma != 1.0:
        inv_gamma = 1.0 / brightness_gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_lut = table
    
    frame_count = 0
    next_object_id = 0
    tracked_objects = {}
    
    # Status detection variables
    status = "IDLE"
    status_color = (150, 150, 150)
    prev_status = "IDLE"  # track transitions
    idle_debounce_counter = 0
    recent_arrivals = deque(maxlen=int(fps * 3)) # 3 second window
    recent_departures = deque(maxlen=int(fps * 3))

    # Event logging setup
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    event_dir = os.path.join("pipeline_output", video_name)
    os.makedirs(event_dir, exist_ok=True)
    events_csv = os.path.join(event_dir, "events.csv")
    with open(events_csv, 'w') as f:
        f.write("event_id,status,video_timestamp,frame_number,snapshot\n")
    event_count = 0
    print(f"Events will be logged to: {events_csv}")

    # Variables for periodic auto-ROI
    belt_detector = BeltDetector(AutoROIConfig())
    next_redetect_t = 0.0
    roi_buffer = []
    current_belt_roi = None
    manual_roi_locked = False
    motion_history = None

    # Object must be visible for >= sustained_secs before it influences the status.
    # This allows for very tight ROIs where the bag is only visible for a split second.
    CONFIRM_FRAMES = int(fps * sustained_secs)

    
    roi_mask = None
    roi_coords = None
    # ref_area / ref_aspect_ratio set by 's' key (optional refinement)
    ref_area = None
    ref_aspect_ratio = None
    # ref_angle initialised from --belt_angle arg; 's' key can override it later
    ref_angle = float(belt_angle) % 180 if belt_angle is not None else None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        t = frame_count / fps

        if frame_count == 0:
            if roi_str is not None:
                coords = roi_str.replace(" ", ",").split(",")
                coords = [int(v) for v in coords if v]
                if len(coords) == 8:
                    roi_poly = np.array([
                        [coords[0], coords[1]],
                        [coords[2], coords[3]],
                        [coords[4], coords[5]],
                        [coords[6], coords[7]]
                    ], dtype=np.int32)
                else:
                    print("Invalid ROI coordinates length. Falling back to manual selection.")
                    roi_poly = select_belt_roi(frame)
            else:
                choice = ""
                while choice not in ['1', '2']:
                    print("\n" + "="*50)
                    print("ROI SELECTION MODE:")
                    print("1. Manual ROI (Draw on screen)")
                    print("2. Auto ROI (Detect automatically)")
                    print("="*50)
                    choice = input("Select an option [1 or 2]: ").strip()
                
                if choice == '1':
                    roi_poly = select_belt_roi(frame)
                else:
                    roi_poly = None
            
            if roi_poly is not None:
                current_belt_roi = polygon_to_belt_roi(roi_poly)
                belt_connected = True
                
                manual_roi_locked = True
                if roi_str is not None:
                    print("Manual ROI locked from command line. Periodic auto-detection disabled.")
                else:
                    print("Manual ROI locked from mouse selection. Periodic auto-detection disabled.")
                    
                roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(roi_mask, [roi_poly], 255)
                roi_coords = roi_poly
            else:
                print("No manual ROI selected. Automatic detection enabled.")

        # --- PERIODIC AUTO ROI ---
        if not manual_roi_locked and (not belt_connected or t >= next_redetect_t):
            roi_buffer.append(frame.copy())
            if len(roi_buffer) >= belt_detector.cfg.roi_sample_frames:
                sample_frames_for_class = list(roi_buffer)
                new_belt = belt_detector.detect(roi_buffer, previous=current_belt_roi)
                roi_buffer = []
                next_redetect_t = t + 10.0  # Force 10 second update
                
                if new_belt is not None:
                    current_belt_roi = new_belt
                    belt_connected = True
                    roi_poly = current_belt_roi.box_points().astype(np.int32)
                    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(roi_mask, [roi_poly], 255)
                    roi_coords = roi_poly
                    pts_str = " ".join([f"{pt[0]},{pt[1]}" for pt in roi_poly])
                    print(f"[roi] Auto-detected new ROI at t={t:.1f}s")
                    print(f"[roi] To reuse this ROI, pass: --roi \"{pts_str}\"")
                else:
                    if current_belt_roi is None and frame_count < fps * 2:
                        print("Initial auto-detection failed, will retry...")

        # --- CHECK IF BELT IS ROOFED OR OPEN ONCE CONNECTED ---
        if belt_connected and not belt_classified and current_belt_roi is not None:
            print(f"[pipeline] Belt connected! Running Stage 1: Classifying belt type...")
            classification = belt_classifier.classify_from_roi(
                sample_frames_for_class if 'sample_frames_for_class' in locals() and sample_frames_for_class else [frame],
                current_belt_roi,
                video_path,
                save_overlays=save_roof_overlays
            )
            belt_type = classification.belt_type
            belt_classified = True
            print(f"[pipeline] Belt type: {belt_type.value} (confidence={classification.confidence:.2f})")
            if belt_type == BeltType.ROOFED:
                print("[pipeline] Roofed belt detected — Baggage detection will be disabled.")
            elif belt_type == BeltType.OPEN:
                print("[pipeline] Open belt detected — Baggage detection enabled.")

        # --- WAIT FOR BELT TO CONNECT BEFORE STATUS DETECTION ---
        if not belt_connected or current_belt_roi is None:
            # We do not count baggage or change status until the conveyor belt is connected!
            frame_count += 1
            continue

        # --- DISABLE BAGGAGE DETECTION FOR ROOFED BELTS ---
        if belt_classified and belt_type == BeltType.ROOFED:
            print("[pipeline] Roofed belt detected. Stopping detection immediately.")
            break

        # Apply Gamma Correction for night videos
        if gamma_lut is not None:
            frame = cv2.LUT(frame, gamma_lut)
                
        # Apply Gaussian Blur to kill camera sensor noise/grain (crucial for night videos)
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Optionally enhance contrast (CLAHE) before background subtraction
        # This helps detect objects through transparent glass/plastic covers
        if clahe is not None:
            lab = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            enhanced = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            fgMask = backSub.apply(enhanced)
        else:
            fgMask = backSub.apply(blurred_frame)
        _, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
        
        # Draw ROI polygon for visual reference
        if roi_coords is not None:
            cv2.polylines(frame, [roi_coords], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Noise reduction + fragment reconnection
        # 1. MORPH_OPEN with 5x5 removes small noise without destroying bag fragments
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)
        
        # 2. MORPH_CLOSE with 25x25 aggressively reconnects bag fragments split by railings
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        
        thresh = cv2.dilate(thresh, open_kernel, iterations=2)
        
        # Find contours on the ENTIRE frame so large humans aren't chopped by the ROI edge
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_detections = []
        for contour in contours:
            area_c = cv2.contourArea(contour)
            # Filter by size: Evaluate the FULL body/object size first!
            if area_c < min_area or area_c > max_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            rect = cv2.minAreaRect(contour)
            (cx_rect, cy_rect), (width_r, height_r), angle = rect
            
            # Now, only process objects whose centroid is actually inside the Active Zone (ROI)
            if roi_mask is not None:
                cx_int, cy_int = int(cx_rect), int(cy_rect)
                if cy_int < 0 or cy_int >= roi_mask.shape[0] or cx_int < 0 or cx_int >= roi_mask.shape[1]:
                    continue
                if roi_mask[cy_int, cx_int] == 0:
                    continue
            
            if width_r <= 0 or height_r <= 0:
                continue
                
            aspect_ratio = max(width_r, height_r) / min(width_r, height_r)
            
            # Filter out extremely long/thin artifacts like railings or wing edges
            if aspect_ratio > 4.0:
                continue
                
            if width_r < height_r:
                angle = angle + 90
            normalized_angle = angle % 180
            
            # Robust Precision Filter 0: Reject Standing Humans
            # A human is tall (aspect_ratio > 1.5) and vertically oriented (angle roughly 90 degrees)
            if aspect_ratio > 1.5 and (60 < normalized_angle < 120):
                continue
                
            # Robust Precision Filter 1: Solidity
            # Suitcases are rigid blocks (Solidity usually > 0.85). 
            # Humans, arms, and legs have irregular shapes with lower solidity.
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area_c / hull_area
                if solidity < 0.70:
                    continue  # Reject humans and irregular shapes

            area = width_r * height_r
            
            # Filter by belt angle if --belt_angle was set (or overridden by 's' key)
            if ref_angle is not None:
                angle_diff = abs(normalized_angle - ref_angle)
                angle_diff = min(angle_diff, 180 - angle_diff)
                if angle_diff > angle_threshold:
                    continue

            # Filter by reference suitcase size/shape if set via 's' key
            if ref_area is not None:
                area_diff = abs(area - ref_area) / ref_area
                aspect_diff = abs(aspect_ratio - ref_aspect_ratio) / ref_aspect_ratio
                if area_diff > 0.30 or aspect_diff > 0.20:
                    continue  
            
            x, y, w, h = cv2.boundingRect(contour)
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            
            # Hard reject if centroid falls outside the polygon ROI
            if roi_coords is not None:
                inside = cv2.pointPolygonTest(roi_coords, (float(cx), float(cy)), False)
                if inside < 0:
                    continue
            
            current_detections.append((cx, cy, (x, y, w, h)))
            
        new_tracked_objects = {}
        used_detections = set()
        active_confirmed = 0
        total_dy = 0
        total_dx = 0
        moving_count = 0
        
        # Track objects
        for obj_id, obj in tracked_objects.items():
            best_match = None
            best_dist = float('inf')
            
            for i, (cx, cy, bbox) in enumerate(current_detections):
                if i in used_detections:
                    continue
                dist = distance(obj.centroids[-1], (cx, cy))
                if dist < max_distance and dist < best_dist:
                    best_match = i
                    best_dist = dist
                    
            if best_match is not None:
                cx, cy, bbox = current_detections[best_match]
                obj.centroids.append((cx, cy))
                obj.bboxes.append(bbox)
                obj.frame_indices.append(frame_count)
                obj.frames_visible += 1
                obj.frames_missing = 0
                used_detections.add(best_match)
                new_tracked_objects[obj_id] = obj
                
                # Check how far the object has moved since it was first seen
                dx = obj.centroids[-1][0] - obj.centroids[0][0]
                dy = obj.centroids[-1][1] - obj.centroids[0][1]
                dist_moved = math.sqrt(dx**2 + dy**2)
                
                # Calculate the angle of movement (0-360)
                move_angle = math.degrees(math.atan2(dy, dx))
                if move_angle < 0:
                    move_angle += 360
                    
                # The movement must be parallel to the belt line (modulo 180)
                is_moving_along_belt = True
                if ref_angle is not None and dist_moved > 5:
                    move_angle_180 = move_angle % 180
                    angle_diff = abs(move_angle_180 - ref_angle)
                    angle_diff = min(angle_diff, 180 - angle_diff)
                    if angle_diff > angle_threshold + 10:  # slight tolerance for wiggling
                        is_moving_along_belt = False
                
                # An object must be visible >= 1 sec, moving far enough, AND moving along the belt
                if obj.frames_visible >= CONFIRM_FRAMES:
                    if dist_moved >= 15 and is_moving_along_belt:
                        active_confirmed += 1
                        color = (0, 255, 0)  # Green: confirmed suitcase moving on belt

                        total_dy += dy
                        total_dx += dx
                        moving_count += 1

                        if not obj.reported:
                            recent_arrivals.append(1)
                            obj.reported = True
                    elif dist_moved >= 15 and not is_moving_along_belt:
                        color = None  # Originally Red: moving, but NOT along belt
                    else:
                        color = None  # Originally Cyan: not moving far enough yet
                elif obj.frames_visible >= CONFIRM_FRAMES:
                    color = None  # Originally Orange: seen but not yet status-confirmed
                else:
                    color = None  # too new, don't draw

                if color is not None:
                    age_secs = obj.frames_visible / fps
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"ID {obj_id} ({age_secs:.1f}s)",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            else:
                obj.frames_missing += 1
                if obj.frames_missing < max_missing:
                    new_tracked_objects[obj_id] = obj
                elif obj.reported and not obj.finished:
                    recent_departures.append(1)
                    obj.finished = True
                    
        for i, (cx, cy, bbox) in enumerate(current_detections):
            if i not in used_detections:
                new_obj = TrackedObject(next_object_id, (cx, cy), bbox, frame_count)
                new_tracked_objects[next_object_id] = new_obj
                next_object_id += 1
                
        tracked_objects = new_tracked_objects
        
        # Add zeros to queues if nothing happened this frame
        if len(recent_arrivals) < fps * 3:
            recent_arrivals.append(0)
        else:
            recent_arrivals.popleft()
            recent_arrivals.append(0)
            
        if len(recent_departures) < fps * 3:
            recent_departures.append(0)
        else:
            recent_departures.popleft()
            recent_departures.append(0)

        # ── Per-object status: set by any object visible >= 1.5 seconds ──────────
        # Only objects that have been tracked for >= CONFIRM_FRAMES contribute.
        if active_confirmed == 0 or moving_count == 0:
            # Debounce IDLE status: don't instantly switch to IDLE if a track is lost for a split second
            idle_debounce_counter += 1
            if idle_debounce_counter > fps * 2:  # Must be empty for 2 full seconds to become IDLE
                status = "IDLE"
                status_color = (150, 150, 150)  # gray
        else:
            idle_debounce_counter = 0  # Reset debounce
            avg_dy = total_dy / moving_count
            avg_dx = total_dx / moving_count
            
            # Use the dominant axis of movement to determine status
            if abs(avg_dx) > abs(avg_dy):
                # Moving mostly horizontally
                if avg_dx < -5:
                    status       = "UNLOADING"
                    status_color = (0, 165, 255)
                elif avg_dx > 5:
                    status       = "LOADING"
                    status_color = (0, 255, 255)
                else:
                    idle_debounce_counter += 1
                    if idle_debounce_counter > fps * 2:
                        status = "IDLE"
                        status_color = (150, 150, 150)
            else:
                # Moving mostly vertically
                if avg_dy < -5:
                    status       = "LOADING"
                    status_color = (0, 255, 255)
                elif avg_dy > 5:
                    status       = "UNLOADING"
                    status_color = (0, 165, 255)
                else:
                    idle_debounce_counter += 1
                    if idle_debounce_counter > fps * 2:
                        status = "IDLE"
                        status_color = (150, 150, 150)

        video_secs = frame_count / fps    # ──────────────────────────────────────────────────────────────────────
                
        # ── Event detection: IDLE → LOADING / UNLOADING ──────────────────
        if prev_status == "IDLE" and status in ("LOADING", "UNLOADING"):
            event_count += 1
            ts_str = f"{int(video_secs//3600):02d}:{int((video_secs%3600)//60):02d}:{video_secs%60:05.2f}"
            event_snapshot = os.path.join(event_dir, f"event_{event_count:03d}_{status}_{ts_str.replace(':','').replace('.','')}.jpg")
            cv2.imwrite(event_snapshot, frame)
            
            print(f"[EVENT #{event_count}] {status} started at {ts_str} (frame {frame_count}) → {event_snapshot}")
            
            with open(events_csv, 'a') as f:
                f.write(f"{event_count},{status},{ts_str},{frame_count},{event_snapshot}\n")

        prev_status = status
        # ─────────────────────────────────────────────────────────────────

        # Draw minimal HUD in top-left corner
        hud_x = 10
        hud_y = 20
        
        cv2.putText(frame, f"STATUS: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Objects: {len(tracked_objects)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        
        minutes, seconds = divmod(int(t), 60)
        hours, minutes = divmod(minutes, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{int((t % 1) * 100):02d}"
        cv2.putText(frame, f"Time: {time_str}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
        if show_video:
            cv2.imshow('Status Detector', frame)
            key = cv2.waitKey(frame_delay) & 0xFF
        else:
            key = 0xFF
            
        if frame_count % 30 == 0:
            os.makedirs("output_frames", exist_ok=True)
            cv2.imwrite(os.path.join("output_frames", f"output_{frame_count:04d}.jpg"), frame)
            
        if key == ord('q'):
            break
        elif key == ord('s'):
            display_frame = frame.copy()
            cv2.putText(display_frame, "Select Reference Suitcase. Press 'c' to cancel.", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ref_roi = cv2.selectROI("Select Reference", display_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Reference")
            
            if ref_roi != (0, 0, 0, 0):
                rx, ry, rw, rh = int(ref_roi[0]), int(ref_roi[1]), int(ref_roi[2]), int(ref_roi[3])
                roi_mask_ref = np.zeros(thresh.shape, dtype=np.uint8)
                roi_mask_ref[ry:ry+rh, rx:rx+rw] = 255
                ref_thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask_ref)
                
                ref_contours, _ = cv2.findContours(ref_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(ref_contours) > 0:
                    largest_ref = max(ref_contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(largest_ref)
                    _, (width, height), angle = rect
                    if width > 0 and height > 0:
                        ref_area = width * height
                        ref_aspect_ratio = max(width, height) / min(width, height)
                        if width < height:
                            angle = angle + 90
                        ref_angle = angle % 180
                        print(f"Reference Captured: Area={ref_area:.1f}, Aspect={ref_aspect_ratio:.2f}, Angle={ref_angle:.1f}")
            
        frame_count += 1
        
    cap.release()
    cv2.destroyAllWindows()
    
    # ── Post-processing: Filter Events ──
    def parse_time(ts_str):
        parts = ts_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    def format_time(total_seconds):
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = total_seconds % 60
        return f"{h:02d}:{m:02d}:{s:05.2f}"

    raw_events_for_filter = []
    if os.path.exists(events_csv):
        with open(events_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_events_for_filter.append({
                    "timestamp": parse_time(row["video_timestamp"]),
                    "event_type": row["status"]
                })
                
    filtered = filter_events(
        raw_events_for_filter, 
        window_size_seconds=30.0, 
        min_detections_in_window=2, 
        merge_gap_seconds=15.0
    )
    
    filtered_events_csv = os.path.join(event_dir, "filtered_events.csv")
    with open(filtered_events_csv, 'w') as f:
        f.write("event_id,event_type,start_time,end_time,duration_seconds,detection_count\n")
        for i, ev in enumerate(filtered, 1):
            duration = ev['end_time'] - ev['start_time']
            f.write(f"{i},{ev['event_type']},{format_time(ev['start_time'])},{format_time(ev['end_time'])},{duration:.2f},{ev['detection_count']}\n")
            
    print(f"Status detection finished. Filtered {len(raw_events_for_filter)} raw events into {len(filtered)} clean events.")
    print(f"Saved to: {filtered_events_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--min_area', type=int, default=800, help="Minimum contour area to detect")
    parser.add_argument('--max_area', type=int, default=25000, help="Maximum contour area to detect")
    parser.add_argument('--sustained_secs', type=float, default=1.5, help="Sustained seconds tracked before confirming")
    parser.add_argument('--belt_angle', type=float, default=30.0,
                        help="Expected belt angle in degrees (0=horizontal, 90=vertical).")
    parser.add_argument('--angle_threshold', type=float, default=20,
                        help="Max allowed deviation from belt_angle in degrees (default: 20)")
    parser.add_argument('--var_threshold', type=int, default=16,
                        help="Background subtractor sensitivity. Lower = more sensitive (default: 16). Try 4-8 for glass/transparent-cover belts.")
    parser.add_argument('--max_missing', type=int, default=15,
                        help="Max consecutive frames a tracked object can disappear before being dropped (default: 15). Raise for transparent-cover belts.")
    parser.add_argument('--use_clahe', action='store_true',
                        help="Apply CLAHE contrast enhancement before detection. Use for glass/low-contrast belts.")
    parser.add_argument('--brightness_gamma', type=float, default=1.0,
                        help="Gamma correction for night videos. >1.0 makes dark areas brighter (e.g. 1.5 to 2.0). Default 1.0.")
    parser.add_argument('--roi', type=str, default=None,
                        help="Optional exact 4-point polygon ROI as 'X1,Y1 X2,Y2 X3,Y3 X4,Y4'")
    parser.add_argument('--save_roof_overlays', action='store_true', default=True,
                        help="Save diagnostic overlay images from belt type classification to roof_detection_output/")
    parser.add_argument('--show_video', action='store_true',
                        help="Show real-time video playback window (will run at 1x speed)")
    args = parser.parse_args()

    run_status_detector(
        args.video,
        min_area=args.min_area,
        max_area=args.max_area,
        sustained_secs=args.sustained_secs,
        belt_angle=args.belt_angle,
        angle_threshold=args.angle_threshold,
        var_threshold=args.var_threshold,
        use_clahe=args.use_clahe,
        brightness_gamma=args.brightness_gamma,
        max_missing=args.max_missing,
        roi_str=args.roi,
        save_roof_overlays=args.save_roof_overlays,
        show_video=args.show_video,
    )
