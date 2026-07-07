from typing import List, Dict
import time
import math
import cv2
import numpy as np

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class TrackedObject:
    def __init__(self, obj_id, centroid, bbox, frame_idx):
        self.id = obj_id
        self.centroids = [centroid]
        self.bboxes = [bbox]
        self.frame_indices = [frame_idx]
        self.frames_visible = 1
        self.frames_missing = 0

def run_detector(video_path: str,
                 roi_mask: np.ndarray,
                 roi_polygon: np.ndarray,
                 headless: bool = False) -> List[Dict]:
    raw_events: List[Dict] = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay = max(1, int(1000 / fps))
    frame_count = 0
    t0 = time.time()

    min_area = 800
    max_area = 25000
    sustained_secs = 1.5
    var_threshold = 16
    max_distance = 60
    max_missing = 15
    CONFIRM_FRAMES = int(fps * sustained_secs)
    
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=var_threshold, detectShadows=True)
    next_object_id = 0
    tracked_objects = {}
    idle_debounce_counter = 0
    status = "IDLE"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
            
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        masked_frame = cv2.bitwise_and(blurred_frame, blurred_frame, mask=roi_mask)
        
        fgMask = backSub.apply(masked_frame)
        _, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
        
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        thresh = cv2.dilate(thresh, open_kernel, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_detections = []
        for contour in contours:
            area_c = cv2.contourArea(contour)
            if area_c < min_area or area_c > max_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            rect = cv2.minAreaRect(contour)
            _, (width_r, height_r), angle = rect
            
            if width_r <= 0 or height_r <= 0:
                continue
                
            aspect_ratio = max(width_r, height_r) / min(width_r, height_r)
            if aspect_ratio > 4.0:
                continue
                
            if width_r < height_r:
                angle = angle + 90
            normalized_angle = angle % 180
            
            if aspect_ratio > 1.5 and (60 < normalized_angle < 120):
                continue
                
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                if area_c / hull_area < 0.70:
                    continue
                    
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            
            if cv2.pointPolygonTest(roi_polygon, (float(cx), float(cy)), False) < 0:
                continue
                
            current_detections.append((cx, cy, (x, y, w, h)))

        new_tracked_objects = {}
        used_detections = set()
        active_confirmed = 0
        total_dy = 0
        total_dx = 0
        moving_count = 0
        
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
                
                dx = obj.centroids[-1][0] - obj.centroids[0][0]
                dy = obj.centroids[-1][1] - obj.centroids[0][1]
                dist_moved = math.sqrt(dx**2 + dy**2)
                
                if obj.frames_visible >= CONFIRM_FRAMES and dist_moved >= 15:
                    active_confirmed += 1
                    total_dy += dy
                    total_dx += dx
                    moving_count += 1
            else:
                obj.frames_missing += 1
                if obj.frames_missing < max_missing:
                    new_tracked_objects[obj_id] = obj
                    
        for i, (cx, cy, bbox) in enumerate(current_detections):
            if i not in used_detections:
                new_tracked_objects[next_object_id] = TrackedObject(next_object_id, (cx, cy), bbox, frame_count)
                next_object_id += 1
                
        tracked_objects = new_tracked_objects

        if active_confirmed == 0 or moving_count == 0:
            idle_debounce_counter += 1
            if idle_debounce_counter > fps * 2:
                status = "IDLE"
        else:
            idle_debounce_counter = 0
            avg_dy = total_dy / moving_count
            avg_dx = total_dx / moving_count
            
            if abs(avg_dx) > abs(avg_dy):
                status = "LOADING" if avg_dx > 5 else "UNLOADING" if avg_dx < -5 else status
            else:
                status = "UNLOADING" if avg_dy > 5 else "LOADING" if avg_dy < -5 else status

        if status in ("LOADING", "UNLOADING"):
            raw_events.append({
                "timestamp": frame_count / fps,
                "event_type": status
            })

        if not headless:
            cv2.polylines(frame, [roi_polygon], True, (0, 255, 255), 2)
            cv2.putText(frame, f"STATUS: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Detector", frame)
            if (cv2.waitKey(frame_delay) & 0xFF) == ord('q'):
                break
        elif frame_count % 250 == 0 and frame_count > 0:
            rate = frame_count / max(time.time() - t0, 1e-3)
            print(f"[{frame_count}/{total}] {rate:.1f} fps")

        frame_count += 1

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    return raw_events
