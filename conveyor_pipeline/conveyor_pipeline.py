"""
Conveyor Belt Event Detection Pipeline

This script serves as the main entry point for the conveyor belt motion detection
and classification system. It accepts a video input and a belt bounding box,
generates a region of interest (ROI), runs a real-time motion detector,
and finally filters the noise using a sliding-window algorithm.

Arguments:
    --video: Path to input video file (required).
    --belt-bbox: The manual bounding box of the belt as X,Y,W,H (required).
    --output: Path to save the output JSON of confirmed events (optional).
    --headless: Skip UI and process as fast as possible.
    --window-secs: Size of the sliding window for noise filtering (default 2.0).
    --min-detections: Minimum detections in window to consider an event real (default 5).
    --merge-gap: Merge confirmed events separated by less than this (default 3.0).

Outputs:
    Prints a clean list of confirmed events to stdout, and optionally saves to JSON.
    Format: [{"start_time": ..., "end_time": ..., "event_type": ...}, ...]
"""

import argparse
import json
import sys
import time
import cv2
import numpy as np

from roi import compute_roi, polygon_to_mask
from events_filter import filter_events
from detector import run_detector

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--belt-bbox", type=str, required=True, help="Manual belt bbox as X,Y,W,H")
    p.add_argument("--roi-padding", type=float, default=0.05)
    p.add_argument("--window-secs", type=float, default=2.0)
    p.add_argument("--min-detections", type=int, default=5)
    p.add_argument("--merge-gap", type=float, default=3.0)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--headless", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, first_frame = cap.read()
    if not ok:
        sys.exit(f"Could not read video: {args.video}")
    cap.release()

    bbox_str = args.belt_bbox.replace(" ", ",").split(",")
    bbox_str = [x for x in bbox_str if x]
    
    if len(bbox_str) == 4:
        bbox = tuple(int(v) for v in bbox_str)
        print(f"[config] Manual belt bbox = {bbox}")
        polygon = compute_roi(bbox, first_frame.shape, args.roi_padding)
    elif len(bbox_str) == 8:
        coords = [int(v) for v in bbox_str]
        polygon = np.array([
            [coords[0], coords[1]],
            [coords[2], coords[3]],
            [coords[4], coords[5]],
            [coords[6], coords[7]]
        ], dtype=np.int32)
        print(f"[config] Exact manual polygon ROI = {polygon.tolist()}")
    else:
        sys.exit("Invalid --belt-bbox format. Provide either 4 values (X,Y,W,H) or 8 values for a 4-point polygon (X1,Y1 X2,Y2 ...)")

    mask = polygon_to_mask(polygon, first_frame.shape)

    t0 = time.time()
    raw_events = run_detector(
        video_path=args.video,
        roi_mask=mask,
        roi_polygon=polygon,
        headless=args.headless
    )
    print(f"[detect] {len(raw_events)} raw events generated in {time.time() - t0:.1f}s")

    clean_events = filter_events(
        events=raw_events,
        window_size_seconds=args.window_secs,
        min_detections_in_window=args.min_detections,
        merge_gap_seconds=args.merge_gap,
    )
    print(f"[filter] {len(clean_events)} confirmed real events after sliding-window.")

    out_json = json.dumps(clean_events, indent=2)
    print("\n=== CONFIRMED EVENTS ===")
    print(out_json)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(out_json)
        print(f"\n[out] saved {args.output}")

if __name__ == "__main__":
    main()
