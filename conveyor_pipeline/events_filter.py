from typing import List, Dict
from collections import defaultdict

def filter_events(events: List[Dict],
                  window_size_seconds: float,
                  min_detections_in_window: int,
                  merge_gap_seconds: float) -> List[Dict]:
    """
    Filter raw detections into clean start/end events using a sliding window.
    
    events: list of {"timestamp": float, "event_type": str}
    window_size_seconds: width of the sliding window
    min_detections_in_window: minimum detections required to confirm a real event
    merge_gap_seconds: gaps smaller than this between confirmed windows are merged
    """
    if not events:
        return []

    # Process each event type independently
    by_type = defaultdict(list)
    for e in events:
        by_type[e["event_type"]].append(float(e["timestamp"]))

    confirmed = []
    for etype, times in by_type.items():
        times.sort()
        confirmed_intervals = []
        left = 0
        for right in range(len(times)):
            while times[right] - times[left] > window_size_seconds:
                left += 1
            if (right - left + 1) >= min_detections_in_window:
                confirmed_intervals.append((times[left], times[right]))

        if not confirmed_intervals:
            continue

        confirmed_intervals.sort()
        merged = [list(confirmed_intervals[0])]
        for s, e in confirmed_intervals[1:]:
            if s - merged[-1][1] <= merge_gap_seconds:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        for s, e in merged:
            confirmed.append({
                "start_time": s,
                "end_time":   e,
                "event_type": etype,
            })

    confirmed.sort(key=lambda x: x["start_time"])
    return confirmed
