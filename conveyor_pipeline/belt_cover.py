import cv2
import numpy as np

def intensity_profile_analysis(frame: np.ndarray, bbox) -> float:
    """
    Samples intensity across the belt. A roofed belt tends to have a more uniform
    intensity profile. Returns the standard deviation of pixel intensities in the bbox.
    """
    x, y, w, h = bbox
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    return float(roi.std())

def edge_density(frame: np.ndarray, bbox, strip_height_frac: float = 0.25) -> float:
    """
    Computes horizontal edge density in a strip immediately above the belt.
    High horizontal edge density may indicate a roof structure.
    """
    x, y, w, h = bbox
    strip_h = int(h * strip_height_frac)
    y0 = max(0, y - strip_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    strip = gray[y0:y, x:x+w]
    if strip.size == 0:
        return 0.0
    gy = cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.abs(gy)))

def temporal_lighting_variance(video_path: str, bbox, sample_seconds: float = 30.0) -> float:
    """
    Computes the variance of mean frame brightness over a time window.
    An open belt shows lighting variation over time. A roofed belt does not.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(fps * sample_seconds)
    x, y, w, h = bbox
    means = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        means.append(gray[y:y+h, x:x+w].mean())
    cap.release()
    return float(np.var(means)) if means else 0.0
