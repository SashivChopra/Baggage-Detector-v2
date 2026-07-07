from typing import Tuple
import cv2
import numpy as np

def compute_roi(belt_bbox: Tuple[int, int, int, int],
                frame_shape: Tuple[int, int],
                padding_fraction: float = 0.05) -> np.ndarray:
    """
    Derives the ROI polygon from a provided conveyor belt bounding box.
    belt_bbox: (x, y, w, h) of the conveyor belt region
    padding_fraction: fractional padding to add around the belt bbox
    Returns: ROI polygon coordinates (4, 2)
    """
    x, y, w, h = belt_bbox
    H, W = frame_shape[:2]

    pad = int(round(max(w, h) * padding_fraction))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)

    return np.array([[x0, y0], [x1, y0],
                     [x1, y1], [x0, y1]], dtype=np.int32)


def polygon_to_mask(polygon: np.ndarray,
                    frame_shape: Tuple[int, int]) -> np.ndarray:
    """Binary uint8 mask: 255 inside polygon, 0 outside."""
    H, W = frame_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask
