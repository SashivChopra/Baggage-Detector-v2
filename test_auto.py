import cv2
import numpy as np
from belt_detection import BeltDetector
from status_detector import AutoROIConfig

cap = cv2.VideoCapture("videos/conv_full_D02.mp4")
ret, frame = cap.read()
if ret:
    detector = BeltDetector(AutoROIConfig())
    auto_belt = detector.detect([frame])
    if auto_belt is not None:
        poly = auto_belt.box_points().astype(np.int32)
        print("DETECTED:", " ".join([f"{p[0]},{p[1]}" for p in poly]))
    else:
        print("FAILED TO DETECT")
