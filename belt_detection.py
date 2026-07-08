from __future__ import annotations
"""Automatic conveyor-belt ROI detection.

Strategy (self-planned) and reasoning
-------------------------------------
Candidate cues considered:

1. Color segmentation          -> belt loaders carry high-visibility YELLOW
                                  side rails. Yellow is rare elsewhere in a
                                  ramp scene, survives night floodlighting,
                                  and is viewpoint-invariant. Weakness:
                                  yellow tarmac paint and hi-vis vests.
2. Line / edge detection       -> the rails are the longest straight,
                                  moderately inclined structures in frame.
                                  Weakness: fuselage/engine edges also fire.
3. Shape/geometry heuristics   -> belts always incline UP toward the hold;
                                  used to disambiguate which end is which.
4. Motion cues                 -> only valid while bags move; belt must be
                                  detectable while Idle, so motion is not
                                  usable as a primary cue.

Chosen method: fuse (1) + (2), disambiguate with (3).

* Yellow pixels are filtered at connected-component level: rails are
  elongated and wider than tall, which rejects vests (blobby) and most
  ground paint.
* Long, moderately inclined Hough segments (3-50 deg) add edge evidence,
  which keeps detection alive if the rails are partially occluded.
* A robust Huber line fit over the fused evidence gives the belt AXIS;
  yellow evidence is weighted 3x because it is the more specific cue.
  Huber loss lets the fit shrug off residual outliers (paint, fuselage
  lines) without hand-tuned masking.
* The ROI is a rotated band around the axis. The HOLD end is the axis
  endpoint with the smaller image y: a belt loader always ramps upward
  into the cargo hold, so "higher in the image" == "aircraft side".

Detection runs on several sampled frames and the per-frame results are
median-fused, then the ROI is LOCKED until the next scheduled
re-detection (config.roi_redetect_every_s). A sanity gate rejects
re-detections that jump implausibly far from the locked ROI.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BeltROI:
    p_ground: np.ndarray      # axis endpoint at ground/feed end (x, y)
    p_hold: np.ndarray        # axis endpoint at aircraft-hold end (x, y)
    halfwidth: float          # band half-width in px

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p_hold - self.p_ground))

    @property
    def angle_deg(self) -> float:
        d = self.p_hold - self.p_ground
        return float(np.degrees(np.arctan2(d[1], d[0])))

    @property
    def center(self) -> np.ndarray:
        return (self.p_ground + self.p_hold) / 2.0

    def box_points(self) -> np.ndarray:
        """Corners of the rotated ROI band (for drawing)."""
        d = self.p_hold - self.p_ground
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-5)
        hw = self.halfwidth
        return np.array([self.p_ground - n * hw, self.p_ground + n * hw,
                         self.p_hold + n * hw, self.p_hold - n * hw])

    def strip_warp(self, strip_h: int) -> tuple[np.ndarray, tuple[int, int]]:
        """Affine warp mapping the band to an axis-aligned strip.

        In strip coordinates the GROUND end is at x=0 and the HOLD end at
        x=strip_w, so +x motion always means "toward the aircraft"
        (Loading) regardless of camera orientation.
        """
        strip_w = max(32, int(round(self.length)))
        d = self.p_hold - self.p_ground
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-5)
        src = np.float32([self.p_ground - n * self.halfwidth,
                          self.p_hold - n * self.halfwidth,
                          self.p_ground + n * self.halfwidth])
        dst = np.float32([[0, 0], [strip_w, 0], [0, strip_h]])
        return cv2.getAffineTransform(src, dst), (strip_w, strip_h)


class BeltDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self._clahe = cv2.createCLAHE(2.0, (8, 8))

    # ------------------------------------------------------------ evidence
    def _rail_color_evidence(self, bgr: np.ndarray) -> np.ndarray:
        """Yellow rail pixels, filtered per connected component."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self.cfg.rail_hsv_lo, self.cfg.rail_hsv_hi)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros_like(m)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 25:
                continue
            elong = max(w, h) / max(1, min(w, h))
            if elong >= 2.0 and w >= 0.8 * h:      # rail-like: long + not vertical
                keep[lbl == i] = 255
        return keep

    def _line_segments(self, bgr: np.ndarray) -> list[tuple]:
        """Long, moderately inclined straight segments (rail/canopy geometry).

        Returns (x1, y1, x2, y2, length, angle_deg) per segment. The 3-degree
        incline floor is what keeps tarmac paint out: painted stand markings
        and lane stripes are horizontal, belts always ramp upward.
        """
        g = self._clahe.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        edges = cv2.Canny(g, 60, 150)
        h, w = g.shape
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                                minLineLength=int(0.22 * w), maxLineGap=8)
        segs = []
        if lines is not None:
            lines = lines.reshape(-1, 4)
            for x1, y1, x2, y2 in lines:
                ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                a = abs(ang)
                a = min(a, 180.0 - a)
                if 3.0 < a < 50.0:
                    segs.append((x1, y1, x2, y2,
                                 float(np.hypot(x2 - x1, y2 - y1)), ang))
        return segs

    # ----------------------------------------------------------- per frame
    def _fit_axis(self, pts: np.ndarray, frame_w: int) -> BeltROI | None:
        """Robust Huber line fit over an evidence point set -> candidate ROI."""
        if pts is None or len(pts) < self.cfg.roi_min_points:
            return None
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        if not (np.isfinite(vx) and np.isfinite(vy) and np.isfinite(x0) and np.isfinite(y0)):
            return None
        u = np.array([vx, vy], dtype=np.float32)
        c = np.array([x0, y0], dtype=np.float32)
        t = (pts - c).dot(u)
        lo, hi = np.percentile(t, 3), np.percentile(t, 97)
        if hi - lo < 0.15 * frame_w:               # implausibly short belt
            return None
        e1, e2 = c + lo * u, c + hi * u
        p_hold, p_ground = (e1, e2) if e1[1] < e2[1] else (e2, e1)   # elevated end = hold
        halfwidth = self.cfg.roi_halfwidth_frac * float(hi - lo)
        roi = BeltROI(p_ground=p_ground, p_hold=p_hold, halfwidth=halfwidth)
        a = abs(roi.angle_deg)
        a = min(a, 180.0 - a)
        if not (3.0 <= a <= 55.0):                 # belts incline; hard gate, not a
            return None                            # soft penalty - flat = ground paint
        return roi

    @staticmethod
    def _near_line(pts: np.ndarray | None, p1, p2, max_dist: float) -> np.ndarray:
        """Boolean mask of points within max_dist of the (extended) line p1-p2."""
        if pts is None or len(pts) == 0:
            return np.zeros(0, dtype=bool)
        d = np.array(p2, dtype=np.float32) - np.array(p1, dtype=np.float32)
        norm = float(np.linalg.norm(d))
        if not np.isfinite(norm) or norm < 1e-5:
            return np.zeros(len(pts), dtype=bool)
        n = np.array([-d[1], d[0]]) / norm
        s = (pts - np.array(p1, dtype=np.float32)).dot(n)
        return np.abs(s) <= max_dist

    def _detect_single(self, bgr: np.ndarray, motion_history: np.ndarray = None,
                       rois_last: list[BeltROI] | None = None) -> BeltROI | None:
        """Frame-level detection via Hough lines + Yellow + Motion + Prior.

        WHY hypothesis-driven: a single global fit (or fits over pooled point
        sets) fails whenever a pollutant owns the majority of the evidence -
        e.g. yellow "B737" stand markings painted on the tarmac outweigh the
        rails when a canopy covers most of them, and the fit lands flat on
        the ground. Here, only long INCLINED Hough segments may PROPOSE an
        axis (ground paint is horizontal and cannot), and yellow evidence
        only VERIFIES proposals as an absolute, capped vote - so however
        much paint the scene contains, it can neither create nor promote a
        ground-plane candidate.
        """
        yellow = self._rail_color_evidence(bgr)
        py = cv2.findNonZero(yellow)
        py = py.reshape(-1, 2).astype(np.float32) if py is not None else None
        
        pm = None
        if motion_history is not None:
            pm_idx = cv2.findNonZero((motion_history > 30).astype(np.uint8))
            pm = pm_idx.reshape(-1, 2).astype(np.float32) if pm_idx is not None else None

        segs = self._line_segments(bgr)
        h, w = bgr.shape[:2]
        band = self.cfg.roi_hypo_band_frac * w      # co-linearity tolerance

        best, best_score = None, -1e9
        seg_samples = []                             # all segment points, for extent
        for x1, y1, x2, y2, L, ang in segs:
            n_samples = max(2, int(L / 4))
            ts = np.linspace(0, 1, n_samples, dtype=np.float32)[:, None]
            seg_samples.append(np.array([x1, y1], np.float32)
                               + ts * np.array([x2 - x1, y2 - y1], np.float32))
        for sx1, sy1, sx2, sy2, slen, sang in segs:  # each long segment seeds one
            p1, p2 = (sx1, sy1), (sx2, sy2)
            # aligned-segment support: total length of segments that agree in
            # angle AND lie on this line (rails + canopy edge reinforce)
            support_px, member_pts = 0.0, []
            for x1, y1, x2, y2, L, ang in segs:
                d_ang = abs(ang - sang)
                d_ang = min(d_ang, 180.0 - d_ang)
                mid = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], np.float32)
                if d_ang <= 6.0 and self._near_line(mid, p1, p2, band)[0]:
                    support_px += L
                    n_samples = max(2, int(L / 4))
                    ts = np.linspace(0, 1, n_samples, dtype=np.float32)[:, None]
                    member_pts.append(np.array([x1, y1], np.float32)
                                      + ts * np.array([x2 - x1, y2 - y1], np.float32))
            # yellow verification: absolute + capped, so paint majorities
            # elsewhere in the frame are irrelevant
            yellow_px = 0
            if py is not None:
                near = self._near_line(py, p1, p2, band)
                yellow_px = int(np.count_nonzero(near))
                
            # motion verification: moving objects along this line strongly indicate a belt
            motion_px = 0
            if pm is not None:
                near_m = self._near_line(pm, p1, p2, band)
                motion_px = int(np.count_nonzero(near_m))
                
            score = (support_px
                     + 0.8 * min(yellow_px, 2 * w)
                     + 1.5 * min(motion_px, 3 * w)
                     + 0.25 * ((sy1 + sy2) / 2.0 / h) * w)  # reward lower-in-frame edges (prefer belt over canopy)
            # continuity prior: damp round-to-round flip-flopping between
            # parallel structures (lower rail vs. canopy edge) by favoring
            # hypotheses consistent with the currently locked axis
            if rois_last is not None:
                for prev in rois_last:
                    d_prev = abs(sang - prev.angle_deg)
                    d_prev = min(d_prev, 180.0 - d_prev)
                    mid_prev = prev.center[None, :].astype(np.float32)
                    if d_prev <= 8.0 and self._near_line(mid_prev, p1, p2, 2.0 * band)[0]:
                        score += 0.5 * w
            if score <= best_score:
                continue
            pts = member_pts
            if py is not None and yellow_px > 0:
                pts = member_pts + [py[self._near_line(py, p1, p2, band)]]
            roi = self._fit_axis(np.concatenate(pts), w)
            if roi is None:
                continue
            best, best_score = roi, score

        if best is None:
            return None
        # Extent refinement: the hypothesis fixes the DIRECTION, but evidence
        # tight to one seed line under-spans the belt (perspective bows the
        # rails slightly). Re-derive the endpoints from ALL evidence inside a
        # wider corridor around the winning axis; the corridor is still far
        # too narrow for ground paint to reach.
        all_pts = [np.concatenate(seg_samples)] if seg_samples else []
        if py is not None:
            all_pts.append(py)
        if all_pts:
            ap = np.concatenate(all_pts)
            near = self._near_line(ap, best.p_ground, best.p_hold, 2.5 * band)
            ap = ap[near]
            if len(ap) >= self.cfg.roi_min_points:
                d = best.p_hold - best.p_ground
                u = d / (np.linalg.norm(d) + 1e-5)
                t = (ap - best.p_ground).dot(u)
                lo, hi = np.percentile(t, 2), np.percentile(t, 98)
                p1 = best.p_ground + lo * u
                p2 = best.p_ground + hi * u
                p_hold, p_ground = (p1, p2) if p1[1] < p2[1] else (p2, p1)
                best = BeltROI(p_ground=p_ground, p_hold=p_hold,
                               halfwidth=self.cfg.roi_halfwidth_frac * float(hi - lo))
        return best

    # --------------------------------------------------------- multi frame
    def detect(self, frames: list[np.ndarray], motion_history: np.ndarray = None, previous: BeltROI = None) -> BeltROI | None:
        """Process a sequence of frames, returning the median stable ROI."""
        if not frames:
            return None
            
        rois = []
        for i, f in enumerate(frames):
            # Only use motion_history on the middle frame for efficiency
            mh = motion_history if i == len(frames) // 2 else None
            roi = self._detect_single(f, mh, [previous] if previous else None)
            if roi:
                rois.append(roi)
        if not rois:
            return previous
        roi = BeltROI(
            p_ground=np.median([r.p_ground for r in rois], axis=0),
            p_hold=np.median([r.p_hold for r in rois], axis=0),
            halfwidth=float(np.median([r.halfwidth for r in rois])),
        )
        if previous is not None:
            diag = np.hypot(*frames[0].shape[:2])
            d_ang = abs(roi.angle_deg - previous.angle_deg)
            d_ang = min(d_ang, 360.0 - d_ang)
            d_cen = np.linalg.norm(roi.center - previous.center)
            if (d_ang > self.cfg.roi_max_angle_jump_deg
                    or d_cen > self.cfg.roi_max_center_jump_frac * diag):
                return previous                     # reject implausible jump, keep lock
        return roi

