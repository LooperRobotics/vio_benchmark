"""
AprilGrid detection using the `aprilgrid` pip package.

Why not pupil_apriltags (apriltag3)?
  Kalibr boards are generated with blackTagBorder=2 (2-cell black border).
  apriltag3 expects 1-cell border → samples data bits from wrong positions → 0 detections.
  The `aprilgrid` package ships `t36h11` (2-cell, kalibr convention) as a first-class family.
"""

import numpy as np
import cv2
from aprilgrid import Detector as _Detector


class AprilGridDetector:
    # Corner ordering matches kalibr exactly: [0=BL, 1=BR, 2=TR, 3=TL]
    # No remapping needed.

    def __init__(self, tag_rows, tag_cols, tag_size, tag_spacing,
                 code_offset=0, tag_family='t36h11'):
        self.tag_rows    = tag_rows
        self.tag_cols    = tag_cols
        self.tag_size    = tag_size
        self.tag_spacing = tag_spacing
        self.code_offset = code_offset
        self.grid_rows   = 2 * tag_rows
        self.grid_cols   = 2 * tag_cols
        self._board_pts  = self._build_board_points()
        self._det        = _Detector(tag_family)

    # ── board geometry ────────────────────────────────────────────────────────

    def _build_board_points(self):
        """3D corner positions in board frame (kalibr layout)."""
        pts = np.zeros((self.grid_rows * self.grid_cols, 3), dtype=np.float64)
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                x = (c // 2) * (1 + self.tag_spacing) * self.tag_size \
                    + (c % 2) * self.tag_size
                y = (r // 2) * (1 + self.tag_spacing) * self.tag_size \
                    + (r % 2) * self.tag_size
                pts[r * self.grid_cols + c] = [x, y, 0.0]
        return pts

    def _tag_corner_indices(self, tag_id):
        """
        Grid corner indices for one tag in kalibr order [BL, BR, TR, TL].
        Returns None if tag_id is out of range.
        """
        local_id = tag_id - self.code_offset
        if local_id < 0 or local_id >= self.tag_rows * self.tag_cols:
            return None
        gc   = self.grid_cols
        base = (local_id // self.tag_cols) * gc * 2 + (local_id % self.tag_cols) * 2
        return [base, base + 1, base + gc + 1, base + gc]  # BL, BR, TR, TL

    # ── detection ─────────────────────────────────────────────────────────────

    def _detect_corners(self, gray):
        """
        Steps 1-4: detect corners with kalibr-compatible filters.
        Returns (pts_2d, pts_3d) float32 or (None, None).
        """
        detections = self._det.detect(gray)
        pts_2d, pts_3d = [], []

        for det in detections:
            indices = self._tag_corner_indices(det.tag_id)
            if indices is None:
                continue
            corners = np.array(det.corners, dtype=np.float32).reshape(4, 2)
            for j, grid_idx in enumerate(indices):
                pts_2d.append(corners[j])
                pts_3d.append(self._board_pts[grid_idx])

        if len(pts_2d) < 16:   # minTagsForValidObs=4 × 4 corners (kalibr default)
            return None, None

        pts_2d_raw = np.array(pts_2d, dtype=np.float32)
        pts_3d     = np.array(pts_3d, dtype=np.float32)

        # border distance filter: discard corners within 4px of image edge (kalibr default)
        h, w = gray.shape[:2]
        border = 4.0
        in_bounds = (
            (pts_2d_raw[:, 0] >= border) & (pts_2d_raw[:, 0] <= w - border) &
            (pts_2d_raw[:, 1] >= border) & (pts_2d_raw[:, 1] <= h - border)
        )
        pts_2d_raw = pts_2d_raw[in_bounds]
        pts_3d     = pts_3d[in_bounds]
        if len(pts_2d_raw) < 16:
            return None, None

        # subpixel refinement — window (2,2) and eps=0.1 match kalibr
        pts_2d_ref = cv2.cornerSubPix(
            gray,
            pts_2d_raw.reshape(-1, 1, 2),
            winSize=(2, 2),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
        ).reshape(-1, 2)

        # maxSubpixDisplacement2=1.5 (kalibr): discard corners that moved too much
        disp2 = np.sum((pts_2d_ref - pts_2d_raw) ** 2, axis=1)
        valid = disp2 <= 1.5
        pts_2d = pts_2d_ref[valid]
        pts_3d = pts_3d[valid]

        if len(pts_2d) < 16:
            return None, None

        return pts_2d, pts_3d

    def _angle_filter(self, pts_2d, pts_3d, K):
        """
        Back-projection angle filter (kalibr PinholeProjection::estimateTransformation):
        discard corners whose back-projected ray deviates > 80° from optical axis.
        cos(80°) ≈ 0.1736
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        # normalized image coords
        xn = (pts_2d[:, 0] - cx) / fx
        yn = (pts_2d[:, 1] - cy) / fy
        # z-component of normalized ray
        denom = np.sqrt(xn**2 + yn**2 + 1.0)
        cos_angle = 1.0 / denom          # cos of angle from optical axis
        valid = cos_angle > np.cos(np.radians(80.0))  # ≈ 0.1736
        return pts_2d[valid], pts_3d[valid]

    def detect_and_solve(self, gray, K, D, max_reproj_px=1.0):
        """
        Full pipeline matching kalibr's detection + pose estimation:
          1. AprilTag detection
          2. Border distance filter       (kalibr: minBorderDistance=4px)
          3. Subpixel refinement          (kalibr: window 2×2, eps=0.1)
          4. Subpix displacement filter   (kalibr: maxSubpixDisplacement2=1.5)
          5. Back-projection angle filter (kalibr: cos(80°)≈0.1736)
          6. solvePnPRansac + refinement  (we keep RANSAC; kalibr uses plain solvePnP)
          7. Reprojection error check     (kalibr optional; we keep at 1.0px)

        Returns T_cam_board (4x4) or None.
        """
        # steps 1-4: detect corners with kalibr-compatible filters
        pts_2d_raw, pts_3d = self._detect_corners(gray)
        if pts_2d_raw is None:
            return None

        # step 5: back-projection angle filter (kalibr cos(80°) threshold)
        pts_2d, pts_3d = self._angle_filter(pts_2d_raw, pts_3d, K)
        if len(pts_2d) < 16:
            return None

        # step 6: solvePnPRansac + iterative refinement on inliers
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, D,
            iterationsCount=200, reprojectionError=1.5,
            confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None or len(inliers) < 16:
            return None

        idx = inliers.flatten()
        _, rvec, tvec = cv2.solvePnP(
            pts_3d[idx], pts_2d[idx], K, D,
            rvec=rvec, tvec=tvec,
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # step 7: reprojection error gate
        proj, _ = cv2.projectPoints(pts_3d[idx], rvec, tvec, K, D)
        reproj_err = np.mean(np.linalg.norm(
            proj.reshape(-1, 2) - pts_2d[idx], axis=1))
        if reproj_err > max_reproj_px:
            return None

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T
