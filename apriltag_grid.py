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

# kalibr: minTagsForValidObs=4, i.e. 16 corners
MIN_CORNERS = 16

# corners further than this from their reprojection are dropped and the pose
# re-solved (raw pixels, so it means the same thing for every lens)
OUTLIER_PX = 2.0
OUTLIER_ROUNDS = 2


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

    def _detect_corners(self, gray, stats=None):
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

        if len(pts_2d) < MIN_CORNERS:   # kalibr: minTagsForValidObs=4
            _bump(stats, 'too_few_tags')
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
        if len(pts_2d_raw) < MIN_CORNERS:
            _bump(stats, 'at_image_border')
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

        if len(pts_2d) < MIN_CORNERS:
            _bump(stats, 'subpix_moved_too_far')
            return None, None

        return pts_2d, pts_3d

    def _angle_filter(self, pts_2d, K):
        """
        Back-projection angle filter (kalibr PinholeProjection::estimateTransformation):
        mask out corners whose back-projected ray deviates > 80° from optical axis.
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
        return cos_angle > np.cos(np.radians(80.0))  # ≈ 0.1736

    @staticmethod
    def _reproj_err(pts_3d, pts_2d_src, rvec, tvec, K, D, distort_points):
        """
        Per-corner reprojection error in RAW image pixels.

        Projecting with (K, D=0) lands in the distortion-free camera PnP was
        solved in; `distort_points` maps that back through the real lens, so
        the error is measured where the corner was actually observed. Without
        it the fisheye stretch (sec²θ, ~4× at 60° off-axis) would inflate the
        error of peripheral corners and they would look like outliers.
        """
        proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)
        if distort_points is not None:
            proj = distort_points(proj)
        return np.linalg.norm(proj - pts_2d_src, axis=1)

    def detect_and_solve(self, gray, K, D, max_reproj_px=1.0,
                         undistort_points=None, distort_points=None, stats=None):
        """
        Full pipeline matching kalibr's detection + pose estimation:
          1. AprilTag detection
          2. Border distance filter       (kalibr: minBorderDistance=4px)
          3. Subpixel refinement          (kalibr: window 2×2, eps=0.1)
          4. Subpix displacement filter   (kalibr: maxSubpixDisplacement2=1.5)
          4b. Corner undistortion         (optional, for fisheye — see undistort.py)
          5. Back-projection angle filter (kalibr: cos(80°)≈0.1736)
          6. solvePnP on all corners      (as kalibr does) + outlier rejection
          7. Reprojection error check     (kalibr optional; we keep at 1.0px)

        `undistort_points` maps refined corners into a distortion-free camera
        and `distort_points` maps back, so errors stay measurable in raw
        pixels; `K`/`D` must describe that distortion-free camera. Subpixel
        refinement happens first, on raw pixels, which is where the corner is.

        `stats`, if given, is a dict counting which stage each frame died at.

        Returns T_cam_board (4x4) or None.
        """
        # steps 1-4: detect corners with kalibr-compatible filters
        pts_2d, pts_3d = self._detect_corners(gray, stats)
        if pts_2d is None:
            return None
        pts_src = pts_2d   # where the corners were actually observed

        # step 4b: lens model applied to the corners themselves
        if undistort_points is not None:
            pts_2d, ok = undistort_points(pts_2d)
            pts_2d, pts_3d, pts_src = pts_2d[ok], pts_3d[ok], pts_src[ok]
            if len(pts_2d) < MIN_CORNERS:
                _bump(stats, 'folded_past_horizon')
                return None

        # step 5: back-projection angle filter (kalibr cos(80°) threshold)
        keep = self._angle_filter(pts_2d, K)
        pts_2d, pts_3d, pts_src = pts_2d[keep], pts_3d[keep], pts_src[keep]
        if len(pts_2d) < MIN_CORNERS:
            _bump(stats, 'beyond_80deg')
            return None

        # step 6: solve on ALL corners, like kalibr.
        # Not solvePnPRansac: the board is planar, so each coplanar minimal
        # sample RANSAC draws is subject to the planar pose ambiguity and
        # converges somewhere else — the consensus never forms. Measured on a
        # real frame: 6 of 144 inliers, i.e. only the minimal sample itself,
        # where solving on all corners reprojects at 0.3 px. Outliers are
        # rejected by reprojection error instead.
        ok, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, D,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            _bump(stats, 'pnp_failed')
            return None

        for _ in range(OUTLIER_ROUNDS):
            err = self._reproj_err(pts_3d, pts_src, rvec, tvec, K, D, distort_points)
            keep = err <= OUTLIER_PX
            if keep.all():
                break
            if keep.sum() < MIN_CORNERS:
                _bump(stats, 'too_many_outliers')
                return None
            pts_2d, pts_3d, pts_src = pts_2d[keep], pts_3d[keep], pts_src[keep]
            _, rvec, tvec = cv2.solvePnP(
                pts_3d, pts_2d, K, D, rvec=rvec, tvec=tvec,
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

        # step 7: reprojection error gate, in raw pixels
        err = self._reproj_err(pts_3d, pts_src, rvec, tvec, K, D, distort_points)
        if err.mean() > max_reproj_px:
            _bump(stats, 'reproj_too_large')
            return None

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        _bump(stats, 'ok')
        return T


def _bump(stats, key):
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1
