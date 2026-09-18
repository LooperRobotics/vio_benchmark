"""
Lens-distortion handling driven by sensor_msgs/CameraInfo.

Two strategies, because one does not fit both lens types:

  image   Rectify the whole frame, then detect on it. Right for pinhole
          (`plumb_bob`, `rational_polynomial`) lenses: distortion is mild, the
          remap is cheap, and straightened tag edges help the quad detector.

  points  Detect on the raw frame, then undistort the detected corners. Right
          for fisheye: a pinhole image plane needs tan(theta), which blows up
          towards 90°, so an equidistant lens with a >120° field cannot be
          rectified without either cropping most of the frame or giving up
          centre resolution. Corner undistortion keeps the full field at
          native resolution, and tag quads are small enough that the detector
          copes with the local distortion.

Either way PnP runs on a distortion-free pinhole camera (`K`, `D` = 0). That
also avoids handing equidistant coefficients to solvePnP, which only knows the
radtan model and would silently mis-interpret them.
"""

import numpy as np
import cv2

# distortion_model strings that mean Kannala-Brandt / fisheye
FISHEYE_MODELS = ('equidistant', 'fisheye', 'kannala_brandt')

MODE_OFF    = 'off'
MODE_IMAGE  = 'image'
MODE_POINTS = 'points'
MODE_AUTO   = 'auto'

# below this, distortion coefficients are treated as exactly zero
_EPS = 1e-9

# a corner that does not survive a model round-trip within this many pixels sat
# beyond the 90° horizon and folded onto the wrong side of the image plane
_ROUND_TRIP_PX = 1.0

_ON_WORDS  = ('true', 'yes', 'on', '1', MODE_AUTO)
_OFF_WORDS = ('false', 'no', 'off', '0', MODE_OFF, 'none')


def parse_mode(value):
    """config value → MODE_OFF / MODE_IMAGE / MODE_POINTS / MODE_AUTO."""
    if isinstance(value, bool):
        return MODE_AUTO if value else MODE_OFF
    word = str(value).strip().lower()
    if word in _ON_WORDS:
        return MODE_AUTO
    if word in _OFF_WORDS:
        return MODE_OFF
    if word in (MODE_IMAGE, MODE_POINTS):
        return word
    raise ValueError(f'unknown undistort mode {value!r} — expected one of '
                     f'auto, image, points, false')


class Undistorter:
    """
    Removes lens distortion either from the image or from detected corners.

    After `undistort()` / `undistort_points()`, use `self.K` and `self.D`
    (all-zero) for PnP instead of the CameraInfo intrinsics.
    """

    def __init__(self, K, D, model='plumb_bob', size=None, mode=MODE_AUTO):
        self.K_raw = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.D_raw = np.asarray(D, dtype=np.float64).reshape(-1)
        self.model = (model or 'plumb_bob').lower()
        self.fisheye = self.model in FISHEYE_MODELS

        requested = parse_mode(mode)
        distorted = bool(self.D_raw.size) and bool(np.any(np.abs(self.D_raw) > _EPS))

        if not distorted or requested == MODE_OFF:
            # nothing to do — e.g. an already-rectified topic
            self.mode = MODE_OFF
        elif requested == MODE_AUTO:
            self.mode = MODE_POINTS if self.fisheye else MODE_IMAGE
        else:
            self.mode = requested

        self.active = self.mode != MODE_OFF

        # intrinsics of the distortion-free camera PnP sees; keeping the
        # original matrix means no rescaling and no shifted principal point
        self.K = self.K_raw.copy()
        self.D = np.zeros(5, dtype=np.float64)
        self._D_fisheye = self.D_raw[:4].reshape(4, 1) if self.fisheye else None

        self._size = None            # (w, h) the maps were built for
        self._map1 = self._map2 = None

        if self.mode == MODE_IMAGE and size is not None:
            self._build(size)

    # ── image rectification ───────────────────────────────────────────────────

    def _build(self, size):
        """Build the remap LUTs for image size (w, h)."""
        w, h = int(size[0]), int(size[1])

        if self.fisheye:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K_raw, self._D_fisheye, np.eye(3), self.K, (w, h), cv2.CV_16SC2)
        else:
            map1, map2 = cv2.initUndistortRectifyMap(
                self.K_raw, self.D_raw, np.eye(3), self.K, (w, h), cv2.CV_16SC2)

        self._map1, self._map2, self._size = map1, map2, (w, h)

    def undistort(self, gray):
        """Rectified copy of `gray`; `gray` itself unless mode is 'image'."""
        if self.mode != MODE_IMAGE:
            return gray

        h, w = gray.shape[:2]
        if self._size != (w, h):
            # CameraInfo size and the actual stream can disagree (e.g. nv12
            # padding); rebuild for what we are really given
            self._build((w, h))

        return cv2.remap(gray, self._map1, self._map2, cv2.INTER_LINEAR)

    # ── corner undistortion ───────────────────────────────────────────────────

    def undistort_points(self, pts_2d):
        """
        (pts, valid) with distortion removed; `pts_2d` unchanged unless mode is
        'points'. `valid` drops corners that sit beyond the 90° horizon, where
        the pinhole image plane folds and undistortPoints returns a mirrored
        position that would otherwise corrupt PnP.
        """
        pts_2d = np.asarray(pts_2d, dtype=np.float64).reshape(-1, 2)
        if self.mode != MODE_POINTS or len(pts_2d) == 0:
            return pts_2d.astype(np.float32), np.ones(len(pts_2d), dtype=bool)

        src = pts_2d.reshape(-1, 1, 2)
        if self.fisheye:
            out = cv2.fisheye.undistortPoints(
                src, self.K_raw, self._D_fisheye, np.eye(3), self.K)
        else:
            out = cv2.undistortPoints(src, self.K_raw, self.D_raw, None, self.K)
        out = out.reshape(-1, 2)

        err = np.linalg.norm(self.distort_points(out) - pts_2d, axis=1)
        return out.astype(np.float32), err <= _ROUND_TRIP_PX

    def distort_points(self, pts_2d):
        """
        Inverse of `undistort_points`: distortion-free pixels → raw pixels.

        Lets reprojection errors be measured where the corner was observed,
        instead of in the stretched undistorted plane.
        """
        pts_2d = np.asarray(pts_2d, dtype=np.float64).reshape(-1, 2)
        if self.mode != MODE_POINTS or len(pts_2d) == 0:
            return pts_2d

        xn = (pts_2d[:, 0] - self.K[0, 2]) / self.K[0, 0]
        yn = (pts_2d[:, 1] - self.K[1, 2]) / self.K[1, 1]
        rays = np.stack([xn, yn, np.ones_like(xn)], axis=1)
        zero = np.zeros(3, dtype=np.float64)

        if self.fisheye:
            out, _ = cv2.fisheye.projectPoints(rays.reshape(1, -1, 3), zero, zero,
                                               self.K_raw, self._D_fisheye)
        else:
            out, _ = cv2.projectPoints(rays, zero, zero, self.K_raw, self.D_raw)
        return out.reshape(-1, 2)

    # ── reporting ─────────────────────────────────────────────────────────────

    def max_half_angle_deg(self, size):
        """Half-angle (deg) of the widest corner of an image of `size`=(w,h)."""
        w, h = size
        dx = max(self.K_raw[0, 2], w - self.K_raw[0, 2])
        dy = max(self.K_raw[1, 2], h - self.K_raw[1, 2])
        r  = float(np.hypot(dx, dy))
        f  = float(0.5 * (self.K_raw[0, 0] + self.K_raw[1, 1]))
        # equidistant: theta = r / f;  pinhole: theta = atan(r / f)
        theta = r / f if self.fisheye else np.arctan(r / f)
        return float(np.degrees(theta))

    def describe(self, size=None):
        if not self.active:
            return f'undistortion off (no distortion in camera_info, model={self.model})'

        coeffs = ', '.join(f'{c:.4f}' for c in self.D_raw)
        head = (f'undistortion on, mode={self.mode} '
                f'(model={self.model}, D=[{coeffs}])')

        if self.mode == MODE_POINTS:
            return head + ' — detection runs on the raw frame, corners are undistorted'

        note = ''
        if size is not None:
            kept = np.degrees(np.arctan(
                max(self.K[1, 2], size[1] - self.K[1, 2]) / self.K[1, 1]))
            note = (f', rectified frame keeps {kept:.0f}° of the lens\'s '
                    f'{self.max_half_angle_deg(size):.0f}° half-field')
        return head + f' — frame is rectified, K unchanged{note}'
