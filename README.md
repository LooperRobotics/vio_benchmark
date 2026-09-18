# VIO Benchmark

Evaluate VIO accuracy using an AprilTag calibration board and ROS2.

## How it works

1. **Start segment** — hold the calibration board in view at the beginning of the trajectory. AprilTag detections give ground-truth camera poses; Umeyama alignment (`evo`) registers the VIO world frame to the board frame (`T_board_world`).
2. **Walk** — the robot completes a loop. VIO poses are recorded continuously.
3. **End segment** — return to the board. `T_board_world` is applied to VIO poses and ATE (translation) is computed against the AprilTag ground truth.
4. **Output** — relative precision (`ATE RMSE / trajectory length`), `results.png`, and `ate_results.txt`.

## Requirements

- Docker (recommended) — see `Dockerfile`
- ROS2 Humble, `ros-humble-cv-bridge`
- Python packages: `aprilgrid`, `evo`, `scipy`, `matplotlib`, `numpy<2`, `pyyaml`

## Usage

### Build and run with Docker

```bash
# build image
docker build -t vio_benchmark .

# run (mounts this directory into the container)
./run.sh python3 main.py config_A500.yaml
```

### Run directly

```bash
source /opt/ros/humble/setup.bash
python3 main.py          # defaults to config_A500.yaml in the current directory
python3 main.py my.yaml  # custom config
```

Nothing has to be configured for the camera itself: the message type on
`image_topic` and the lens model both come from the ROS graph and
`camera_info_topic` at startup.

### State machine (press Enter to advance)

```
IDLE → COLLECT_START → RUNNING → COLLECT_END → DONE
```

| State | Action |
|---|---|
| `IDLE` | Place calibration board in camera view, press Enter |
| `COLLECT_START` | Move camera in front of board; press Enter when done |
| `RUNNING` | Walk the full trajectory; return near board, press Enter |
| `COLLECT_END` | Hold board in view again; press Enter when done |
| `DONE` | Results are computed and saved to `results/` |

### What a run prints

```
[vio_benchmark]: Image transport: sensor_msgs/msg/CompressedImage on /…/image_raw/compressed
[vio_benchmark]: T_imu_cam received (…_imu → …_left)
[vio_benchmark]: Camera intrinsics: fx=563.6 fy=563.7 cx=585.7 cy=804.2
[vio_benchmark]: undistortion on, mode=points (model=equidistant, D=[…])
                 — detection runs on the raw frame, corners are undistorted
…
  → 422 start frames captured. Detecting AprilGrid in background…
[START] AprilGrid detected in 310/422 frames
[START] frames dropped at: too_few_tags=104, reproj_too_large=8
```

Check these four lines before trusting a result:

1. **Image transport** — if no frames arrive at all, the subscription type or
   the QoS does not match the publisher (see *Image transport* below).
2. **T_imu_cam received** — otherwise `body_frame` / `camera_frame` do not
   match `/tf_static` and the run cannot finish.
3. **Camera intrinsics** / **undistortion** — `mode=points` for fisheye,
   `mode=image` for pinhole. A principal point far from the image centre here
   means something is wrong with `camera_info`.
4. **detected in N/M frames** with the `dropped at` breakdown.

### Reading the `dropped at` breakdown

Frames without the board in view land in `too_few_tags` and are normal — you
walk away from the board during `COLLECT_START`. The other buckets point at
real problems:

| Bucket | Meaning | Usual cause |
|---|---|---|
| `too_few_tags` | fewer than 4 tags decoded | board out of view, too far, motion blur, or the wrong `tagFamily` |
| `at_image_border` | tags found but clipped by the 4px border | board half out of frame |
| `subpix_moved_too_far` | subpixel refinement moved corners >1.5px² | blur, heavy JPEG compression |
| `folded_past_horizon` | corners beyond the lens's 90° horizon | board at the extreme edge of a fisheye |
| `beyond_80deg` | corners >80° off-axis | same, one step earlier |
| `pnp_failed` | solver did not converge | almost always a geometry mismatch |
| `too_many_outliers` | pose found, but corners disagree with it | **`tagSize` / `tagSpacing` / `tagRows/Cols` do not match the physical board** |
| `reproj_too_large` | mean reprojection >1px | bad intrinsics, or a board printed at the wrong scale |

If nearly every frame lands in `too_many_outliers` or `reproj_too_large`,
check the board spec first — the A200 and A500 boards differ in `tagSize`
(0.022 m vs 0.055 m) and the model number is printed in the board's lower left
corner.

## Configuration (`config.yaml`)

The config.yaml files for A200 and A500 are provided, and they correspond one-to-one with the model numbers in the lower left corner of the calibration board.

```yaml
ros:
  vio_topic:         /camera/camera/vio_image       # PoseStamped
  image_topic:       /camera/camera/infra1/image_rect_raw   # Image or CompressedImage
  static_tf_topic:   /tf_static
  camera_info_topic: /camera/camera/infra1/camera_info
  body_frame:        camera_camera_imu             # parent frame of T_imu_cam
  camera_frame:      camera_camera_left            # child frame

image:
  undistort:  auto         # auto | image | points | false
  compressed: auto         # auto | true | false — message type on image_topic

apriltag:
  target_type: aprilgrid
  tagFamily:   t36h11      # kalibr 2-cell border convention
  tagRows:     6
  tagCols:     6
  tagSize:     0.055       # metres
  tagSpacing:  0.3         # ratio relative to tagSize

output:
  results_dir:  results/
  vio_log_file: vio_poses.csv
```

### Image transport

`image_topic` may carry either `sensor_msgs/msg/Image` or
`sensor_msgs/msg/CompressedImage`. With `image.compressed: auto` (the default)
the type advertised on the topic is looked up in the ROS graph at startup; set
the option to `true` / `false` to force one. This matters because a
subscription with the wrong type never matches the publisher — no error, just
no frames. If no publisher is up yet after 2 s, the type is guessed from the
`/compressed` suffix and a warning is logged.

| Message | Handling |
|---|---|
| `Image`, `mono8` | used directly |
| `Image`, `nv12` | the Y (luma) plane is unpacked, chroma is dropped |
| `CompressedImage`, jpeg/png | decoded with `cv2.imdecode(..., IMREAD_GRAYSCALE)` |

Any other raw encoding is rejected with an error on every frame — the AprilGrid
detector needs single-channel 8-bit luma. `compressedDepth` is rejected too.

Prefer a raw topic when one is available: JPEG is lossy, which costs subpixel
corner accuracy and therefore ATE.

### Undistortion

Distortion is removed using `camera_info_topic` so that PnP always runs on a
distortion-free pinhole camera. This also matters for correctness: `solvePnP`
only knows the radtan model, so equidistant coefficients passed to it directly
would be silently mis-interpreted.

`image.undistort: auto` (the default) picks the strategy from
`distortion_model`:

| Mode | Used for | What happens |
|---|---|---|
| `image` | `plumb_bob`, `rational_polynomial` | the frame is rectified (`cv2.initUndistortRectifyMap`, `K` unchanged), then detected on — straight tag edges help the quad detector |
| `points` | `equidistant` (fisheye) | detection runs on the raw frame; the refined corners are undistorted afterwards |

Why fisheye is different: a pinhole image plane needs `tan θ`, which diverges
towards 90°, so a lens with a 160°+ field cannot be rectified without either
cropping most of the frame or throwing away centre resolution. On the A500
(`fx≈564`, 1200×1600, ~103° half-field) OpenCV's
`estimateNewCameraMatrixForUndistortRectify` returns `f≈332, cy≈1417`, and the
rectified frame covers only −77°…+29° vertically — a board held low drops out
of the image entirely. Undistorting the corners instead keeps the full field at
native resolution. Corners past the horizon, where the inverse model folds onto
the wrong side, are rejected by a round-trip check.

Set the mode explicitly (`image` / `points`) to override, or `false` to feed
raw pixels to the detector and the CameraInfo coefficients to PnP. If
CameraInfo carries no distortion — an already-rectified topic such as
`image_rect_raw` — undistortion stays off regardless.

## Output

| File | Description |
|---|---|
| `results/results.png` | Top-down trajectory, per-frame ATE, summary |
| `results/ate_results.txt` | ATE stats and relative precision |
| `results/T_board_world.txt` | 4×4 SE3 alignment matrix |
| `results/vio_poses.csv` | Full VIO pose log (stamp, tx, ty, tz, qx, qy, qz, qw) |
| `results/debug_start_frame0.png` | First start-segment frame, exactly as fed to the detector |
| `results/debug_end_frame0.png` | Same, for the end segment |

`test_detect.py` runs the bare detector on one of those frames, which is the
quickest way to tell a detection problem from a geometry problem:

```bash
./run.sh python3 test_detect.py results/debug_start_frame0.png
```

## Notes

- VIO timestamps and image timestamps are assumed to be exactly 1-to-1 (same stamp).
- Image subscriptions use default (RELIABLE) QoS; a BEST_EFFORT publisher will not match it.
- `/tf_static` is subscribed with `TRANSIENT_LOCAL + RELIABLE` QoS to receive latched transforms.
- Distortion is removed from `camera_info` (frame or corners, per lens model), so PnP always sees a distortion-free camera.
- Detection pipeline mirrors kalibr thresholds: 4px border filter, subpix window 2×2, max displacement² 1.5px, back-projection angle < 80°, reprojection error < 1.0px.
- Pose is solved on **all** corners (as kalibr does), not with `solvePnPRansac`. The board is planar, so every coplanar minimal sample RANSAC draws is subject to the planar pose ambiguity and no consensus forms — on a real frame it returned 6 of 144 corners as inliers where solving on all of them reprojects at 0.3px. Outlier corners are dropped by reprojection error (2px, 2 rounds) instead.
- Reprojection errors are measured in **raw** image pixels, i.e. mapped back through the lens model. In the undistorted plane a fisheye stretches errors by `sec²θ` (~4× at 60° off-axis), which would make peripheral corners look like outliers.
- Each segment prints where rejected frames died (`frames dropped at: too_few_tags=…, reproj_too_large=…`).
- Iterative 2σ outlier rejection is applied to both the alignment and ATE computation.
