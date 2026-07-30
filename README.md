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
python3 main.py          # uses config_A500.yaml in current directory defaultly
python3 main.py my.yaml  # custom config
```

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

## Configuration (`config.yaml`)

The config.yaml files for A200 and A500 are provided, and they correspond one-to-one with the model numbers in the lower left corner of the calibration board.

```yaml
ros:
  vio_topic:         /camera/camera/vio_image       # PoseStamped
  image_topic:       /camera/camera/infra1/image_rect_raw   # Image, mono8 or nv12
  static_tf_topic:   /tf_static
  camera_info_topic: /camera/camera/infra1/camera_info
  body_frame:        camera_camera_imu             # parent frame of T_imu_cam
  camera_frame:      camera_camera_left            # child frame

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

### Image encodings

`image_topic` must be `sensor_msgs/msg/Image` (compressed topics are not
supported) with one of these encodings:

| Encoding | Handling |
|---|---|
| `mono8` | used directly |
| `nv12` | the Y (luma) plane is unpacked, chroma is dropped |

Any other encoding is rejected with an error on every frame — the AprilGrid
detector needs single-channel 8-bit luma.

## Output

| File | Description |
|---|---|
| `results/results.png` | Top-down trajectory, per-frame ATE, summary |
| `results/ate_results.txt` | ATE stats and relative precision |
| `results/T_board_world.txt` | 4×4 SE3 alignment matrix |
| `results/vio_poses.csv` | Full VIO pose log (stamp, tx, ty, tz, qx, qy, qz, qw) |
| `results/debug_start_frame0.png` | First start-segment frame (for inspection) |

## Notes

- VIO timestamps and image timestamps are assumed to be exactly 1-to-1 (same stamp).
- `/tf_static` is subscribed with `TRANSIENT_LOCAL + RELIABLE` QoS to receive latched transforms.
- Detection pipeline mirrors kalibr thresholds: 4px border filter, subpix window 2×2, max displacement² 1.5px, back-projection angle < 80°, reprojection error < 1.0px.
- Iterative 2σ outlier rejection is applied to both the alignment and ATE computation.
