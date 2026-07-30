#!/usr/bin/env python3
"""
VIO Benchmark — ROS2 node with keyboard-driven state machine.

Usage:
    python3 main.py           (reads config.yaml in current directory)
    python3 main.py my.yaml   (custom config path)

States (advance with ENTER):
    IDLE → COLLECT_START → RUNNING → COLLECT_END → DONE
"""

import sys
import os
import threading
import time
import yaml
import csv

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge

from apriltag_grid import AprilGridDetector
from pose_math import (mat4_from_ros_pose, mat4_from_tf,
                       align_frames, compute_ate, compute_trajectory_length)
from visualizer import plot_results

# ── states ────────────────────────────────────────────────────────────────────
IDLE          = 'IDLE'
COLLECT_START = 'COLLECT_START'
RUNNING       = 'RUNNING'
COLLECT_END   = 'COLLECT_END'
DONE          = 'DONE'

_HINTS = {
    IDLE:          'Put calibration board in view, then press ENTER to start collecting.',
    COLLECT_START: 'Collecting start frames… move away from board, then press ENTER.',
    RUNNING:       'Walk your trajectory. Return near the board, then press ENTER.',
    COLLECT_END:   'Collecting end frames… press ENTER when done.',
    DONE:          'Processing — please wait.',
}

# image encodings accepted on image_topic
SUPPORTED_ENCODINGS = ('mono8', 'nv12')


def _nv12_luma(msg):
    """
    Extract the Y (luma) plane of an NV12 image as a contiguous uint8 array.

    NV12 stores a full-resolution Y plane followed by a half-resolution
    interleaved UV plane, so the buffer holds height * 1.5 rows. AprilGrid
    detection only needs luma, so chroma is dropped.
    """
    buf    = np.frombuffer(msg.data, dtype=np.uint8)
    stride = msg.step if msg.step >= msg.width else msg.width

    # Most publishers set height to the luma height; some report the full
    # buffer height (height * 3 / 2) instead.
    h = msg.height
    if len(buf) < stride * h * 3 // 2:
        h = msg.height * 2 // 3
    if h <= 0 or len(buf) < stride * h:
        raise ValueError(f'nv12 buffer too small: {len(buf)} bytes for '
                         f'{msg.width}x{msg.height} (step={msg.step})')

    return np.ascontiguousarray(buf[:stride * h].reshape(h, stride)[:, :msg.width])


class VIOBenchmark(Node):

    def __init__(self, cfg):
        super().__init__('vio_benchmark')
        self.cfg   = cfg
        self.bridge = CvBridge()

        self._state      = IDLE
        self._lock       = threading.Lock()
        self._last_encoding = None   # log the encoding once, and on change
        self._vio_active = True   # set False when file should close

        # data buffers
        self._start_frames = []   # (stamp_ns: int, image: np.ndarray)
        self._end_frames   = []
        self._vio_log      = []   # (stamp_ns: int, T_world_imu: np.ndarray 4x4)
        self._start_poses  = None # filled by background thread

        # calibration (filled from ROS topics)
        self._K        = None   # camera matrix 3x3
        self._D        = None   # distortion coefficients
        self._T_imu_cam = None  # 4x4, pose of camera in IMU frame

        # AprilGrid detector
        ap = cfg['apriltag']
        self._detector = AprilGridDetector(
            tag_rows=ap['tagRows'], tag_cols=ap['tagCols'],
            tag_size=ap['tagSize'], tag_spacing=ap['tagSpacing'],
            code_offset=ap.get('codeOffset', 0),
            tag_family=ap.get('tagFamily', 'tag36h11'),
        )

        # VIO CSV log (written throughout, closed at DONE)
        out = cfg['output']
        os.makedirs(out['results_dir'], exist_ok=True)
        vio_path = os.path.join(out['results_dir'], out['vio_log_file'])
        self._vio_file   = open(vio_path, 'w', newline='')
        self._vio_writer = csv.writer(self._vio_file)
        self._vio_writer.writerow(['stamp_ns', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        self.get_logger().info(f'VIO log → {vio_path}')

        # ROS2 subscribers
        ros = cfg['ros']
        # /tf_static uses TRANSIENT_LOCAL (latched) — must match or messages are missed
        _tf_qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Image,       ros['image_topic'],       self._img_cb,    10)
        self.create_subscription(PoseStamped, ros['vio_topic'],         self._vio_cb,    10)
        self.create_subscription(TFMessage,   ros['static_tf_topic'],   self._tf_cb,     _tf_qos)
        self.create_subscription(CameraInfo,  ros['camera_info_topic'], self._caminfo_cb, 1)

        # keyboard thread
        threading.Thread(target=self._keyboard_loop, daemon=True).start()

        print('\n' + '='*50)
        print('  VIO Benchmark')
        print('='*50)
        self._print_state()

    # ── state machine ─────────────────────────────────────────────────────────

    def _print_state(self):
        print(f'\n[{self._state}] {_HINTS[self._state]}')

    def _keyboard_loop(self):
        while True:
            input()                # block until Enter
            with self._lock:
                self._advance()

    def _advance(self):
        """Called under self._lock."""
        s = self._state

        if s == IDLE:
            self._state = COLLECT_START

        elif s == COLLECT_START:
            n = len(self._start_frames)
            if n == 0:
                print('[WARN] No frames collected yet — move the camera in front of the board.')
                return
            self._state = RUNNING
            print(f'  → {n} start frames captured. Detecting AprilGrid in background…')
            threading.Thread(target=self._bg_detect_start, daemon=True).start()

        elif s == RUNNING:
            self._state = COLLECT_END

        elif s == COLLECT_END:
            n = len(self._end_frames)
            if n == 0:
                print('[WARN] No end frames collected yet.')
                return
            self._state = DONE
            self._vio_active = False
            print(f'  → {n} end frames captured. Computing results…')
            self._vio_file.flush()
            self._vio_file.close()
            threading.Thread(target=self._bg_report, daemon=True).start()

        elif s == DONE:
            print('[INFO] Already done.')
            return

        self._print_state()

    # ── image decoding ────────────────────────────────────────────────────────

    def _to_gray(self, msg):
        """
        sensor_msgs/Image → single-channel uint8, as the AprilGrid detector
        expects. Only mono8 and nv12 are supported; cv_bridge does not know
        nv12, so its luma plane is unpacked by hand.
        """
        enc = msg.encoding.lower()

        if enc != self._last_encoding:
            self._last_encoding = enc
            self.get_logger().info(f'Image encoding: {msg.encoding} '
                                   f'({msg.width}x{msg.height}, step={msg.step})')

        if enc == 'nv12':
            return _nv12_luma(msg)
        if enc == 'mono8':
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        raise ValueError(f'unsupported encoding — expected one of '
                         f'{", ".join(SUPPORTED_ENCODINGS)}')

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _img_cb(self, msg):
        with self._lock:
            s = self._state
        if s not in (COLLECT_START, COLLECT_END):
            return

        stamp_ns = _stamp_ns(msg.header.stamp)
        try:
            img = self._to_gray(msg)
        except Exception as e:
            self.get_logger().error(
                f'Cannot decode image (encoding={msg.encoding!r}): {e}',
                throttle_duration_sec=5.0)
            return

        with self._lock:
            if self._state == COLLECT_START:
                self._start_frames.append((stamp_ns, img.copy()))
            elif self._state == COLLECT_END:
                self._end_frames.append((stamp_ns, img.copy()))

    def _vio_cb(self, msg):
        with self._lock:
            if not self._vio_active:
                return

        stamp_ns = _stamp_ns(msg.header.stamp)
        p, q = msg.pose.position, msg.pose.orientation
        T = mat4_from_ros_pose(p, q)

        with self._lock:
            self._vio_log.append((stamp_ns, T))

        # write outside the lock to avoid blocking callbacks
        self._vio_writer.writerow([
            stamp_ns, p.x, p.y, p.z, q.x, q.y, q.z, q.w,
        ])

    def _tf_cb(self, msg):
        with self._lock:
            if self._T_imu_cam is not None:
                return

        parent = self.cfg['ros']['body_frame']    # imu
        child  = self.cfg['ros']['camera_frame']  # left_cam

        for tf in msg.transforms:
            if tf.header.frame_id == parent and tf.child_frame_id == child:
                T = mat4_from_tf(tf.transform)
                with self._lock:
                    self._T_imu_cam = T
                self.get_logger().info(f'T_imu_cam received ({parent} → {child})')
                return

    def _caminfo_cb(self, msg):
        with self._lock:
            if self._K is not None:
                return
            self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self._D = np.array(msg.d, dtype=np.float64)
        self.get_logger().info(
            f'Camera intrinsics: fx={self._K[0,0]:.1f} fy={self._K[1,1]:.1f} '
            f'cx={self._K[0,2]:.1f} cy={self._K[1,2]:.1f}'
        )

    # ── background processing ─────────────────────────────────────────────────

    def _calib_snapshot(self):
        """Return a thread-safe snapshot of calibration data, or (None,None,None)."""
        with self._lock:
            K          = self._K.copy()        if self._K is not None        else None
            D          = self._D.copy()        if self._D is not None        else None
            T_imu_cam  = self._T_imu_cam.copy() if self._T_imu_cam is not None else None
        return K, D, T_imu_cam

    def _detect_frames(self, frames, label):
        K, D, _ = self._calib_snapshot()
        if K is None:
            print(f'[ERROR] No camera intrinsics for {label} detection. '
                  'Check camera_info_topic.')
            return []
        # save first frame for visual inspection
        out_dir = self.cfg['output']['results_dir']
        if frames:
            _, sample = frames[0]
            dbg_path = os.path.join(out_dir, f'debug_{label.lower()}_frame0.png')
            cv2.imwrite(dbg_path, sample)
            print(f'[{label}] sample frame: shape={sample.shape} dtype={sample.dtype} '
                  f'min={sample.min()} max={sample.max()} → saved {dbg_path}')

        poses = []
        for stamp_ns, img in frames:
            T_cam_board = self._detector.detect_and_solve(img, K, D)
            if T_cam_board is not None:
                poses.append((stamp_ns, T_cam_board))
        print(f'[{label}] AprilGrid detected in {len(poses)}/{len(frames)} frames')
        return poses

    def _bg_detect_start(self):
        with self._lock:
            frames = list(self._start_frames)
        poses = self._detect_frames(frames, 'START')
        with self._lock:
            self._start_poses = poses

    def _bg_report(self):
        # wait for start detection to finish (max 30 s)
        for _ in range(60):
            with self._lock:
                ready = self._start_poses is not None
            if ready:
                break
            time.sleep(0.5)

        with self._lock:
            end_frames = list(self._end_frames)
            vio_log    = list(self._vio_log)
            T_imu_cam  = self._T_imu_cam.copy() if self._T_imu_cam is not None else None
            start_poses = list(self._start_poses or [])

        if T_imu_cam is None:
            print('[ERROR] T_imu_cam not received from /tf_static. '
                  f'Looking for parent={self.cfg["ros"]["body_frame"]} '
                  f'child={self.cfg["ros"]["camera_frame"]}')
            return

        end_poses = self._detect_frames(end_frames, 'END')

        # ── alignment (evo Umeyama) ───────────────────────────────────────────
        T_board_world, n_used = align_frames(start_poses, vio_log, T_imu_cam)
        if T_board_world is None:
            print(f'[ERROR] Alignment failed — only {n_used} matched start frames '
                  f'(need ≥5).\n'
                  f'       • Check board visibility during COLLECT_START\n'
                  f'       • Check /tf_static frame IDs')
            return
        print(f'[ALIGN] T_board_world estimated from {n_used} frames')

        # ── ATE (evo APE translation_part) ───────────────────────────────────
        ate = compute_ate(end_poses, vio_log, T_board_world, T_imu_cam)
        if not ate:
            print('[ERROR] No matched end frames for ATE.\n'
                  '        • Check timestamps between VIO and image topics\n'
                  '        • Check board is visible during COLLECT_END')
            return

        # ── trajectory length & relative precision ────────────────────────────
        traj_len = compute_trajectory_length(vio_log, T_imu_cam)
        rel_prec = ate['rmse'] / traj_len if traj_len > 0 else float('nan')

        print(f'\n{"="*50}')
        print(f'  End segment : {ate["n"]} matched frames')
        print(f'  ATE rmse    : {ate["rmse"]:.4f} m')
        print(f'  ATE mean    : {ate["mean"]:.4f} m')
        print(f'  ATE max     : {ate["max"]:.4f} m')
        print(f'  ATE std     : {ate["std"]:.4f} m')
        print(f'  Traj length : {traj_len:.2f} m')
        print(f'  Rel. prec.  : {rel_prec*100:.4f} %  ({rel_prec:.6f})')
        print(f'{"="*50}\n')

        # ── save ─────────────────────────────────────────────────────────────
        out_dir = self.cfg['output']['results_dir']
        np.savetxt(os.path.join(out_dir, 'T_board_world.txt'),
                   T_board_world, fmt='%.8f',
                   header='T_board_world (4x4): SE3 alignment from VIO world to board frame')

        with open(os.path.join(out_dir, 'ate_results.txt'), 'w') as f:
            f.write(f'end_segment_frames:   {ate["n"]}\n')
            f.write(f'ate_rmse_m:           {ate["rmse"]:.6f}\n')
            f.write(f'ate_mean_m:           {ate["mean"]:.6f}\n')
            f.write(f'ate_max_m:            {ate["max"]:.6f}\n')
            f.write(f'ate_std_m:            {ate["std"]:.6f}\n')
            f.write(f'trajectory_length_m:  {traj_len:.4f}\n')
            f.write(f'relative_precision:   {rel_prec:.6f}\n')
            f.write(f'relative_precision_%: {rel_prec*100:.4f}\n')

        plot_results(
            vio_log=vio_log,
            T_board_world=T_board_world,
            ate=ate,
            traj_len=traj_len,
            out_dir=out_dir,
            start_poses=start_poses,
            end_poses=end_poses,
        )
        print(f'[DONE] All results saved to {out_dir}')


# ── helpers ───────────────────────────────────────────────────────────────────

def _stamp_ns(stamp):
    return stamp.sec * 10**9 + stamp.nanosec


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config_A500.yaml'
    if not os.path.exists(config_path):
        print(f'Config file not found: {config_path}')
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rclpy.init(args=sys.argv)
    node = VIOBenchmark(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
