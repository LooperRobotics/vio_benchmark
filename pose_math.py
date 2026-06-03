"""SE3 utilities and evo-based trajectory alignment / error computation."""

import copy
import numpy as np
from scipy.spatial.transform import Rotation

from evo.core.trajectory import PoseTrajectory3D
from evo.core import geometry as evo_geom
from evo.core.metrics import APE, PoseRelation


# ── SE3 primitives ────────────────────────────────────────────────────────────

def mat4_from_ros_pose(position, orientation):
    q = [orientation.x, orientation.y, orientation.z, orientation.w]
    R = Rotation.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [position.x, position.y, position.z]
    return T


def mat4_from_tf(transform):
    t = transform.translation
    r = transform.rotation
    R = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def se3_inv(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ── VIO log helpers ───────────────────────────────────────────────────────────

def vio_dict(vio_log):
    """stamp_ns (int) → T_world_imu (4x4). O(1) lookup."""
    return {s: T for s, T in vio_log}


# ── evo trajectory helpers ────────────────────────────────────────────────────

def _make_traj(stamp_ns_list, T_list):
    """Build evo PoseTrajectory3D from parallel lists of stamp_ns and 4x4 SE3."""
    ts = np.array(stamp_ns_list, dtype=np.float64) * 1e-9   # ns → s
    return PoseTrajectory3D(poses_se3=list(T_list), timestamps=ts)


# ── alignment ─────────────────────────────────────────────────────────────────

def align_frames(april_poses, vio_log, T_imu_cam, min_frames=5):
    """
    Estimate T_board_world from the start segment using evo Umeyama alignment.

    Conventions (T_AB = pose of B in A frame):
        april_poses → T_cam_board  (solvePnP output)
        T_board_cam = inv(T_cam_board)   ← camera in board frame (ground truth)
        T_world_cam = T_world_imu @ T_imu_cam  ← camera in world frame (VIO)

    Alignment finds R, t such that:
        p_board ≈ R @ p_world + t
    i.e. T_board_world = [[R, t], [0,1]]

    Returns (T_board_world 4x4, n_used) or (None, n_used).
    """
    vio = vio_dict(vio_log)

    stamps, T_board_cam_list, T_world_cam_list = [], [], []
    for stamp_ns, T_cam_board in april_poses:
        T_world_imu = vio.get(stamp_ns)
        if T_world_imu is None:
            continue
        stamps.append(stamp_ns)
        T_board_cam_list.append(se3_inv(T_cam_board))
        T_world_cam_list.append(T_world_imu @ T_imu_cam)

    n_used = len(stamps)
    if n_used < min_frames:
        return None, n_used

    traj_ref = _make_traj(stamps, T_board_cam_list)   # ground truth (board frame)
    traj_est = _make_traj(stamps, T_world_cam_list)   # VIO (world frame)

    # Umeyama: p_ref ≈ s * R @ p_est + t  (s=1 for rigid)
    R, t, _s = evo_geom.umeyama_alignment(
        traj_est.positions_xyz.T,
        traj_ref.positions_xyz.T,
        with_scale=False,
    )

    T_board_world = np.eye(4)
    T_board_world[:3, :3] = R
    T_board_world[:3, 3] = t

    # ── iterative outlier rejection: recompute after dropping >2σ frames ──────
    for iteration in range(3):
        traj_est_aligned = copy.deepcopy(traj_est)
        traj_est_aligned.transform(T_board_world)
        ape_iter = APE(PoseRelation.full_transformation)
        ape_iter.process_data((traj_ref, traj_est_aligned))
        errors = np.array(ape_iter.error)
        threshold = errors.mean() + 2.0 * errors.std()
        mask = errors <= threshold
        if mask.sum() < min_frames or mask.all():
            break
        # rebuild trajectories with inliers only and re-run Umeyama
        idx_in         = [i for i in range(len(stamps)) if mask[i]]
        stamps         = [stamps[i]          for i in idx_in]
        T_board_cam_list = [T_board_cam_list[i] for i in idx_in]
        T_world_cam_list = [T_world_cam_list[i] for i in idx_in]
        traj_ref = _make_traj(stamps, T_board_cam_list)
        traj_est = _make_traj(stamps, T_world_cam_list)
        R, t, _s = evo_geom.umeyama_alignment(
            traj_est.positions_xyz.T,
            traj_ref.positions_xyz.T,
            with_scale=False,
        )
        T_board_world[:3, :3] = R
        T_board_world[:3, 3] = t
        n_used = int(mask.sum())
        print(f'[ALIGN] iter {iteration+1}: {n_used} inliers (dropped {mask.size - n_used} outliers)')

    # final quality report (mirrors evo_ape -r full -a)
    traj_est_final = copy.deepcopy(traj_est)
    traj_est_final.transform(T_board_world)
    ape_final = APE(PoseRelation.full_transformation)
    ape_final.process_data((traj_ref, traj_est_final))
    s = ape_final.get_all_statistics()
    print(f'[ALIGN] final APE (full, {n_used} frames): '
          f'rmse={s["rmse"]:.4f}  mean={s["mean"]:.4f}  max={s["max"]:.4f}')

    return T_board_world, n_used


# ── ATE (end segment) ─────────────────────────────────────────────────────────

def compute_ate(april_poses, vio_log, T_board_world, T_imu_cam):
    """
    Compute APE (translation) for the end segment using evo.

    For each matched frame:
        T_board_cam_gt  = inv(T_cam_board)          ← AprilTag ground truth
        T_board_cam_vio = T_board_world @ T_world_cam ← VIO prediction

    Returns dict: {rmse, mean, std, median, min, max, n, errors_m, stamps_ns}
    or empty dict if no matched frames.
    """
    vio = vio_dict(vio_log)

    stamps, T_gt_list, T_est_list = [], [], []
    for stamp_ns, T_cam_board in april_poses:
        T_world_imu = vio.get(stamp_ns)
        if T_world_imu is None:
            continue
        stamps.append(stamp_ns)
        T_gt_list.append(se3_inv(T_cam_board))
        T_est_list.append(T_board_world @ T_world_imu @ T_imu_cam)

    if not stamps:
        return {}

    traj_ref = _make_traj(stamps, T_gt_list)
    traj_est = _make_traj(stamps, T_est_list)

    # iterative 2σ outlier rejection on translation error
    for _ in range(3):
        ape_iter = APE(PoseRelation.translation_part)
        ape_iter.process_data((traj_ref, traj_est))
        errors = np.array(ape_iter.error)
        threshold = errors.mean() + 2.0 * errors.std()
        mask = errors <= threshold
        if mask.all() or mask.sum() < 4:
            break
        idx_in   = [i for i in range(len(stamps)) if mask[i]]
        stamps   = [stamps[i]    for i in idx_in]
        T_gt_list  = [T_gt_list[i]  for i in idx_in]
        T_est_list = [T_est_list[i] for i in idx_in]
        traj_ref = _make_traj(stamps, T_gt_list)
        traj_est = _make_traj(stamps, T_est_list)
        print(f'[ATE] outlier rejection: kept {mask.sum()}/{len(mask)} frames '
              f'(threshold={threshold:.4f} m)')

    ape = APE(PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_est))
    stats = ape.get_all_statistics()
    stats['n'] = len(stamps)
    stats['errors_m'] = list(ape.error)
    stats['stamps_ns'] = stamps
    return stats


# ── trajectory length ─────────────────────────────────────────────────────────

def compute_trajectory_length(vio_log, T_imu_cam):
    """
    Total camera path length in metres using evo (equivalent to evo_traj -v).
    """
    if len(vio_log) < 2:
        return 0.0
    stamps = [s for s, _ in vio_log]
    poses  = [T_world_imu @ T_imu_cam for _, T_world_imu in vio_log]
    traj   = _make_traj(stamps, poses)
    return traj.get_infos()['path length (m)']
