"""Trajectory and error visualization."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pose_math import se3_inv


def plot_results(vio_log, T_board_world, ate, traj_len, out_dir,
                 start_poses=None, end_poses=None):
    """
    Generate results.png with three subplots:
      1. Top-down VIO trajectory with board origin and start-segment camera positions
      2. Per-frame position ATE (end segment)
      3. ATE summary text
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── Plot 1: top-down trajectory ───────────────────────────────────────────
    ax = axes[0]
    if vio_log:
        xs = [T[0, 3] for _, T in vio_log]
        ys = [T[1, 3] for _, T in vio_log]
        ax.plot(xs, ys, color='royalblue', lw=0.8, label='VIO trajectory')
        ax.plot(xs[0],  ys[0],  'g^', ms=9, zorder=5, label='traj start')
        ax.plot(xs[-1], ys[-1], 'rs', ms=9, zorder=5, label='traj end')

    # board origin in world frame
    T_world_board = se3_inv(T_board_world)
    bo = T_world_board[:3, 3]
    ax.plot(bo[0], bo[1], 'k*', ms=14, zorder=6, label='board origin')

    # camera positions from start-segment AprilTag detections (reprojection-error filtered)
    if start_poses:
        cam_pts = []
        for _, T_cam_board in start_poses:
            T_world_cam = T_world_board @ se3_inv(T_cam_board)
            cam_pts.append(T_world_cam[:2, 3])
        cam_pts = np.array(cam_pts)
        ax.scatter(cam_pts[:, 0], cam_pts[:, 1],
                   c='darkorange', s=18, zorder=4, alpha=0.7, label=f'start detections ({len(cam_pts)})')

    # camera positions from end-segment AprilTag detections (reprojection-error filtered)
    if end_poses:
        cam_pts_end = []
        for _, T_cam_board in end_poses:
            T_world_cam = T_world_board @ se3_inv(T_cam_board)
            cam_pts_end.append(T_world_cam[:2, 3])
        cam_pts_end = np.array(cam_pts_end)
        ax.scatter(cam_pts_end[:, 0], cam_pts_end[:, 1],
                   c='mediumseagreen', s=18, zorder=4, alpha=0.7, label=f'end detections ({len(cam_pts_end)})')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('VIO Trajectory (top-down, XY)')
    ax.legend(fontsize=7, loc='best')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, lw=0.4)

    # ── Plot 2: per-frame ATE ─────────────────────────────────────────────────
    ax = axes[1]
    errors_m = ate.get('errors_m', [])
    if errors_m:
        ts = np.arange(len(errors_m))
        ax.plot(ts, errors_m, color='crimson', lw=1.2, marker='.', ms=4)
        ax.axhline(ate['rmse'], color='crimson', ls='--', lw=1.0,
                   label=f'RMSE = {ate["rmse"]:.4f} m')
        ax.axhline(ate['mean'], color='orange', ls=':', lw=1.0,
                   label=f'mean = {ate["mean"]:.4f} m')
        ax.set_xlabel('Frame index (end segment)')
        ax.set_ylabel('Position error [m]')
        ax.set_title('Position ATE — end segment')
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.4)

    # ── Plot 3: summary text ──────────────────────────────────────────────────
    ax = axes[2]
    ax.axis('off')
    rel_prec = ate.get('rmse', 0) / traj_len if traj_len > 0 else float('nan')
    summary = (
        f"── ATE (end segment) ──\n"
        f"  frames  : {ate.get('n', 0)}\n"
        f"  RMSE    : {ate.get('rmse', 0):.4f} m\n"
        f"  mean    : {ate.get('mean', 0):.4f} m\n"
        f"  max     : {ate.get('max', 0):.4f} m\n"
        f"  std     : {ate.get('std', 0):.4f} m\n"
        f"\n── Trajectory ──\n"
        f"  length  : {traj_len:.2f} m\n"
        f"\n── Relative Precision ──\n"
        f"  RMSE / length\n"
        f"  = {rel_prec*100:.4f} %"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Summary')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[VIZ] Saved {out_path}')
