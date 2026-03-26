#!/usr/bin/env python3
"""Plot 2026_03_26 flight data: position (EKF2+mocap), velocity, RPY, cmd_raw moments, actual RPM moments."""

import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ── Constants ──
C_T = 1.386e-07
k_m = 0.01569
l = 0.265
MaxBit = 8191
MaxRpm = 9800

lx1 = l * np.sin(np.pi / 3); ly1 = l * np.cos(np.pi / 3)
lx2 = 0.0;                    ly2 = l
lx3 = -l * np.sin(np.pi / 3); ly3 = l * np.cos(np.pi / 3)
lx4 = -l * np.sin(np.pi / 3); ly4 = -l * np.cos(np.pi / 3)
lx5 = 0.0;                    ly5 = -l
lx6 = l * np.sin(np.pi / 3);  ly6 = -l * np.cos(np.pi / 3)

K_forward = np.array([
    [1, 1, 1, 1, 1, 1],
    [ly1, ly2, ly3, ly4, ly5, ly6],
    [-lx1, -lx2, -lx3, -lx4, -lx5, -lx6],
    [-k_m, k_m, -k_m, k_m, -k_m, k_m]
])


def parse_cmd_raw(data):
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    if off % 2 != 0: off += 1
    cmds = np.array(struct.unpack_from('<6h', data, off), dtype=np.float64)
    return sec + nsec * 1e-9, cmds


def parse_actual_rpm(data):
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    off = (off + 3) & ~3
    rpms = np.array(struct.unpack_from('<6i', data, off), dtype=np.float64)
    return sec + nsec * 1e-9, rpms


def parse_odom(data):
    """Parse Odometry CDR2 → (t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz)."""
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, 44)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 68)
    vx, vy, vz = struct.unpack_from('<3d', data, 388)
    wx, wy, wz = struct.unpack_from('<3d', data, 412)
    t = sec + nsec * 1e-9
    return t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz


def parse_pose_stamped(data):
    """Parse PoseStamped CDR2 → (t, px, py, pz, qx, qy, qz, qw)."""
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, 28)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 52)
    t = sec + nsec * 1e-9
    return t, px, py, pz, qx, qy, qz, qw


def parse_wrench_stamped(data):
    """Parse WrenchStamped CDR2 → (t, fx, fy, fz, tx, ty, tz).
    frame_id='base_link' (10 bytes), force at off=28, torque at off=52.
    """
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    fx, fy, fz = struct.unpack_from('<3d', data, 28)
    tx, ty, tz = struct.unpack_from('<3d', data, 52)
    t = sec + nsec * 1e-9
    return t, fx, fy, fz, tx, ty, tz


def load_bag(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name FROM topics')
    topics = {name: tid for tid, name in c.fetchall()}

    # ── odom (EKF2) ──
    tid = topics['/mavros/local_position/odom']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    odom_t, ekf_px, ekf_py, ekf_pz = [], [], [], []
    ekf_vx_w, ekf_vy_w, ekf_vz_w = [], [], []
    ekf_roll, ekf_pitch, ekf_yaw = [], [], []
    ekf_wx, ekf_wy, ekf_wz = [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = parse_odom(data)
        q = np.array([qx, qy, qz, qw])
        norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm < 1e-10:
            continue
        R = Rotation.from_quat(q / norm)
        roll, pitch, yaw = R.as_euler('xyz', degrees=True)
        v_world = R.as_matrix() @ np.array([vx, vy, vz])
        odom_t.append(t)
        ekf_px.append(px); ekf_py.append(py); ekf_pz.append(pz)
        ekf_vx_w.append(v_world[0]); ekf_vy_w.append(v_world[1]); ekf_vz_w.append(v_world[2])
        ekf_roll.append(roll); ekf_pitch.append(pitch); ekf_yaw.append(yaw)
        ekf_wx.append(wx); ekf_wy.append(wy); ekf_wz.append(wz)

    # ── mocap (/S550/pose) ──
    tid = topics['/S550/pose']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    mocap_t, mocap_px, mocap_py, mocap_pz = [], [], [], []
    mocap_roll, mocap_pitch, mocap_yaw = [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw = parse_pose_stamped(data)
        q = np.array([qx, qy, qz, qw])
        norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm < 1e-10:
            continue
        r = Rotation.from_quat(q / norm)
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        mocap_t.append(t)
        mocap_px.append(px); mocap_py.append(py); mocap_pz.append(pz)
        mocap_roll.append(roll); mocap_pitch.append(pitch); mocap_yaw.append(yaw)

    # ── cmd_raw ──
    tid = topics['/uav/cmd_raw']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    cmd_t, cmd_Mx, cmd_My, cmd_Mz, cmd_F = [], [], [], [], []
    for data, in c.fetchall():
        t, cmds = parse_cmd_raw(data)
        rpms = cmds * MaxRpm / MaxBit
        thrusts = C_T * rpms ** 2
        u = K_forward @ thrusts
        cmd_t.append(t)
        cmd_F.append(u[0])
        cmd_Mx.append(u[1]); cmd_My.append(u[2]); cmd_Mz.append(u[3])

    # ── actual_rpm ──
    tid = topics['/uav/actual_rpm']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    rpm_t, rpm_Mx, rpm_My, rpm_Mz, rpm_F = [], [], [], [], []
    rpm_raw = []
    for data, in c.fetchall():
        t, rpms = parse_actual_rpm(data)
        thrusts = C_T * rpms ** 2
        u = K_forward @ thrusts
        rpm_t.append(t)
        rpm_F.append(u[0])
        rpm_Mx.append(u[1]); rpm_My.append(u[2]); rpm_Mz.append(u[3])
        rpm_raw.append(rpms.copy())

    # ── hgdo wrench ──
    tid = topics['/hgdo/wrench']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    hgdo_t, hgdo_fx, hgdo_fy, hgdo_fz = [], [], [], []
    hgdo_tx, hgdo_ty, hgdo_tz = [], [], []
    for data, in c.fetchall():
        t, fx, fy, fz, tx, ty, tz = parse_wrench_stamped(data)
        hgdo_t.append(t)
        hgdo_fx.append(fx); hgdo_fy.append(fy); hgdo_fz.append(fz)
        hgdo_tx.append(tx); hgdo_ty.append(ty); hgdo_tz.append(tz)

    conn.close()

    # Convert to arrays and align time
    odom_t = np.array(odom_t); mocap_t = np.array(mocap_t)
    cmd_t = np.array(cmd_t); rpm_t = np.array(rpm_t)
    hgdo_t = np.array(hgdo_t)
    t0 = odom_t[0]
    odom_t -= t0; mocap_t -= t0; cmd_t -= t0; rpm_t -= t0; hgdo_t -= t0

    # Subtract initial pz offset (both EKF2 and mocap)
    ekf_pz = np.array(ekf_pz)
    ekf_pz_offset = ekf_pz[0]
    ekf_pz = ekf_pz - ekf_pz_offset

    mocap_pz = np.array(mocap_pz)
    mocap_pz_offset = mocap_pz[0]
    mocap_pz = mocap_pz - mocap_pz_offset

    return dict(
        odom_t=odom_t,
        ekf_px=np.array(ekf_px), ekf_py=np.array(ekf_py), ekf_pz=ekf_pz,
        ekf_pz_offset=ekf_pz_offset,
        ekf_vx=np.array(ekf_vx_w), ekf_vy=np.array(ekf_vy_w), ekf_vz=np.array(ekf_vz_w),
        ekf_roll=np.array(ekf_roll), ekf_pitch=np.array(ekf_pitch), ekf_yaw=np.array(ekf_yaw),
        ekf_wx=np.array(ekf_wx), ekf_wy=np.array(ekf_wy), ekf_wz=np.array(ekf_wz),
        mocap_t=mocap_t,
        mocap_px=np.array(mocap_px), mocap_py=np.array(mocap_py), mocap_pz=mocap_pz,
        mocap_pz_offset=mocap_pz_offset,
        mocap_roll=np.array(mocap_roll), mocap_pitch=np.array(mocap_pitch), mocap_yaw=np.array(mocap_yaw),
        cmd_t=cmd_t, cmd_F=np.array(cmd_F),
        cmd_Mx=np.array(cmd_Mx), cmd_My=np.array(cmd_My), cmd_Mz=np.array(cmd_Mz),
        rpm_t=rpm_t, rpm_F=np.array(rpm_F),
        rpm_Mx=np.array(rpm_Mx), rpm_My=np.array(rpm_My), rpm_Mz=np.array(rpm_Mz),
        rpm_raw=np.array(rpm_raw),
        hgdo_t=hgdo_t,
        hgdo_fx=np.array(hgdo_fx), hgdo_fy=np.array(hgdo_fy), hgdo_fz=np.array(hgdo_fz),
        hgdo_tx=np.array(hgdo_tx), hgdo_ty=np.array(hgdo_ty), hgdo_tz=np.array(hgdo_tz),
    )


def plot_bag(bag_name, db_path):
    d = load_bag(db_path)
    base = f'/home/user/drone_control_pkgs/bag_folder/{bag_name}'

    # ── 1. Position + Velocity (EKF2 + mocap, mocap pz offset subtracted) ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax = axes[0]
    ax.plot(d['odom_t'], d['ekf_px'], 'tab:red', lw=0.8, label='EKF2 x')
    ax.plot(d['odom_t'], d['ekf_py'], 'tab:blue', lw=0.8, label='EKF2 y')
    ax.plot(d['odom_t'], d['ekf_pz'], 'tab:green', lw=0.8, label='EKF2 z')
    ax.plot(d['mocap_t'], d['mocap_px'], 'tab:red', lw=0.8, ls='--', alpha=0.6, label='mocap x')
    ax.plot(d['mocap_t'], d['mocap_py'], 'tab:blue', lw=0.8, ls='--', alpha=0.6, label='mocap y')
    ax.plot(d['mocap_t'], d['mocap_pz'], 'tab:green', lw=0.8, ls='--', alpha=0.6, label='mocap z')
    ax.set_ylabel('Position (m)')
    ax.set_title(f'Position - EKF2 (solid, pz0={d["ekf_pz_offset"]:.3f}m sub) vs Mocap (dashed, pz0={d["mocap_pz_offset"]:.3f}m sub) ({bag_name})')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(d['odom_t'], d['ekf_vx'], 'tab:red', lw=0.8, label='vx')
    ax.plot(d['odom_t'], d['ekf_vy'], 'tab:blue', lw=0.8, label='vy')
    ax.plot(d['odom_t'], d['ekf_vz'], 'tab:green', lw=0.8, label='vz')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Linear velocity - world frame ({bag_name})')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'{base}_position_velocity.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')

    # ── Detect liftoff time: first time cmd_raw total thrust > mg ──
    # Note: actual RPM never exceeded W (29N max vs 30.9N W) — drone flipped before liftoff.
    # Using cmd thrust as proxy for when controller intended liftoff.
    m = 3.144  # kg
    W = m * 9.81
    cmd_total_thrust = d['cmd_F']
    liftoff_t = None
    for j in range(len(cmd_total_thrust)):
        if cmd_total_thrust[j] > W:
            liftoff_t = d['cmd_t'][j]
            break

    # ── 2. cmd_raw moment + angle paired by axis (3 rows x 2 cols) ──
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)

    for i, (m_label, m_key, a_label, ekf_key, mocap_key) in enumerate([
        ('Mx (roll)', 'cmd_Mx', 'Roll', 'ekf_roll', 'mocap_roll'),
        ('My (pitch)', 'cmd_My', 'Pitch', 'ekf_pitch', 'mocap_pitch'),
        ('Mz (yaw)', 'cmd_Mz', 'Yaw', 'ekf_yaw', 'mocap_yaw'),
    ]):
        # Left: moment
        ax = axes[i, 0]
        ax.plot(d['cmd_t'], d[m_key], 'tab:blue', lw=0.8)
        if liftoff_t is not None:
            ax.axvline(liftoff_t, color='k', ls='--', lw=0.8, alpha=0.6, label=f'liftoff {liftoff_t:.1f}s')
            ax.legend(loc='upper right', fontsize=8)
        ax.set_ylabel('Moment (Nm)')
        ax.set_title(f'{m_label} from cmd_raw ({bag_name})')
        ax.grid(True, alpha=0.3)

        # Right: angle
        ax = axes[i, 1]
        ax.plot(d['odom_t'], d[ekf_key], 'tab:blue', lw=0.8, label='EKF2')
        ax.plot(d['mocap_t'], d[mocap_key], 'tab:red', lw=0.8, alpha=0.7, label='Mocap')
        if liftoff_t is not None:
            ax.axvline(liftoff_t, color='k', ls='--', lw=0.8, alpha=0.6)
        ax.set_ylabel(f'{a_label} (deg)')
        ax.set_title(f'{a_label} ({bag_name})')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Time (s)')

    plt.tight_layout()
    out = f'{base}_cmd_moments_rpy.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')

    # ── 3. Actual RPM per motor ──
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    colors = ['tab:red', 'tab:orange', 'tab:olive', 'tab:green', 'tab:blue', 'tab:purple']
    for i in range(6):
        ax.plot(d['rpm_t'], d['rpm_raw'][:, i], color=colors[i], lw=0.7, alpha=0.8, label=f'M{i+1}')
    ax.set_ylabel('RPM')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Actual RPM per motor ({bag_name})')
    ax.legend(loc='upper right', fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f'{base}_actual_rpm.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')

    # ── 4. HGDO (disturbance observer) ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax = axes[0]
    ax.plot(d['hgdo_t'], d['hgdo_fx'], 'tab:red', lw=0.8, label='fx')
    ax.plot(d['hgdo_t'], d['hgdo_fy'], 'tab:blue', lw=0.8, label='fy')
    ax.plot(d['hgdo_t'], d['hgdo_fz'], 'tab:green', lw=0.8, label='fz')
    ax.set_ylabel('Force (N)')
    ax.set_title(f'HGDO estimated disturbance force ({bag_name})')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(d['hgdo_t'], d['hgdo_tx'], 'tab:red', lw=0.8, label='tx (roll)')
    ax.plot(d['hgdo_t'], d['hgdo_ty'], 'tab:blue', lw=0.8, label='ty (pitch)')
    ax.plot(d['hgdo_t'], d['hgdo_tz'], 'tab:green', lw=0.8, label='tz (yaw)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'HGDO estimated disturbance torque ({bag_name})')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'{base}_hgdo.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')


# ── Run for both bags ──
for bag in ['2026_03_26_06', '2026_03_26_07']:
    db = f'/home/user/drone_control_pkgs/bag_folder/{bag}/{bag}_0.db3'
    print(f'\n=== {bag} ===')
    plot_bag(bag, db)
