#!/usr/bin/env python3
"""Plot 2026_03_26 flight data: position (EKF2+mocap), velocity, RPY, cmd_raw moments, actual RPM moments."""

import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def apply_lpf(signal, fc, ts):
    """Apply 1st-order exponential LPF (exact ZOH).
    y[k+1] = beta * y[k] + (1 - beta) * x[k], beta = exp(-2*pi*fc*ts)
    """
    beta = np.exp(-2 * np.pi * fc * ts)
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for k in range(1, len(signal)):
        out[k] = beta * out[k - 1] + (1 - beta) * signal[k]
    return out


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
    ekf_vx_b, ekf_vy_b, ekf_vz_b = [], [], []  # body frame velocity
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
        ekf_vx_b.append(vx); ekf_vy_b.append(vy); ekf_vz_b.append(vz)

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
    cmd_rpms_list = []
    for data, in c.fetchall():
        t, cmds = parse_cmd_raw(data)
        rpms = cmds * MaxRpm / MaxBit
        thrusts = C_T * rpms ** 2
        u = K_forward @ thrusts
        cmd_t.append(t)
        cmd_F.append(u[0])
        cmd_Mx.append(u[1]); cmd_My.append(u[2]); cmd_Mz.append(u[3])
        cmd_rpms_list.append(rpms.copy())

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

    # ── nmpc/control (pure MPC output) ──
    tid = topics['/nmpc/control']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    mpc_t, mpc_F, mpc_Mx, mpc_My, mpc_Mz = [], [], [], [], []
    for data, in c.fetchall():
        t, fx, fy, fz, tx, ty, tz = parse_wrench_stamped(data)
        mpc_t.append(t)
        mpc_F.append(fz)
        mpc_Mx.append(tx); mpc_My.append(ty); mpc_Mz.append(tz)

    # ── RC in (kill switch detection) ──
    kill_t = None
    if '/mavros/rc/in' in topics:
        tid = topics['/mavros/rc/in']
        c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
        for data, in c.fetchall():
            sec = struct.unpack_from('<I', data, 4)[0]
            nsec = struct.unpack_from('<I', data, 8)[0]
            ch8 = struct.unpack_from('<H', data, 24 + 8 * 2)[0]  # channel 8 (SE)
            if ch8 < 1200:
                kill_t = sec + nsec * 1e-9
                break

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
    mpc_t = np.array(mpc_t)
    hgdo_t = np.array(hgdo_t)
    t0 = odom_t[0]
    odom_t -= t0; mocap_t -= t0; cmd_t -= t0; rpm_t -= t0; mpc_t -= t0; hgdo_t -= t0
    if kill_t is not None:
        kill_t -= t0

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
        ekf_vx_b=np.array(ekf_vx_b), ekf_vy_b=np.array(ekf_vy_b), ekf_vz_b=np.array(ekf_vz_b),
        mocap_t=mocap_t,
        mocap_px=np.array(mocap_px), mocap_py=np.array(mocap_py), mocap_pz=mocap_pz,
        mocap_pz_offset=mocap_pz_offset,
        mocap_roll=np.array(mocap_roll), mocap_pitch=np.array(mocap_pitch), mocap_yaw=np.array(mocap_yaw),
        cmd_t=cmd_t, cmd_F=np.array(cmd_F),
        cmd_Mx=np.array(cmd_Mx), cmd_My=np.array(cmd_My), cmd_Mz=np.array(cmd_Mz),
        cmd_rpms=np.array(cmd_rpms_list),
        rpm_t=rpm_t, rpm_F=np.array(rpm_F),
        rpm_Mx=np.array(rpm_Mx), rpm_My=np.array(rpm_My), rpm_Mz=np.array(rpm_Mz),
        rpm_raw=np.array(rpm_raw),
        hgdo_t=hgdo_t,
        mpc_t=mpc_t, mpc_F=np.array(mpc_F),
        mpc_Mx=np.array(mpc_Mx), mpc_My=np.array(mpc_My), mpc_Mz=np.array(mpc_Mz),
        hgdo_fx=np.array(hgdo_fx), hgdo_fy=np.array(hgdo_fy), hgdo_fz=np.array(hgdo_fz),
        hgdo_tx=np.array(hgdo_tx), hgdo_ty=np.array(hgdo_ty), hgdo_tz=np.array(hgdo_tz),
        kill_t=kill_t,
    )


def add_kill_line(ax, kill_t):
    """Add a red dashed vertical line at kill time."""
    if kill_t is not None:
        ax.axvline(kill_t, color='red', ls='--', lw=1.2, alpha=0.8, label=f'kill {kill_t:.1f}s')


def plot_bag(bag_name, db_path):
    d = load_bag(db_path)
    base = f'/home/user/drone_control_pkgs/bag_folder/{bag_name}'
    kill_t = d['kill_t']

    # ── 1. Position + Velocity + Position Error (EKF2 + mocap) ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
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
    ax.set_title(f'Linear velocity - world frame ({bag_name})')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Position error (EKF2 - mocap), interpolate mocap onto ekf time
    mocap_px_i = np.interp(d['odom_t'], d['mocap_t'], d['mocap_px'])
    mocap_py_i = np.interp(d['odom_t'], d['mocap_t'], d['mocap_py'])
    mocap_pz_i = np.interp(d['odom_t'], d['mocap_t'], d['mocap_pz'])
    ax = axes[2]
    ax.plot(d['odom_t'], d['ekf_px'] - mocap_px_i, 'tab:red', lw=0.8, label='ex')
    ax.plot(d['odom_t'], d['ekf_py'] - mocap_py_i, 'tab:blue', lw=0.8, label='ey')
    ax.plot(d['odom_t'], d['ekf_pz'] - mocap_pz_i, 'tab:green', lw=0.8, label='ez')
    ax.set_ylabel('Position error (m)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Position error (EKF2 - Mocap) ({bag_name})')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        add_kill_line(ax, kill_t)

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

    # ── 2. MPC moment + angle + angular velocity (6 rows, dual y-axis) ──
    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)

    for i, (m_label, mpc_key, cmd_key, a_label, ekf_key, mocap_key, w_label, w_key) in enumerate([
        ('Mx', 'mpc_Mx', 'cmd_Mx', 'Roll', 'ekf_roll', 'mocap_roll', 'wx', 'ekf_wx'),
        ('My', 'mpc_My', 'cmd_My', 'Pitch', 'ekf_pitch', 'mocap_pitch', 'wy', 'ekf_wy'),
        ('Mz', 'mpc_Mz', 'cmd_Mz', 'Yaw', 'ekf_yaw', 'mocap_yaw', 'wz', 'ekf_wz'),
    ]):
        # Row 1: Moment + Angle
        ax1 = axes[i * 2]
        ln1 = ax1.plot(d['mpc_t'], d[mpc_key], color='tab:blue', lw=0.8, label=f'MPC {m_label} (Nm)')
        ln4 = ax1.plot(d['cmd_t'], d[cmd_key], color='tab:cyan', lw=0.6, alpha=0.5, label=f'cmd_raw {m_label} (Nm)')
        ax1.set_ylabel(f'{m_label} Moment (Nm)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ln2 = ax2.plot(d['odom_t'], d[ekf_key], color='tab:red', lw=0.8, label=f'{a_label} EKF2 (deg)')
        ln3 = ax2.plot(d['mocap_t'], d[mocap_key], color='tab:orange', lw=0.8, alpha=0.7, label=f'{a_label} Mocap (deg)')
        ax2.set_ylabel(f'{a_label} (deg)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ln3 = ax2.plot(d['mocap_t'], d[mocap_key], color='tab:orange', lw=0.8, alpha=0.7, label=f'{a_label} Mocap (deg)')
        ax2.set_ylabel(f'{a_label} (deg)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        if liftoff_t is not None:
            ax1.axvline(liftoff_t, color='k', ls='--', lw=0.8, alpha=0.6, label=f'liftoff {liftoff_t:.1f}s')
        if kill_t is not None:
            ax1.axvline(kill_t, color='red', ls='--', lw=1.2, alpha=0.8, label=f'kill {kill_t:.1f}s')

        lns = ln1 + ln4 + ln2 + ln3
        if liftoff_t is not None:
            lns += [ax1.get_lines()[-2 if kill_t is not None else -1]]
        if kill_t is not None:
            lns += [ax1.get_lines()[-1]]
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper right', fontsize=8)
        ax1.set_title(f'MPC {m_label} + {a_label} ({bag_name})')

        # Row 2: Moment + Angular velocity
        ax3 = axes[i * 2 + 1]
        ln5 = ax3.plot(d['mpc_t'], d[mpc_key], color='tab:blue', lw=0.8, label=f'MPC {m_label} (Nm)')
        ln6 = ax3.plot(d['cmd_t'], d[cmd_key], color='tab:cyan', lw=0.6, alpha=0.5, label=f'cmd_raw {m_label} (Nm)')
        ax3.set_ylabel(f'{m_label} Moment (Nm)', color='tab:blue')
        ax3.tick_params(axis='y', labelcolor='tab:blue')
        ax3.grid(True, alpha=0.3)

        ax4 = ax3.twinx()
        ln7 = ax4.plot(d['odom_t'], d[w_key], color='tab:green', lw=0.8, label=f'{w_label} (rad/s)')
        ax4.set_ylabel(f'{w_label} (rad/s)', color='tab:green')
        ax4.tick_params(axis='y', labelcolor='tab:green')

        if liftoff_t is not None:
            ax3.axvline(liftoff_t, color='k', ls='--', lw=0.8, alpha=0.6)
        if kill_t is not None:
            ax3.axvline(kill_t, color='red', ls='--', lw=1.2, alpha=0.8)

        lns_w = ln5 + ln6 + ln7
        labs_w = [l.get_label() for l in lns_w]
        ax3.legend(lns_w, labs_w, loc='upper right', fontsize=8)
        ax3.set_title(f'MPC {m_label} + {w_label} ({bag_name})')

    axes[5].set_xlabel('Time (s)')

    plt.tight_layout()
    out = f'{base}_mpc_moments_rpy.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')

    # ── 3. Actual RPM + cmd_raw RPM per motor ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = ['tab:red', 'tab:orange', 'tab:olive', 'tab:green', 'tab:blue', 'tab:purple']

    ax = axes[0]
    for i in range(6):
        ax.plot(d['rpm_t'], d['rpm_raw'][:, i], color=colors[i], lw=0.7, alpha=0.8, label=f'M{i+1}')
    ax.set_ylabel('RPM')
    ax.set_title(f'Actual RPM per motor ({bag_name})')
    ax.legend(loc='upper right', fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cmd_rpms = d['cmd_rpms']
    for i in range(6):
        ax.plot(d['cmd_t'], cmd_rpms[:, i], color=colors[i], lw=0.7, alpha=0.8, label=f'M{i+1}')
    ax.set_ylabel('RPM')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Cmd raw RPM per motor ({bag_name})')
    ax.legend(loc='upper right', fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        add_kill_line(ax, kill_t)

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

    for ax in axes:
        add_kill_line(ax, kill_t)

    plt.tight_layout()
    out = f'{base}_hgdo.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')

    # ── 5. RPY comparison (initial offset subtracted) ──
    mocap_roll_i = np.interp(d['odom_t'], d['mocap_t'], d['mocap_roll'])
    mocap_pitch_i = np.interp(d['odom_t'], d['mocap_t'], d['mocap_pitch'])
    mocap_yaw_i = np.interp(d['odom_t'], d['mocap_t'], d['mocap_yaw'])

    # Subtract initial values
    ekf_roll0 = d['ekf_roll'][0]; ekf_pitch0 = d['ekf_pitch'][0]; ekf_yaw0 = d['ekf_yaw'][0]
    mocap_roll0 = mocap_roll_i[0]; mocap_pitch0 = mocap_pitch_i[0]; mocap_yaw0 = mocap_yaw_i[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (label, ekf_sig, ekf0, mocap_sig, mocap0) in enumerate([
        ('Roll', d['ekf_roll'], ekf_roll0, mocap_roll_i, mocap_roll0),
        ('Pitch', d['ekf_pitch'], ekf_pitch0, mocap_pitch_i, mocap_pitch0),
        ('Yaw', d['ekf_yaw'], ekf_yaw0, mocap_yaw_i, mocap_yaw0),
    ]):
        ax = axes[i]
        ax.plot(d['odom_t'], ekf_sig - ekf0, 'tab:blue', lw=0.8, label=f'EKF2 (init={ekf0:.2f})')
        ax.plot(d['odom_t'], mocap_sig - mocap0, 'tab:red', lw=0.8, alpha=0.7, label=f'Mocap (init={mocap0:.2f})')
        ax.set_ylabel(f'{label} (deg)')
        ax.set_title(f'{label} - initial offset subtracted ({bag_name})')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[2].set_xlabel('Time (s)')
    for ax in axes:
        add_kill_line(ax, kill_t)
    plt.tight_layout()
    out = f'{base}_rpy_comparison.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')

    # ── 6. RPY error (EKF2 - Mocap, raw without offset subtraction) ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (label, ekf_sig, mocap_sig) in enumerate([
        ('Roll', d['ekf_roll'], mocap_roll_i),
        ('Pitch', d['ekf_pitch'], mocap_pitch_i),
        ('Yaw', d['ekf_yaw'], mocap_yaw_i),
    ]):
        ax = axes[i]
        err = ekf_sig - mocap_sig
        ax.plot(d['odom_t'], err, 'tab:blue', lw=0.8)
        ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
        ax.set_ylabel(f'{label} error (deg)')
        ax.set_title(f'{label} error (EKF2 - Mocap), mean={np.mean(err):.3f}, std={np.std(err):.3f} ({bag_name})')
        ax.grid(True, alpha=0.3)
    axes[2].set_xlabel('Time (s)')
    for ax in axes:
        add_kill_line(ax, kill_t)
    plt.tight_layout()
    out = f'{base}_rpy_error.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')

    # ── 7. LPF comparison: body-frame velocity + angular velocity ──
    # Compute average sample time from odom timestamps
    dt_arr = np.diff(d['odom_t'])
    ts = np.median(dt_arr)
    # Plot order: raw (bottom) → 20 Hz → 15 Hz → 10 Hz (top, smoothest)
    cutoffs_draw = [20, 15, 10]
    lpf_colors_draw = ['tab:purple', 'tab:green', 'tab:orange']

    fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True)
    vel_labels = [
        ('vx_b', d['ekf_vx_b'], 'Body vx (m/s)'),
        ('vy_b', d['ekf_vy_b'], 'Body vy (m/s)'),
        ('vz_b', d['ekf_vz_b'], 'Body vz (m/s)'),
        ('wx', d['ekf_wx'], 'wx (rad/s)'),
        ('wy', d['ekf_wy'], 'wy (rad/s)'),
        ('wz', d['ekf_wz'], 'wz (rad/s)'),
    ]
    for idx, (name, sig, ylabel) in enumerate(vel_labels):
        ax = axes[idx]
        ax.plot(d['odom_t'], sig, 'tab:blue', lw=0.4, alpha=0.35, label='raw')
        for fc, clr in zip(cutoffs_draw, lpf_colors_draw):
            filtered = apply_lpf(sig, fc, ts)
            ax.plot(d['odom_t'], filtered, color=clr, lw=1.0, label=f'LPF {fc}Hz')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name} raw vs LPF (ts={ts*1000:.1f}ms) ({bag_name})')
        ax.legend(loc='upper right', fontsize=8, ncol=4)
        ax.grid(True, alpha=0.3)
        add_kill_line(ax, kill_t)
    axes[5].set_xlabel('Time (s)')
    plt.tight_layout()
    out = f'{base}_velocity_lpf.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')

    # ── 8. RPM vs angular velocity noise (scatter) ──
    # Interpolate RPM onto odom time
    rpm_raw = d['rpm_raw']  # (N, 6)
    rpm_mean_per_sample = np.mean(np.abs(rpm_raw), axis=1)  # mean of 6 motors
    rpm_interp = np.interp(d['odom_t'], d['rpm_t'], rpm_mean_per_sample)

    # Windowed analysis: detrend + std per window
    win_sec = 0.3
    win_samples = max(int(win_sec / ts), 5)
    n = len(d['odom_t'])
    n_windows = n // win_samples

    win_rpm, win_wx_std, win_wy_std, win_wz_std = [], [], [], []
    for i in range(n_windows):
        s = i * win_samples
        e = s + win_samples
        win_rpm.append(np.mean(rpm_interp[s:e]))
        # Detrend (remove linear trend) then take std = high-freq noise
        for wlist, sig in [(win_wx_std, d['ekf_wx']),
                           (win_wy_std, d['ekf_wy']),
                           (win_wz_std, d['ekf_wz'])]:
            seg = sig[s:e]
            t_seg = np.arange(len(seg))
            if len(seg) > 1:
                coef = np.polyfit(t_seg, seg, 1)
                detrended = seg - np.polyval(coef, t_seg)
                wlist.append(np.std(detrended))
            else:
                wlist.append(0.0)

    win_rpm = np.array(win_rpm)
    win_wx_std = np.array(win_wx_std)
    win_wy_std = np.array(win_wy_std)
    win_wz_std = np.array(win_wz_std)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    for ax, wstd, label in zip(axes,
                                [win_wx_std, win_wy_std, win_wz_std],
                                ['wx', 'wy', 'wz']):
        ax.scatter(win_rpm, wstd, s=8, alpha=0.5, edgecolors='none')
        ax.set_xlabel('Mean RPM (abs, 6-motor avg)')
        ax.set_ylabel(f'{label} noise std (rad/s)')
        ax.set_title(f'{label} noise vs RPM (window={win_sec}s, detrended) ({bag_name})')
        ax.grid(True, alpha=0.3)
        # Add reference lines
        ax.axhline(0.000245, color='green', ls='--', lw=1, alpha=0.7, label='static std (0.000245)')
        ax.axhline(0.05, color='red', ls='--', lw=1, alpha=0.7, label='sim worst (0.05)')
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = f'{base}_rpm_vs_noise.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')


# ── Run for selected bags ──
for bag in ['2026_03_26_07', '2026_03_31_01', '2026_03_31_02']:
    db = f'/home/user/drone_control_pkgs/bag_folder/{bag}/{bag}_0.db3'
    print(f'\n=== {bag} ===')
    plot_bag(bag, db)
