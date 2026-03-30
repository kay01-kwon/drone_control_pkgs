#!/usr/bin/env python3
"""Plot 2026_03_30_sim: NMPC+HGDO simulation with 25ms state delay."""

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


def parse_odom(data, pos_off=44, q_off=68, v_off=388, w_off=412):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, pos_off)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, q_off)
    vx, vy, vz = struct.unpack_from('<3d', data, v_off)
    wx, wy, wz = struct.unpack_from('<3d', data, w_off)
    t = sec + nsec * 1e-9
    return t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz


def parse_wrench_stamped(data):
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

    # ── odom_sim (delayed, used by controller) ──
    tid = topics['/mavros/local_position/odom_sim']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    sim_t, sim_px, sim_py, sim_pz = [], [], [], []
    sim_vx_w, sim_vy_w, sim_vz_w = [], [], []
    sim_roll, sim_pitch, sim_yaw = [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = parse_odom(
            data, pos_off=52, q_off=76, v_off=396, w_off=420)
        q = np.array([qx, qy, qz, qw]); norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm < 1e-10: continue
        R = Rotation.from_quat(q / norm)
        roll, pitch, yaw = R.as_euler('xyz', degrees=True)
        v_world = R.as_matrix() @ np.array([vx, vy, vz])
        sim_t.append(t)
        sim_px.append(px); sim_py.append(py); sim_pz.append(pz)
        sim_vx_w.append(v_world[0]); sim_vy_w.append(v_world[1]); sim_vz_w.append(v_world[2])
        sim_roll.append(roll); sim_pitch.append(pitch); sim_yaw.append(yaw)

    # ── odom (ground truth) ──
    tid = topics['/mavros/local_position/odom']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    gt_t, gt_px, gt_py, gt_pz = [], [], [], []
    gt_vx_w, gt_vy_w, gt_vz_w = [], [], []
    gt_roll, gt_pitch, gt_yaw = [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = parse_odom(
            data, pos_off=36, q_off=60, v_off=380, w_off=404)
        q = np.array([qx, qy, qz, qw]); norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm < 1e-10: continue
        R = Rotation.from_quat(q / norm)
        roll, pitch, yaw = R.as_euler('xyz', degrees=True)
        v_world = R.as_matrix() @ np.array([vx, vy, vz])
        gt_t.append(t)
        gt_px.append(px); gt_py.append(py); gt_pz.append(pz)
        gt_vx_w.append(v_world[0]); gt_vy_w.append(v_world[1]); gt_vz_w.append(v_world[2])
        gt_roll.append(roll); gt_pitch.append(pitch); gt_yaw.append(yaw)

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
    rpm_t, rpm_raw = [], []
    for data, in c.fetchall():
        t, rpms = parse_actual_rpm(data)
        rpm_t.append(t)
        rpm_raw.append(rpms.copy())

    # ── nmpc/control ──
    tid = topics['/nmpc/control']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    mpc_t, mpc_F, mpc_Mx, mpc_My, mpc_Mz = [], [], [], [], []
    for data, in c.fetchall():
        t, fx, fy, fz, tx, ty, tz = parse_wrench_stamped(data)
        mpc_t.append(t)
        mpc_F.append(fz)
        mpc_Mx.append(tx); mpc_My.append(ty); mpc_Mz.append(tz)

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

    # Convert and align time
    sim_t = np.array(sim_t); gt_t = np.array(gt_t)
    cmd_t = np.array(cmd_t); rpm_t = np.array(rpm_t)
    mpc_t = np.array(mpc_t); hgdo_t = np.array(hgdo_t)
    t0 = gt_t[0]
    sim_t -= t0; gt_t -= t0; cmd_t -= t0; rpm_t -= t0; mpc_t -= t0; hgdo_t -= t0

    # Subtract initial pz
    sim_pz = np.array(sim_pz); sim_pz = sim_pz - sim_pz[0]
    gt_pz = np.array(gt_pz); gt_pz = gt_pz - gt_pz[0]

    return dict(
        sim_t=sim_t,
        sim_px=np.array(sim_px), sim_py=np.array(sim_py), sim_pz=sim_pz,
        sim_vx=np.array(sim_vx_w), sim_vy=np.array(sim_vy_w), sim_vz=np.array(sim_vz_w),
        sim_roll=np.array(sim_roll), sim_pitch=np.array(sim_pitch), sim_yaw=np.array(sim_yaw),
        gt_t=gt_t,
        gt_px=np.array(gt_px), gt_py=np.array(gt_py), gt_pz=gt_pz,
        gt_vx=np.array(gt_vx_w), gt_vy=np.array(gt_vy_w), gt_vz=np.array(gt_vz_w),
        gt_roll=np.array(gt_roll), gt_pitch=np.array(gt_pitch), gt_yaw=np.array(gt_yaw),
        cmd_t=cmd_t, cmd_F=np.array(cmd_F),
        cmd_Mx=np.array(cmd_Mx), cmd_My=np.array(cmd_My), cmd_Mz=np.array(cmd_Mz),
        cmd_rpms=np.array(cmd_rpms_list),
        rpm_t=rpm_t, rpm_raw=np.array(rpm_raw),
        mpc_t=mpc_t, mpc_F=np.array(mpc_F),
        mpc_Mx=np.array(mpc_Mx), mpc_My=np.array(mpc_My), mpc_Mz=np.array(mpc_Mz),
        hgdo_t=hgdo_t,
        hgdo_fx=np.array(hgdo_fx), hgdo_fy=np.array(hgdo_fy), hgdo_fz=np.array(hgdo_fz),
        hgdo_tx=np.array(hgdo_tx), hgdo_ty=np.array(hgdo_ty), hgdo_tz=np.array(hgdo_tz),
    )


def plot_sim(bag_name, db_path):
    d = load_bag(db_path)
    base = f'/home/user/drone_control_pkgs/bag_folder/{bag_name}'

    # Liftoff detection
    m = 3.144
    W = m * 9.81
    liftoff_t = None
    for j in range(len(d['cmd_F'])):
        if d['cmd_F'][j] > W:
            liftoff_t = d['cmd_t'][j]
            break

    # ── 1. Position + Velocity (odom_sim only, no GT) ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax = axes[0]
    ax.plot(d['sim_t'], d['sim_px'], 'tab:red', lw=0.8, label='x')
    ax.plot(d['sim_t'], d['sim_py'], 'tab:blue', lw=0.8, label='y')
    ax.plot(d['sim_t'], d['sim_pz'], 'tab:green', lw=0.8, label='z')
    ax.set_ylabel('Position (m)')
    ax.set_title(f'Position ({bag_name})')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(d['sim_t'], d['sim_vx'], 'tab:red', lw=0.8, label='vx')
    ax.plot(d['sim_t'], d['sim_vy'], 'tab:blue', lw=0.8, label='vy')
    ax.plot(d['sim_t'], d['sim_vz'], 'tab:green', lw=0.8, label='vz')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Linear velocity - world frame ({bag_name})')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'{base}_position_velocity.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')

    # ── 2. MPC moment + angle (3 rows, dual y-axis, 0-lines aligned) ──
    # Fix roll singularity: unwrap ±180 jumps, then subtract initial value
    sim_roll_rad = np.deg2rad(d['sim_roll'])
    sim_roll_unwrap = np.rad2deg(np.unwrap(sim_roll_rad))
    sim_roll_clean = sim_roll_unwrap - sim_roll_unwrap[0]  # zero-referenced

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (m_label, mpc_key, cmd_key, a_label, angle_data) in enumerate([
        ('Mx', 'mpc_Mx', 'cmd_Mx', 'Roll', sim_roll_clean),
        ('My', 'mpc_My', 'cmd_My', 'Pitch', d['sim_pitch']),
        ('Mz', 'mpc_Mz', 'cmd_Mz', 'Yaw', d['sim_yaw']),
    ]):
        ax1 = axes[i]
        ln1 = ax1.plot(d['mpc_t'], d[mpc_key], color='tab:blue', lw=0.8, label=f'MPC {m_label} (Nm)')
        ln4 = ax1.plot(d['cmd_t'], d[cmd_key], color='tab:cyan', lw=0.6, alpha=0.5, label=f'cmd_raw {m_label} (Nm)')
        ax1.set_ylabel(f'{m_label} Moment (Nm)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ln2 = ax2.plot(d['sim_t'], angle_data, color='tab:red', lw=0.8, label=f'{a_label} (deg)')
        ax2.set_ylabel(f'{a_label} (deg)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Align 0-lines: make both axes symmetric around 0
        m_max = max(abs(ax1.get_ylim()[0]), abs(ax1.get_ylim()[1]))
        ax1.set_ylim(-m_max, m_max)
        a_max = max(abs(ax2.get_ylim()[0]), abs(ax2.get_ylim()[1]))
        ax2.set_ylim(-a_max, a_max)

        if liftoff_t is not None:
            ax1.axvline(liftoff_t, color='k', ls='--', lw=0.8, alpha=0.6, label=f'liftoff {liftoff_t:.1f}s')

        lns = ln1 + ln4 + ln2
        if liftoff_t is not None:
            lns += ax1.get_lines()[-1:]
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper right', fontsize=8)
        ax1.set_title(f'MPC {m_label} + {a_label} ({bag_name})')

    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    out = f'{base}_mpc_moments_rpy.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')

    # ── 3. Actual RPM + cmd_raw RPM ──
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
    for i in range(6):
        ax.plot(d['cmd_t'], d['cmd_rpms'][:, i], color=colors[i], lw=0.7, alpha=0.8, label=f'M{i+1}')
    ax.set_ylabel('RPM')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Cmd raw RPM per motor ({bag_name})')
    ax.legend(loc='upper right', fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'{base}_actual_rpm.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')

    # ── 4. HGDO ──
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

    # ── 5. RPY (odom_sim only, initial offset subtracted, roll clipped) ──
    sim_roll0 = d['sim_roll'][0]; sim_pitch0 = d['sim_pitch'][0]; sim_yaw0 = d['sim_yaw'][0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, (label, sim_sig, sim0) in enumerate([
        ('Roll', sim_roll_clean, sim_roll_clean[0]),
        ('Pitch', d['sim_pitch'], sim_pitch0),
        ('Yaw', d['sim_yaw'], sim_yaw0),
    ]):
        ax = axes[i]
        ax.plot(d['sim_t'], sim_sig - sim0, 'tab:blue', lw=0.8, label=f'odom_sim (init={sim0:.2f})')
        ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
        ax.set_ylabel(f'{label} (deg)')
        ax.set_title(f'{label} - initial offset subtracted ({bag_name})')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    out = f'{base}_rpy_comparison.png'
    plt.savefig(out, dpi=150); plt.close()
    print(f'Saved: {out}')


# ── Run ──
bag = '2026_03_30_sim'
db = f'/home/user/drone_control_pkgs/bag_folder/{bag}/{bag}_0.db3'
print(f'\n=== {bag} ===')
plot_sim(bag, db)
