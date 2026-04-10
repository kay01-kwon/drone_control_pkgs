#!/usr/bin/env python3
"""Analyse quaternion double-cover issue in 2026_03_26_07 and 2026_03_31_04 bags.

Plots:
  Row 1: qw component with zero-crossing highlights
  Row 2: Roll / Pitch (Euler)
  Row 3: MPC moments (Mx, My) — shows if controller reacted violently at qw sign flip
"""

import sqlite3, struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ── Parsers (CDR2) ─────────────────────────────────────────────────
def parse_odom(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, 44)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 68)
    vx, vy, vz = struct.unpack_from('<3d', data, 388)
    wx, wy, wz = struct.unpack_from('<3d', data, 412)
    return sec + nsec * 1e-9, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz

def parse_wrench_stamped(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    fx, fy, fz = struct.unpack_from('<3d', data, 28)
    tx, ty, tz = struct.unpack_from('<3d', data, 52)
    return sec + nsec * 1e-9, fx, fy, fz, tx, ty, tz

def parse_pose_stamped(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, 28)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 52)
    return sec + nsec * 1e-9, px, py, pz, qx, qy, qz, qw


# ── Load one bag ───────────────────────────────────────────────────
def load_bag(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name FROM topics')
    topics = {name: tid for tid, name in c.fetchall()}

    # odom (EKF2)
    tid = topics['/mavros/local_position/odom']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    odom_t, qws, qxs, qys, qzs = [], [], [], [], []
    rolls, pitches, yaws = [], [], []
    wxs, wys, wzs = [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = parse_odom(data)
        q = np.array([qx, qy, qz, qw])
        norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm < 1e-10:
            continue
        q_n = q / norm
        r = Rotation.from_quat(q_n)
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        odom_t.append(t)
        qws.append(qw); qxs.append(qx); qys.append(qy); qzs.append(qz)
        rolls.append(roll); pitches.append(pitch); yaws.append(yaw)
        wxs.append(np.degrees(wx)); wys.append(np.degrees(wy)); wzs.append(np.degrees(wz))

    # MPC control wrench
    mpc_t, mpc_Mx, mpc_My, mpc_Mz = [], [], [], []
    if '/nmpc/control' in topics:
        tid = topics['/nmpc/control']
        c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
        for data, in c.fetchall():
            t, fx, fy, fz, tx, ty, tz = parse_wrench_stamped(data)
            mpc_t.append(t)
            mpc_Mx.append(tx); mpc_My.append(ty); mpc_Mz.append(tz)

    # mocap (/S550/pose) for RPY comparison
    mocap_t, mocap_rolls, mocap_pitches, mocap_yaws = [], [], [], []
    mocap_qws = []
    if '/S550/pose' in topics:
        tid = topics['/S550/pose']
        c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
        for data, in c.fetchall():
            t, px, py, pz, qx, qy, qz, qw = parse_pose_stamped(data)
            q = np.array([qx, qy, qz, qw])
            norm = np.linalg.norm(q)
            if not np.isfinite(norm) or norm < 1e-10:
                continue
            r = Rotation.from_quat(q / norm)
            roll, pitch, yaw = r.as_euler('xyz', degrees=True)
            mocap_t.append(t)
            mocap_qws.append(qw)
            mocap_rolls.append(roll); mocap_pitches.append(pitch); mocap_yaws.append(yaw)

    conn.close()
    return {
        'odom_t': np.array(odom_t), 'qw': np.array(qws), 'qx': np.array(qxs),
        'qy': np.array(qys), 'qz': np.array(qzs),
        'roll': np.array(rolls), 'pitch': np.array(pitches), 'yaw': np.array(yaws),
        'wx': np.array(wxs), 'wy': np.array(wys), 'wz': np.array(wzs),
        'mpc_t': np.array(mpc_t), 'mpc_Mx': np.array(mpc_Mx),
        'mpc_My': np.array(mpc_My), 'mpc_Mz': np.array(mpc_Mz),
        'mocap_t': np.array(mocap_t), 'mocap_qw': np.array(mocap_qws),
        'mocap_roll': np.array(mocap_rolls), 'mocap_pitch': np.array(mocap_pitches),
        'mocap_yaw': np.array(mocap_yaws),
    }


# ── Find qw zero-crossings ────────────────────────────────────────
def find_zero_crossings(t, qw):
    """Return times where qw changes sign."""
    crossings = []
    for i in range(1, len(qw)):
        if qw[i-1] * qw[i] < 0:
            # linear interpolation for exact crossing
            frac = abs(qw[i-1]) / (abs(qw[i-1]) + abs(qw[i]))
            tc = t[i-1] + frac * (t[i] - t[i-1])
            crossings.append(tc)
    return crossings


# ── Plot one bag ───────────────────────────────────────────────────
def plot_bag(data, title, save_path):
    t0 = data['odom_t'][0]
    ot = data['odom_t'] - t0
    mt = data['mpc_t'] - t0 if len(data['mpc_t']) > 0 else np.array([])
    moct = data['mocap_t'] - t0 if len(data['mocap_t']) > 0 else np.array([])

    crossings = find_zero_crossings(ot, data['qw'])

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(title, fontsize=15, fontweight='bold')

    # ── Row 1: Quaternion components ──
    ax = axes[0]
    ax.plot(ot, data['qw'], label='qw', linewidth=1.2, color='tab:blue')
    ax.plot(ot, data['qx'], label='qx', linewidth=0.8, alpha=0.6, color='tab:orange')
    ax.plot(ot, data['qy'], label='qy', linewidth=0.8, alpha=0.6, color='tab:green')
    ax.plot(ot, data['qz'], label='qz', linewidth=0.8, alpha=0.6, color='tab:red')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    for tc in crossings:
        ax.axvline(tc, color='red', linewidth=1.5, alpha=0.7, linestyle='--')
    ax.set_ylabel('Quaternion')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Quaternion Components (EKF2)  |  qw zero-crossings: {len(crossings)}')

    # ── Row 2: Roll / Pitch ──
    ax = axes[1]
    ax.plot(ot, data['roll'], label='Roll (EKF2)', linewidth=1.0, color='tab:red')
    ax.plot(ot, data['pitch'], label='Pitch (EKF2)', linewidth=1.0, color='tab:blue')
    if len(moct) > 0:
        ax.plot(moct, data['mocap_roll'], label='Roll (mocap)', linewidth=0.7, alpha=0.5, linestyle='--', color='tab:red')
        ax.plot(moct, data['mocap_pitch'], label='Pitch (mocap)', linewidth=0.7, alpha=0.5, linestyle='--', color='tab:blue')
    for tc in crossings:
        ax.axvline(tc, color='red', linewidth=1.5, alpha=0.7, linestyle='--')
    ax.set_ylabel('Angle [deg]')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Roll / Pitch')

    # ── Row 3: Yaw ──
    ax = axes[2]
    ax.plot(ot, data['yaw'], label='Yaw (EKF2)', linewidth=1.0, color='tab:green')
    if len(moct) > 0:
        ax.plot(moct, data['mocap_yaw'], label='Yaw (mocap)', linewidth=0.7, alpha=0.5, linestyle='--', color='tab:green')
    for tc in crossings:
        ax.axvline(tc, color='red', linewidth=1.5, alpha=0.7, linestyle='--')
    ax.set_ylabel('Angle [deg]')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Yaw')

    # ── Row 4: MPC Moments ──
    ax = axes[3]
    if len(mt) > 0:
        ax.plot(mt, data['mpc_Mx'], label='Mx (roll)', linewidth=1.0, color='tab:red')
        ax.plot(mt, data['mpc_My'], label='My (pitch)', linewidth=1.0, color='tab:blue')
        ax.plot(mt, data['mpc_Mz'], label='Mz (yaw)', linewidth=0.8, color='tab:green', alpha=0.7)
    for tc in crossings:
        ax.axvline(tc, color='red', linewidth=1.5, alpha=0.7, linestyle='--')
    ax.set_ylabel('Moment [Nm]')
    ax.set_xlabel('Time [s]')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('MPC Control Moments')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'Saved: {save_path}')
    plt.close()


# ── Main ───────────────────────────────────────────────────────────
bags = [
    ('bag_folder/2026_03_26_07/2026_03_26_07_0.db3',
     '2026-03-26-07 (before double-cover fix)',
     'bag_folder/double_cover_analysis_0326_07.png'),
    ('bag_folder/2026_03_31_04/2026_03_31_04_0.db3',
     '2026-03-31-04 (before double-cover fix)',
     'bag_folder/double_cover_analysis_0331_04.png'),
]

for db_path, title, save_path in bags:
    print(f'\nLoading {db_path} ...')
    data = load_bag(db_path)
    crossings = find_zero_crossings(data['odom_t'] - data['odom_t'][0], data['qw'])
    print(f'  qw zero-crossings: {len(crossings)}')
    if crossings:
        print(f'  crossing times: {[f"{c:.2f}s" for c in crossings]}')
    print(f'  qw range: [{data["qw"].min():.4f}, {data["qw"].max():.4f}]')
    print(f'  Roll range: [{data["roll"].min():.1f}, {data["roll"].max():.1f}] deg')
    print(f'  Pitch range: [{data["pitch"].min():.1f}, {data["pitch"].max():.1f}] deg')
    plot_bag(data, title, save_path)
