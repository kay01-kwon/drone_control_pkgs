#!/usr/bin/env python3
"""Plot velocity with LPF at 2 Hz and 5 Hz for bags 2026_03_31_05 and 2026_03_26_07."""

import sqlite3
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ── LPF ──
def apply_lpf(signal, fc, ts):
    """1st-order exponential LPF. beta = exp(-2*pi*fc*ts)"""
    beta = np.exp(-2 * np.pi * fc * ts)
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for k in range(1, len(signal)):
        out[k] = beta * out[k - 1] + (1 - beta) * signal[k]
    return out


def parse_odom(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, 44)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 68)
    vx, vy, vz = struct.unpack_from('<3d', data, 388)
    wx, wy, wz = struct.unpack_from('<3d', data, 412)
    t = sec + nsec * 1e-9
    return t, px, py, pz, qx, qy, qz, qw, vx, vy, vz


def load_velocity(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name FROM topics')
    topics = {name: tid for tid, name in c.fetchall()}

    tid = topics['/mavros/local_position/odom']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))

    ts, vxs, vys, vzs = [], [], [], []
    wxs, wys, wzs = [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw, vx, vy, vz = parse_odom(data)
        q = np.array([qx, qy, qz, qw])
        norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm < 1e-10:
            continue
        R = Rotation.from_quat(q / norm)
        v_world = R.as_matrix() @ np.array([vx, vy, vz])
        ts.append(t)
        vxs.append(v_world[0]); vys.append(v_world[1]); vzs.append(v_world[2])
        # angular velocity (body frame) from odom
        raw = struct.unpack_from('<3d', data, 412)
        wxs.append(raw[0]); wys.append(raw[1]); wzs.append(raw[2])

    conn.close()

    ts = np.array(ts)
    t0 = ts[0]
    ts -= t0
    return ts, np.array(vxs), np.array(vys), np.array(vzs), np.array(wxs), np.array(wys), np.array(wzs)


def plot_bag_velocity(ts, vx, vy, vz, wx, wy, wz, bag_label, filename):
    dt = np.median(np.diff(ts))

    vx_2 = apply_lpf(vx, 2.0, dt)
    vy_2 = apply_lpf(vy, 2.0, dt)
    vz_2 = apply_lpf(vz, 2.0, dt)

    vx_5 = apply_lpf(vx, 5.0, dt)
    vy_5 = apply_lpf(vy, 5.0, dt)
    vz_5 = apply_lpf(vz, 5.0, dt)

    wx_2 = apply_lpf(wx, 2.0, dt)
    wy_2 = apply_lpf(wy, 2.0, dt)
    wz_2 = apply_lpf(wz, 2.0, dt)

    wx_5 = apply_lpf(wx, 5.0, dt)
    wy_5 = apply_lpf(wy, 5.0, dt)
    wz_5 = apply_lpf(wz, 5.0, dt)

    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(f'Velocity & Angular Velocity LPF Comparison — {bag_label}', fontsize=14)

    labels = ['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)',
              'ωx (rad/s)', 'ωy (rad/s)', 'ωz (rad/s)']
    raw_data = [vx, vy, vz, wx, wy, wz]
    lpf2_data = [vx_2, vy_2, vz_2, wx_2, wy_2, wz_2]
    lpf5_data = [vx_5, vy_5, vz_5, wx_5, wy_5, wz_5]

    for i, ax in enumerate(axes):
        ax.plot(ts, raw_data[i], color='lightgray', linewidth=0.5, label='Raw', alpha=0.7)
        ax.plot(ts, lpf5_data[i], color='tab:blue', linewidth=1.0, label='LPF 5 Hz')
        ax.plot(ts, lpf2_data[i], color='tab:red', linewidth=1.2, label='LPF 2 Hz')
        ax.set_ylabel(labels[i])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f'Saved: {filename}')


# ── Main ──
bags = [
    ('2026_03_31_05/2026_03_31_05_0.db3', '2026_03_31_05'),
    ('2026_03_26_07/2026_03_26_07_0.db3', '2026_03_26_07'),
]

for db_rel, label in bags:
    ts, vx, vy, vz, wx, wy, wz = load_velocity(db_rel)
    out_file = f'{label}_velocity_lpf_2_5hz.png'
    plot_bag_velocity(ts, vx, vy, vz, wx, wy, wz, label, out_file)
