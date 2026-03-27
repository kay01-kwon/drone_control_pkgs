#!/usr/bin/env python3
"""EKF2 vs Mocap delay analysis using normalized cross-correlation.

Analyzes: position (px,py,pz), linear velocity (vx,vy,vz),
          attitude (roll,pitch,yaw), angular velocity (wx,wy,wz)

Usage: python3 ekf2_mocap_delay_analysis.py
"""

import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import correlate


def parse_odom(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, 44)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 68)
    vx, vy, vz = struct.unpack_from('<3d', data, 388)
    wx, wy, wz = struct.unpack_from('<3d', data, 412)
    return sec + nsec * 1e-9, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz


def parse_pose_stamped(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, 28)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 52)
    return sec + nsec * 1e-9, px, py, pz, qx, qy, qz, qw


def analyze_delay(db_path, t_start=10, t_end=25):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name FROM topics')
    topics = {name: tid for tid, name in c.fetchall()}

    # EKF2
    tid = topics['/mavros/local_position/odom']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    ekf_t = []
    ekf_px, ekf_py, ekf_pz = [], [], []
    ekf_vx, ekf_vy, ekf_vz = [], [], []
    ekf_roll, ekf_pitch, ekf_yaw = [], [], []
    ekf_wx, ekf_wy, ekf_wz = [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = parse_odom(data)
        q = np.array([qx, qy, qz, qw])
        n = np.linalg.norm(q)
        if n < 1e-10:
            continue
        R = Rotation.from_quat(q / n)
        r, p, y = R.as_euler('xyz', degrees=True)
        v_world = R.as_matrix() @ np.array([vx, vy, vz])
        ekf_t.append(t)
        ekf_px.append(px); ekf_py.append(py); ekf_pz.append(pz)
        ekf_vx.append(v_world[0]); ekf_vy.append(v_world[1]); ekf_vz.append(v_world[2])
        ekf_roll.append(r); ekf_pitch.append(p); ekf_yaw.append(y)
        ekf_wx.append(wx); ekf_wy.append(wy); ekf_wz.append(wz)

    # Mocap
    tid = topics['/S550/pose']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    mocap_t = []
    mocap_px, mocap_py, mocap_pz = [], [], []
    mocap_roll, mocap_pitch, mocap_yaw = [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw = parse_pose_stamped(data)
        q = np.array([qx, qy, qz, qw])
        n = np.linalg.norm(q)
        if n < 1e-10:
            continue
        r, p, y = Rotation.from_quat(q / n).as_euler('xyz', degrees=True)
        mocap_t.append(t)
        mocap_px.append(px); mocap_py.append(py); mocap_pz.append(pz)
        mocap_roll.append(r); mocap_pitch.append(p); mocap_yaw.append(y)
    conn.close()

    ekf_t = np.array(ekf_t)
    mocap_t = np.array(mocap_t)
    dt_ekf = np.median(np.diff(ekf_t))
    dt_mocap = np.median(np.diff(mocap_t))
    t0 = ekf_t[0]

    # Mocap velocity via numerical differentiation (central difference)
    mocap_px = np.array(mocap_px); mocap_py = np.array(mocap_py); mocap_pz = np.array(mocap_pz)
    mocap_roll = np.array(mocap_roll); mocap_pitch = np.array(mocap_pitch); mocap_yaw = np.array(mocap_yaw)

    mocap_vx = np.gradient(mocap_px, mocap_t)
    mocap_vy = np.gradient(mocap_py, mocap_t)
    mocap_vz = np.gradient(mocap_pz, mocap_t)
    mocap_wx = np.deg2rad(np.gradient(mocap_roll, mocap_t))
    mocap_wy = np.deg2rad(np.gradient(mocap_pitch, mocap_t))
    mocap_wz = np.deg2rad(np.gradient(mocap_yaw, mocap_t))

    # Analysis window
    mask = ((ekf_t - t0) > t_start) & ((ekf_t - t0) < t_end)
    N = mask.sum()
    print(f'EKF2 rate: {1/dt_ekf:.1f} Hz, Mocap rate: {1/dt_mocap:.1f} Hz')
    print(f'Analysis window: {t_start}-{t_end}s, N={N} samples\n')

    # ── Signals to analyze ──
    signals = [
        ('px', np.array(ekf_px), np.interp(ekf_t, mocap_t, mocap_px)),
        ('py', np.array(ekf_py), np.interp(ekf_t, mocap_t, mocap_py)),
        ('pz', np.array(ekf_pz), np.interp(ekf_t, mocap_t, mocap_pz)),
        ('vx', np.array(ekf_vx), np.interp(ekf_t, mocap_t, mocap_vx)),
        ('vy', np.array(ekf_vy), np.interp(ekf_t, mocap_t, mocap_vy)),
        ('vz', np.array(ekf_vz), np.interp(ekf_t, mocap_t, mocap_vz)),
        ('Roll', np.array(ekf_roll), np.interp(ekf_t, mocap_t, mocap_roll)),
        ('Pitch', np.array(ekf_pitch), np.interp(ekf_t, mocap_t, mocap_pitch)),
        ('Yaw', np.array(ekf_yaw), np.interp(ekf_t, mocap_t, mocap_yaw)),
        ('wx', np.array(ekf_wx), np.interp(ekf_t, mocap_t, mocap_wx)),
        ('wy', np.array(ekf_wy), np.interp(ekf_t, mocap_t, mocap_wy)),
        ('wz', np.array(ekf_wz), np.interp(ekf_t, mocap_t, mocap_wz)),
    ]

    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 2, figsize=(16, 3 * n_signals))

    for row, (name, ekf_sig, mocap_sig) in enumerate(signals):
        e = ekf_sig[mask] - np.mean(ekf_sig[mask])
        m = mocap_sig[mask] - np.mean(mocap_sig[mask])

        std_e = np.std(e)
        std_m = np.std(m)
        if std_e < 1e-10 or std_m < 1e-10:
            print(f'{name:6s}: signal too flat, skipping')
            continue

        # Normalize
        e_norm = e / std_e
        m_norm = m / std_m

        # Cross-correlation: R(tau) = (1/N) * sum_n e_norm(n) * m_norm(n + tau)
        corr = correlate(e_norm, m_norm, mode='full') / N
        lags = np.arange(-len(m_norm) + 1, len(e_norm))
        lag_ms = lags * dt_ekf * 1000

        # Peak
        peak_idx = np.argmax(corr)
        peak_lag = lags[peak_idx]
        peak_ms = peak_lag * dt_ekf * 1000
        peak_corr = corr[peak_idx]

        # Parabolic interpolation for sub-sample precision
        if 0 < peak_idx < len(corr) - 1:
            y_m1 = corr[peak_idx - 1]
            y_0 = corr[peak_idx]
            y_p1 = corr[peak_idx + 1]
            denom = y_m1 - 2 * y_0 + y_p1
            if abs(denom) > 1e-12:
                delta = 0.5 * (y_m1 - y_p1) / denom
            else:
                delta = 0
            refined_ms = (peak_lag + delta) * dt_ekf * 1000
        else:
            refined_ms = peak_ms

        print(f'{name:6s}: lag={peak_lag:+3d} samples = {peak_ms:+7.1f} ms  '
              f'(refined={refined_ms:+7.2f} ms)  corr={peak_corr:.4f}')

        # Left: cross-correlation
        ax = axes[row, 0]
        zoom_corr = np.abs(lag_ms) < 300
        ax.plot(lag_ms[zoom_corr], corr[zoom_corr], 'tab:blue', lw=1.0)
        ax.axvline(peak_ms, color='r', ls='--', lw=0.8, label=f'peak={peak_ms:.0f}ms')
        ax.axvline(refined_ms, color='g', ls='--', lw=0.8, label=f'refined={refined_ms:.1f}ms')
        ax.axvline(0, color='k', ls='-', lw=0.5, alpha=0.5)
        ax.set_ylabel('Corr')
        ax.set_title(f'{name}: Cross-correlation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: zoomed time series
        ax = axes[row, 1]
        t_rel = (ekf_t - t0)[mask]
        zoom = (t_rel > 15) & (t_rel < 17)
        ax.plot(t_rel[zoom], ekf_sig[mask][zoom], 'tab:blue', lw=1.2, marker='.', ms=2, label='EKF2')
        ax.plot(t_rel[zoom], mocap_sig[mask][zoom], 'tab:red', lw=1.2, marker='.', ms=2, label='Mocap')
        ax.set_title(f'{name}: zoomed 15-17s')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Lag (ms), negative = EKF2 lags mocap')
    axes[-1, 1].set_xlabel('Time (s)')
    plt.tight_layout()

    out = db_path.rsplit('/', 2)[0] + '/ekf2_mocap_delay_all.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    bag = '2026_03_26_07'
    db = f'/home/user/drone_control_pkgs/bag_folder/{bag}/{bag}_0.db3'
    analyze_delay(db)
