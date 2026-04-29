#!/usr/bin/env python3
"""Analyze ball-joint attitude tests att_01 and att_02 (2026-04-30)."""

import sqlite3, struct, os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

BASE = os.path.dirname(__file__)

def parse_odom(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    if off % 4: off += 4 - off % 4
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    if off % 4: off += 4 - off % 4
    off += 4
    px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
    off += 36 * 8
    vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
    wx, wy, wz = struct.unpack_from('<3d', blob, off)
    return px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz

def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    if off % 4: off += 4 - off % 4
    fx, fy, fz = struct.unpack_from('<3d', blob, off); off += 24
    tx, ty, tz = struct.unpack_from('<3d', blob, off)
    return fx, fy, fz, tx, ty, tz

def parse_cmd_raw(blob):
    vals = struct.unpack_from('<6H', blob, 18)
    return np.array(vals, dtype=float) * (9800.0 / 8191.0)

def parse_actual_rpm(blob):
    vals = struct.unpack_from('<6i', blob, 20)
    return np.array(vals, dtype=float)

def parse_imu(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    if off % 4: off += 4 - off % 4
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
    off += 9 * 8
    wx, wy, wz = struct.unpack_from('<3d', blob, off); off += 24
    off += 9 * 8
    ax, ay, az = struct.unpack_from('<3d', blob, off)
    return wx, wy, wz, ax, ay, az

def parse_float64(blob):
    return struct.unpack_from('<d', blob, 4)[0]

def quat_to_rpy(qw, qx, qy, qz):
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (qw * qy - qz * qx); sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny, cosy)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def load_topic(db, topic_name, parser):
    cur = db.execute("SELECT m.timestamp, m.data FROM messages m JOIN topics t ON m.topic_id=t.id WHERE t.name=? ORDER BY m.timestamp", (topic_name,))
    times, data = [], []
    for ts, blob in cur.fetchall():
        try:
            d = parser(blob); times.append(ts * 1e-9); data.append(d)
        except: pass
    return np.array(times), data

def analyze_bag(db_path, label, out_dir):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    db = sqlite3.connect(db_path)
    odom_ts, odom_raw = load_topic(db, '/mavros/local_position/odom', parse_odom); odom = np.array(odom_raw)
    ctrl_ts, ctrl_raw = load_topic(db, '/nmpc/control', parse_wrench); ctrl = np.array(ctrl_raw)
    dob_ts, dob_raw = load_topic(db, '/hgdo/wrench', parse_wrench); dob = np.array(dob_raw)
    cmd_ts, cmd_raw = load_topic(db, '/uav/cmd_raw', parse_cmd_raw); cmd_rpm = np.array(cmd_raw)
    arpm_ts, arpm_raw = load_topic(db, '/uav/actual_rpm', parse_actual_rpm); actual_rpm = np.array(arpm_raw)
    imu_ts, imu_raw = load_topic(db, '/mavros/imu/data', parse_imu); imu = np.array(imu_raw)
    thrust_ts, thrust_raw = load_topic(db, '/des_thrust', parse_float64)
    thrust_data = np.array(thrust_raw) if thrust_raw else None
    db.close()

    t0 = odom_ts[0]
    odom_t = odom_ts - t0; ctrl_t = ctrl_ts - t0; dob_t = dob_ts - t0
    cmd_t = cmd_ts - t0; arpm_t = arpm_ts - t0; imu_t = imu_ts - t0
    thrust_t = thrust_ts - t0 if thrust_ts is not None and len(thrust_ts) > 0 else None

    rpy = np.array([quat_to_rpy(r[6], r[7], r[8], r[9]) for r in odom])
    ang_vel = imu[:, 0:3]

    avg_cmd = cmd_rpm.mean(axis=1)
    active_mask = avg_cmd > 2500
    if np.any(active_mask):
        s = np.argmax(active_mask)
        e = len(active_mask) - 1 - np.argmax(active_mask[::-1])
        t_start, t_end = cmd_t[s], cmd_t[e]
        print(f"Active: {t_start:.1f}s - {t_end:.1f}s (dur={t_end-t_start:.1f}s)")
    else:
        t_start, t_end = 0, odom_t[-1]

    odom_active = (odom_t >= t_start) & (odom_t <= t_end)
    if np.any(odom_active):
        rpy_a = rpy[odom_active]
        print(f"\nAttitude Stats (active):")
        for i, a in enumerate(['roll', 'pitch', 'yaw']):
            print(f"  {a}: mean={rpy_a[:,i].mean():.2f}° σ={rpy_a[:,i].std():.2f}° range=[{rpy_a[:,i].min():.1f}, {rpy_a[:,i].max():.1f}]")

    if thrust_data is not None and len(thrust_data) > 0:
        unique_thrust = np.unique(thrust_data)
        print(f"\nThrust levels: {unique_thrust}")

    air_cmd_mask = (cmd_t >= t_start) & (cmd_t <= t_end)
    if np.any(air_cmd_mask):
        sat_pct = [(cmd_rpm[air_cmd_mask, i] > 7200).mean() * 100 for i in range(6)]
        print(f"Motor saturation (>7200): [{', '.join(f'{s:.0f}' for s in sat_pct)}]%")
        print(f"Mean RPMs: [{', '.join(f'{cmd_rpm[air_cmd_mask, i].mean():.0f}' for i in range(6))}]")
        print(f"Max RPMs: [{', '.join(f'{cmd_rpm[air_cmd_mask, i].max():.0f}' for i in range(6))}]")
        print(f"Min RPMs: [{', '.join(f'{cmd_rpm[air_cmd_mask, i].min():.0f}' for i in range(6))}]")
        rpm_range = cmd_rpm[air_cmd_mask].max(axis=0) - cmd_rpm[air_cmd_mask].min(axis=0)
        print(f"RPM range: [{', '.join(f'{r:.0f}' for r in rpm_range)}]")

    # DOB stats
    air_dob_mask = (dob_t >= t_start) & (dob_t <= t_end)
    if np.any(air_dob_mask):
        dob_a = dob[air_dob_mask]
        print(f"\nDOB torque [Nm] (active):")
        for i, a in enumerate(['tx', 'ty', 'tz']):
            print(f"  {a}: mean={dob_a[:,3+i].mean():.4f} σ={dob_a[:,3+i].std():.4f}")

    # Ctrl moments stats
    air_ctrl_mask = (ctrl_t >= t_start) & (ctrl_t <= t_end)
    if np.any(air_ctrl_mask):
        ctrl_a = ctrl[air_ctrl_mask]
        print(f"\nControl moments [Nm] (active):")
        for i, a in enumerate(['Mx', 'My', 'Mz']):
            print(f"  {a}: mean={ctrl_a[:,3+i].mean():.4f} σ={ctrl_a[:,3+i].std():.4f} range=[{ctrl_a[:,3+i].min():.3f}, {ctrl_a[:,3+i].max():.3f}]")

    return {
        'odom_t': odom_t, 'rpy': rpy, 'ctrl_t': ctrl_t, 'ctrl': ctrl,
        'dob_t': dob_t, 'dob': dob, 'cmd_t': cmd_t, 'cmd_rpm': cmd_rpm,
        'arpm_t': arpm_t, 'actual_rpm': actual_rpm, 'imu_t': imu_t,
        'ang_vel': ang_vel, 'thrust_t': thrust_t, 'thrust_data': thrust_data,
        't_start': t_start, 't_end': t_end, 'label': label
    }

# Analyze both bags
db1 = os.path.join(BASE, '..', '2026_04_30_att_01', '2026_04_30_att_01_0.db3')
db2 = os.path.join(BASE, '..', '2026_04_30_att_02', '2026_04_30_att_02_0.db3')

r1 = analyze_bag(db1, 'att_01: Q_qz=0.2, Mz_clamp=0.15', BASE)
r2 = analyze_bag(db2, 'att_02: Q_qz=2.8, Mz_clamp=0.15', BASE)

# ─── Comparison Plots ─────────────────────────────────────
OUT = os.path.join(BASE, '..', '2026_04_30_att_02')

fig, axes = plt.subplots(3, 2, figsize=(18, 10), sharex='col')
for col, (r, title) in enumerate([(r1, 'att_01: Q_qz=0.2'), (r2, 'att_02: Q_qz=2.8')]):
    rpy_labels = ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]']
    for i, ax in enumerate(axes[:, col]):
        ax.plot(r['odom_t'], r['rpy'][:, i], 'b-', linewidth=0.6)
        ax.set_ylabel(rpy_labels[i]); ax.grid(True, alpha=0.3)
        ax.axvspan(r['t_start'], r['t_end'], alpha=0.08, color='green')
        odom_active = (r['odom_t'] >= r['t_start']) & (r['odom_t'] <= r['t_end'])
        if np.any(odom_active):
            ra = r['rpy'][odom_active, i]
            ax.set_title(f'{title} — {rpy_labels[i].split()[0]} σ={ra.std():.2f}°') if i == 0 else None
            ax.axhline(ra.mean(), color='r', linestyle='--', alpha=0.4,
                       label=f'mean={ra.mean():.1f}, σ={ra.std():.2f}')
            ax.legend(loc='upper right', fontsize=8)
    axes[0, col].set_title(f'{title}')
    axes[-1, col].set_xlabel('Time [s]')
fig.suptitle('Attitude Comparison — 2026-04-30 Ball Joint Tests', fontsize=14)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'attitude_comparison.png'), dpi=150)
print("\nSaved attitude_comparison.png")

# Motor RPM comparison
fig, axes = plt.subplots(2, 2, figsize=(18, 8), sharex='col')
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
for col, (r, title) in enumerate([(r1, 'att_01: Q_qz=0.2'), (r2, 'att_02: Q_qz=2.8')]):
    for i in range(6):
        axes[0, col].plot(r['cmd_t'], r['cmd_rpm'][:, i], color=colors[i], linewidth=0.4, alpha=0.7, label=f'M{i+1}')
    axes[0, col].set_ylabel('Cmd RPM'); axes[0, col].set_title(f'{title} — Cmd RPM')
    axes[0, col].axhline(7300, color='r', linestyle=':', alpha=0.5)
    axes[0, col].legend(ncol=6, fontsize=7); axes[0, col].grid(True, alpha=0.3)
    axes[0, col].axvspan(r['t_start'], r['t_end'], alpha=0.08, color='green')
    for i in range(6):
        axes[1, col].plot(r['arpm_t'], r['actual_rpm'][:, i], color=colors[i], linewidth=0.4, alpha=0.7, label=f'M{i+1}')
    axes[1, col].set_ylabel('Actual RPM'); axes[1, col].set_xlabel('Time [s]')
    axes[1, col].legend(ncol=6, fontsize=7); axes[1, col].grid(True, alpha=0.3)
    axes[1, col].axvspan(r['t_start'], r['t_end'], alpha=0.08, color='green')
fig.suptitle('Motor RPM Comparison — 2026-04-30', fontsize=14)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'motor_comparison.png'), dpi=150)
print("Saved motor_comparison.png")

# Control moments comparison
fig, axes = plt.subplots(3, 2, figsize=(18, 10), sharex='col')
for col, (r, title) in enumerate([(r1, 'att_01: Q_qz=0.2'), (r2, 'att_02: Q_qz=2.8')]):
    moment_labels = ['Mx [Nm]', 'My [Nm]', 'Mz [Nm]']
    for i, ax in enumerate(axes[:, col]):
        ax.plot(r['ctrl_t'], r['ctrl'][:, 3+i], 'b-', linewidth=0.6)
        ax.set_ylabel(moment_labels[i]); ax.grid(True, alpha=0.3)
        ax.axvspan(r['t_start'], r['t_end'], alpha=0.08, color='green')
        if i == 2:
            ax.axhline(0.15, color='r', linestyle='--', alpha=0.5, label='clamp ±0.15')
            ax.axhline(-0.15, color='r', linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)
    axes[0, col].set_title(f'{title}')
    axes[-1, col].set_xlabel('Time [s]')
fig.suptitle('Control Moments — 2026-04-30', fontsize=14)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'moments_comparison.png'), dpi=150)
print("Saved moments_comparison.png")

# IMU angular velocity PSD comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for row, (r, title) in enumerate([(r1, 'att_01: Q_qz=0.2'), (r2, 'att_02: Q_qz=2.8')]):
    imu_active = (r['imu_t'] >= r['t_start']) & (r['imu_t'] <= r['t_end'])
    if np.any(imu_active):
        ang_a = r['ang_vel'][imu_active]; t_imu_a = r['imu_t'][imu_active]
        dt = np.median(np.diff(t_imu_a)); fs = 1.0/dt
        for j, name in enumerate(['ωx (roll)', 'ωy (pitch)', 'ωz (yaw)']):
            sig = ang_a[:, j] - ang_a[:, j].mean()
            nperseg = min(512, len(sig)//2)
            if nperseg > 16:
                f, Pxx = signal.welch(sig, fs=fs, nperseg=nperseg)
                axes[row, j].semilogy(f, Pxx, 'b-', linewidth=0.8)
                axes[row, j].set_xlabel('Freq [Hz]'); axes[row, j].set_ylabel('PSD')
                axes[row, j].set_title(f'{title} — {name}')
                axes[row, j].grid(True, alpha=0.3); axes[row, j].set_xlim([0, 50])
fig.suptitle('IMU Angular Velocity PSD — 2026-04-30', fontsize=14)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'imu_psd_comparison.png'), dpi=150)
print("Saved imu_psd_comparison.png")

plt.close('all')
print("\nDone!")
