#!/usr/bin/env python3
"""Analyze ball-joint attitude test — QArray=[70,70,2.8, 0.1,0.1,0.02], R=0.1, rotor_max=8000."""

import sqlite3, struct, os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

DB = os.path.join(os.path.dirname(__file__), '2026_04_29_att_01_0.db3')

MASS = 3.146
G = 9.81

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
    off += 9 * 8  # orientation covariance
    wx, wy, wz = struct.unpack_from('<3d', blob, off); off += 24
    off += 9 * 8  # angular velocity covariance
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

print("Loading bag data...")
db = sqlite3.connect(DB)
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
ang_vel = imu[:, 0:3]  # wx, wy, wz from IMU

# Detect active period (motors running based on cmd RPM)
avg_cmd = cmd_rpm.mean(axis=1)
active_mask = avg_cmd > 2500
if np.any(active_mask):
    s = np.argmax(active_mask)
    e = len(active_mask) - 1 - np.argmax(active_mask[::-1])
    t_start, t_end = cmd_t[s], cmd_t[e]
    print(f"Active: {t_start:.1f}s - {t_end:.1f}s (dur={t_end-t_start:.1f}s)")
else:
    t_start, t_end = 0, odom_t[-1]

# Hover stats during active period
odom_active = (odom_t >= t_start) & (odom_t <= t_end)
if np.any(odom_active):
    rpy_a = rpy[odom_active]
    print(f"\n=== Attitude Stats (active) ===")
    for i, a in enumerate(['roll', 'pitch', 'yaw']):
        print(f"  {a}: mean={rpy_a[:,i].mean():.2f}° σ={rpy_a[:,i].std():.2f}° range=[{rpy_a[:,i].min():.2f}, {rpy_a[:,i].max():.2f}]")

# Thrust info
if thrust_data is not None and len(thrust_data) > 0:
    print(f"\n=== Thrust Commands ===")
    for i, (t, v) in enumerate(zip(thrust_t, thrust_data)):
        print(f"  t={t:.1f}s: {v:.1f} N")

# Motor saturation
air_cmd_mask = (cmd_t >= t_start) & (cmd_t <= t_end)
if np.any(air_cmd_mask):
    sat_pct = [(cmd_rpm[air_cmd_mask, i] > 7900).mean() * 100 for i in range(6)]
    print(f"\nMotor saturation (>7900): [{', '.join(f'{s:.0f}' for s in sat_pct)}]%")
    print(f"Mean RPMs: [{', '.join(f'{cmd_rpm[air_cmd_mask, i].mean():.0f}' for i in range(6))}]")

# DOB stats
air_dob_mask = (dob_t >= t_start) & (dob_t <= t_end)
if np.any(air_dob_mask):
    dob_a = dob[air_dob_mask]
    print(f"\nDOB torque [Nm] (active):")
    for i, a in enumerate(['tx', 'ty', 'tz']):
        print(f"  {a}: mean={dob_a[:,3+i].mean():.4f} σ={dob_a[:,3+i].std():.4f}")

OUT = os.path.dirname(__file__)
TITLE = 'Ball Joint — Q=[70,70,2.8], R=0.1'

# ─── Plot 1: Attitude ────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
rpy_labels = ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]']
for i, ax in enumerate(axes):
    ax.plot(odom_t, rpy[:, i], 'b-', linewidth=0.8)
    ax.set_ylabel(rpy_labels[i]); ax.grid(True, alpha=0.3)
    ax.axvspan(t_start, t_end, alpha=0.1, color='green')
    if np.any(odom_active):
        ax.axhline(rpy_a[:, i].mean(), color='r', linestyle='--', alpha=0.5,
                   label=f'mean={rpy_a[:,i].mean():.2f}, σ={rpy_a[:,i].std():.2f}')
        ax.legend(loc='upper right')
axes[0].set_title(f'Attitude — {TITLE}')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'attitude.png'), dpi=150); print("Saved attitude.png")

# ─── Plot 2: Control Wrench (moments) & Thrust ───────────
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
# Thrust
axes[0].plot(ctrl_t, ctrl[:, 2], 'b-', linewidth=0.8, label='f_z (ctrl)')
if thrust_data is not None and len(thrust_data) > 0:
    axes[0].plot(thrust_t, thrust_data, 'r-', linewidth=2, marker='o', markersize=4, label='des_thrust')
axes[0].axhline(MASS * G, color='gray', linestyle=':', alpha=0.5, label=f'W={MASS*G:.1f}N')
axes[0].set_ylabel('Thrust [N]'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_title(f'Control Wrench — {TITLE}')
axes[0].axvspan(t_start, t_end, alpha=0.1, color='green')
# Moments
for i, (label, c) in enumerate([('Mx', 'C0'), ('My', 'C1'), ('Mz', 'C2')]):
    axes[1].plot(ctrl_t, ctrl[:, 3+i], color=c, linewidth=0.8, label=label)
axes[1].set_ylabel('Moment [Nm]'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].axvspan(t_start, t_end, alpha=0.1, color='green')
# DOB torque
for i, (label, c) in enumerate([('DOB tx', 'C0'), ('DOB ty', 'C1'), ('DOB tz', 'C2')]):
    axes[2].plot(dob_t, dob[:, 3+i], color=c, linewidth=0.8, label=label)
axes[2].set_ylabel('DOB Torque [Nm]'); axes[2].set_xlabel('Time [s]')
axes[2].legend(); axes[2].grid(True, alpha=0.3)
axes[2].axvspan(t_start, t_end, alpha=0.1, color='green')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'control_wrench.png'), dpi=150); print("Saved control_wrench.png")

# ─── Plot 3: Motor RPMs ──────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
for i in range(6):
    axes[0].plot(cmd_t, cmd_rpm[:, i], color=colors[i], linewidth=0.5, alpha=0.7, label=f'M{i+1}')
axes[0].set_ylabel('Cmd RPM'); axes[0].set_title(f'Motor RPMs — {TITLE}')
axes[0].legend(ncol=6, fontsize=8); axes[0].grid(True, alpha=0.3)
axes[0].axhline(8000, color='r', linestyle=':', alpha=0.5, label='limit')
axes[0].axvspan(t_start, t_end, alpha=0.1, color='green')
for i in range(6):
    axes[1].plot(arpm_t, actual_rpm[:, i], color=colors[i], linewidth=0.5, alpha=0.7, label=f'M{i+1}')
axes[1].set_ylabel('Actual RPM'); axes[1].set_xlabel('Time [s]')
axes[1].legend(ncol=6, fontsize=8); axes[1].grid(True, alpha=0.3)
axes[1].axvspan(t_start, t_end, alpha=0.1, color='green')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'motor_rpm.png'), dpi=150); print("Saved motor_rpm.png")

# ─── Plot 4: Per-motor cmd vs actual ─────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
for i in range(6):
    r, c = i // 2, i % 2
    ax = axes[r, c]
    ax.plot(cmd_t, cmd_rpm[:, i], 'b-', linewidth=0.5, alpha=0.7, label='cmd')
    ax.plot(arpm_t, actual_rpm[:, i], 'r-', linewidth=0.5, alpha=0.7, label='actual')
    ax.set_ylabel('RPM'); ax.set_title(f'Motor {i+1}')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
    ax.axhline(8000, color='gray', linestyle=':', alpha=0.3)
    ax.axvspan(t_start, t_end, alpha=0.08, color='green')
axes[2, 0].set_xlabel('Time [s]'); axes[2, 1].set_xlabel('Time [s]')
fig.suptitle(f'Cmd vs Actual RPM — {TITLE}', fontsize=13)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'motor_cmd_vs_actual.png'), dpi=150); print("Saved motor_cmd_vs_actual.png")

# ─── Plot 5: IMU Angular Velocity PSD (vibration analysis) ──
imu_active = (imu_t >= t_start) & (imu_t <= t_end)
if np.any(imu_active):
    t_imu_a = imu_t[imu_active]; ang_a = ang_vel[imu_active]
    dt_imu = np.median(np.diff(t_imu_a)); fs_imu = 1.0 / dt_imu
    print(f"\nIMU sample rate: {fs_imu:.1f} Hz")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for j, name in enumerate(['wx (roll rate)', 'wy (pitch rate)', 'wz (yaw rate)']):
        sig = ang_a[:, j] - ang_a[:, j].mean()
        nperseg = min(512, len(sig) // 2)
        if nperseg > 16:
            f, Pxx = signal.welch(sig, fs=fs_imu, nperseg=nperseg)
            axes[j].semilogy(f, Pxx, 'b-', linewidth=0.8)
            pk_mask = f > 1
            if np.any(pk_mask):
                pk_idx = np.argmax(Pxx[pk_mask])
                pk_f = f[pk_mask][pk_idx]
                axes[j].axvline(pk_f, color='r', linestyle='--', alpha=0.5, label=f'peak={pk_f:.1f}Hz')
            axes[j].set_xlabel('Frequency [Hz]'); axes[j].set_ylabel('PSD [(rad/s)²/Hz]')
            axes[j].set_title(name); axes[j].legend(fontsize=8); axes[j].grid(True, alpha=0.3)
            axes[j].set_xlim([0, min(fs_imu/2, 100)])
    fig.suptitle(f'IMU Angular Velocity PSD — {TITLE}', fontsize=13)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'imu_vibration.png'), dpi=150); print("Saved imu_vibration.png")

# ─── Plot 6: Angular velocity time domain ────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
w_labels = ['ωx [rad/s]', 'ωy [rad/s]', 'ωz [rad/s]']
for i, ax in enumerate(axes):
    ax.plot(imu_t, ang_vel[:, i], 'b-', linewidth=0.5)
    ax.set_ylabel(w_labels[i]); ax.grid(True, alpha=0.3)
    ax.axvspan(t_start, t_end, alpha=0.1, color='green')
    if np.any(imu_active):
        ax.axhline(ang_a[:, i].mean(), color='r', linestyle='--', alpha=0.5,
                   label=f'mean={ang_a[:,i].mean():.3f}, σ={ang_a[:,i].std():.3f}')
        ax.legend(loc='upper right')
axes[0].set_title(f'Angular Velocity — {TITLE}')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'angular_velocity.png'), dpi=150); print("Saved angular_velocity.png")

plt.close('all')
print("\nDone!")
