#!/usr/bin/env python3
"""Analyze free flight hover data — Kd tuning experiment (2026_04_28_02_Kd)."""

import sqlite3, struct, os
import numpy as np
import matplotlib.pyplot as plt

DB = os.path.join(os.path.dirname(__file__), '2026_04_28_02_Kd_0.db3')

# ─── Parsers ──────────────────────────────────────────────

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

def parse_ref(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    if off % 4: off += 4 - off % 4
    px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
    vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
    psi = struct.unpack_from('<d', blob, off)[0]; off += 8
    psi_dot = struct.unpack_from('<d', blob, off)[0]
    return px, py, pz, vx, vy, vz, psi, psi_dot

def quat_to_rpy(qw, qx, qy, qz):
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny, cosy)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

# ─── Load data ────────────────────────────────────────────

def load_topic(db, topic_name, parser):
    cur = db.execute(
        "SELECT m.timestamp, m.data FROM messages m "
        "JOIN topics t ON m.topic_id=t.id "
        "WHERE t.name=? ORDER BY m.timestamp", (topic_name,))
    rows = cur.fetchall()
    times, data = [], []
    for ts, blob in rows:
        try:
            d = parser(blob)
            times.append(ts * 1e-9)
            data.append(d)
        except Exception:
            pass
    return np.array(times), data

print("Loading bag data...")
db = sqlite3.connect(DB)

odom_ts, odom_raw = load_topic(db, '/mavros/local_position/odom', parse_odom)
odom = np.array(odom_raw)

ref_ts, ref_raw = load_topic(db, '/nmpc/ref', parse_ref)
ref_data = np.array(ref_raw) if len(ref_raw) > 0 else None

ctrl_ts, ctrl_raw = load_topic(db, '/nmpc/control', parse_wrench)
ctrl = np.array(ctrl_raw)

dob_ts, dob_raw = load_topic(db, '/hgdo/wrench', parse_wrench)
dob = np.array(dob_raw)

cmd_ts, cmd_raw = load_topic(db, '/uav/cmd_raw', parse_cmd_raw)
cmd_rpm = np.array(cmd_raw)

arpm_ts, arpm_raw = load_topic(db, '/uav/actual_rpm', parse_actual_rpm)
actual_rpm = np.array(arpm_raw)

db.close()

# ─── Time offset ──────────────────────────────────────────
t0 = odom_ts[0]
odom_t = odom_ts - t0
ref_t = ref_ts - t0 if ref_ts is not None and len(ref_ts) > 0 else None
ctrl_t = ctrl_ts - t0
dob_t = dob_ts - t0
cmd_t = cmd_ts - t0
arpm_t = arpm_ts - t0

# ─── Position offset ─────────────────────────────────────
px0, py0, pz0 = odom[0, 0], odom[0, 1], odom[0, 2]
pos = odom[:, 0:3].copy()
pos[:, 0] -= px0
pos[:, 1] -= py0
pos[:, 2] -= pz0

# ─── RPY ──────────────────────────────────────────────────
rpy = np.array([quat_to_rpy(r[6], r[7], r[8], r[9]) for r in odom])
vel = odom[:, 3:6]

# ─── Detect airborne phase ───────────────────────────────
z_threshold = 0.05
airborne_mask = pos[:, 2] > z_threshold
if np.any(airborne_mask):
    airborne_start_idx = np.argmax(airborne_mask)
    airborne_end_idx = len(airborne_mask) - 1 - np.argmax(airborne_mask[::-1])
    t_start = odom_t[airborne_start_idx]
    t_end = odom_t[airborne_end_idx]
    print(f"Airborne phase: {t_start:.1f}s - {t_end:.1f}s (duration: {t_end-t_start:.1f}s)")
    print(f"Max altitude: {pos[:, 2].max():.3f} m")
else:
    t_start, t_end = 0, odom_t[-1]
    print("No airborne phase detected!")

# ─── Hover statistics ────────────────────────────────────
hover_mask = airborne_mask
if np.any(hover_mask):
    p_hover = pos[hover_mask]
    v_hover = vel[hover_mask]
    rpy_hover = rpy[hover_mask]

    print(f"\n=== Hover Statistics ===")
    print(f"Position [m]:")
    for i, ax in enumerate(['x', 'y', 'z']):
        print(f"  {ax}: mean={p_hover[:, i].mean():.4f}, "
              f"std={p_hover[:, i].std():.4f}, "
              f"range=[{p_hover[:, i].min():.4f}, {p_hover[:, i].max():.4f}]")
    print(f"Velocity [m/s]:")
    for i, ax in enumerate(['vx', 'vy', 'vz']):
        print(f"  {ax}: mean={v_hover[:, i].mean():.4f}, "
              f"std={v_hover[:, i].std():.4f}")
    print(f"Attitude [deg]:")
    for i, ax in enumerate(['roll', 'pitch', 'yaw']):
        print(f"  {ax}: mean={rpy_hover[:, i].mean():.2f}, "
              f"std={rpy_hover[:, i].std():.2f}")

OUT = os.path.dirname(__file__)

# ─── Plot 1: Position ────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
labels = ['x [m]', 'y [m]', 'z [m]']
for i, ax in enumerate(axes):
    ax.plot(odom_t, pos[:, i], 'b-', linewidth=0.8, label='actual')
    if ref_data is not None and len(ref_data) > 0:
        ax.plot(ref_t, ref_data[:, i], 'r--', linewidth=1.5, label='ref', marker='o', markersize=3)
    ax.set_ylabel(labels[i])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if np.any(airborne_mask):
        ax.axvspan(t_start, t_end, alpha=0.1, color='green')
axes[0].set_title('Position (offset from start) — Kd tuning')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'position.png'), dpi=150)
print("Saved position.png")

# ─── Plot 2: RPY ─────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
rpy_labels = ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]']
for i, ax in enumerate(axes):
    ax.plot(odom_t, rpy[:, i], 'b-', linewidth=0.8)
    ax.set_ylabel(rpy_labels[i])
    ax.grid(True, alpha=0.3)
    if np.any(airborne_mask):
        ax.axvspan(t_start, t_end, alpha=0.1, color='green')
    if np.any(hover_mask):
        ax.axhline(rpy_hover[:, i].mean(), color='r', linestyle='--', alpha=0.5,
                    label=f'mean={rpy_hover[:, i].mean():.2f}, σ={rpy_hover[:, i].std():.2f}')
        ax.legend(loc='upper right')
axes[0].set_title('Attitude (RPY) — Kd tuning')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'attitude.png'), dpi=150)
print("Saved attitude.png")

# ─── Plot 3: Motor RPMs ──────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
for i in range(6):
    axes[0].plot(cmd_t, cmd_rpm[:, i], color=colors[i], linewidth=0.5, alpha=0.7, label=f'M{i+1}')
axes[0].set_ylabel('Cmd RPM')
axes[0].set_title('Motor Commands — Kd tuning')
axes[0].legend(ncol=6, fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(7300, color='r', linestyle=':', alpha=0.5, label='max')

for i in range(6):
    axes[1].plot(arpm_t, actual_rpm[:, i], color=colors[i], linewidth=0.5, alpha=0.7, label=f'M{i+1}')
axes[1].set_ylabel('Actual RPM')
axes[1].set_xlabel('Time [s]')
axes[1].legend(ncol=6, fontsize=8)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'motor_rpm.png'), dpi=150)
print("Saved motor_rpm.png")

# ─── Plot 4: Per-motor cmd vs actual ─────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
for i in range(6):
    r, c = i // 2, i % 2
    ax = axes[r, c]
    ax.plot(cmd_t, cmd_rpm[:, i], 'b-', linewidth=0.5, alpha=0.7, label='cmd')
    ax.plot(arpm_t, actual_rpm[:, i], 'r-', linewidth=0.5, alpha=0.7, label='actual')
    ax.set_ylabel('RPM')
    ax.set_title(f'Motor {i+1}')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(7300, color='gray', linestyle=':', alpha=0.3)
    if np.any(airborne_mask):
        ax.axvspan(t_start, t_end, alpha=0.08, color='green')
axes[2, 0].set_xlabel('Time [s]')
axes[2, 1].set_xlabel('Time [s]')
fig.suptitle('Cmd vs Actual RPM per Motor — Kd tuning', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'motor_cmd_vs_actual.png'), dpi=150)
print("Saved motor_cmd_vs_actual.png")

# ─── Plot 5: Control wrench & DOB wrench ─────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

axes[0].plot(ctrl_t, ctrl[:, 2], 'b-', linewidth=0.8, label='f_col (NMPC)')
axes[0].axhline(3.146 * 9.81, color='r', linestyle='--', alpha=0.5, label='W=30.9N')
axes[0].set_ylabel('Force z [N]')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Control & DOB Wrench — Kd tuning')

for i, (label, c) in enumerate([('Mx', 'C0'), ('My', 'C1'), ('Mz', 'C2')]):
    axes[1].plot(ctrl_t, ctrl[:, 3+i], color=c, linewidth=0.8, label=label)
axes[1].set_ylabel('Moment [Nm]')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

for i, (label, c) in enumerate([('fx', 'C0'), ('fy', 'C1'), ('fz', 'C2')]):
    axes[2].plot(dob_t, dob[:, i], color=c, linewidth=0.8, label=f'DOB {label}')
axes[2].set_ylabel('DOB Force [N]')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

for i, (label, c) in enumerate([('tx', 'C0'), ('ty', 'C1'), ('tz', 'C2')]):
    axes[3].plot(dob_t, dob[:, 3+i], color=c, linewidth=0.8, label=f'DOB {label}')
axes[3].set_ylabel('DOB Torque [Nm]')
axes[3].set_xlabel('Time [s]')
axes[3].legend()
axes[3].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'control_dob.png'), dpi=150)
print("Saved control_dob.png")

# ─── Plot 6: XY trajectory (top view) ────────────────────
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
if np.any(airborne_mask):
    ax.plot(pos[airborne_mask, 0], pos[airborne_mask, 1], 'b-', linewidth=0.8, label='trajectory')
    ax.plot(pos[airborne_mask, 0][0], pos[airborne_mask, 1][0], 'go', markersize=10, label='takeoff')
    ax.plot(pos[airborne_mask, 0][-1], pos[airborne_mask, 1][-1], 'rs', markersize=10, label='landing')
if ref_data is not None and len(ref_data) > 0:
    ax.plot(ref_data[:, 0], ref_data[:, 1], 'r+', markersize=15, label='ref')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('XY Trajectory (top view) — Kd tuning')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'xy_trajectory.png'), dpi=150)
print("Saved xy_trajectory.png")

# ─── Plot 7: Velocity ────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
vel_labels = ['vx [m/s]', 'vy [m/s]', 'vz [m/s]']
for i, ax in enumerate(axes):
    ax.plot(odom_t, vel[:, i], 'b-', linewidth=0.8)
    ax.set_ylabel(vel_labels[i])
    ax.grid(True, alpha=0.3)
    if np.any(airborne_mask):
        ax.axvspan(t_start, t_end, alpha=0.1, color='green')
axes[0].set_title('Velocity — Kd tuning')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'velocity.png'), dpi=150)
print("Saved velocity.png")

# ─── Plot 8: Frequency analysis (position error PSD) ─────
from scipy import signal

if np.any(airborne_mask):
    t_air = odom_t[airborne_mask]
    p_air = pos[airborne_mask]
    v_air = vel[airborne_mask]
    dt = np.median(np.diff(t_air))
    fs = 1.0 / dt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for i, (ax_name, data_arr, ylabel) in enumerate([
        (['x', 'y', 'z'], p_air, 'Position PSD [m²/Hz]'),
        (['vx', 'vy', 'vz'], v_air, 'Velocity PSD [(m/s)²/Hz]'),
    ]):
        for j in range(3):
            ax = axes[i, j]
            sig = data_arr[:, j] - data_arr[:, j].mean()
            nperseg = min(512, len(sig) // 2)
            if nperseg > 16:
                f, Pxx = signal.welch(sig, fs=fs, nperseg=nperseg)
                ax.semilogy(f, Pxx, 'b-', linewidth=0.8)
                peak_idx = np.argmax(Pxx[1:]) + 1
                ax.axvline(f[peak_idx], color='r', linestyle='--', alpha=0.5,
                           label=f'peak={f[peak_idx]:.2f}Hz')
                ax.set_xlabel('Frequency [Hz]')
                ax.set_ylabel(ylabel)
                ax.set_title(f'{ax_name[j]}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 5])
    fig.suptitle('Power Spectral Density (airborne) — Kd tuning', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'frequency.png'), dpi=150)
    print("Saved frequency.png")

plt.close('all')
print("\nDone!")
