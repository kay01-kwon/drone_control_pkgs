#!/usr/bin/env python3
"""Settling time analysis for ball-joint attitude test."""

import sqlite3, struct, os
import numpy as np
import matplotlib.pyplot as plt

DB = os.path.join(os.path.dirname(__file__), '2026_04_29_att_01_0.db3')

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

print("Loading...")
db = sqlite3.connect(DB)
odom_ts, odom_raw = load_topic(db, '/mavros/local_position/odom', parse_odom); odom = np.array(odom_raw)
ctrl_ts, ctrl_raw = load_topic(db, '/nmpc/control', parse_wrench); ctrl = np.array(ctrl_raw)
db.close()

t0 = odom_ts[0]
odom_t = odom_ts - t0
ctrl_t = ctrl_ts - t0
rpy = np.array([quat_to_rpy(r[6], r[7], r[8], r[9]) for r in odom])

# Find settling time: time from initial disturbance to stay within ±band of steady state
def find_settling_time(t, signal, t_start, t_settle_search_end, band_deg=2.0):
    """Find 2% settling time after t_start."""
    mask = (t >= t_start) & (t <= t_settle_search_end)
    t_seg = t[mask]; sig_seg = signal[mask]
    if len(sig_seg) < 10:
        return np.nan, np.nan

    # Steady state = mean of last 30% of segment
    n_ss = max(10, int(0.3 * len(sig_seg)))
    ss_val = sig_seg[-n_ss:].mean()

    # Find last time signal exceeds ±band from steady state
    outside = np.abs(sig_seg - ss_val) > band_deg
    if not np.any(outside):
        return 0.0, ss_val

    last_outside_idx = np.max(np.where(outside))
    t_settle = t_seg[last_outside_idx] - t_seg[0]
    return t_settle, ss_val

# Analyze initial transient (thrust applied around t=2s)
t_thrust_on = 2.0
t_search_end = 15.0

print(f"\n=== Settling Time Analysis (band=±2°) ===")
print(f"Thrust on at t≈{t_thrust_on:.1f}s, search window: {t_thrust_on}-{t_search_end}s\n")

for i, name in enumerate(['Roll', 'Pitch', 'Yaw']):
    ts, ss = find_settling_time(odom_t, rpy[:, i], t_thrust_on, t_search_end, band_deg=2.0)
    ts5, ss5 = find_settling_time(odom_t, rpy[:, i], t_thrust_on, t_search_end, band_deg=5.0)
    print(f"{name}:")
    print(f"  Steady state ≈ {ss:.2f}°")
    print(f"  ±2° settling time: {ts:.2f}s")
    print(f"  ±5° settling time: {ts5:.2f}s")

# Also compute from a narrower band (±1°)
print(f"\n=== Settling Time Analysis (band=±1°) ===")
for i, name in enumerate(['Roll', 'Pitch', 'Yaw']):
    ts, ss = find_settling_time(odom_t, rpy[:, i], t_thrust_on, t_search_end, band_deg=1.0)
    print(f"  {name}: {ts:.2f}s (ss={ss:.2f}°)")

# Analyze multiple segments for roll/pitch to get average settling
print(f"\n=== Cross-zero analysis (roll/pitch oscillation period) ===")
for i, name in enumerate(['Roll', 'Pitch']):
    mask = (odom_t >= 5.0) & (odom_t <= 40.0)
    sig = rpy[mask, i]; t_seg = odom_t[mask]
    sig_c = sig - sig.mean()

    # Find zero crossings
    crossings = np.where(np.diff(np.sign(sig_c)))[0]
    if len(crossings) > 2:
        periods = np.diff(t_seg[crossings])
        half_periods = periods[periods > 0.1]  # filter noise
        full_period = 2 * np.median(half_periods)
        freq = 1.0 / full_period
        omega = 2 * np.pi * freq
        print(f"  {name}: median half-period={np.median(half_periods):.3f}s, "
              f"full period≈{full_period:.3f}s, freq≈{freq:.2f}Hz, ω≈{omega:.1f} rad/s")

# ─── Plot: Settling time visualization ────────────────────
OUT = os.path.dirname(__file__)
TITLE = 'Ball Joint — Q=[70,70,2.8], R=0.1'

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
rpy_labels = ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]']
bands = [2.0, 2.0, 5.0]  # Different band for yaw since it drifts more

for i, ax in enumerate(axes):
    ax.plot(odom_t, rpy[:, i], 'b-', linewidth=0.8)
    ax.set_ylabel(rpy_labels[i]); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 20])

    # Compute steady state from 10-15s window
    mask_ss = (odom_t >= 10) & (odom_t <= 15)
    if np.any(mask_ss):
        ss = rpy[mask_ss, i].mean()
        band = bands[i]
        ax.axhline(ss, color='r', linestyle='-', alpha=0.5, linewidth=1)
        ax.axhline(ss + band, color='r', linestyle='--', alpha=0.3)
        ax.axhline(ss - band, color='r', linestyle='--', alpha=0.3)
        ax.fill_between([0, 20], ss - band, ss + band, alpha=0.05, color='red')

        ts_val, _ = find_settling_time(odom_t, rpy[:, i], t_thrust_on, t_search_end, band_deg=band)
        ax.axvline(t_thrust_on + ts_val, color='g', linestyle='--', alpha=0.7,
                   label=f't_s(±{band}°)={ts_val:.2f}s, ss={ss:.1f}°')
        ax.legend(loc='upper right', fontsize=9)

    ax.axvline(t_thrust_on, color='orange', linestyle=':', alpha=0.5, label='thrust on')

axes[0].set_title(f'Settling Time — {TITLE}')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'settling_time.png'), dpi=150)
print("\nSaved settling_time.png")

# ─── Plot: Control moment zoom ────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
moment_labels = ['Mx [Nm]', 'My [Nm]', 'Mz [Nm]']
for i, ax in enumerate(axes):
    ax.plot(ctrl_t, ctrl[:, 3+i], 'b-', linewidth=0.8)
    ax.set_ylabel(moment_labels[i]); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 20])

    # Show clamp for Mz
    if i == 2:
        ax.axhline(0.05, color='r', linestyle='--', alpha=0.5, label='clamp ±0.05')
        ax.axhline(-0.05, color='r', linestyle='--', alpha=0.5)
        ax.legend()

axes[0].set_title(f'Control Moments — {TITLE}')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'moments_zoom.png'), dpi=150)
print("Saved moments_zoom.png")

plt.close('all')
print("\nDone!")
