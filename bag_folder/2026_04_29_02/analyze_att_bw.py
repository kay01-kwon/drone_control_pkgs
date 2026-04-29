#!/usr/bin/env python3
"""Estimate attitude closed-loop bandwidth from flight data."""

import sqlite3, struct, os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

DB_01 = os.path.join(os.path.dirname(__file__), '..', '2026_04_29_01', '2026_04_29_01_0.db3')
DB_02 = os.path.join(os.path.dirname(__file__), '..', '2026_04_29_02', '2026_04_29_02_0.db3')

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

def quat_to_rpy(qw, qx, qy, qz):
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (qw * qy - qz * qx); sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw

def load_topic(db, topic_name, parser):
    cur = db.execute("SELECT m.timestamp, m.data FROM messages m JOIN topics t ON m.topic_id=t.id WHERE t.name=? ORDER BY m.timestamp", (topic_name,))
    times, data = [], []
    for ts, blob in cur.fetchall():
        try:
            d = parser(blob); times.append(ts * 1e-9); data.append(d)
        except: pass
    return np.array(times), data

def analyze_att_bw(db_path, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    db = sqlite3.connect(db_path)
    odom_ts, odom_raw = load_topic(db, '/mavros/local_position/odom', parse_odom); odom = np.array(odom_raw)
    ctrl_ts, ctrl_raw = load_topic(db, '/nmpc/control', parse_wrench); ctrl = np.array(ctrl_raw)
    dob_ts, dob_raw = load_topic(db, '/hgdo/wrench', parse_wrench); dob = np.array(dob_raw)
    db.close()

    t0 = odom_ts[0]
    odom_t = odom_ts - t0; ctrl_t = ctrl_ts - t0; dob_t = dob_ts - t0

    pos = odom[:, 0:3].copy()
    pos[:, 0] -= odom[0, 0]; pos[:, 1] -= odom[0, 1]; pos[:, 2] -= odom[0, 2]
    rpy = np.array([quat_to_rpy(r[6], r[7], r[8], r[9]) for r in odom])

    airborne_mask = pos[:, 2] > 0.05
    if not np.any(airborne_mask):
        print("No airborne data!"); return None
    s = np.argmax(airborne_mask)
    e = len(airborne_mask) - 1 - np.argmax(airborne_mask[::-1])
    t_start, t_end = odom_t[s], odom_t[e]
    print(f"Airborne: {t_start:.1f}s - {t_end:.1f}s ({t_end-t_start:.1f}s)")

    # Pure PD force = ctrl + dob
    dob_fx_interp = np.interp(ctrl_t, dob_t, dob[:, 0])
    dob_fy_interp = np.interp(ctrl_t, dob_t, dob[:, 1])
    pd_fx = ctrl[:, 0] + dob_fx_interp
    pd_fy = ctrl[:, 1] + dob_fy_interp

    # q_des from PD force (small angle, yaw-corrected)
    yaw_at_ctrl = np.interp(ctrl_t, odom_t, rpy[:, 2])
    cos_psi = np.cos(yaw_at_ctrl); sin_psi = np.sin(yaw_at_ctrl)
    # Body frame forces
    fb_x = pd_fx * cos_psi + pd_fy * sin_psi
    fb_y = -pd_fx * sin_psi + pd_fy * cos_psi
    pitch_des = -fb_x / (MASS * G)  # rad
    roll_des = fb_y / (MASS * G)    # rad

    # Interpolate actual attitude to ctrl timestamps
    roll_actual = np.interp(ctrl_t, odom_t, rpy[:, 0])
    pitch_actual = np.interp(ctrl_t, odom_t, rpy[:, 1])

    # Airborne mask for ctrl timestamps
    air_mask = (ctrl_t >= t_start + 2) & (ctrl_t <= t_end - 2)  # trim edges
    if np.sum(air_mask) < 100:
        print("Not enough airborne ctrl data"); return None

    t_air = ctrl_t[air_mask]
    dt = np.median(np.diff(t_air)); fs = 1.0 / dt
    print(f"Ctrl sample rate: {fs:.1f} Hz, dt={dt*1000:.1f} ms")

    results = {}
    for axis, des, act, name in [
        ('roll', roll_des[air_mask], roll_actual[air_mask], 'Roll'),
        ('pitch', pitch_des[air_mask], pitch_actual[air_mask], 'Pitch'),
    ]:
        # Cross-correlation for time delay
        des_c = des - des.mean()
        act_c = act - act.mean()
        corr = np.correlate(act_c, des_c, 'full')
        lags = np.arange(-len(des_c)+1, len(des_c)) * dt
        peak_idx = np.argmax(corr)
        delay = lags[peak_idx]
        print(f"\n{name}:")
        print(f"  Cross-corr delay: {delay*1000:.1f} ms")

        # Transfer function estimate via Welch
        nperseg = min(256, len(des_c) // 4)
        if nperseg > 32:
            f, Pxx = signal.welch(des_c, fs=fs, nperseg=nperseg)
            f, Pyy = signal.welch(act_c, fs=fs, nperseg=nperseg)
            f, Pxy = signal.csd(des_c, act_c, fs=fs, nperseg=nperseg)
            f, Pyx = signal.csd(act_c, des_c, fs=fs, nperseg=nperseg)

            H_mag = np.sqrt(Pyy / np.maximum(Pxx, 1e-20))
            coherence = np.abs(Pxy)**2 / (Pxx * Pyy + 1e-20)
            H_phase = np.angle(Pxy)  # phase of cross-spectrum

            # -3dB bandwidth
            H_mag_db = 20 * np.log10(np.maximum(H_mag, 1e-10))
            dc_gain = H_mag_db[1:4].mean() if len(H_mag_db) > 4 else H_mag_db[1]
            bw_mask = H_mag_db < (dc_gain - 3)
            valid_bw = f[bw_mask]
            bw_hz = valid_bw[valid_bw > 0.1].min() if len(valid_bw[valid_bw > 0.1]) > 0 else float('nan')
            bw_rad = bw_hz * 2 * np.pi
            print(f"  DC gain: {dc_gain:.1f} dB")
            print(f"  -3dB BW: {bw_hz:.2f} Hz = {bw_rad:.1f} rad/s")

            results[axis] = {
                'f': f, 'H_mag_db': H_mag_db, 'coherence': coherence,
                'H_phase': H_phase, 'delay': delay, 'bw_hz': bw_hz, 'bw_rad': bw_rad,
                't': t_air, 'des': des, 'act': act, 'dc_gain': dc_gain
            }

    return results

# Analyze both flights
r1 = analyze_att_bw(DB_01, 'Kp=[1,1,5], Kd=[1.4,1.4,3.5]')
r2 = analyze_att_bw(DB_02, 'Kp=[1.5,1.5,5], Kd=[1.7,1.7,3.5]')

OUT = os.path.dirname(__file__)

# ─── Plot 1: Time-domain tracking ────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=False)
datasets = [
    (r1, 'Kp=1.0', 0, 'roll'), (r1, 'Kp=1.0', 1, 'pitch'),
    (r2, 'Kp=1.5', 2, 'roll'), (r2, 'Kp=1.5', 3, 'pitch'),
]
for r, lbl, idx, axis in datasets:
    if r and axis in r:
        d = r[axis]
        ax = axes[idx]
        t_plot = d['t'] - d['t'][0]
        ax.plot(t_plot, np.degrees(d['des']), 'r-', linewidth=0.8, alpha=0.7, label=f'{axis}_des')
        ax.plot(t_plot, np.degrees(d['act']), 'b-', linewidth=0.8, alpha=0.7, label=f'{axis}_act')
        ax.set_ylabel(f'{axis.title()} [deg]')
        ax.set_title(f'{lbl} — {axis.title()} (delay={d["delay"]*1000:.0f}ms, BW={d["bw_rad"]:.1f} rad/s)')
        ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(15, t_plot[-1])])
axes[-1].set_xlabel('Time [s]')
fig.suptitle('Attitude Tracking: q_des (from PD) vs q_actual', fontsize=13)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'att_bw_tracking.png'), dpi=150)
print("\nSaved att_bw_tracking.png")

# ─── Plot 2: Bode-like magnitude + coherence ─────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for col, (r, lbl) in enumerate([(r1, 'Kp=1.0'), (r2, 'Kp=1.5')]):
    if r is None: continue
    for axis, color in [('roll', 'C0'), ('pitch', 'C1')]:
        if axis not in r: continue
        d = r[axis]
        f_plot = d['f']
        mask = (f_plot > 0.05) & (f_plot < 15)

        # Magnitude
        axes[0, col].plot(f_plot[mask], d['H_mag_db'][mask], color=color, linewidth=1.2, label=f'{axis} (BW={d["bw_hz"]:.1f}Hz)')
        axes[0, col].axhline(d['dc_gain'] - 3, color='gray', linestyle=':', alpha=0.5)
        if not np.isnan(d['bw_hz']):
            axes[0, col].axvline(d['bw_hz'], color=color, linestyle='--', alpha=0.5)

        # Coherence
        axes[1, col].plot(f_plot[mask], d['coherence'][mask], color=color, linewidth=1.2, label=axis)

    axes[0, col].set_ylabel('|H| [dB]'); axes[0, col].set_title(f'{lbl} — Magnitude')
    axes[0, col].legend(); axes[0, col].grid(True, alpha=0.3); axes[0, col].set_ylim([-20, 10])
    axes[1, col].set_ylabel('Coherence'); axes[1, col].set_xlabel('Frequency [Hz]')
    axes[1, col].set_title(f'{lbl} — Coherence')
    axes[1, col].legend(); axes[1, col].grid(True, alpha=0.3); axes[1, col].set_ylim([0, 1.05])
    axes[1, col].axhline(0.5, color='gray', linestyle=':', alpha=0.5)

fig.suptitle('Attitude Closed-Loop Transfer Function Estimate', fontsize=13)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'att_bw_bode.png'), dpi=150)
print("Saved att_bw_bode.png")

plt.close('all')
print("\nDone!")
