#!/usr/bin/env python3
"""Yaw tracking analysis: ref_psi vs actual yaw, drift, rate.

Pulls:
  • /nmpc/ref         → ref_psi, ref_psi_dot
  • /mavros/local_position/odom → quaternion → actual yaw, ω_z

Plots over the whole flight and zoom around bursts:
  1. Yaw absolute (actual vs ref, unwrapped)
  2. Yaw error (act − ref, wrapped to ±180°)
  3. Yaw rate (ω_z from odom and ref_psi_dot)
  4. Coupling with x/y error (does yaw drift correlate with burst events?)

Usage:
  python3 _yaw_track.py <bag_subdir> [<date>] [<tag>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
DATE = sys.argv[2] if len(sys.argv) > 2 else '2026_05_14_free_flight'
TAG_OVR = sys.argv[3] if len(sys.argv) > 3 else None
BAG_DIR = os.path.join(_HERE, DATE, BAG)
db = glob.glob(os.path.join(BAG_DIR, '*.db3'))[0]
parts = BAG.split('/')
OUT_DIR = os.path.join(_HERE, DATE, *parts[:-1])
TAG = TAG_OVR if TAG_OVR else parts[-1]


def _align(off, n):
    return off + (-(off - 4)) % n


def parse_odom(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
    off += 36 * 8
    vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
    wx, wy, wz = struct.unpack_from('<3d', blob, off); off += 24
    return px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz


def parse_ref(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return struct.unpack_from('<8d', blob, off)


def quat_to_yaw(qw, qx, qy, qz):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))


def wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("SELECT id,name FROM topics")
tids = {n: i for i, n in cur.fetchall()}
tid = tids['/mavros/local_position/odom']
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tid}")
t0 = cur.fetchone()[0]


def fetch(topic, parser):
    tid = tids[topic]
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tid} ORDER BY timestamp")
    rows = cur.fetchall()
    t = np.array([(ts - t0) * 1e-9 for ts, _ in rows])
    d = np.array([parser(blob) for _, blob in rows])
    return t, d


ot, odom = fetch('/mavros/local_position/odom', parse_odom)
rt, ref = fetch('/nmpc/ref', parse_ref)
con.close()

# Extract
yaw_act = np.array([quat_to_yaw(odom[i, 3], odom[i, 4], odom[i, 5], odom[i, 6])
                    for i in range(len(ot))])
wz_act = odom[:, 12]
px = odom[:, 0]; py = odom[:, 1]; pz = odom[:, 2]
ref_psi = ref[:, 6]
ref_psi_dot = ref[:, 7]
ref_x = ref[:, 0]; ref_y = ref[:, 1]; ref_z = ref[:, 2]

# Airborne window detection (z > 0.05 m sustained)
z_rel = pz - pz[ot < 5.0].mean() if (ot < 5.0).any() else pz
airborne = z_rel > 0.05
# Find first/last airborne
if airborne.any():
    i_to = np.argmax(airborne)
    i_lo = len(airborne) - 1 - np.argmax(airborne[::-1])
    t_to = ot[i_to]; t_land = ot[i_lo]
else:
    t_to = ot[0]; t_land = ot[-1]

# Interp ref to odom timeline
ref_psi_o = np.interp(ot, rt, ref_psi) if len(rt) > 1 else np.zeros_like(ot)
ref_psi_dot_o = np.interp(ot, rt, ref_psi_dot) if len(rt) > 1 else np.zeros_like(ot)
ref_x_o = np.interp(ot, rt, ref_x) if len(rt) > 1 else np.zeros_like(ot)
ref_y_o = np.interp(ot, rt, ref_y) if len(rt) > 1 else np.zeros_like(ot)

# Yaw error (wrapped)
yaw_err = wrap_pi(yaw_act - ref_psi_o)

# Stats during airborne
m_air = (ot >= t_to + 2.0) & (ot <= t_land - 1.0)
print(f'Airborne window: {t_to:.1f} → {t_land:.1f} s   ({(t_land-t_to):.1f} s)')
print(f'\nYaw tracking stats (airborne):')
print(f'  ref_psi:        mean={np.degrees(ref_psi_o[m_air]).mean():+.2f}°  '
      f'std={np.degrees(ref_psi_o[m_air]).std():.2f}°  '
      f'range=[{np.degrees(ref_psi_o[m_air]).min():+.2f}, {np.degrees(ref_psi_o[m_air]).max():+.2f}]')
print(f'  yaw_act:        mean={np.degrees(yaw_act[m_air]).mean():+.2f}°  '
      f'std={np.degrees(yaw_act[m_air]).std():.2f}°  '
      f'range=[{np.degrees(yaw_act[m_air]).min():+.2f}, {np.degrees(yaw_act[m_air]).max():+.2f}]')
print(f'  yaw_err:        mean={np.degrees(yaw_err[m_air]).mean():+.2f}°  '
      f'std={np.degrees(yaw_err[m_air]).std():.2f}°  '
      f'|max|={np.degrees(np.abs(yaw_err[m_air])).max():.2f}°')
print(f'  ω_z:            mean={np.degrees(wz_act[m_air]).mean():+.2f}°/s  '
      f'std={np.degrees(wz_act[m_air]).std():.2f}°/s  '
      f'|max|={np.degrees(np.abs(wz_act[m_air])).max():.2f}°/s')

# Net yaw drift (act - act_initial during airborne)
yaw_drift_total = np.unwrap(yaw_act[m_air])[-1] - np.unwrap(yaw_act[m_air])[0]
print(f'\nNet yaw drift over airborne: {np.degrees(yaw_drift_total):+.2f}°  '
      f'(rate: {np.degrees(yaw_drift_total)/(t_land-t_to):.3f}°/s)')

# Plot
fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)

ax = axes[0]
ax.plot(ot, np.degrees(np.unwrap(yaw_act)), 'r-', lw=1.2, label='yaw_act')
ax.plot(ot, np.degrees(np.unwrap(ref_psi_o)), 'k--', lw=1.0, alpha=0.7, label='ref_psi')
ax.axvline(t_to, color='g', alpha=0.3, lw=1.0)
ax.axvline(t_land, color='r', alpha=0.3, lw=1.0)
ax.set_ylabel('yaw [deg]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')
ax.set_title(f'{TAG} — Yaw tracking analysis')

ax = axes[1]
ax.plot(ot, np.degrees(yaw_err), 'b', lw=1.0)
ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.axvline(t_to, color='g', alpha=0.3, lw=1.0)
ax.axvline(t_land, color='r', alpha=0.3, lw=1.0)
ax.set_ylabel('yaw err [deg]'); ax.grid(alpha=0.3)
ax.set_title(f'Yaw error (wrapped ±180°)   std = {np.degrees(yaw_err[m_air]).std():.2f}°,  '
             f'|max| = {np.degrees(np.abs(yaw_err[m_air])).max():.2f}°')

ax = axes[2]
ax.plot(ot, np.degrees(wz_act), 'r-', lw=0.8, alpha=0.8, label='ω_z actual')
ax.plot(ot, np.degrees(ref_psi_dot_o), 'k--', lw=1.0, alpha=0.7, label='ref ψ_dot')
ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('yaw rate [deg/s]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')

ax = axes[3]
ax.plot(ot, px - ref_x_o, 'r', lw=0.8, alpha=0.8, label='e_x = ref-p_x')
ax.plot(ot, py - ref_y_o, 'g', lw=0.8, alpha=0.8, label='e_y')
ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('pos error [m]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')

# Cross-correlation: does yaw_err correlate with e_x/e_y bursts?
ax = axes[4]
# Sliding-window std of e_x and yaw_err to see if they track together
win_size = max(int(2.0 / np.median(np.diff(ot))), 50)  # ~2s window
e_x = px - ref_x_o; e_y = py - ref_y_o
e_xy_rms = np.sqrt(e_x ** 2 + e_y ** 2)
def rolling_std(x, w):
    out = np.zeros_like(x)
    for i in range(len(x)):
        lo = max(0, i - w // 2); hi = min(len(x), i + w // 2)
        out[i] = x[lo:hi].std()
    return out

e_xy_rs   = rolling_std(e_xy_rms, win_size)
yaw_err_rs = rolling_std(yaw_err, win_size)
ax.plot(ot, e_xy_rs, 'r', lw=1.0, label='2-s rolling std |e_xy| [m]')
ax2 = ax.twinx()
ax2.plot(ot, np.degrees(yaw_err_rs), 'b', lw=1.0, label='2-s rolling std yaw_err [deg]')
ax.set_ylabel('|e_xy| std [m]', color='r')
ax2.set_ylabel('yaw_err std [deg]', color='b')
ax.set_xlabel('time [s]'); ax.grid(alpha=0.3)
ax.set_title('Rolling 2-s std: does yaw error spike coincide with burst?')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_yaw_track.png')
plt.savefig(out, dpi=120)
print(f'\nSaved: {out}')
