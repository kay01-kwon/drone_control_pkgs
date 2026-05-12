#!/usr/bin/env python3
"""Compute time derivative of desired roll/pitch (= required body rate) and
compare with actual body rate and the inner-loop bandwidth design.

If d/dt(desired_rp) exceeds what NMPC + body dynamics can deliver, the
actual will lag and overshoot even if desired looks "smooth" in absolute
terms.

Usage:  python3 des_rp_rate.py [<bag_subdir> [<date_dir>]]
"""

import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG_SUBDIR = sys.argv[1] if len(sys.argv) > 1 else 'Qw_0p6'
DATE_DIR   = sys.argv[2] if len(sys.argv) > 2 else '2026_05_11_free_flight'
DB = glob.glob(os.path.join(_HERE, DATE_DIR, BAG_SUBDIR, '*.db3'))[0]
OUT_DIR = os.path.join(_HERE, DATE_DIR)
TAG = BAG_SUBDIR
print(f'Analyzing: {DB}')


def _align(off, n):
    return off + ((-(off - 4)) % n)


def parse_odom(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
    off += 36 * 8
    off += 24
    wx, wy, wz = struct.unpack_from('<3d', blob, off)
    return np.array([qw, qx, qy, qz, wx, wy, wz])


def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return np.array(struct.unpack_from('<6d', blob, off))


def quat_to_rpy(q):
    qw, qx, qy, qz = q
    roll  = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw   = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    return np.array([roll, pitch, yaw])


def force_to_rp(fx, fy, f_col, psi):
    fz = np.sqrt(max(f_col ** 2 - fx ** 2 - fy ** 2, 0.0))
    n = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    if n < 1e-6: return 0.0, 0.0
    zb = np.array([fx, fy, fz]) / n
    xc = np.array([np.cos(psi), np.sin(psi), 0.0])
    yb = np.cross(zb, xc)
    yn = np.linalg.norm(yb)
    yb = (np.array([-np.sin(psi), np.cos(psi), 0.0]) if yn < 1e-6 else yb / yn)
    xb = np.cross(yb, zb)
    R = np.column_stack((xb, yb, zb))
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    return roll, pitch


conn = sqlite3.connect(DB)
c = conn.cursor()
tid = {n: i for i, n in c.execute('SELECT id, name FROM topics').fetchall()}

def fetch(topic, parser):
    rows = c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
                     (tid[topic],)).fetchall()
    ts = np.array([r[0] for r in rows], dtype=np.float64)
    dat = np.array([parser(bytes(r[1])) for r in rows])
    return ts, dat

odom_ts, od = fetch('/mavros/local_position/odom', parse_odom)
ctrl_ts, ct = fetch('/nmpc/control', parse_wrench)
conn.close()

t0 = min(odom_ts[0], ctrl_ts[0])
odom_t = (odom_ts - t0) * 1e-9
ctrl_t = (ctrl_ts - t0) * 1e-9

rpy = np.array([quat_to_rpy(od[i, 0:4]) for i in range(len(od))])
w = od[:, 4:7]
psi_at_ctrl = np.interp(ctrl_t, odom_t, np.unwrap(rpy[:, 2]))
des_rp = np.zeros((len(ctrl_t), 2))
for i in range(len(ctrl_t)):
    des_rp[i] = force_to_rp(ct[i, 0], ct[i, 1], ct[i, 2], psi_at_ctrl[i])

# Smooth derivative of desired RP (box filter + central diff)
def smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, mode='same')

mask = ctrl_t > 5.0
rp_des_smooth = np.column_stack([smooth(des_rp[mask, k], 5) for k in range(2)])
t_c = ctrl_t[mask]
des_rp_rate = np.column_stack([np.gradient(rp_des_smooth[:, k], t_c) for k in range(2)])

# Actual body rate
mask_o = odom_t > 5.0
w_a = w[mask_o]
t_o = odom_t[mask_o]

# Stats
print('\n== Desired RP and its rate (airborne) ==')
print('-- desired tilt [deg] --')
for k, name in enumerate(['roll_des', 'pitch_des']):
    deg = np.degrees(des_rp[mask, k])
    print(f'  {name:9s}  std={deg.std():6.3f}°  peak±{max(abs(deg.min()), abs(deg.max())):6.3f}°')

print('-- d/dt desired tilt [deg/s] --')
for k, name in enumerate(['d_roll_des',  'd_pitch_des']):
    deg = np.degrees(des_rp_rate[:, k])
    print(f'  {name:11s}  std={deg.std():7.2f}°/s  peak±{max(abs(deg.min()), abs(deg.max())):7.2f}°/s')

print('-- actual body rate ω [deg/s] --')
for k, name in enumerate(['ω_x', 'ω_y', 'ω_z']):
    deg = np.degrees(w_a[:, k])
    print(f'  {name:3s}  std={deg.std():7.2f}°/s  peak±{max(abs(deg.min()), abs(deg.max())):7.2f}°/s')

# Inner-loop design bandwidth from Q_q=70, Q_ω=0.6
# ωn ≈ 10.06 rad/s = 576 deg/s — this is the natural frequency, not max rate.
# But the system can only deliver rates up to its angular acceleration capability.
# α_max ~ M_max / J = (max motor torque ~ 1 N·m) / 0.06 = 16.7 rad/s² ≈ 956 deg/s²
# At ωn = 10 rad/s, peak velocity = amplitude × ωn.
# For 5° amplitude at ωn: peak velocity = 5 × 10/57.3 = 0.87 rad/s = 50°/s
print('\n  Design: ωn_inner = 10.06 rad/s = 576 deg/s (natural freq)')
print('  Inner loop can roughly track desired rates up to ωn × tilt_amplitude.')


# ── Plots ──
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
# Roll
axes[0].plot(t_c, np.degrees(des_rp[mask, 0]),     'k', lw=1.0, label='roll desired')
axes[0].plot(t_o, np.degrees(rpy[mask_o, 0]),      'r', lw=0.8, alpha=0.7, label='roll actual')
axes[0].set_ylabel('Roll [deg]'); axes[0].grid(alpha=0.3)
axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_title(f'{TAG}  —  Roll: desired vs actual')

axes[1].plot(t_c, np.degrees(des_rp_rate[:, 0]), 'k', lw=0.9, label='d/dt roll_des')
axes[1].plot(t_o, np.degrees(w_a[:, 0]),         'r', lw=0.8, alpha=0.7, label='ω_x actual')
axes[1].set_ylabel('Roll rate [deg/s]'); axes[1].grid(alpha=0.3)
axes[1].legend(loc='upper right', fontsize=9)
axes[1].set_title(f'd/dt(roll_des) vs ω_x  (this is what inner loop must deliver)')

# Pitch
axes[2].plot(t_c, np.degrees(des_rp[mask, 1]),     'k', lw=1.0, label='pitch desired')
axes[2].plot(t_o, np.degrees(rpy[mask_o, 1]),      'g', lw=0.8, alpha=0.7, label='pitch actual')
axes[2].set_ylabel('Pitch [deg]'); axes[2].grid(alpha=0.3)
axes[2].legend(loc='upper right', fontsize=9)

axes[3].plot(t_c, np.degrees(des_rp_rate[:, 1]), 'k', lw=0.9, label='d/dt pitch_des')
axes[3].plot(t_o, np.degrees(w_a[:, 1]),         'g', lw=0.8, alpha=0.7, label='ω_y actual')
axes[3].set_ylabel('Pitch rate [deg/s]'); axes[3].set_xlabel('Time [s]')
axes[3].grid(alpha=0.3); axes[3].legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_des_rp_rate.png'), dpi=120)
plt.close()


# ── PSD of desired tilt and rate ──
from numpy.fft import rfft, rfftfreq
def psd(x, dt):
    x = x - x.mean()
    n = len(x)
    X = rfft(x * np.hanning(n))
    f = rfftfreq(n, dt)
    return f, (np.abs(X) ** 2) / n

dt_c = np.diff(t_c).mean()
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
for k, (name, col) in enumerate([('Roll', 'r'), ('Pitch', 'g')]):
    f, P_des = psd(np.degrees(des_rp[mask, k]), dt_c)
    f2, P_rate = psd(np.degrees(des_rp_rate[:, k]), dt_c)
    axes[0].loglog(f,  np.sqrt(P_des),  col, lw=1.0, label=f'{name} desired')
    axes[1].loglog(f2, np.sqrt(P_rate), col, lw=1.0, label=f'd/dt {name} desired')
axes[0].axvline(10.06 / (2 * np.pi), color='k', ls='--', alpha=0.6, label='inner ωn / 2π = 1.6 Hz')
axes[1].axvline(10.06 / (2 * np.pi), color='k', ls='--', alpha=0.6)
axes[0].set_ylabel('amplitude [deg/√Hz]'); axes[0].set_title('desired tilt PSD')
axes[0].grid(alpha=0.3, which='both'); axes[0].legend(loc='upper right', fontsize=9)
axes[1].set_xlabel('frequency [Hz]'); axes[1].set_ylabel('amplitude [(deg/s)/√Hz]')
axes[1].set_title('d/dt desired tilt PSD')
axes[1].grid(alpha=0.3, which='both'); axes[1].legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_des_rp_psd.png'), dpi=120)
plt.close()

print(f'\nSaved:')
print(f'  - {TAG}_des_rp_rate.png')
print(f'  - {TAG}_des_rp_psd.png')
