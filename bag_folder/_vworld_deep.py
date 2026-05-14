#!/usr/bin/env python3
"""Deep-dive on world-frame linear velocity used by the PD's Kd term.

Compares three views:
  (1) v_world = R(q) · v_body        (what the controller actually uses)
  (2) v_world_num = d/dt p_world     (numerical derivative of mocap position)
  (3) v_world − v_world_num           (residual: discrepancy)

Provides:
  • Time series in calm and burst windows
  • PSD of v_world in both windows
  • Coherence (v_world vs v_world_num) — frequency where they agree
  • Numerical-derivative noise level (gauges raw mocap noise floor)

If v_world deviates from d/dt p substantially, EKF2 fusion or body→world
transform is suspect.

Usage:
  python3 _vworld_deep.py <bag_subdir> <t_center> [<win_s>] [<date>] [<tag>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
T_CENTER = float(sys.argv[2])
WIN = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
DATE = sys.argv[4] if len(sys.argv) > 4 else '2026_05_14_free_flight'
TAG_OVR = sys.argv[5] if len(sys.argv) > 5 else None
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
    return px, py, pz, qw, qx, qy, qz, vx, vy, vz


def quat_to_rotm(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ])


con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("SELECT id,name FROM topics")
tids = {n: i for i, n in cur.fetchall()}
tid = tids['/mavros/local_position/odom']
cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tid} ORDER BY timestamp")
rows = cur.fetchall()
con.close()
t_all = np.array([(ts - rows[0][0]) * 1e-9 for ts, _ in rows])
data = np.array([parse_odom(blob) for _, blob in rows])

# Window
m = (t_all >= T_CENTER - WIN / 2) & (t_all <= T_CENTER + WIN / 2)
t = t_all[m]
p = data[m, 0:3]
qs = data[m, 3:7]
v_body = data[m, 7:10]

N = len(t)
v_world_from_odom = np.empty_like(v_body)
for i in range(N):
    R = quat_to_rotm(qs[i, 0], qs[i, 1], qs[i, 2], qs[i, 3])
    v_world_from_odom[i] = R @ v_body[i]

# Numerical derivative of mocap position (central difference)
dt = np.gradient(t)
v_world_num = np.gradient(p, axis=0) / dt[:, None]

# Residual: where EKF2/transform may add or subtract
v_resid = v_world_from_odom - v_world_num

# Stats
print(f'Window: {t[0]:.1f} → {t[-1]:.1f} s  ({N} samples)')
print(f'\nv_world std comparison:')
for i, axis in enumerate('xyz'):
    print(f'  {axis}:  odom={v_world_from_odom[:,i].std():.3f}  '
          f'd/dt p={v_world_num[:,i].std():.3f}  '
          f'resid={v_resid[:,i].std():.3f}  '
          f'resid/odom={100*v_resid[:,i].std()/max(v_world_from_odom[:,i].std(),1e-6):.1f}%')

# Resample to uniform 100 Hz for PSD
fs = 100.0
N_u = int((t[-1] - t[0]) * fs)
t_u = t[0] + np.arange(N_u) / fs
v_o_u = np.array([np.interp(t_u, t, v_world_from_odom[:, i]) for i in range(3)]).T
v_n_u = np.array([np.interp(t_u, t, v_world_num[:, i]) for i in range(3)]).T

nperseg = min(1024, N_u // 4)
freqs, P_ox = signal.welch(v_o_u[:, 0] - v_o_u[:, 0].mean(), fs=fs, nperseg=nperseg)
_,     P_oy = signal.welch(v_o_u[:, 1] - v_o_u[:, 1].mean(), fs=fs, nperseg=nperseg)
_,     P_nx = signal.welch(v_n_u[:, 0] - v_n_u[:, 0].mean(), fs=fs, nperseg=nperseg)
_,     P_ny = signal.welch(v_n_u[:, 1] - v_n_u[:, 1].mean(), fs=fs, nperseg=nperseg)
fc, Cx = signal.coherence(v_o_u[:, 0], v_n_u[:, 0], fs=fs, nperseg=nperseg)
_,  Cy = signal.coherence(v_o_u[:, 1], v_n_u[:, 1], fs=fs, nperseg=nperseg)

# Plot
fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=False)

ax = axes[0]
ax.plot(t, v_world_from_odom[:, 0], 'r',  lw=1.5, label='v_x odom (R·v_body)')
ax.plot(t, v_world_num[:, 0],       'b--', lw=1.0, alpha=0.7, label='v_x = d/dt p_x')
ax.plot(t, v_resid[:, 0],           'g',  lw=0.8, alpha=0.6, label='residual (odom − dp/dt)')
ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('v_x [m/s]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')
ax.set_title(f'{TAG} — v_world comparison @ t={T_CENTER:.0f}±{WIN/2:.0f} s')
ax.set_xlabel('time [s]')

ax = axes[1]
ax.plot(t, v_world_from_odom[:, 1], 'r',  lw=1.5, label='v_y odom')
ax.plot(t, v_world_num[:, 1],       'b--', lw=1.0, alpha=0.7, label='v_y = d/dt p_y')
ax.plot(t, v_resid[:, 1],           'g',  lw=0.8, alpha=0.6, label='residual')
ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('v_y [m/s]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')
ax.set_xlabel('time [s]')

ax = axes[2]
ax.loglog(freqs, P_ox, 'r',  lw=1.2, label='v_x odom PSD')
ax.loglog(freqs, P_nx, 'r--', lw=1.0, alpha=0.6, label='v_x d/dt PSD')
ax.loglog(freqs, P_oy, 'g',  lw=1.2, label='v_y odom PSD')
ax.loglog(freqs, P_ny, 'g--', lw=1.0, alpha=0.6, label='v_y d/dt PSD')
ax.set_xlabel('frequency [Hz]'); ax.set_ylabel('PSD [(m/s)²/Hz]')
ax.grid(alpha=0.3, which='both'); ax.legend(loc='lower left', fontsize=9)
ax.set_title('PSD: odom-derived vs numerical-derivative')

ax = axes[3]
ax.semilogx(fc, Cx, 'r', lw=1.2, label='coherence v_x odom vs d/dt p')
ax.semilogx(fc, Cy, 'g', lw=1.2, label='coherence v_y odom vs d/dt p')
ax.set_xlabel('frequency [Hz]'); ax.set_ylabel('coherence'); ax.set_ylim(0, 1)
ax.grid(alpha=0.3, which='both'); ax.legend(loc='lower left')
ax.set_title('Coherence: where odom v and d/dt p agree')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_vworld_t{int(T_CENTER)}.png')
plt.savefig(out, dpi=120)
print(f'\nSaved: {out}')
