#!/usr/bin/env python3
"""Decompose xy position error into DC / low-freq drift / high-freq jitter.

Answers: is the residual xy wander a steady offset (CoM/disturbance the
PD can't reject because dob_force_xy=false), a slow limit-cycle drift,
or high-frequency jitter fed in through v_world → Kd term?

Bands:
  DC      : mean of e_xy
  drift   : 0.05 – 0.5 Hz band power
  midcycle: 0.5 – 2 Hz
  jitter  : > 2 Hz

Usage:
  python3 _exy_spectrum.py <bag_subdir> [<date>] [<tag>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
DATE = sys.argv[2] if len(sys.argv) > 2 else '2026_05_15_free_flight'
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
    return px, py, pz, vx, vy, vz


def parse_ref(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return struct.unpack_from('<8d', blob, off)


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

px = odom[:, 0]; py = odom[:, 1]; pz = odom[:, 2]
# Airborne window
z_rel = pz - pz[ot < 5.0].mean() if (ot < 5.0).any() else pz
ab = z_rel > 0.05
t_to = ot[np.argmax(ab)]; t_land = ot[len(ab) - 1 - np.argmax(ab[::-1])]
m = (ot >= t_to + 3.0) & (ot <= t_land - 3.0)

t = ot[m]
ref_x = np.interp(t, rt, ref[:, 0]) if len(rt) > 1 else np.zeros_like(t)
ref_y = np.interp(t, rt, ref[:, 1]) if len(rt) > 1 else np.zeros_like(t)
e_x = ref_x - px[m]
e_y = ref_y - py[m]

# Resample uniform
fs = 100.0
N = int((t[-1] - t[0]) * fs)
tu = t[0] + np.arange(N) / fs
ex_u = np.interp(tu, t, e_x)
ey_u = np.interp(tu, t, e_y)

# DC
dc_x, dc_y = ex_u.mean(), ey_u.mean()

# PSD
nperseg = min(2048, N // 4)
fx, Px = signal.welch(ex_u - dc_x, fs=fs, nperseg=nperseg)
fy, Py = signal.welch(ey_u - dc_y, fs=fs, nperseg=nperseg)


def bandpow(f, P, lo, hi):
    mm = (f >= lo) & (f < hi)
    return np.trapezoid(P[mm], f[mm])


bands = [('drift 0.05-0.5Hz', 0.05, 0.5),
         ('midcyc 0.5-2Hz', 0.5, 2.0),
         ('jitter >2Hz', 2.0, fs / 2)]

print(f'Airborne: {t_to:.1f} → {t_land:.1f} s')
print(f'\n=== e_x ===')
print(f'  DC (mean offset)  = {dc_x*100:+.2f} cm')
print(f'  total AC std      = {ex_u.std()*100:.2f} cm')
for name, lo, hi in bands:
    rms = np.sqrt(bandpow(fx, Px, lo, hi))
    print(f'  {name:18s} rms = {rms*100:.2f} cm')
print(f'\n=== e_y ===')
print(f'  DC (mean offset)  = {dc_y*100:+.2f} cm')
print(f'  total AC std      = {ey_u.std()*100:.2f} cm')
for name, lo, hi in bands:
    rms = np.sqrt(bandpow(fy, Py, lo, hi))
    print(f'  {name:18s} rms = {rms*100:.2f} cm')

# Peak frequency in AC
def peak_freq(f, P):
    mm = f > 0.03
    j = np.argmax(P[mm])
    return f[mm][j]
print(f'\nPeak AC freq: e_x={peak_freq(fx,Px):.3f} Hz   e_y={peak_freq(fy,Py):.3f} Hz')

fig, axes = plt.subplots(3, 1, figsize=(13, 10))
ax = axes[0]
ax.plot(t, e_x, 'r', lw=1.0, label=f'e_x (DC={dc_x*100:+.1f} cm, std={ex_u.std()*100:.1f})')
ax.plot(t, e_y, 'g', lw=1.0, label=f'e_y (DC={dc_y*100:+.1f} cm, std={ey_u.std()*100:.1f})')
ax.axhline(dc_x, color='r', ls='--', alpha=0.5)
ax.axhline(dc_y, color='g', ls='--', alpha=0.5)
ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('pos error [m]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')
ax.set_title(f'{TAG} — xy position error decomposition (airborne)')
ax.set_xlabel('time [s]')

ax = axes[1]
ax.loglog(fx, Px, 'r', label='e_x PSD'); ax.loglog(fy, Py, 'g', label='e_y PSD')
for name, lo, hi in bands:
    ax.axvspan(lo, hi, alpha=0.06, color='b')
ax.set_xlabel('freq [Hz]'); ax.set_ylabel('PSD [m²/Hz]')
ax.grid(alpha=0.3, which='both'); ax.legend()
ax.set_title('Position-error PSD (band shading: drift / midcycle / jitter)')

# XY trajectory (top-down)
ax = axes[2]
ax.plot(px[m] - px[m].mean(), py[m] - py[m].mean(), 'b', lw=0.5, alpha=0.7)
ax.scatter([0], [0], c='r', marker='+', s=200, label='mean')
ax.set_xlabel('x - mean [m]'); ax.set_ylabel('y - mean [m]')
ax.axis('equal'); ax.grid(alpha=0.3); ax.legend()
ax.set_title('XY trajectory (top-down) — line vs circle reveals motion type')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_exy_spectrum.png')
plt.savefig(out, dpi=120)
print(f'\nSaved: {out}')
