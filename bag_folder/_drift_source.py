#!/usr/bin/env python3
"""Trace the source of the 0.2 Hz position drift: compare PSDs of
position error, actual roll/pitch, and HGDO force on a common freq axis.

A shared peak reveals the chain:
  - peak in att + pos but not HGDO  → attitude wander drives position
  - peak in HGDO + pos              → force disturbance drives it
  - peak only in pos                → position-loop resonance / estimator

Usage:
  python3 _drift_source.py <bag_subdir> [<date>] [<tag>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
DATE = sys.argv[2] if len(sys.argv) > 2 else '2026_05_29'
TAG_OVR = sys.argv[3] if len(sys.argv) > 3 else None
db = glob.glob(os.path.join(_HERE, DATE, BAG, '*.db3'))[0]
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
    return px, py, pz, qw, qx, qy, qz


def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return struct.unpack_from('<6d', blob, off)


def quat_rp(qw, qx, qy, qz):
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    sp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
    pitch = np.arcsin(sp)
    return roll, pitch


con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("SELECT id,name FROM topics")
tids = {n: i for i, n in cur.fetchall()}
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tids['/mavros/local_position/odom']}")
t0 = cur.fetchone()[0]


def fetch(topic, parser):
    tid = tids[topic]
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tid} ORDER BY timestamp")
    rows = cur.fetchall()
    t = np.array([(ts - t0) * 1e-9 for ts, _ in rows])
    d = np.array([parser(b) for _, b in rows])
    return t, d


ot, odom = fetch('/mavros/local_position/odom', parse_odom)
ht, hg = fetch('/hgdo/wrench', parse_wrench)
con.close()

pz = odom[:, 2]
z_rel = pz - pz[ot < 5.0].mean() if (ot < 5.0).any() else pz
ab = z_rel > 0.05
t_to = ot[np.argmax(ab)]; t_land = ot[len(ab) - 1 - np.argmax(ab[::-1])]
m = (ot >= t_to + 3.0) & (ot <= t_land - 3.0)
t = ot[m]

px = odom[m, 0]; py = odom[m, 1]
rp = np.array([quat_rp(*odom[i, 3:7]) for i in np.where(m)[0]])
roll = np.degrees(rp[:, 0]); pitch = np.degrees(rp[:, 1])
fx = np.interp(t, ht, hg[:, 0]); fy = np.interp(t, ht, hg[:, 1])

fs = 100.0
N = int((t[-1] - t[0]) * fs)
tu = t[0] + np.arange(N) / fs


def rs(x):
    return np.interp(tu, t, x)


nper = min(2048, N // 4)
def psd(x):
    f, P = signal.welch(rs(x) - rs(x).mean(), fs=fs, nperseg=nper)
    return f, P


f, Pex = psd(px); _, Pey = psd(py)
_, Prl = psd(roll); _, Ppt = psd(pitch)
_, Pfx = psd(fx);  _, Pfy = psd(fy)


def peakf(f, P):
    mm = (f > 0.03) & (f < 1.0)
    return f[mm][np.argmax(P[mm])]


print(f'{TAG}  airborne {t_to:.1f}-{t_land:.1f}s')
print(f'Peak freq (0.03-1Hz):')
print(f'  pos e_x={peakf(f,Pex):.3f}  e_y={peakf(f,Pey):.3f} Hz')
print(f'  att roll={peakf(f,Prl):.3f}  pitch={peakf(f,Ppt):.3f} Hz')
print(f'  HGDO fx={peakf(f,Pfx):.3f}  fy={peakf(f,Pfy):.3f} Hz')

fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
axes[0].loglog(f, Pex, 'r', label='e_x'); axes[0].loglog(f, Pey, 'g', label='e_y')
axes[0].set_ylabel('pos PSD [m²/Hz]'); axes[0].legend(); axes[0].grid(alpha=0.3, which='both')
axes[0].axvline(0.22, color='k', ls=':', alpha=0.5, label='0.22Hz')
axes[0].set_title(f'{TAG} — drift source: position vs attitude vs HGDO force PSD')

axes[1].loglog(f, Prl, 'r', label='roll act'); axes[1].loglog(f, Ppt, 'g', label='pitch act')
axes[1].axvline(0.22, color='k', ls=':', alpha=0.5)
axes[1].set_ylabel('att PSD [deg²/Hz]'); axes[1].legend(); axes[1].grid(alpha=0.3, which='both')

axes[2].loglog(f, Pfx, 'r', label='HGDO fx'); axes[2].loglog(f, Pfy, 'g', label='HGDO fy')
axes[2].axvline(0.22, color='k', ls=':', alpha=0.5)
axes[2].set_ylabel('HGDO PSD [N²/Hz]'); axes[2].set_xlabel('freq [Hz]')
axes[2].legend(); axes[2].grid(alpha=0.3, which='both')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_drift_source.png')
plt.savefig(out, dpi=120)
print(f'Saved: {out}')
