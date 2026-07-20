#!/usr/bin/env python3
"""z_th chattering sensitivity analysis (offline, no re-flight needed).

Takes the recorded z(t) and, for a sweep of altitude thresholds, counts how
many ground<->flight mode transitions occur (chattering metric), for:
  (a) single threshold:  mode = z > z_th
  (b) hysteresis: enter at z_th+band, exit at z_th-band, with dwell
This shows (1) how transition count varies with z_th, and (2) how much the
hysteresis suppresses chattering versus a single threshold.

Usage:
  python3 _zth_sensitivity.py <bag_subdir> [<root>] [<tag>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
ROOT = sys.argv[2] if len(sys.argv) > 2 else '.'
TAG = sys.argv[3] if len(sys.argv) > 3 else BAG.replace('/', '_')
db = glob.glob(os.path.join(_HERE, ROOT, BAG, '*.db3'))[0]
OUT_DIR = os.path.join(_HERE, ROOT, os.path.dirname(BAG))


def _align(o, n):
    return o + (-(o - 4)) % n


def parse_oz(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 4)
    s2 = struct.unpack_from('<I', b, o)[0]; o += 4 + s2; o = _align(o, 8)
    return struct.unpack_from('<d', b, o + 16)[0]


con = sqlite3.connect(db); cur = con.cursor()
cur.execute("SELECT id,name FROM topics"); tids = {n: i for i, n in cur.fetchall()}
tid = tids['/mavros/local_position/odom']
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tid}")
t0 = cur.fetchone()[0]
cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tid} ORDER BY timestamp")
rows = cur.fetchall(); con.close()
t = np.array([(x - t0) * 1e-9 for x, _ in rows])
z = np.array([parse_oz(b) for _, b in rows])
z = z - z[t < 5].mean() if (t < 5).any() else z   # zero at ground
fs = 1 / np.median(np.diff(t))

# restrict to takeoff+landing region (where z crosses thresholds): whole flight
# noise floor of z on ground (chattering source)
gmask = t < (t[np.argmax(z > 0.05)] - 1) if (z > 0.05).any() else (t < 5)
z_noise = z[gmask].std() if gmask.sum() > 10 else 0.0


def count_single(zth):
    mode = z > zth
    return int(np.sum(np.abs(np.diff(mode.astype(int)))))


def count_hyst(zth, band, dwell_n):
    enter = zth + band; exit_ = max(zth - band, 0.0)
    inflight = False; ec = 0; xc = 0; trans = 0
    for zz in z:
        if not inflight:
            ec = ec + 1 if zz > enter else 0
            if ec > dwell_n:
                inflight = True; trans += 1; xc = 0
        else:
            xc = xc + 1 if zz < exit_ else 0
            if xc > dwell_n:
                inflight = False; trans += 1; ec = 0
    return trans


zth_sweep = np.arange(0.005, 0.051, 0.0025)
single = [count_single(z_) for z_ in zth_sweep]
hyst = [count_hyst(z_, 0.01, int(0.3 * fs)) for z_ in zth_sweep]

print(f"{TAG}")
print(f"  ground z noise std = {z_noise*1000:.2f} mm  (chattering source)")
print(f"  z_th=0.010m: single-threshold transitions={count_single(0.010)}, "
      f"hysteresis transitions={count_hyst(0.010,0.01,int(0.3*fs))}")
print(f"  min ideal transitions = 2 (one takeoff + one landing)")
print(f"\n  z_th[m]  single  hyst")
for zt, s_, h_ in zip(zth_sweep, single, hyst):
    print(f"   {zt:.3f}    {s_:4d}   {h_:3d}")

fig, axes = plt.subplots(2, 1, figsize=(12, 9))
ax = axes[0]
ax.plot(zth_sweep * 1000, single, 'r-o', ms=4, label='single threshold')
ax.plot(zth_sweep * 1000, hyst, 'b-s', ms=4, label='hysteresis (±10mm, 0.3s dwell)')
ax.axhline(2, color='k', ls='--', alpha=0.5, label='ideal (1 takeoff+1 land)')
ax.axvline(10, color='g', ls=':', alpha=0.6, label='z_th=10mm (paper)')
ax.set_xlabel('altitude threshold z_th [mm]'); ax.set_ylabel('mode transitions (chattering)')
ax.set_yscale('symlog'); ax.grid(alpha=0.3, which='both'); ax.legend()
ax.set_title(f'{TAG} — z_th chattering sensitivity  (ground z noise {z_noise*1000:.1f}mm)')

ax = axes[1]
tm = (t > t[0]) & (t < t[-1])
ax.plot(t[tm], z[tm] * 1000, 'b', lw=0.8, label='z (rel) [mm]')
ax.axhline(10, color='g', ls='--', alpha=0.7, label='z_th=10mm')
ax.axhline(z_noise * 1000, color='r', ls=':', alpha=0.6, label=f'ground noise {z_noise*1000:.1f}mm')
ax.axhline(-z_noise * 1000, color='r', ls=':', alpha=0.6)
ax.set_xlabel('time [s]'); ax.set_ylabel('z [mm]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')
ax.set_ylim(-30, 120)
ax.set_title('z(t): threshold vs ground noise band')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_zth_sensitivity.png')
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
