#!/usr/bin/env python3
"""Plot per-motor actual RPM vs rotor_min/max limits to visualize saturation.
Usage: python3 _motor_sat.py <bag_subdir> [<date>] [<tag>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
DATE = sys.argv[2] if len(sys.argv) > 2 else '2026_05_29'
TAG_OVR = sys.argv[3] if len(sys.argv) > 3 else None
db = glob.glob(os.path.join(_HERE, DATE, BAG, '*.db3'))[0]
parts = BAG.split('/')
OUT_DIR = os.path.join(_HERE, DATE, *parts[:-1])
TAG = TAG_OVR if TAG_OVR else parts[-1]
RMIN, RMAX = 2000, 7300


def _align(o, n):
    return o + (-(o - 4)) % n


def p_rpm(b):
    o = 4 + 8; s = struct.unpack_from('<I', b, o)[0]; o += 4 + s; o = _align(o, 4)
    return list(struct.unpack_from('<6I', b, o))


def p_oz(b):
    o = 4 + 8; s = struct.unpack_from('<I', b, o)[0]; o += 4 + s; o = _align(o, 4)
    s2 = struct.unpack_from('<I', b, o)[0]; o += 4 + s2; o = _align(o, 8)
    return struct.unpack_from('<d', b, o + 16)[0]


con = sqlite3.connect(db); cur = con.cursor()
cur.execute("SELECT id,name FROM topics"); tids = {n: i for i, n in cur.fetchall()}
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tids['/mavros/local_position/odom']}")
t0 = cur.fetchone()[0]


def fetch(top, par):
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tids[top]} ORDER BY timestamp")
    r = cur.fetchall()
    return np.array([(t - t0) * 1e-9 for t, _ in r]), np.array([par(b) for _, b in r])


ot, oz = fetch('/mavros/local_position/odom', p_oz)
rt, act = fetch('/uav/actual_rpm', p_rpm)
con.close()
z = oz - oz[ot < 5].mean() if (ot < 5).any() else oz
ab = z > 0.05
t_to = ot[np.argmax(ab)]; t_land = ot[len(ab) - 1 - np.argmax(ab[::-1])]

fig, axes = plt.subplots(6, 1, figsize=(13, 12), sharex=True)
m = (rt >= t_to + 2) & (rt <= t_land - 2)
for i in range(6):
    ax = axes[i]
    ax.plot(rt, act[:, i], 'b', lw=0.6)
    ax.axhline(RMAX, color='r', ls='--', lw=1.0, alpha=0.7, label=f'max {RMAX}')
    ax.axhline(RMIN, color='orange', ls='--', lw=1.0, alpha=0.5, label=f'min {RMIN}')
    ax.axvspan(t_to, t_land, alpha=0.05, color='g')
    pct = (act[m, i] >= RMAX - 100).mean() * 100
    ax.set_ylabel(f'm{i} RPM')
    ax.set_ylim(RMIN - 500, RMAX + 400)
    ax.grid(alpha=0.3)
    ax.text(0.01, 0.92, f'%near_max={pct:.1f}  mean={act[m,i].mean():.0f}',
            transform=ax.transAxes, va='top', fontsize=9,
            color='red' if pct > 1 else 'black')
    if i == 0:
        ax.legend(loc='lower right', fontsize=8)
axes[0].set_title(f'{TAG} — per-motor actual RPM vs limits (green=airborne)')
axes[-1].set_xlabel('time [s]')
plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_motor_sat.png')
plt.savefig(out, dpi=120)
print(f'Saved: {out}')
