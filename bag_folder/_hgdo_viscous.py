#!/usr/bin/env python3
"""Decompose HGDO xy force into viscous (∝ v) and constant-bias parts.

   f_HGDO_x = a_x · v_world_x + b_x
   f_HGDO_y = a_y · v_world_y + b_y

   a (slope, N per m/s) → viscous damping coefficient (negative = drag)
   b (intercept, N)     → constant force (CoM gravity proj / wind / thrust asym)
   R²                   → how much of HGDO force the viscous model explains

Usage:
  python3 _hgdo_viscous.py <bag_subdir> [<date>] [<tag>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
DATE = sys.argv[2] if len(sys.argv) > 2 else '2026_05_15_free_flight'
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
    off += 36 * 8
    vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
    return px, py, pz, qw, qx, qy, qz, vx, vy, vz


def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return struct.unpack_from('<6d', blob, off)


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

# Airborne crop
pz = odom[:, 2]
z_rel = pz - pz[ot < 5.0].mean() if (ot < 5.0).any() else pz
ab = z_rel > 0.05
t_to = ot[np.argmax(ab)]; t_land = ot[len(ab) - 1 - np.argmax(ab[::-1])]
m = (ot >= t_to + 3.0) & (ot <= t_land - 3.0)
t = ot[m]

# v_world
qs = odom[m, 3:7]; vb = odom[m, 7:10]
vw = np.empty_like(vb)
for i in range(len(t)):
    vw[i] = quat_to_rotm(*qs[i]) @ vb[i]

# HGDO fxy interp to odom timeline
fx = np.interp(t, ht, hg[:, 0])
fy = np.interp(t, ht, hg[:, 1])


def regress(v, f):
    """f = a*v + b ; return a, b, R²"""
    A = np.vstack([v, np.ones_like(v)]).T
    (a, b), *_ = np.linalg.lstsq(A, f, rcond=None)
    pred = a * v + b
    ss_res = ((f - pred) ** 2).sum()
    ss_tot = ((f - f.mean()) ** 2).sum()
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
    return a, b, R2


ax_, bx_, R2x = regress(vw[:, 0], fx)
ay_, by_, R2y = regress(vw[:, 1], fy)

print(f'Airborne: {t_to:.1f} → {t_land:.1f} s\n')
print('HGDO force = a·v_world + b   (a=viscous N/(m/s), b=constant N)\n')
print(f'  x:  a = {ax_:+.3f} N/(m/s)   b = {bx_:+.3f} N   R²(viscous) = {R2x:.3f}')
print(f'  y:  a = {ay_:+.3f} N/(m/s)   b = {by_:+.3f} N   R²(viscous) = {R2y:.3f}')
print(f'\n  v_world std:  vx={vw[:,0].std():.3f}  vy={vw[:,1].std():.3f} m/s')
print(f'  HGDO f std:   fx={fx.std():.3f}  fy={fy.std():.3f} N')
print(f'  HGDO f mean:  fx={fx.mean():+.3f}  fy={fy.mean():+.3f} N')
print(f'\nInterpretation:')
print(f'  constant bias |b|: x={abs(bx_):.3f}, y={abs(by_):.3f} N  → CoM/wind/thrust-asym')
print(f'  viscous slope |a|: x={abs(ax_):.3f}, y={abs(ay_):.3f} N/(m/s)  '
      f'({"drag-like" if ax_<0 and ay_<0 else "not clean drag"})')
# Steady-state position error each component would cause via PD (Kp=2, m=3)
Kp = 2.0; m_d = 3.0
print(f'\n  Steady-state e from constant bias (e=b/(Kp·m)):  '
      f'x={bx_/(Kp*m_d)*100:+.1f} cm,  y={by_/(Kp*m_d)*100:+.1f} cm')

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, v, f, a, b, R2, name in [
        (axes[0], vw[:, 0], fx, ax_, bx_, R2x, 'x'),
        (axes[1], vw[:, 1], fy, ay_, by_, R2y, 'y')]:
    ax.scatter(v, f, s=5, alpha=0.3, c='steelblue')
    xs = np.linspace(v.min(), v.max(), 50)
    ax.plot(xs, a * xs + b, 'r-', lw=2,
            label=f'fit: f = {a:+.2f}·v {b:+.2f}\nR²={R2:.3f}')
    ax.axhline(b, color='g', ls='--', alpha=0.6, label=f'constant b = {b:+.2f} N')
    ax.axhline(0, color='k', alpha=0.3, lw=0.7); ax.axvline(0, color='k', alpha=0.3, lw=0.7)
    ax.set_xlabel(f'v_world_{name} [m/s]'); ax.set_ylabel(f'HGDO f_{name} [N]')
    ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f'HGDO f_{name}: viscous (slope) vs constant (intercept)')
fig.suptitle(f'{TAG} — HGDO force = viscous·v + constant', y=1.02)
plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_hgdo_viscous.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f'\nSaved: {out}')
