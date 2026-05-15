#!/usr/bin/env python3
"""Clean HGDO torque time-series + stats."""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return struct.unpack_from('<6d', blob, off)

def parse_odom_z(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    off += 16  # skip px, py
    pz = struct.unpack_from('<d', blob, off)[0]
    return pz


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

ot, oz = fetch('/mavros/local_position/odom', parse_odom_z)
ht, hg = fetch('/hgdo/wrench', parse_wrench)
con.close()

# Airborne window (z > 0.05 m sustained)
z_rel = oz - oz[ot < 5.0].mean() if (ot < 5.0).any() else oz
airborne = z_rel > 0.05
i_to = np.argmax(airborne); i_lo = len(airborne) - 1 - np.argmax(airborne[::-1])
t_to = ot[i_to]; t_land = ot[i_lo]
print(f'Airborne: {t_to:.1f} → {t_land:.1f} s')

# Crop HGDO to airborne (skip first/last 2s margin)
m = (ht >= t_to + 2.0) & (ht <= t_land - 1.0)
t_h = ht[m]
mx = hg[m, 3]; my = hg[m, 4]; mz = hg[m, 5]
fx = hg[m, 0]; fy = hg[m, 1]; fz = hg[m, 2]

# Static gravity-moment expectation from CoM offset (com_offset xy in yaml)
m_drone = 3.146; g = 9.81
W = m_drone * g
com_off = np.array([-0.01043, 0.00496])  # from yaml
# Static expected:
#   gravity gives -mg z_hat in world.  Body frame (small tilt): expected gravity moment about CoM = r_com × m·g (downward) in body
#   Body-frame: gravity_body = (0,0,-mg) roughly  → moment = r_com × gravity = (com_y*(-mg), -com_x*(-mg), 0) = (-com_y*mg, com_x*mg, 0)
expected_mx = -com_off[1] * W   # roll moment from y offset
expected_my =  com_off[0] * W   # pitch moment from x offset
print(f'\nFrom yaml com_offset = ({com_off[0]:+.4f}, {com_off[1]:+.4f}) m:')
print(f'  expected static HGDO Mx ≈ {expected_mx:+.3f} N·m  (from com_offset_y)')
print(f'  expected static HGDO My ≈ {expected_my:+.3f} N·m  (from com_offset_x)')
print(f'\nMeasured HGDO torque (airborne):')
print(f'  Mx:  mean={mx.mean():+.4f}  std={mx.std():.4f}  range=[{mx.min():+.3f}, {mx.max():+.3f}]')
print(f'  My:  mean={my.mean():+.4f}  std={my.std():.4f}  range=[{my.min():+.3f}, {my.max():+.3f}]')
print(f'  Mz:  mean={mz.mean():+.4f}  std={mz.std():.4f}  range=[{mz.min():+.3f}, {mz.max():+.3f}]')
print(f'\nMeasured HGDO force (airborne):')
print(f'  fx:  mean={fx.mean():+.4f}  std={fx.std():.4f}')
print(f'  fy:  mean={fy.mean():+.4f}  std={fy.std():.4f}')
print(f'  fz:  mean={fz.mean():+.4f}  std={fz.std():.4f}   →  mass mismatch ≈ {-fz.mean()/g:+.3f} kg')

# Inferred CoM offset from measured HGDO bias
# r_y_inferred = -mean_Mx / W
# r_x_inferred = mean_My / W
r_y_inf = -mx.mean() / W
r_x_inf =  my.mean() / W
print(f'\nInferred CoM offset from HGDO mean (assuming static gravity moment):')
print(f'  r_x_offset = {r_x_inf*1000:+.2f} mm    (yaml: {com_off[0]*1000:+.2f} mm)')
print(f'  r_y_offset = {r_y_inf*1000:+.2f} mm    (yaml: {com_off[1]*1000:+.2f} mm)')

fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

for ax, (data, name, exp) in zip(axes, [
        (mx, 'Mx (roll)',  expected_mx),
        (my, 'My (pitch)', expected_my),
        (mz, 'Mz (yaw)',   0.0)]):
    ax.plot(t_h, data, 'b', lw=0.8, alpha=0.8, label=f'HGDO {name}')
    ax.axhline(data.mean(), color='r', ls='--', alpha=0.7,
               label=f'mean = {data.mean():+.4f} N·m')
    ax.axhline(exp, color='g', ls=':', alpha=0.7,
               label=f'expected from CoM yaml = {exp:+.4f}')
    ax.axhline(0, color='k', alpha=0.3, lw=0.7)
    ax.set_ylabel(f'{name} [N·m]')
    ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=9)

axes[0].set_title(f'{TAG} — HGDO torque (airborne {t_to:.1f}→{t_land:.1f} s)')
axes[-1].set_xlabel('time [s]')
plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_hgdo_torque.png')
plt.savefig(out, dpi=120)
print(f'\nSaved: {out}')
