#!/usr/bin/env python3
"""Zoom in on z position around takeoff."""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
DATE = sys.argv[2] if len(sys.argv) > 2 else '2026_05_14_free_flight'
BAG_DIR = os.path.join(_HERE, DATE, BAG)
db = glob.glob(os.path.join(BAG_DIR, '*.db3'))[0]


def _align(off, n):
    rel = off - 4
    pad = (-rel) % n
    return off + pad


def parse_odom(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4
    off += slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4
    off += slen2
    off = _align(off, 8)
    px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
    off += 36 * 8
    vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
    return px, py, pz, vx, vy, vz


def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4
    off += slen
    off = _align(off, 8)
    return struct.unpack_from('<6d', blob, off)


def parse_rpm(blob):
    # HexaActualRpm: 6 fixed-size float32 (no length prefix)
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4
    off += slen
    off = _align(off, 4)
    rpms = struct.unpack_from('<6I', blob, off)
    return list(rpms)


con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("SELECT id,name,type FROM topics")
topics = {n: i for i, n, t in cur.fetchall()}

# odom
tid = topics['/mavros/local_position/odom']
cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tid} ORDER BY timestamp")
odom_t, odom_z, odom_vz = [], [], []
for ts, blob in cur.fetchall():
    px, py, pz, vx, vy, vz = parse_odom(blob)
    odom_t.append(ts); odom_z.append(pz); odom_vz.append(vz)
odom_t = np.array(odom_t) * 1e-9
odom_t -= odom_t[0]
odom_z = np.array(odom_z); odom_vz = np.array(odom_vz)

# wrench (HGDO)
tid = topics['/hgdo/wrench']
cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tid} ORDER BY timestamp")
hg_t, hg_fz = [], []
for ts, blob in cur.fetchall():
    fx, fy, fz, mx, my, mz = parse_wrench(blob)
    hg_t.append(ts); hg_fz.append(fz)
hg_t = (np.array(hg_t) * 1e-9) - odom_t[0] - 0  # already subtracted
# actually fix: use raw timestamps
con.close()

# Recompute properly: subtract first odom's raw time.
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
    out = []
    for ts, blob in cur.fetchall():
        out.append((ts - t0) * 1e-9, *parser(blob)) if False else out.append(((ts - t0) * 1e-9, parser(blob)))
    return out

odom = fetch('/mavros/local_position/odom', parse_odom)
wrench = fetch('/hgdo/wrench', parse_wrench)
rpm = fetch('/uav/actual_rpm', parse_rpm)
con.close()

ot = np.array([x[0] for x in odom])
oz = np.array([x[1][2] for x in odom])
ovz = np.array([x[1][5] for x in odom])

wt = np.array([x[0] for x in wrench])
wfz = np.array([x[1][2] for x in wrench])

rt = np.array([x[0] for x in rpm])
rrpm = np.array([x[1] for x in rpm])

# Detect takeoff: motor mean RPM transition from low (idle) to high
mean_rpm = rrpm.mean(axis=1)
# find first big jump - look for first time RPM crosses 5500 (well above any idle)
i_to = np.argmax(mean_rpm > 5500)
t_takeoff = rt[i_to] if i_to > 0 else rt[0]
# z reference (ground z): mean z before takeoff
mask_ground = ot < t_takeoff
z0 = oz[mask_ground].mean() if mask_ground.sum() > 5 else oz[0]
oz_rel = oz - z0
print(f'Takeoff detected at t={t_takeoff:.2f}s (motor RPM > 5500)')
print(f'Ground z (pre-takeoff mean): {z0:.3f} m')
print(f'Total flight time: {ot[-1]:.1f}s')

# Z stats AFTER takeoff
mask = ot > t_takeoff
print(f'\nZ position post-takeoff:')
print(f'  z at takeoff:      {oz_rel[mask][0]:.3f} m  (abs={oz[mask][0]:.3f})')
print(f'  z at takeoff+1s:   {oz_rel[(ot>t_takeoff+1.0)&(ot<t_takeoff+1.05)].mean():.3f} m')
print(f'  z at takeoff+2s:   {oz_rel[(ot>t_takeoff+2.0)&(ot<t_takeoff+2.05)].mean():.3f} m')
print(f'  z at takeoff+5s:   {oz_rel[(ot>t_takeoff+5.0)&(ot<t_takeoff+5.05)].mean():.3f} m')
print(f'  z at takeoff+10s:  {oz_rel[(ot>t_takeoff+10.0)&(ot<t_takeoff+10.05)].mean():.3f} m')
print(f'  z at takeoff+20s:  {oz_rel[(ot>t_takeoff+20.0)&(ot<t_takeoff+20.05)].mean():.3f} m')

# Find when z first exceeds 0.1m, 0.15m, 0.2m
def first_t_above(z_thr):
    cond = (ot > t_takeoff) & (oz_rel > z_thr)
    if not cond.any(): return None
    idx = np.argmax(cond)
    return ot[idx] - t_takeoff

print(f'  first z > 0.10 m:  +{first_t_above(0.10):.2f} s after takeoff' if first_t_above(0.10) else '  z never exceeded 0.10 m')
print(f'  first z > 0.15 m:  +{first_t_above(0.15):.2f} s after takeoff' if first_t_above(0.15) else '  z never exceeded 0.15 m')
print(f'  first z > 0.20 m:  +{first_t_above(0.20):.2f} s after takeoff' if first_t_above(0.20) else '  z never exceeded 0.20 m')

# Plot zoomed
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
t_lo = t_takeoff - 2; t_hi = t_takeoff + 25
m1 = (ot >= t_lo) & (ot <= t_hi)
m2 = (rt >= t_lo) & (rt <= t_hi)
m3 = (wt >= t_lo) & (wt <= t_hi)

axes[0].plot(ot[m1], oz_rel[m1], 'b'); axes[0].axhline(0.10, color='k', ls='--', alpha=0.4, label='0.10 m')
axes[0].axhline(0.20, color='k', ls=':', alpha=0.4, label='0.20 m')
axes[0].axvline(t_takeoff, color='r', alpha=0.5)
axes[0].set_ylabel('z rel [m]'); axes[0].grid(alpha=0.3); axes[0].legend()
axes[0].set_title(f'{BAG} — takeoff zoom (red line = motor spool-up)')

axes[1].plot(ot[m1], ovz[m1], 'b')
axes[1].axvline(t_takeoff, color='r', alpha=0.5)
axes[1].set_ylabel('vz [m/s]'); axes[1].grid(alpha=0.3)

axes[2].plot(rt[m2], mean_rpm[m2], 'g')
axes[2].axvline(t_takeoff, color='r', alpha=0.5)
axes[2].set_ylabel('mean RPM'); axes[2].grid(alpha=0.3)

axes[3].plot(wt[m3], wfz[m3], 'm')
axes[3].axvline(t_takeoff, color='r', alpha=0.5)
axes[3].set_ylabel('HGDO fz [N]'); axes[3].set_xlabel('time [s]'); axes[3].grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(_HERE, DATE, *BAG.split('/')[:-1], f'{BAG.split("/")[-1]}_z_takeoff_zoom.png')
plt.savefig(out, dpi=120)
print(f'\nSaved: {out}')
