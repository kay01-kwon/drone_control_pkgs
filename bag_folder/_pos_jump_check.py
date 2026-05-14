#!/usr/bin/env python3
"""Check for position jumps (mocap glitches) vs mavros-odom-filtered jumps.

Compares:
  (1) /S550/pose                 — raw mocap pose
  (2) /mavros/local_position/odom — EKF2-fused pose
And looks for high-rate spikes in dp/dt that suggest mocap dropout/jump.

Usage:
  python3 _pos_jump_check.py <bag_subdir> [<date>] [<tag>]
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


def parse_pose(blob):
    """geometry_msgs/PoseStamped"""
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off)
    return px, py, pz


def parse_odom(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    px, py, pz = struct.unpack_from('<3d', blob, off)
    return px, py, pz


con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("SELECT id,name FROM topics")
tids = {n: i for i, n in cur.fetchall()}

tid = tids['/mavros/local_position/odom']
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tid}")
t0 = cur.fetchone()[0]


def fetch(topic, parser):
    if topic not in tids: return None, None
    tid = tids[topic]
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tid} ORDER BY timestamp")
    rows = cur.fetchall()
    t = np.array([(ts - t0) * 1e-9 for ts, _ in rows])
    d = np.array([parser(blob) for _, blob in rows])
    return t, d


ot, odom_p = fetch('/mavros/local_position/odom', parse_odom)
mt, moc_p  = fetch('/S550/pose', parse_pose)
con.close()

# Numerical derivatives (instantaneous "velocity" between consecutive samples)
def diff_p(t, p):
    dt = np.diff(t)
    dp = np.diff(p, axis=0)
    # Reject samples with dt <= 0 or weird
    dt = np.where(dt > 1e-6, dt, 1e-3)
    return t[:-1] + dt / 2, dp / dt[:, None]


ot_d, odom_v = diff_p(ot, odom_p)
mt_d, moc_v  = diff_p(mt, moc_p) if mt is not None else (None, None)

# Stats
print(f'Bag: {BAG}')
print(f'\nOdom samples: {len(ot)} over {ot[-1]-ot[0]:.1f}s  (dt med = {np.median(np.diff(ot))*1000:.1f} ms)')
if mt is not None:
    print(f'Mocap samples: {len(mt)} over {mt[-1]-mt[0]:.1f}s  (dt med = {np.median(np.diff(mt))*1000:.1f} ms)')

# Find big jumps: |Δp| in single sample > threshold
def find_jumps(t, p, thr_m=0.05):
    """Return (time, axis, magnitude) of big single-sample jumps."""
    dp = np.diff(p, axis=0)
    mag = np.linalg.norm(dp, axis=1)
    idx = np.where(mag > thr_m)[0]
    return [(t[i+1], dp[i], mag[i]) for i in idx]


odom_jumps = find_jumps(ot, odom_p, thr_m=0.05)
moc_jumps  = find_jumps(mt, moc_p, thr_m=0.05) if mt is not None else []

print(f'\nOdom single-sample jumps > 5 cm: {len(odom_jumps)}')
for tt, dp, mag in odom_jumps[:20]:
    print(f'  t={tt:.2f}s  dp=({dp[0]:+.3f}, {dp[1]:+.3f}, {dp[2]:+.3f})  |dp|={mag*100:.1f} cm')

print(f'\nMocap single-sample jumps > 5 cm: {len(moc_jumps)}')
for tt, dp, mag in moc_jumps[:20]:
    print(f'  t={tt:.2f}s  dp=({dp[0]:+.3f}, {dp[1]:+.3f}, {dp[2]:+.3f})  |dp|={mag*100:.1f} cm')

# Velocity max (effective)
print(f'\nMax |v| from differentiation (m/s):')
print(f'  odom:  vx={np.max(np.abs(odom_v[:,0])):.2f}  vy={np.max(np.abs(odom_v[:,1])):.2f}  vz={np.max(np.abs(odom_v[:,2])):.2f}')
if mt is not None:
    print(f'  mocap: vx={np.max(np.abs(moc_v[:,0])):.2f}  vy={np.max(np.abs(moc_v[:,1])):.2f}  vz={np.max(np.abs(moc_v[:,2])):.2f}')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
labels = ['x', 'y', 'z']
for k in range(3):
    ax = axes[k]
    ax.plot(ot, odom_p[:, k], 'b',  lw=0.8, alpha=0.8, label='odom (EKF2)')
    if mt is not None:
        ax.plot(mt, moc_p[:, k], 'r--', lw=0.6, alpha=0.7, label='mocap raw')
    # Mark jumps
    for tt, dp, mag in odom_jumps:
        ax.axvline(tt, color='b', alpha=0.15, lw=0.5)
    for tt, dp, mag in moc_jumps:
        ax.axvline(tt, color='r', alpha=0.15, lw=0.5)
    ax.set_ylabel(f'{labels[k]} [m]')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
axes[0].set_title(f'{TAG} — position: mocap raw vs mavros odom  (vertical lines = jumps > 5 cm)')
axes[-1].set_xlabel('time [s]')
plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_pos_jump_check.png')
plt.savefig(out, dpi=120)
print(f'\nSaved: {out}')
