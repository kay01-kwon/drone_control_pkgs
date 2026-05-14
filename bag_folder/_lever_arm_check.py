#!/usr/bin/env python3
"""Verify the lever-arm hypothesis: linear velocity at measurement point
is contaminated by ω × r_offset.

Test:  v_x_world  =  v_x_CoM  +  ω_y_world · r_z
       v_y_world  =  v_y_CoM  -  ω_x_world · r_z

Linear regression of v_xy on ω_yx yields the effective r_z offset; the
R² of the fit tells you how much of the apparent linear-velocity
fluctuation is rotational artifact.

Usage:
  python3 _lever_arm_check.py <bag_subdir> <t_center> [<win_s>] [<date>] [<tag>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    wx, wy, wz = struct.unpack_from('<3d', blob, off); off += 24
    return px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz


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
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tid}")
t0 = cur.fetchone()[0]
cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tid} ORDER BY timestamp")
rows = cur.fetchall()
con.close()

t_all = np.array([(ts - t0) * 1e-9 for ts, _ in rows])
data = np.array([parse_odom(blob) for _, blob in rows])

# Crop window
m = (t_all >= T_CENTER - WIN / 2) & (t_all <= T_CENTER + WIN / 2)
t = t_all[m]
v_body = data[m, 7:10]
w_body = data[m, 10:13]
qs = data[m, 3:7]

# Transform to world
N = len(t)
v_world = np.empty_like(v_body)
w_world = np.empty_like(w_body)
for i in range(N):
    R = quat_to_rotm(qs[i, 0], qs[i, 1], qs[i, 2], qs[i, 3])
    v_world[i] = R @ v_body[i]
    w_world[i] = R @ w_body[i]

# Pull out
vx = v_world[:, 0]; vy = v_world[:, 1]
wx = w_world[:, 0]; wy = w_world[:, 1]


def regression(x, y):
    """y ≈ a·x + b ; return (a, b, R²)"""
    x = x - x.mean(); y0 = y - y.mean()
    a = (x * y0).sum() / (x * x).sum()
    pred = a * x
    ss_res = ((y0 - pred) ** 2).sum()
    ss_tot = (y0 ** 2).sum()
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
    return a, y.mean() - a * (x.mean() + x.mean()), R2  # intercept w.r.t. raw scale


# v_x = v_x_CoM + ω_y · r_z   →   regress vx on wy, slope = r_z
r_z_x, _, R2_x = regression(wy, vx)
r_z_y, _, R2_y = regression(-wx, vy)  # v_y = v_y_CoM - ω_x · r_z  →  slope of vy on (-ω_x) = r_z

# Residual = "true" v_CoM estimate
vx_corrected = vx - r_z_x * wy
vy_corrected = vy - (-r_z_y) * (-wx)   # equivalent to vy + r_z_y · wx

print(f'Window: {t[0]:.1f} → {t[-1]:.1f} s   ({len(t)} samples)')
print(f'\n--- v_x_world vs ω_y_world (slope = r_z) ---')
print(f'  fitted r_z = {r_z_x*100:+.2f} cm    R² = {R2_x:.3f}')
print(f'  v_x std raw:        {vx.std():.3f} m/s')
print(f'  v_x std corrected:  {vx_corrected.std():.3f} m/s   (reduction {100*(1-vx_corrected.std()/vx.std()):.1f}%)')

print(f'\n--- v_y_world vs -ω_x_world (slope = r_z) ---')
print(f'  fitted r_z = {r_z_y*100:+.2f} cm    R² = {R2_y:.3f}')
print(f'  v_y std raw:        {vy.std():.3f} m/s')
print(f'  v_y std corrected:  {vy_corrected.std():.3f} m/s   (reduction {100*(1-vy_corrected.std()/vy.std()):.1f}%)')

print(f'\nMean r_z estimate: {0.5*(r_z_x+r_z_y)*100:+.2f} cm')


def align_zero(ax1, ax2):
    A = max(abs(ax1.get_ylim()[0]), abs(ax1.get_ylim()[1]))
    B = max(abs(ax2.get_ylim()[0]), abs(ax2.get_ylim()[1]))
    ax1.set_ylim(-A, A); ax2.set_ylim(-B, B)


fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

# Time series 1: vx and ω_y · r_z prediction
ax = axes[0]
ax.plot(t, vx, 'r', lw=1.5, label='v_x_world (measured)')
ax.plot(t, r_z_x * wy, 'b--', lw=1.2, label=f'ω_y · r_z  (r_z = {r_z_x*100:+.1f} cm)')
ax.plot(t, vx_corrected, 'g', lw=1.2, alpha=0.8, label='v_x corrected (raw − ω_y·r_z)')
ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('v_x [m/s]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')
ax.set_title(f'{TAG} — Lever-arm test  '
             f'(r_z_x = {r_z_x*100:+.1f} cm, R²={R2_x:.3f};  r_z_y = {r_z_y*100:+.1f} cm, R²={R2_y:.3f})')

# Time series 2: vy and -ω_x · r_z prediction
ax = axes[1]
ax.plot(t, vy, 'r', lw=1.5, label='v_y_world (measured)')
ax.plot(t, -r_z_y * wx, 'b--', lw=1.2, label=f'-ω_x · r_z  (r_z = {r_z_y*100:+.1f} cm)')
ax.plot(t, vy_corrected, 'g', lw=1.2, alpha=0.8, label='v_y corrected')
ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('v_y [m/s]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')

# Scatter 1: vx vs ω_y
ax = axes[2]
ax.scatter(wy, vx, s=8, alpha=0.5, c='steelblue')
xs = np.linspace(wy.min(), wy.max(), 50)
ax.plot(xs, r_z_x * xs, 'r--', lw=1.5,
        label=f'fit: v_x = {r_z_x:+.3f}·ω_y    R²={R2_x:.3f}')
ax.set_xlabel('ω_y world [rad/s]'); ax.set_ylabel('v_x world [m/s]')
ax.grid(alpha=0.3); ax.legend()
ax.set_title('Scatter: v_x vs ω_y  →  slope = r_z')

# Scatter 2: vy vs -ω_x
ax = axes[3]
ax.scatter(-wx, vy, s=8, alpha=0.5, c='steelblue')
xs = np.linspace(-wx.max(), -wx.min(), 50)
ax.plot(xs, r_z_y * xs, 'r--', lw=1.5,
        label=f'fit: v_y = {r_z_y:+.3f}·(-ω_x)    R²={R2_y:.3f}')
ax.set_xlabel('-ω_x world [rad/s]'); ax.set_ylabel('v_y world [m/s]')
ax.grid(alpha=0.3); ax.legend()
ax.set_title('Scatter: v_y vs -ω_x  →  slope = r_z')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_lever_arm_t{int(T_CENTER)}.png')
plt.savefig(out, dpi=120)
print(f'\nSaved: {out}')
