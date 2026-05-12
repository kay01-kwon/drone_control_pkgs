#!/usr/bin/env python3
"""Decompose the d/dt of desired roll/pitch into PD vs HGDO contributions.

  F_des_w = m·a_des_pd  +  (−R(q)·f_hgdo_body)
            └── PD ──┘     └─── HGDO comp ───┘

If R(q) varies with attitude (which it does), HGDO comp rotates with the
body even when f_hgdo_body is constant — generating high d/dt(desired)
even if HGDO output itself looks slow.

Usage:  python3 des_rp_rate_src.py [<bag_subdir> [<date_dir>]]
"""

import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG_SUBDIR = sys.argv[1] if len(sys.argv) > 1 else 'Qw_0p6'
DATE_DIR   = sys.argv[2] if len(sys.argv) > 2 else '2026_05_11_free_flight'
DB = glob.glob(os.path.join(_HERE, DATE_DIR, BAG_SUBDIR, '*.db3'))[0]
OUT_DIR = os.path.join(_HERE, DATE_DIR)
TAG = BAG_SUBDIR
print(f'Analyzing: {DB}')


def _align(off, n):
    return off + ((-(off - 4)) % n)


def parse_odom(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off)
    return np.array([qw, qx, qy, qz])


def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return np.array(struct.unpack_from('<6d', blob, off))


def quat_to_rpy(q):
    qw, qx, qy, qz = q
    roll  = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw   = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    return np.array([roll, pitch, yaw])


def quat_to_rotm(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx ** 2 + qy ** 2)]])


def force_to_rp(fx, fy, f_col, psi):
    fz_sq = f_col ** 2 - fx ** 2 - fy ** 2
    fz = np.sqrt(max(fz_sq, 0.0))
    n = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    if n < 1e-6: return 0.0, 0.0
    zb = np.array([fx, fy, fz]) / n
    xc = np.array([np.cos(psi), np.sin(psi), 0.0])
    yb = np.cross(zb, xc)
    yn = np.linalg.norm(yb)
    yb = (np.array([-np.sin(psi), np.cos(psi), 0.0]) if yn < 1e-6 else yb / yn)
    xb = np.cross(yb, zb)
    R = np.column_stack((xb, yb, zb))
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    return roll, pitch


conn = sqlite3.connect(DB)
c = conn.cursor()
tid = {n: i for i, n in c.execute('SELECT id, name FROM topics').fetchall()}

def fetch(topic, parser):
    rows = c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
                     (tid[topic],)).fetchall()
    ts = np.array([r[0] for r in rows], dtype=np.float64)
    dat = np.array([parser(bytes(r[1])) for r in rows])
    return ts, dat

odom_ts, q_arr = fetch('/mavros/local_position/odom', parse_odom)
ctrl_ts, ctrl  = fetch('/nmpc/control', parse_wrench)
hgdo_ts, hgdo  = fetch('/hgdo/wrench',  parse_wrench)
conn.close()

t0 = min(odom_ts[0], ctrl_ts[0], hgdo_ts[0])
odom_t = (odom_ts - t0) * 1e-9
ctrl_t = (ctrl_ts - t0) * 1e-9
hgdo_t = (hgdo_ts - t0) * 1e-9
rpy = np.array([quat_to_rpy(q_arr[i]) for i in range(len(q_arr))])

# At each ctrl timestamp: total des_rp, PD-only des_rp, HGDO-only des_rp
roll_at_ctrl  = np.interp(ctrl_t, odom_t, rpy[:, 0])
pit_at_ctrl   = np.interp(ctrl_t, odom_t, rpy[:, 1])
psi_at_ctrl   = np.interp(ctrl_t, odom_t, np.unwrap(rpy[:, 2]))
hgdo_at_ctrl  = np.column_stack([np.interp(ctrl_t, hgdo_t, hgdo[:, k]) for k in range(3)])

# Build R(q) at each ctrl timestamp (using actual rpy at that time)
def rpy_to_rotm(r, p, y):
    cr, sr = np.cos(r), np.sin(r); cp, sp = np.cos(p), np.sin(p); cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]])

hgdo_w = np.zeros((len(ctrl_t), 3))
for i in range(len(ctrl_t)):
    R_wb = rpy_to_rotm(roll_at_ctrl[i], pit_at_ctrl[i], psi_at_ctrl[i])
    hgdo_w[i] = R_wb @ hgdo_at_ctrl[i]

# Total des = from /nmpc/control fx, fy, f_col (post-DOB)
# PD-only des = total + HGDO compensation back = (fx + hgdo_w_x, fy + hgdo_w_y, ...)
# HGDO-only des = uses just -hgdo_w_xy with same f_col
fz_sq = ctrl[:, 2] ** 2 - ctrl[:, 0] ** 2 - ctrl[:, 1] ** 2
Fdes_z = np.sqrt(np.maximum(fz_sq, 0.0))
Fdes_w = np.column_stack([ctrl[:, 0], ctrl[:, 1], Fdes_z])
Fpd_w  = Fdes_w + hgdo_w
fcol_pd = np.linalg.norm(Fpd_w, axis=1)

des_total = np.zeros((len(ctrl_t), 2))
des_pd    = np.zeros((len(ctrl_t), 2))
des_hgdo  = np.zeros((len(ctrl_t), 2))
for i in range(len(ctrl_t)):
    des_total[i] = force_to_rp(ctrl[i, 0],  ctrl[i, 1],  ctrl[i, 2], psi_at_ctrl[i])
    des_pd[i]    = force_to_rp(Fpd_w[i, 0], Fpd_w[i, 1], fcol_pd[i], psi_at_ctrl[i])
    des_hgdo[i]  = force_to_rp(-hgdo_w[i, 0], -hgdo_w[i, 1], ctrl[i, 2], psi_at_ctrl[i])

mask = ctrl_t > 5.0
t_c = ctrl_t[mask]
def smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, mode='same')
def rate(x):
    xs = smooth(x, 5)
    return np.gradient(xs, t_c)

rate_total_r = rate(des_total[mask, 0])
rate_total_p = rate(des_total[mask, 1])
rate_pd_r    = rate(des_pd   [mask, 0])
rate_pd_p    = rate(des_pd   [mask, 1])
rate_hgdo_r  = rate(des_hgdo [mask, 0])
rate_hgdo_p  = rate(des_hgdo [mask, 1])


def stat(name, x_radps):
    deg = np.degrees(x_radps)
    print(f'  {name:25s} std={deg.std():7.2f}°/s  peak±{max(abs(deg.min()), abs(deg.max())):7.2f}°/s')

print('\n== d/dt desired RP decomposition ==')
print('-- Roll --')
stat('d/dt(des_total)', rate_total_r)
stat('d/dt(des_PD)',    rate_pd_r)
stat('d/dt(des_HGDO)',  rate_hgdo_r)
print('-- Pitch --')
stat('d/dt(des_total)', rate_total_p)
stat('d/dt(des_PD)',    rate_pd_p)
stat('d/dt(des_HGDO)',  rate_hgdo_p)

# ratio
print('\n-- HGDO share of rate variance --')
for ax, rh, rt in [('Roll', rate_hgdo_r, rate_total_r),
                   ('Pitch', rate_hgdo_p, rate_total_p)]:
    print(f'  {ax:5s} std(d/dt HGDO)/std(d/dt total) = {rh.std()/max(rt.std(),1e-9):5.2f}')


# ── Plots: per axis, overlay total / PD / HGDO derivatives ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(t_c, np.degrees(rate_total_r), 'k', lw=1.0, label='d/dt total')
axes[0].plot(t_c, np.degrees(rate_pd_r),    'b', lw=0.9, alpha=0.85, label='d/dt PD only')
axes[0].plot(t_c, np.degrees(rate_hgdo_r),  'r', lw=0.9, alpha=0.85, label='d/dt HGDO only')
axes[0].set_ylabel('d/dt(Roll des) [deg/s]'); axes[0].grid(alpha=0.3)
axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_title(f'{TAG}  —  d/dt of desired roll: PD vs HGDO contributions')

axes[1].plot(t_c, np.degrees(rate_total_p), 'k', lw=1.0, label='d/dt total')
axes[1].plot(t_c, np.degrees(rate_pd_p),    'b', lw=0.9, alpha=0.85, label='d/dt PD only')
axes[1].plot(t_c, np.degrees(rate_hgdo_p),  'r', lw=0.9, alpha=0.85, label='d/dt HGDO only')
axes[1].set_ylabel('d/dt(Pitch des) [deg/s]'); axes[1].set_xlabel('Time [s]')
axes[1].grid(alpha=0.3); axes[1].legend(loc='upper right', fontsize=9)
axes[1].set_title('d/dt of desired pitch: PD vs HGDO contributions')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_des_rp_rate_src.png'), dpi=120)
plt.close()

print(f'\nSaved: {TAG}_des_rp_rate_src.png')
