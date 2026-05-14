#!/usr/bin/env python3
"""Zoom around position-burst event with all key signals overlaid.
Usage: python3 _burst_zoom.py <bag_subdir> <t_center> [<window_sec>] [<date_dir>]
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
BAG_DIR = os.path.join(_HERE, DATE, BAG)
db = glob.glob(os.path.join(BAG_DIR, '*.db3'))[0]
parts = BAG.split('/')
OUT_DIR = os.path.join(_HERE, DATE, *parts[:-1])
TAG = parts[-1]


def _align(off, n):
    rel = off - 4
    return off + (-rel) % n


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


def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return struct.unpack_from('<6d', blob, off)


def parse_rpm(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    return list(struct.unpack_from('<6I', blob, off))


def quat_to_rpy(qw, qx, qy, qz):
    r = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    sp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
    p = np.arcsin(sp)
    y = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    return r, p, y


def force_to_des_rp(fx, fy, fz, psi):
    n = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    if n < 1e-6: return 0.0, 0.0
    zb = np.array([fx, fy, fz]) / n
    xc = np.array([np.cos(psi), np.sin(psi), 0.0])
    yb = np.cross(zb, xc)
    yb_n = np.linalg.norm(yb)
    yb = yb / yb_n if yb_n > 1e-6 else np.array([-np.sin(psi), np.cos(psi), 0.0])
    xb = np.cross(yb, zb)
    R = np.column_stack((xb, yb, zb))
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-R[2, 0])
    return roll, pitch


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
ct, ctrl = fetch('/nmpc/control', parse_wrench)
ht, hg = fetch('/hgdo/wrench', parse_wrench)
rt, rpm_act = fetch('/uav/actual_rpm', parse_rpm)
con.close()

# Cropping window
t_lo = T_CENTER - WIN / 2
t_hi = T_CENTER + WIN / 2
mO = (ot >= t_lo) & (ot <= t_hi)
mC = (ct >= t_lo) & (ct <= t_hi)
mH = (ht >= t_lo) & (ht <= t_hi)
mR = (rt >= t_lo) & (rt <= t_hi)

t_o = ot[mO]
p_x = odom[mO, 0]; p_y = odom[mO, 1]; p_z = odom[mO, 2]
v_x = odom[mO, 7]; v_y = odom[mO, 8]; v_z = odom[mO, 9]
w_x = odom[mO, 10]; w_y = odom[mO, 11]; w_z = odom[mO, 12]
# Actual RP
rp_act = np.array([quat_to_rpy(odom[i, 3], odom[i, 4], odom[i, 5], odom[i, 6])
                   for i in np.where(mO)[0]])
roll_act = np.rad2deg(rp_act[:, 0]); pitch_act = np.rad2deg(rp_act[:, 1])
yaw_act = np.rad2deg(rp_act[:, 2])

# Desired RP from /nmpc/control (force.xy world, force.z body) — same convention as analyze script
t_c = ct[mC]
fx_w = ctrl[mC, 0]; fy_w = ctrl[mC, 1]; f_col = ctrl[mC, 2]
Mx_cmd = ctrl[mC, 3]; My_cmd = ctrl[mC, 4]; Mz_cmd = ctrl[mC, 5]
psi_c = np.interp(t_c, t_o, np.unwrap(np.deg2rad(yaw_act)))
rp_des = np.array([force_to_des_rp(fx_w[i], fy_w[i],
                                    np.sqrt(max(f_col[i] ** 2 - fx_w[i] ** 2 - fy_w[i] ** 2, 0.0)),
                                    psi_c[i]) for i in range(len(t_c))])
roll_des = np.rad2deg(rp_des[:, 0]); pitch_des = np.rad2deg(rp_des[:, 1])

# HGDO
t_h = ht[mH]
hg_fx = hg[mH, 0]; hg_fy = hg[mH, 1]; hg_fz = hg[mH, 2]
hg_mx = hg[mH, 3]; hg_my = hg[mH, 4]; hg_mz = hg[mH, 5]

# Motor mean RPM
t_r = rt[mR]
mean_rpm = rpm_act[mR].mean(axis=1)

print(f'Zoom window: t={t_lo:.1f} → {t_hi:.1f} s')
print(f'Position range in window: x=[{p_x.min():.2f}, {p_x.max():.2f}]  y=[{p_y.min():.2f}, {p_y.max():.2f}]  z=[{p_z.min():.2f}, {p_z.max():.2f}]')
print(f'Roll range: [{roll_act.min():.1f}, {roll_act.max():.1f}] deg  std={roll_act.std():.2f}')
print(f'Pitch range: [{pitch_act.min():.1f}, {pitch_act.max():.1f}] deg  std={pitch_act.std():.2f}')
print(f'ω range: wx=[{w_x.min():.2f}, {w_x.max():.2f}]  wy=[{w_y.min():.2f}, {w_y.max():.2f}]  rad/s')

fig, axes = plt.subplots(7, 1, figsize=(13, 16), sharex=True)

axes[0].plot(t_o, p_x, 'r', label='x'); axes[0].plot(t_o, p_y, 'g', label='y'); axes[0].plot(t_o, p_z, 'b', label='z')
axes[0].set_ylabel('pos [m]'); axes[0].grid(alpha=0.3); axes[0].legend(loc='upper right')
axes[0].set_title(f'{TAG} — burst zoom around t={T_CENTER:.1f} s (±{WIN/2:.1f} s)')

axes[1].plot(t_o, v_x, 'r', label='vx'); axes[1].plot(t_o, v_y, 'g', label='vy'); axes[1].plot(t_o, v_z, 'b', label='vz')
axes[1].set_ylabel('v [m/s]'); axes[1].grid(alpha=0.3); axes[1].legend(loc='upper right')

axes[2].plot(t_o, roll_act, 'r-', alpha=0.6, label='roll act')
axes[2].plot(t_c, roll_des, 'r--', alpha=0.9, label='roll des')
axes[2].plot(t_o, pitch_act, 'g-', alpha=0.6, label='pitch act')
axes[2].plot(t_c, pitch_des, 'g--', alpha=0.9, label='pitch des')
axes[2].set_ylabel('roll/pitch [deg]'); axes[2].grid(alpha=0.3); axes[2].legend(ncol=2, loc='upper right')

axes[3].plot(t_o, np.rad2deg(w_x), 'r', label='ωx'); axes[3].plot(t_o, np.rad2deg(w_y), 'g', label='ωy'); axes[3].plot(t_o, np.rad2deg(w_z), 'b', label='ωz')
axes[3].set_ylabel('ω [deg/s]'); axes[3].grid(alpha=0.3); axes[3].legend(loc='upper right')

axes[4].plot(t_h, hg_fx, 'r', label='HGDO fx'); axes[4].plot(t_h, hg_fy, 'g', label='HGDO fy'); axes[4].plot(t_h, hg_fz, 'b', label='HGDO fz')
axes[4].set_ylabel('HGDO force [N]'); axes[4].grid(alpha=0.3); axes[4].legend(loc='upper right')

axes[5].plot(t_h, hg_mx, 'r', label='HGDO Mx'); axes[5].plot(t_h, hg_my, 'g', label='HGDO My'); axes[5].plot(t_h, hg_mz, 'b', label='HGDO Mz')
axes[5].plot(t_c, Mx_cmd, 'r:', alpha=0.6, label='Mx cmd')
axes[5].plot(t_c, My_cmd, 'g:', alpha=0.6, label='My cmd')
axes[5].plot(t_c, Mz_cmd, 'b:', alpha=0.6, label='Mz cmd')
axes[5].set_ylabel('M [N·m]'); axes[5].grid(alpha=0.3); axes[5].legend(loc='upper right', ncol=2, fontsize=8)

axes[6].plot(t_r, mean_rpm, 'm')
axes[6].set_ylabel('mean RPM'); axes[6].set_xlabel('time [s]'); axes[6].grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_burst_zoom_t{int(T_CENTER)}.png')
plt.savefig(out, dpi=120)
print(f'Saved: {out}')
