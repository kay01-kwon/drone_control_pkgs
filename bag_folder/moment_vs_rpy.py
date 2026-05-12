#!/usr/bin/env python3
"""Overlay motor-commanded moments with actual RPY (axis-aligned).
Shows directly whether the commanded torque maps cleanly to the
expected attitude angle response, and lets us spot yaw drift / coupling.

Usage:  python3 moment_vs_rpy.py [<bag_subdir> [<date_dir>]]
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
    off += 32 + 36 * 8 + 24
    wx, wy, wz = struct.unpack_from('<3d', blob, off)
    return np.array([qw, qx, qy, qz, wx, wy, wz])

def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return np.array(struct.unpack_from('<6d', blob, off))


def parse_pose_p(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    px, py, pz = struct.unpack_from('<3d', blob, off)
    return np.array([px, py, pz])


def quat_to_rpy(q):
    qw, qx, qy, qz = q
    roll  = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw   = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    return np.array([roll, pitch, yaw])


conn = sqlite3.connect(DB)
c = conn.cursor()
tid = {n: i for i, n in c.execute('SELECT id, name FROM topics').fetchall()}

def fetch(topic, parser):
    rows = c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
                     (tid[topic],)).fetchall()
    ts = np.array([r[0] for r in rows], dtype=np.float64)
    dat = np.array([parser(bytes(r[1])) for r in rows])
    return ts, dat

odom_ts, odom = fetch('/mavros/local_position/odom', parse_odom)
ctrl_ts, ctrl = fetch('/nmpc/control', parse_wrench)
pos_ts = None
if '/S550/pose' in tid:
    pos_ts, mocap_p = fetch('/S550/pose', parse_pose_p)
conn.close()

t0 = min(odom_ts[0], ctrl_ts[0])
odom_t = (odom_ts - t0) * 1e-9
ctrl_t = (ctrl_ts - t0) * 1e-9
pos_t  = (pos_ts  - t0) * 1e-9 if pos_ts is not None else None

rpy = np.array([quat_to_rpy(odom[i, 0:4]) for i in range(len(odom))])
rpy_unwrap = np.column_stack([rpy[:, 0], rpy[:, 1], np.unwrap(rpy[:, 2])])
w = odom[:, 4:7]


# ── Stats over flight segment (skip first 5 s) ──
mask = odom_t > 5.0
mask_c = ctrl_t > 5.0
print(f'\n== {TAG}  —  RPY and ω stats (airborne) ==')
for k, ax in enumerate(['roll', 'pitch', 'yaw']):
    deg = np.degrees(rpy_unwrap[mask, k])
    print(f'  {ax:5s}  mean = {deg.mean():+7.3f}°   std = {deg.std():6.3f}°   '
          f'range [{deg.min():+7.2f}, {deg.max():+7.2f}]')
print('-- body angular velocity ω --')
for k, ax in enumerate(['wx', 'wy', 'wz']):
    val = np.degrees(w[mask, k])
    print(f'  {ax:5s}  mean = {val.mean():+7.3f}°/s   std = {val.std():6.3f}°/s')
print('-- /nmpc/control torque (motor cmd, post-DOB) --')
for k, ax in enumerate(['Mx', 'My', 'Mz']):
    val = ctrl[mask_c, 3 + k]
    print(f'  {ax:5s}  mean = {val.mean():+8.4f} N·m   std = {val.std():8.4f}   '
          f'range [{val.min():+8.4f}, {val.max():+8.4f}]')


# ── 3-row plot: Mx↔Roll, My↔Pitch, Mz↔Yaw with twin axes ──
fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
labels = [('Mx', 'Roll', 0), ('My', 'Pitch', 1), ('Mz', 'Yaw', 2)]
for k, (Mname, rname, idx) in enumerate(labels):
    ax_M = axes[k]
    ax_R = ax_M.twinx()
    l1, = ax_M.plot(ctrl_t, ctrl[:, 3 + idx], 'b', lw=1.0, alpha=0.8, label=f'{Mname} motor cmd')
    l2, = ax_R.plot(odom_t, np.degrees(rpy_unwrap[:, idx]), 'r', lw=1.1, alpha=0.85, label=f'{rname} actual')
    ax_M.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax_R.axhline(0, color='gray', lw=0.5, alpha=0.5, ls='--')
    ax_M.set_ylabel(f'{Mname} [N·m]', color='b')
    ax_R.set_ylabel(f'{rname} [deg]', color='r')
    ax_M.tick_params(axis='y', labelcolor='b')
    ax_R.tick_params(axis='y', labelcolor='r')
    ax_M.grid(alpha=0.3)
    ax_M.legend([l1, l2], [l1.get_label(), l2.get_label()], loc='upper right', fontsize=9)
    ax_M.set_title(f'{Mname} motor cmd  vs  {rname} actual')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_moment_vs_rpy.png'), dpi=120)
plt.close()


# ── Yaw vs XY position drift (if mocap available) ──
if pos_t is not None:
    p_off = mocap_p[(pos_t > 5.0)].mean(axis=0)
    p_zero = mocap_p - p_off

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(pos_t, p_zero[:, 0], 'r', lw=0.9, label='x')
    axes[0].plot(pos_t, p_zero[:, 1], 'g', lw=0.9, label='y')
    axes[0].set_ylabel('Position [m]'); axes[0].grid(alpha=0.3); axes[0].legend(loc='upper right')
    axes[0].set_title('Mocap XY position drift')

    axes[1].plot(odom_t, np.degrees(rpy_unwrap[:, 2]), 'b', lw=1.0)
    axes[1].axhline(np.degrees(rpy_unwrap[mask, 2]).mean(), color='gray', ls='--', alpha=0.5)
    axes[1].set_ylabel('Yaw [deg]'); axes[1].grid(alpha=0.3)
    axes[1].set_title('Yaw actual (drift / oscillation)')

    axes[2].plot(ctrl_t, ctrl[:, 5], 'b', lw=0.9)
    axes[2].axhline(0, color='gray', lw=0.5, alpha=0.5)
    axes[2].set_ylabel('Mz [N·m]'); axes[2].set_xlabel('Time [s]')
    axes[2].grid(alpha=0.3)
    axes[2].set_title('Mz motor cmd')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{TAG}_yaw_position_link.png'), dpi=120)
    plt.close()

print(f'\nSaved:')
print(f'  - {TAG}_moment_vs_rpy.png')
if pos_t is not None: print(f'  - {TAG}_yaw_position_link.png')
