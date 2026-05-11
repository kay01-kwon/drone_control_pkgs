#!/usr/bin/env python3
"""Attitude-stand RPY analysis (ball-joint, attitude-only).

Compares actual RPY from IMU AHRS and EKF2 odom (mocap if present),
NMPC commanded torque, and HGDO disturbance torque.  The reference
attitude on a ball-joint stand is RPY = 0 — any persistent offset
points to a trim/bias problem that must be fixed before debugging
free-flight position control.

Usage:  python3 att_stand_rpy.py [<bag_subdir> [<date_dir>]]
"""

import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG_SUBDIR = sys.argv[1] if len(sys.argv) > 1 else 'Qw_0_6'
DATE_DIR   = sys.argv[2] if len(sys.argv) > 2 else '2026_04_30_att_data_set'
DB = glob.glob(os.path.join(_HERE, DATE_DIR, BAG_SUBDIR, '*.db3'))[0]
OUT_DIR = os.path.join(_HERE, DATE_DIR)
TAG = BAG_SUBDIR
print(f'Analyzing: {DB}')


def _align(off, n):
    return off + ((-(off - 4)) % n)


def parse_imu_quat(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off)
    return np.array([qw, qx, qy, qz])


def parse_pose_quat(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    _px, _py, _pz, qx, qy, qz, qw = struct.unpack_from('<7d', blob, off)
    return np.array([qw, qx, qy, qz])


def parse_odom_quat(blob):
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


# ── Load ──
conn = sqlite3.connect(DB)
c = conn.cursor()
tid = {n: i for i, n in c.execute('SELECT id, name FROM topics').fetchall()}

def fetch(topic, parser):
    rows = c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
                     (tid[topic],)).fetchall()
    ts = np.array([r[0] for r in rows], dtype=np.float64)
    dat = np.array([parser(bytes(r[1])) for r in rows])
    return ts, dat

imu_ts,  imu_q  = fetch('/mavros/imu/data',            parse_imu_quat)
odom_ts, odom_q = fetch('/mavros/local_position/odom', parse_odom_quat)
ctrl_ts, ctrl   = fetch('/nmpc/control',               parse_wrench)
hgdo_ts, hgdo   = fetch('/hgdo/wrench',                parse_wrench)
moc_ts = moc_q = None
if '/S550/pose' in tid:
    moc_ts, moc_q = fetch('/S550/pose', parse_pose_quat)
conn.close()

t0 = min(imu_ts[0], odom_ts[0], ctrl_ts[0], hgdo_ts[0])
imu_t  = (imu_ts  - t0) * 1e-9
odom_t = (odom_ts - t0) * 1e-9
ctrl_t = (ctrl_ts - t0) * 1e-9
hgdo_t = (hgdo_ts - t0) * 1e-9
moc_t  = (moc_ts  - t0) * 1e-9 if moc_ts is not None else None

imu_rpy  = np.array([quat_to_rpy(imu_q[i])  for i in range(len(imu_q))])
odom_rpy = np.array([quat_to_rpy(odom_q[i]) for i in range(len(odom_q))])
moc_rpy  = (np.array([quat_to_rpy(moc_q[i]) for i in range(len(moc_q))])
            if moc_q is not None else None)


# ── Steady-state window selection ──
# Use a stationary segment in the middle of the flight, away from start/landing.
# Pick the segment with lowest IMU angular rate variance.
def airborne_window(t, rpy, win_s=20.0):
    """Find the longest contiguous window with rpy std < 5 deg (approximately stationary)."""
    # Compute rolling std with 1s window
    dt = np.diff(t).mean()
    win = max(int(1.0 / dt), 5)
    rstd = np.array([rpy[max(0, i - win):i + win, 0].std() for i in range(len(t))])
    # Find longest contiguous segment with rstd < 10 deg/s (in radians, 0.175)
    quiet = rstd < np.radians(5)
    # find the longest True run
    runs = []
    start = None
    for i, q in enumerate(quiet):
        if q and start is None: start = i
        elif not q and start is not None:
            runs.append((start, i)); start = None
    if start is not None: runs.append((start, len(quiet)))
    if not runs: return 0, len(t) - 1
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    return runs[0]

i0_o, i1_o = airborne_window(odom_t, odom_rpy)
T_LO, T_HI = odom_t[i0_o], odom_t[i1_o - 1]
print(f'Steady-state window: [{T_LO:.1f}, {T_HI:.1f}] s')

def window_mean_std(t, x, T_LO=T_LO, T_HI=T_HI):
    mask = (t >= T_LO) & (t <= T_HI)
    return x[mask].mean(), x[mask].std()


# ── Stats ──
print(f'\n== {TAG}  —  steady-state attitude trim ==')
for src, t, rpy in [('IMU AHRS', imu_t, imu_rpy),
                    ('EKF2 odom', odom_t, odom_rpy)] + (
                   [('mocap', moc_t, moc_rpy)] if moc_rpy is not None else []):
    print(f'  {src}:')
    for k, name in enumerate(['roll', 'pitch', 'yaw']):
        m, s = window_mean_std(t, np.degrees(rpy[:, k]))
        print(f'    {name:5s}  mean = {m:+7.3f} deg   std = {s:6.3f} deg')

print('\n  NMPC commanded torque (motor cmd, /nmpc/control):')
for k, name in enumerate(['Mx', 'My', 'Mz']):
    m, s = window_mean_std(ctrl_t, ctrl[:, 3 + k])
    print(f'    {name}  mean = {m:+8.5f} N·m   std = {s:8.5f}')

print('\n  HGDO disturbance torque (/hgdo/wrench):')
for k, name in enumerate(['Mx', 'My', 'Mz']):
    m, s = window_mean_std(hgdo_t, hgdo[:, 3 + k])
    print(f'    {name}  mean = {m:+8.5f} N·m   std = {s:8.5f}')

# Pure NMPC = motor cmd + HGDO
hgdo_at_ctrl = np.column_stack([
    np.interp(ctrl_t, hgdo_t, hgdo[:, 3 + k]) for k in range(3)])
pure_nmpc = ctrl[:, 3:6] + hgdo_at_ctrl
print('\n  Pure NMPC (= motor cmd + HGDO):')
for k, name in enumerate(['Mx', 'My', 'Mz']):
    m, s = window_mean_std(ctrl_t, pure_nmpc[:, k])
    print(f'    {name}  mean = {m:+8.5f} N·m   std = {s:8.5f}')


# ── Plots ──
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for k, name in enumerate(['Roll', 'Pitch', 'Yaw']):
    axes[k].axhline(0, color='gray', lw=0.8, alpha=0.5)
    axes[k].plot(odom_t, np.degrees(odom_rpy[:, k]), 'r', lw=0.9, alpha=0.85, label='EKF2 odom')
    axes[k].plot(imu_t,  np.degrees(imu_rpy [:, k]), 'b', lw=0.9, alpha=0.65, label='IMU AHRS')
    if moc_rpy is not None:
        axes[k].plot(moc_t, np.degrees(moc_rpy[:, k]), 'k', lw=1.2, alpha=0.85, label='mocap')
    axes[k].axvspan(T_LO, T_HI, color='yellow', alpha=0.10, label='steady-state window')
    axes[k].set_ylabel(f'{name} [deg]'); axes[k].grid(alpha=0.3)
    axes[k].legend(loc='upper right', fontsize=9)
axes[0].set_title(f'{TAG} ball-joint  —  attitude (reference RPY = 0)')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_att_rpy.png'), dpi=120)
plt.close()

# Torque decomposition steady-state
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for k, name in enumerate(['Mx', 'My', 'Mz']):
    axes[k].axhline(0, color='gray', lw=0.8, alpha=0.5)
    axes[k].plot(ctrl_t, ctrl[:, 3 + k],   'k', lw=1.2, label='motor cmd (/nmpc/control)')
    axes[k].plot(hgdo_t, hgdo[:, 3 + k],   'r', lw=1.0, alpha=0.85, label='HGDO')
    axes[k].plot(ctrl_t, pure_nmpc[:, k],  'b', lw=1.0, alpha=0.85, label='pure NMPC = motor + HGDO')
    axes[k].axvspan(T_LO, T_HI, color='yellow', alpha=0.10)
    axes[k].set_ylabel(f'{name} [N·m]'); axes[k].grid(alpha=0.3)
    axes[k].legend(loc='upper right', fontsize=9)
axes[0].set_title(f'{TAG} ball-joint  —  torques in steady-state window (yellow band)')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_att_torque.png'), dpi=120)
plt.close()

print('\nSaved:')
print(f'  - {TAG}_att_rpy.png')
print(f'  - {TAG}_att_torque.png')
