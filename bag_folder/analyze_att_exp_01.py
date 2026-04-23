#!/usr/bin/env python3
"""Bag analysis: 2026_04_24_att_exp_01 — NMPC attitude with HGDO."""

import sqlite3, struct, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, '2026_04_24_att_exp_01/2026_04_24_att_exp_01_0.db3')
os.chdir(_HERE)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

tid = {}
c.execute('SELECT id, name FROM topics')
for row in c.fetchall():
    tid[row[1]] = row[0]

# ── Parse helpers ──

def parse_odom(blob):
    """nav_msgs/Odometry CDR → (px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz)"""
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    if off % 4: off += 4 - off % 4
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    if off % 4: off += 4 - off % 4
    off += 4  # XCDR2 padding
    px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
    off += 36 * 8
    vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
    wx, wy, wz = struct.unpack_from('<3d', blob, off)
    return np.array([px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz])


def parse_wrench(blob):
    """geometry_msgs/WrenchStamped CDR → (fx,fy,fz, tx,ty,tz)"""
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    if off % 4: off += 4 - off % 4
    vals = struct.unpack_from('<6d', blob, off)
    return np.array(vals)


def parse_pose(blob):
    """geometry_msgs/PoseStamped CDR → (px,py,pz, qw,qx,qy,qz)"""
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    if off % 4: off += 4 - off % 4
    px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off)
    return np.array([px, py, pz, qw, qx, qy, qz])


def parse_rc(blob):
    """mavros_msgs/RCIn CDR → channel values"""
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    if off % 4: off += 4 - off % 4
    n_ch = struct.unpack_from('<I', blob, off)[0]; off += 4
    channels = struct.unpack_from(f'<{n_ch}H', blob, off)
    return np.array(channels, dtype=np.float64)


def quat_to_rpy(q):
    qw, qx, qy, qz = q
    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    sinp = np.clip(2*(qw*qy - qz*qx), -1, 1)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    return np.array([roll, pitch, yaw])


def quat_to_rotm(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2+qz**2), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),     1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw),    1-2*(qx**2+qy**2)],
    ])


# ── Extract data ──

def fetch(topic_name, parser):
    topic_id = tid[topic_name]
    c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (topic_id,))
    ts_list, data_list = [], []
    for ts, data in c.fetchall():
        try:
            parsed = parser(bytes(data))
            ts_list.append(ts)
            data_list.append(parsed)
        except Exception:
            pass
    return np.array(ts_list, dtype=np.float64), np.array(data_list)


odom_ts, odom = fetch('/mavros/local_position/odom', parse_odom)
ctrl_ts, ctrl = fetch('/nmpc/control', parse_wrench)
hgdo_ts, hgdo = fetch('/hgdo/wrench', parse_wrench)
pose_ts, pose = fetch('/S550/pose', parse_pose)

conn.close()

t0 = odom_ts[0]
odom_t = (odom_ts - t0) * 1e-9
ctrl_t = (ctrl_ts - t0) * 1e-9
hgdo_t = (hgdo_ts - t0) * 1e-9
pose_t = (pose_ts - t0) * 1e-9

# Derived
rpy = np.array([quat_to_rpy(odom[i, 6:10]) for i in range(len(odom))])
rpy_deg = np.degrees(rpy)

v_world = np.zeros((len(odom), 3))
for i in range(len(odom)):
    R = quat_to_rotm(odom[i, 6:10])
    v_world[i] = R @ odom[i, 3:6]

# S550/pose RPY
pose_rpy = np.array([quat_to_rpy(pose[i, 3:7]) for i in range(len(pose))])
pose_rpy_deg = np.degrees(pose_rpy)

# ── PLOT 1: Full overview ──

fig, axes = plt.subplots(6, 1, figsize=(16, 22), sharex=True)

ax = axes[0]
ax.plot(odom_t, rpy_deg[:, 0], 'r', alpha=0.8, label='Roll')
ax.plot(odom_t, rpy_deg[:, 1], 'g', alpha=0.8, label='Pitch')
ax.plot(odom_t, rpy_deg[:, 2], 'b', alpha=0.8, label='Yaw')
ax.set_ylabel('Angle [deg]')
ax.set_title('Roll / Pitch / Yaw (odom)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(odom_t, odom[:, 10], 'r', alpha=0.8, label='wx')
ax.plot(odom_t, odom[:, 11], 'g', alpha=0.8, label='wy')
ax.plot(odom_t, odom[:, 12], 'b', alpha=0.8, label='wz')
ax.set_ylabel('Angular vel [rad/s]')
ax.set_title('Angular Velocity (body)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(ctrl_t, ctrl[:, 3], 'r', alpha=0.8, label='Mx (NMPC)')
ax.plot(ctrl_t, ctrl[:, 4], 'g', alpha=0.8, label='My (NMPC)')
ax.plot(ctrl_t, ctrl[:, 5], 'b', alpha=0.8, label='Mz (NMPC)')
ax.set_ylabel('Moment [Nm]')
ax.set_title('NMPC Control Moments')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes[3]
ax.plot(hgdo_t, hgdo[:, 3], 'r', alpha=0.8, label='tau_x HGDO')
ax.plot(hgdo_t, hgdo[:, 4], 'g', alpha=0.8, label='tau_y HGDO')
ax.plot(hgdo_t, hgdo[:, 5], 'b', alpha=0.8, label='tau_z HGDO')
ax.set_ylabel('Torque [Nm]')
ax.set_title('HGDO Disturbance Torque')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes[4]
ax.plot(odom_t, v_world[:, 0], 'r', alpha=0.8, label='Vx world')
ax.plot(odom_t, v_world[:, 1], 'g', alpha=0.8, label='Vy world')
ax.plot(odom_t, v_world[:, 2], 'b', alpha=0.6, label='Vz world')
ax.set_ylabel('Velocity [m/s]')
ax.set_title('World-frame Velocity')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes[5]
ax.plot(odom_t, odom[:, 0], 'r', alpha=0.8, label='px')
ax.plot(odom_t, odom[:, 1], 'g', alpha=0.8, label='py')
ax.plot(odom_t, odom[:, 2], 'b', alpha=0.8, label='pz')
ax.set_ylabel('Position [m]')
ax.set_xlabel('Time [s]')
ax.set_title('Position')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Mark ctrl start
ctrl_start = ctrl_t[0] if len(ctrl_t) > 0 else 0
for a in axes:
    a.axvline(x=ctrl_start, color='gray', linestyle=':', alpha=0.5, label='ctrl start' if a == axes[0] else None)

plt.tight_layout()
plt.savefig('att_exp_01_overview.png', dpi=150)
print('Saved att_exp_01_overview.png')

# ── PLOT 2: NMPC + HGDO compensation detail ──

fig2, axes2 = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

ax = axes2[0]
ax.plot(odom_t, rpy_deg[:, 0], 'r', alpha=0.8, label='Roll')
ax.plot(odom_t, rpy_deg[:, 1], 'g', alpha=0.8, label='Pitch')
ax.set_ylabel('Angle [deg]')
ax.set_title('Roll / Pitch')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes2[1]
Mx_interp = np.interp(odom_t, ctrl_t, ctrl[:, 3])
My_interp = np.interp(odom_t, ctrl_t, ctrl[:, 4])
Mz_interp = np.interp(odom_t, ctrl_t, ctrl[:, 5])
ax.plot(odom_t, Mx_interp, 'r', alpha=0.8, label='Mx NMPC')
ax.plot(odom_t, My_interp, 'g', alpha=0.8, label='My NMPC')
ax.plot(odom_t, Mz_interp, 'b', alpha=0.8, label='Mz NMPC')
ax.set_ylabel('Moment [Nm]')
ax.set_title('NMPC Moments')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes2[2]
tx_interp = np.interp(odom_t, hgdo_t, hgdo[:, 3])
ty_interp = np.interp(odom_t, hgdo_t, hgdo[:, 4])
tz_interp = np.interp(odom_t, hgdo_t, hgdo[:, 5])
ax.plot(odom_t, tx_interp, 'r', alpha=0.8, label='tau_x HGDO')
ax.plot(odom_t, ty_interp, 'g', alpha=0.8, label='tau_y HGDO')
ax.plot(odom_t, tz_interp, 'b', alpha=0.8, label='tau_z HGDO')
ax.set_ylabel('Torque [Nm]')
ax.set_title('HGDO Torque')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes2[3]
ax.plot(odom_t, Mx_interp - tx_interp, 'r', alpha=0.8, label='Mx_comp (NMPC-HGDO)')
ax.plot(odom_t, My_interp - ty_interp, 'g', alpha=0.8, label='My_comp (NMPC-HGDO)')
ax.plot(odom_t, np.clip(Mz_interp - tz_interp, -0.05, 0.05), 'b', alpha=0.8, label='Mz_comp (clipped)')
ax.set_ylabel('Moment [Nm]')
ax.set_xlabel('Time [s]')
ax.set_title('Compensated Moments (M_nmpc - tau_hgdo)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

for a in axes2:
    a.axvline(x=ctrl_start, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('att_exp_01_compensation.png', dpi=150)
print('Saved att_exp_01_compensation.png')

# ── PLOT 3: S550/pose vs odom comparison ──

fig3, axes3 = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

ax = axes3[0]
ax.plot(odom_t, rpy_deg[:, 0], 'r', alpha=0.8, label='Roll (odom)')
ax.plot(odom_t, rpy_deg[:, 1], 'g', alpha=0.8, label='Pitch (odom)')
ax.plot(odom_t, rpy_deg[:, 2], 'b', alpha=0.8, label='Yaw (odom)')
ax.plot(pose_t, pose_rpy_deg[:, 0], 'r--', alpha=0.5, label='Roll (S550/pose)')
ax.plot(pose_t, pose_rpy_deg[:, 1], 'g--', alpha=0.5, label='Pitch (S550/pose)')
ax.plot(pose_t, pose_rpy_deg[:, 2], 'b--', alpha=0.5, label='Yaw (S550/pose)')
ax.set_ylabel('Angle [deg]')
ax.set_title('Odom vs S550/pose')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes3[1]
ax.plot(odom_t, odom[:, 0], 'r', alpha=0.8, label='px odom')
ax.plot(odom_t, odom[:, 1], 'g', alpha=0.8, label='py odom')
ax.plot(odom_t, odom[:, 2], 'b', alpha=0.8, label='pz odom')
ax.plot(pose_t, pose[:, 0], 'r--', alpha=0.5, label='px S550')
ax.plot(pose_t, pose[:, 1], 'g--', alpha=0.5, label='py S550')
ax.plot(pose_t, pose[:, 2], 'b--', alpha=0.5, label='pz S550')
ax.set_ylabel('Position [m]')
ax.set_title('Position: Odom vs S550/pose')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes3[2]
ax.plot(odom_t, odom[:, 10], 'r', alpha=0.8, label='wx')
ax.plot(odom_t, odom[:, 11], 'g', alpha=0.8, label='wy')
ax.plot(odom_t, odom[:, 12], 'b', alpha=0.8, label='wz')
ax.set_ylabel('Angular vel [rad/s]')
ax.set_xlabel('Time [s]')
ax.set_title('Angular Velocity (body)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

for a in axes3:
    a.axvline(x=ctrl_start, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('att_exp_01_pose_compare.png', dpi=150)
print('Saved att_exp_01_pose_compare.png')

# ── Print statistics ──
print(f'\n=== Experiment Summary ===')
print(f'Duration: {odom_t[-1]:.1f}s')
print(f'Ctrl active: {ctrl_t[0]:.1f}s to {ctrl_t[-1]:.1f}s ({ctrl_t[-1]-ctrl_t[0]:.1f}s)')
print(f'RPY (deg): roll=[{rpy_deg[:,0].min():.1f},{rpy_deg[:,0].max():.1f}], pitch=[{rpy_deg[:,1].min():.1f},{rpy_deg[:,1].max():.1f}], yaw=[{rpy_deg[:,2].min():.1f},{rpy_deg[:,2].max():.1f}]')
print(f'Position: x=[{odom[:,0].min():.3f},{odom[:,0].max():.3f}], y=[{odom[:,1].min():.3f},{odom[:,1].max():.3f}], z=[{odom[:,2].min():.3f},{odom[:,2].max():.3f}]')
print(f'NMPC Mx=[{ctrl[:,3].min():.4f},{ctrl[:,3].max():.4f}], My=[{ctrl[:,4].min():.4f},{ctrl[:,4].max():.4f}], Mz=[{ctrl[:,5].min():.4f},{ctrl[:,5].max():.4f}]')
print(f'HGDO tx=[{hgdo[:,3].min():.4f},{hgdo[:,3].max():.4f}], ty=[{hgdo[:,4].min():.4f},{hgdo[:,4].max():.4f}], tz=[{hgdo[:,5].min():.4f},{hgdo[:,5].max():.4f}]')
print(f'Fz (constant): {ctrl[:,2].mean():.3f}N')
