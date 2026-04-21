#!/usr/bin/env python3
"""Bag analysis: odom, /nmpc/control, /hgdo/wrench, /nmpc/ref — yaw step instability."""

import sqlite3, struct, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DB_PATH = 'bag_folder/2026_04_22_sim/2026_04_22_sim_0.db3'

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# ── Topic IDs ──
tid = {}
c.execute('SELECT id, name FROM topics')
for row in c.fetchall():
    tid[row[1]] = row[0]

# ── Parse helpers ──

def parse_odom(blob):
    """nav_msgs/Odometry CDR → (px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz)"""
    off = 4  # skip CDR encapsulation header
    off += 8  # skip stamp (sec + nsec)
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4
    off += slen  # skip frame_id string
    if off % 4: off += 4 - off % 4  # align for next uint32
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4
    off += slen2  # skip child_frame_id string
    if off % 4: off += 4 - off % 4  # align for doubles (4-byte in ROS2 CDR)
    px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
    off += 36 * 8  # skip pose covariance
    vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
    wx, wy, wz = struct.unpack_from('<3d', blob, off); off += 24
    return np.array([px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz])


def parse_wrench(blob):
    """geometry_msgs/WrenchStamped CDR → (fx,fy,fz, tx,ty,tz)"""
    off = 4  # CDR encapsulation
    off += 8  # skip stamp (sec + nsec)
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4
    off += slen  # skip frame_id string (includes null terminator)
    if off % 4: off += 4 - off % 4  # align to 4-byte boundary
    vals = struct.unpack_from('<6d', blob, off)
    return np.array(vals)


def parse_ref(blob):
    """drone_msgs/Ref CDR → (px,py,pz, vx,vy,vz, psi, psi_dot)"""
    off = 20  # CDR(4) + header(16) for empty frame_id
    p = struct.unpack_from('<3d', blob, off); off += 24
    v = struct.unpack_from('<3d', blob, off); off += 24
    psi = struct.unpack_from('<d', blob, off)[0]; off += 8
    psi_dot = struct.unpack_from('<d', blob, off)[0]; off += 8
    return np.array([*p, *v, psi, psi_dot])


def quat_to_rpy(q):
    """[qw,qx,qy,qz] → [roll,pitch,yaw] rad (ZYX)."""
    qw, qx, qy, qz = q
    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    sinp = 2*(qw*qy - qz*qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    return np.array([roll, pitch, yaw])


def quat_to_rotm(q):
    """[qw,qx,qy,qz] → 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2+qz**2), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),     1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw),    1-2*(qx**2+qy**2)],
    ])


# ── Extract all data ──

def fetch(topic_name, parser):
    topic_id = tid[topic_name]
    c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (topic_id,))
    ts_list, data_list = [], []
    for ts, data in c.fetchall():
        ts_list.append(ts)
        data_list.append(parser(bytes(data)))
    return np.array(ts_list, dtype=np.float64), np.array(data_list)

odom_ts, odom = fetch('/mavros/local_position/odom', parse_odom)
ctrl_ts, ctrl = fetch('/nmpc/control', parse_wrench)
hgdo_ts, hgdo = fetch('/hgdo/wrench', parse_wrench)
ref_ts, ref = fetch('/nmpc/ref', parse_ref)

conn.close()

# Align times to t=0
t0 = odom_ts[0]
odom_t = (odom_ts - t0) * 1e-9
ctrl_t = (ctrl_ts - t0) * 1e-9
hgdo_t = (hgdo_ts - t0) * 1e-9
ref_t = (ref_ts - t0) * 1e-9

# ── Derived quantities ──

# Subtract initial position offset
p_offset = odom[0, 0:3].copy()
odom[:, 0:3] -= p_offset

# RPY from odom quaternion
rpy = np.array([quat_to_rpy(odom[i, 6:10]) for i in range(len(odom))])
rpy_deg = np.degrees(rpy)

# World-frame velocity from body velocity
v_world = np.zeros((len(odom), 3))
for i in range(len(odom)):
    R = quat_to_rotm(odom[i, 6:10])
    v_world[i] = R @ odom[i, 3:6]

# ── PLOTS ──

fig, axes = plt.subplots(7, 1, figsize=(16, 24), sharex=True)

# 1) RPY + psi_ref
ax = axes[0]
ax.plot(odom_t, rpy_deg[:, 0], 'r', alpha=0.8, label='Roll')
ax.plot(odom_t, rpy_deg[:, 1], 'g', alpha=0.8, label='Pitch')
ax.plot(odom_t, rpy_deg[:, 2], 'b', alpha=0.8, label='Yaw')
ax.step(ref_t, np.degrees(ref[:, 6]), 'b--', linewidth=2, where='post', label='Yaw ref')
ax.set_ylabel('Angle [deg]')
ax.set_title('Roll / Pitch / Yaw')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 2) World-frame velocity vx, vy
ax = axes[1]
ax.plot(odom_t, v_world[:, 0], 'r', alpha=0.8, label='Vx world')
ax.plot(odom_t, v_world[:, 1], 'g', alpha=0.8, label='Vy world')
ax.plot(odom_t, v_world[:, 2], 'b', alpha=0.6, label='Vz world')
ax.set_ylabel('Velocity [m/s]')
ax.set_title('World-frame Velocity (R @ v_body)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 3) Body-frame velocity (raw odom)
ax = axes[2]
ax.plot(odom_t, odom[:, 3], 'r', alpha=0.8, label='vx body')
ax.plot(odom_t, odom[:, 4], 'g', alpha=0.8, label='vy body')
ax.plot(odom_t, odom[:, 5], 'b', alpha=0.6, label='vz body')
ax.set_ylabel('Velocity [m/s]')
ax.set_title('Body-frame Velocity (raw odom)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 4) NMPC control: Mx, My, Mz
ax = axes[3]
ax.plot(ctrl_t, ctrl[:, 3], 'r', alpha=0.8, label='Mx (torque.x)')
ax.plot(ctrl_t, ctrl[:, 4], 'g', alpha=0.8, label='My (torque.y)')
ax.plot(ctrl_t, ctrl[:, 5], 'b', alpha=0.8, label='Mz (torque.z)')
ax.set_ylabel('Moment [Nm]')
ax.set_title('NMPC Control Moments (torque.x/y/z)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 5) PD F_des (force.x, force.y) from /nmpc/control
ax = axes[4]
ax.plot(ctrl_t, ctrl[:, 0], 'r', alpha=0.8, label='F_des_x (force.x)')
ax.plot(ctrl_t, ctrl[:, 1], 'g', alpha=0.8, label='F_des_y (force.y)')
ax.set_ylabel('Force [N]')
ax.set_title('PD Desired Force (world frame, post-DOB)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 6) HGDO wrench (torque)
ax = axes[5]
ax.plot(hgdo_t, hgdo[:, 3], 'r', alpha=0.8, label='tau_x HGDO')
ax.plot(hgdo_t, hgdo[:, 4], 'g', alpha=0.8, label='tau_y HGDO')
ax.plot(hgdo_t, hgdo[:, 5], 'b', alpha=0.8, label='tau_z HGDO')
ax.set_ylabel('Torque [Nm]')
ax.set_title('HGDO Disturbance Torque')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 7) Position
ax = axes[6]
ax.plot(odom_t, odom[:, 0], 'r', alpha=0.8, label='px')
ax.plot(odom_t, odom[:, 1], 'g', alpha=0.8, label='py')
ax.plot(odom_t, odom[:, 2], 'b', alpha=0.8, label='pz')
ax.step(ref_t, ref[:, 2], 'b--', linewidth=2, where='post', label='pz ref')
ax.set_ylabel('Position [m]')
ax.set_xlabel('Time [s]')
ax.set_title('Position')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

for ax in axes:
    try:
        ylo, yhi = ax.get_ylim()
        if yhi - ylo < 1e-6:
            ax.set_ylim(-1, 1)
        ax.axvline(x=15.0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=29.33, color='gray', linestyle=':', alpha=0.5)
    except Exception:
        pass

plt.tight_layout()
plt.savefig('bag_analysis.png', dpi=150)
print('Saved bag_analysis.png')

# ── Zoomed plot around psi=1.5 step (t=27~40s) ──
fig2, axes2 = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

mask_o = (odom_t >= 27) & (odom_t <= 42)
mask_c = (ctrl_t >= 27) & (ctrl_t <= 42)
mask_h = (hgdo_t >= 27) & (hgdo_t <= 42)

ax = axes2[0]
ax.plot(odom_t[mask_o], rpy_deg[mask_o, 0], 'r', label='Roll')
ax.plot(odom_t[mask_o], rpy_deg[mask_o, 1], 'g', label='Pitch')
ax.plot(odom_t[mask_o], rpy_deg[mask_o, 2], 'b', label='Yaw')
ax.axhline(y=np.degrees(1.5), color='b', linestyle='--', alpha=0.5, label='ψ ref=1.5')
ax.set_ylabel('Angle [deg]')
ax.set_title('RPY around ψ=1.5 step (t=27~42s)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes2[1]
ax.plot(odom_t[mask_o], v_world[mask_o, 0], 'r', label='Vx world')
ax.plot(odom_t[mask_o], v_world[mask_o, 1], 'g', label='Vy world')
ax.set_ylabel('Velocity [m/s]')
ax.set_title('World Velocity (Vx, Vy)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes2[2]
ax.plot(ctrl_t[mask_c], ctrl[mask_c, 0], 'r', label='F_des_x')
ax.plot(ctrl_t[mask_c], ctrl[mask_c, 1], 'g', label='F_des_y')
ax.set_ylabel('Force [N]')
ax.set_title('PD F_des (world)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes2[3]
ax.plot(ctrl_t[mask_c], ctrl[mask_c, 3], 'r', label='Mx')
ax.plot(ctrl_t[mask_c], ctrl[mask_c, 4], 'g', label='My')
ax.plot(ctrl_t[mask_c], ctrl[mask_c, 5], 'b', label='Mz')
ax.set_ylabel('Moment [Nm]')
ax.set_title('NMPC Moments (Mx, My, Mz)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes2[4]
ax.plot(hgdo_t[mask_h], hgdo[mask_h, 3], 'r', label='tau_x HGDO')
ax.plot(hgdo_t[mask_h], hgdo[mask_h, 4], 'g', label='tau_y HGDO')
ax.plot(hgdo_t[mask_h], hgdo[mask_h, 5], 'b', label='tau_z HGDO')
ax.set_ylabel('Torque [Nm]')
ax.set_xlabel('Time [s]')
ax.set_title('HGDO Disturbance Torque')
ax.legend()
ax.grid(True, alpha=0.3)

for ax in axes2:
    try:
        ax.axvline(x=29.33, color='k', linestyle='--', alpha=0.5)
    except Exception:
        pass

plt.tight_layout()
plt.savefig('bag_analysis_zoom.png', dpi=150)
print('Saved bag_analysis_zoom.png')

# ── Correlation: Vx vs My, Vy vs Mx (scatter during t=29~42) ──
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Interpolate ctrl Mx/My onto odom timestamps for correlation
from numpy import interp
Mx_interp = interp(odom_t[mask_o], ctrl_t, ctrl[:, 3])
My_interp = interp(odom_t[mask_o], ctrl_t, ctrl[:, 4])

ax1.scatter(v_world[mask_o, 0], My_interp, s=2, c=odom_t[mask_o], cmap='viridis')
ax1.set_xlabel('Vx world [m/s]')
ax1.set_ylabel('My [Nm]')
ax1.set_title('Vx vs My (t=27~42s)')
ax1.grid(True, alpha=0.3)

ax2.scatter(v_world[mask_o, 1], Mx_interp, s=2, c=odom_t[mask_o], cmap='viridis')
ax2.set_xlabel('Vy world [m/s]')
ax2.set_ylabel('Mx [Nm]')
ax2.set_title('Vy vs Mx (t=27~42s)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bag_correlation.png', dpi=150)
print('Saved bag_correlation.png')

# ── Position error & Velocity error plots ──

# Interpolate ref onto odom timestamps (ref is 1 Hz step)
ref_p_interp = np.zeros((len(odom), 3))
ref_v_interp = np.zeros((len(odom), 3))
for ax_i in range(3):
    ref_p_interp[:, ax_i] = np.interp(odom_t, ref_t, ref[:, ax_i])
    ref_v_interp[:, ax_i] = np.interp(odom_t, ref_t, ref[:, 3 + ax_i])

e_p = ref_p_interp - odom[:, 0:3]   # position error
e_v = ref_v_interp - v_world         # velocity error (world frame)

fig4, axes4 = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

ax = axes4[0]
ax.plot(odom_t, e_p[:, 0], 'r', alpha=0.8, label='e_px')
ax.plot(odom_t, e_p[:, 1], 'g', alpha=0.8, label='e_py')
ax.plot(odom_t, e_p[:, 2], 'b', alpha=0.8, label='e_pz')
ax.set_ylabel('Position Error [m]')
ax.set_title('Position Error (ref - odom)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes4[1]
ax.plot(odom_t, e_v[:, 0], 'r', alpha=0.8, label='e_vx')
ax.plot(odom_t, e_v[:, 1], 'g', alpha=0.8, label='e_vy')
ax.plot(odom_t, e_v[:, 2], 'b', alpha=0.8, label='e_vz')
ax.set_ylabel('Velocity Error [m/s]')
ax.set_title('Velocity Error (ref_v - R@v_body, world frame)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Zoomed: t=27~42s
ax = axes4[2]
ax.plot(odom_t[mask_o], e_p[mask_o, 0], 'r', alpha=0.8, label='e_px')
ax.plot(odom_t[mask_o], e_p[mask_o, 1], 'g', alpha=0.8, label='e_py')
ax.plot(odom_t[mask_o], e_p[mask_o, 2], 'b', alpha=0.8, label='e_pz')
ax.set_ylabel('Position Error [m]')
ax.set_title('Position Error — zoom ψ=1.5 step (t=27~42s)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes4[3]
ax.plot(odom_t[mask_o], e_v[mask_o, 0], 'r', alpha=0.8, label='e_vx')
ax.plot(odom_t[mask_o], e_v[mask_o, 1], 'g', alpha=0.8, label='e_vy')
ax.plot(odom_t[mask_o], e_v[mask_o, 2], 'b', alpha=0.8, label='e_vz')
ax.set_ylabel('Velocity Error [m/s]')
ax.set_xlabel('Time [s]')
ax.set_title('Velocity Error — zoom ψ=1.5 step (t=27~42s)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

for ax in axes4:
    ax.axvline(x=15.0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=29.33, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('bag_error.png', dpi=150)
print('Saved bag_error.png')

# ── Print error statistics ──
print("\n=== Position / Velocity Error around ψ=1.5 step (t=29~42s) ===")
mask_post = (odom_t >= 29.3) & (odom_t <= 42)
for i, lbl in enumerate(['x', 'y', 'z']):
    print(f"e_p{lbl} range: [{e_p[mask_post,i].min():+.4f}, {e_p[mask_post,i].max():+.4f}] m")
for i, lbl in enumerate(['x', 'y', 'z']):
    print(f"e_v{lbl} range: [{e_v[mask_post,i].min():+.4f}, {e_v[mask_post,i].max():+.4f}] m/s")

print("\n=== Position / Velocity Error around ψ=0.5 step (t=15~28s) ===")
mask_05 = (odom_t >= 15.0) & (odom_t <= 28)
for i, lbl in enumerate(['x', 'y', 'z']):
    print(f"e_p{lbl} range: [{e_p[mask_05,i].min():+.4f}, {e_p[mask_05,i].max():+.4f}] m")
for i, lbl in enumerate(['x', 'y', 'z']):
    print(f"e_v{lbl} range: [{e_v[mask_05,i].min():+.4f}, {e_v[mask_05,i].max():+.4f}] m/s")

# ── Print key statistics ──
print("\n=== Statistics around ψ=1.5 step (t=29~42s) ===")
mask_post = (odom_t >= 29.3) & (odom_t <= 42)
print(f"Roll  range: [{rpy_deg[mask_post,0].min():+.1f}, {rpy_deg[mask_post,0].max():+.1f}] deg")
print(f"Pitch range: [{rpy_deg[mask_post,1].min():+.1f}, {rpy_deg[mask_post,1].max():+.1f}] deg")
print(f"Yaw   range: [{rpy_deg[mask_post,2].min():+.1f}, {rpy_deg[mask_post,2].max():+.1f}] deg")
print(f"Vx world  std: {v_world[mask_post,0].std():.4f} m/s")
print(f"Vy world  std: {v_world[mask_post,1].std():.4f} m/s")

mask_c_post = (ctrl_t >= 29.3) & (ctrl_t <= 42)
print(f"Mx range: [{ctrl[mask_c_post,3].min():+.4f}, {ctrl[mask_c_post,3].max():+.4f}] Nm")
print(f"My range: [{ctrl[mask_c_post,4].min():+.4f}, {ctrl[mask_c_post,4].max():+.4f}] Nm")
print(f"Mz range: [{ctrl[mask_c_post,5].min():+.4f}, {ctrl[mask_c_post,5].max():+.4f}] Nm")

# Also check psi=0.5 step for comparison
print("\n=== Statistics around ψ=0.5 step (t=15~28s) ===")
mask_05 = (odom_t >= 15.0) & (odom_t <= 28)
print(f"Roll  range: [{rpy_deg[mask_05,0].min():+.1f}, {rpy_deg[mask_05,0].max():+.1f}] deg")
print(f"Pitch range: [{rpy_deg[mask_05,1].min():+.1f}, {rpy_deg[mask_05,1].max():+.1f}] deg")
print(f"Yaw   range: [{rpy_deg[mask_05,2].min():+.1f}, {rpy_deg[mask_05,2].max():+.1f}] deg")
print(f"Vx world  std: {v_world[mask_05,0].std():.4f} m/s")
print(f"Vy world  std: {v_world[mask_05,1].std():.4f} m/s")

mask_c_05 = (ctrl_t >= 15.0) & (ctrl_t <= 28)
print(f"Mx range: [{ctrl[mask_c_05,3].min():+.4f}, {ctrl[mask_c_05,3].max():+.4f}] Nm")
print(f"My range: [{ctrl[mask_c_05,4].min():+.4f}, {ctrl[mask_c_05,4].max():+.4f}] Nm")
print(f"Mz range: [{ctrl[mask_c_05,5].min():+.4f}, {ctrl[mask_c_05,5].max():+.4f}] Nm")
