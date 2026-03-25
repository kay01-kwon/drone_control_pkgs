#!/usr/bin/env python3
"""Plot cmd_raw-derived moments vs roll/pitch from bag data (before LPF)."""

import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ── Constants ──
C_T = 1.465e-07      # N/RPM^2
k_m = 0.01569         # Nm/N
l = 0.265             # arm length (m)
MaxBit = 8191
MaxRpm = 9800

# ── Hexacopter K_forward matrix ──
# Rotor positions (60 deg intervals)
angles = [np.pi/3, np.pi/2, 2*np.pi/3, 4*np.pi/3, 3*np.pi/2, 5*np.pi/3]
# From code: lx = l*sin(angle_offset), ly = l*cos(angle_offset)
# Rotor 1: angle=60deg, Rotor2: 90deg, etc.
lx1 = l * np.sin(np.pi/3);  ly1 = l * np.cos(np.pi/3)
lx2 = 0.0;                   ly2 = l
lx3 = -l * np.sin(np.pi/3); ly3 = l * np.cos(np.pi/3)
lx4 = -l * np.sin(np.pi/3); ly4 = -l * np.cos(np.pi/3)
lx5 = 0.0;                   ly5 = -l
lx6 = l * np.sin(np.pi/3);  ly6 = -l * np.cos(np.pi/3)

K_forward = np.array([
    [1, 1, 1, 1, 1, 1],
    [ly1, ly2, ly3, ly4, ly5, ly6],
    [-lx1, -lx2, -lx3, -lx4, -lx5, -lx6],
    [-k_m, k_m, -k_m, k_m, -k_m, k_m]
])


def parse_cmd_raw(data):
    """Parse HexaCmdRaw CDR message → (timestamp, cmd[6])."""
    off = 4  # CDR header
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    if off % 2 != 0: off += 1  # align to 2 bytes for int16
    cmds = np.array(struct.unpack_from('<6h', data, off), dtype=np.float64)
    return sec + nsec * 1e-9, cmds


def cdr_align(off, alignment, cdr_start=4):
    """Align offset for CDR encoding (relative to CDR data start)."""
    rel = off - cdr_start
    rem = rel % alignment
    if rem != 0:
        off += alignment - rem
    return off


def parse_odom(data):
    """Parse nav_msgs/Odometry CDR → (timestamp, roll, pitch, yaw)."""
    off = 4  # CDR header
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    # frame_id string
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    off = cdr_align(off, 4)
    # child_frame_id string
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    # Align to 8 bytes for float64 position
    off = cdr_align(off, 8)
    # Pose: position (3 x float64) + orientation (4 x float64)
    px, py, pz = struct.unpack_from('<3d', data, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', data, off); off += 32

    t = sec + nsec * 1e-9
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if norm < 1e-10:
        return t, 0.0, 0.0, 0.0
    r = Rotation.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    return t, roll, pitch, yaw


def load_bag(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Get topic IDs
    c.execute('SELECT id, name FROM topics')
    topics = {name: tid for tid, name in c.fetchall()}

    # ── Load cmd_raw ──
    tid = topics['/uav/cmd_raw']
    c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    cmd_times, moments_x, moments_y, moments_z, forces = [], [], [], [], []
    for ts, data in c.fetchall():
        t, cmds = parse_cmd_raw(data)
        # cmd_raw → RPM → thrust
        rpms = cmds * MaxRpm / MaxBit
        thrusts = C_T * rpms ** 2
        # K_forward @ thrusts → [F, Mx, My, Mz]
        u = K_forward @ thrusts
        cmd_times.append(t)
        forces.append(u[0])
        moments_x.append(u[1])
        moments_y.append(u[2])
        moments_z.append(u[3])

    # ── Load odom ──
    tid = topics['/mavros/local_position/odom']
    c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    odom_times, rolls, pitches = [], [], []
    for ts, data in c.fetchall():
        t, roll, pitch, yaw = parse_odom(data)
        odom_times.append(t)
        rolls.append(roll)
        pitches.append(pitch)

    conn.close()

    # Convert to arrays and make time relative
    cmd_times = np.array(cmd_times)
    odom_times = np.array(odom_times)
    t0 = cmd_times[0]  # reference time = cmd_raw (moment) start
    cmd_times -= t0
    odom_times -= t0

    return (cmd_times, np.array(forces), np.array(moments_x),
            np.array(moments_y), np.array(moments_z),
            odom_times, np.array(rolls), np.array(pitches))


# ── Load data ──
db_path = '/home/user/drone_control_pkgs/bag_folder/2026_03_22_nmpc_2/2026_03_22_nmpc_2_0.db3'
(cmd_t, F, Mx, My, Mz, odom_t, roll, pitch) = load_bag(db_path)

# ── Plot ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Roll + Mx
ax1 = axes[0]
color_angle = 'tab:blue'
color_moment = 'tab:red'

ax1.set_ylabel('Roll (deg)', color=color_angle, fontsize=12)
ax1.plot(odom_t, roll, color=color_angle, alpha=0.8, label='Roll')
ax1.tick_params(axis='y', labelcolor=color_angle)

ax1_twin = ax1.twinx()
ax1_twin.set_ylabel('Mx (Nm)', color=color_moment, fontsize=12)
ax1_twin.plot(cmd_t, Mx, color=color_moment, alpha=0.6, linewidth=0.8, label='Mx (from cmd_raw)')
ax1_twin.tick_params(axis='y', labelcolor=color_moment)
ax1.set_title('Roll angle vs Roll moment (Mx) from cmd_raw', fontsize=13)
ax1.grid(True, alpha=0.3)

# Pitch + My
ax2 = axes[1]
ax2.set_ylabel('Pitch (deg)', color=color_angle, fontsize=12)
ax2.plot(odom_t, pitch, color=color_angle, alpha=0.8, label='Pitch')
ax2.tick_params(axis='y', labelcolor=color_angle)

ax2_twin = ax2.twinx()
ax2_twin.set_ylabel('My (Nm)', color=color_moment, fontsize=12)
ax2_twin.plot(cmd_t, My, color=color_moment, alpha=0.6, linewidth=0.8, label='My (from cmd_raw)')
ax2_twin.tick_params(axis='y', labelcolor=color_moment)
ax2.set_title('Pitch angle vs Pitch moment (My) from cmd_raw', fontsize=13)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/drone_control_pkgs/bag_folder/moment_vs_roll_pitch.png', dpi=150)
plt.close()
print('Saved: bag_folder/moment_vs_roll_pitch.png')
