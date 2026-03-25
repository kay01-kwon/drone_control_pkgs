#!/usr/bin/env python3
"""Plot moments from actual RPM & cmd_raw, roll/pitch, and angular velocities (4 rows)."""

import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ── Constants ──
C_T = 1.465e-07       # N/RPM^2
k_m = 0.01569          # Nm/N
l = 0.265              # arm length (m)
MaxBit = 8191
MaxRpm = 9800

# ── K_forward matrix (same as existing code) ──
lx1 = l * np.sin(np.pi / 3); ly1 = l * np.cos(np.pi / 3)
lx2 = 0.0;                    ly2 = l
lx3 = -l * np.sin(np.pi / 3); ly3 = l * np.cos(np.pi / 3)
lx4 = -l * np.sin(np.pi / 3); ly4 = -l * np.cos(np.pi / 3)
lx5 = 0.0;                    ly5 = -l
lx6 = l * np.sin(np.pi / 3);  ly6 = -l * np.cos(np.pi / 3)

K_forward = np.array([
    [1, 1, 1, 1, 1, 1],
    [ly1, ly2, ly3, ly4, ly5, ly6],
    [-lx1, -lx2, -lx3, -lx4, -lx5, -lx6],
    [-k_m, k_m, -k_m, k_m, -k_m, k_m]
])


def cdr_align(off, alignment, cdr_start=4):
    rel = off - cdr_start
    rem = rel % alignment
    if rem != 0:
        off += alignment - rem
    return off


def parse_cmd_raw(data):
    """Parse HexaCmdRaw CDR → (timestamp, cmd[6])."""
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    if off % 2 != 0: off += 1  # align to 2 bytes for int16
    cmds = np.array(struct.unpack_from('<6h', data, off), dtype=np.float64)
    return sec + nsec * 1e-9, cmds


def parse_actual_rpm(data):
    """Parse HexaActualRpm CDR → (timestamp, rpm[6])."""
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    off = (off + 3) & ~3
    rpms = np.array(struct.unpack_from('<6i', data, off), dtype=np.float64)
    return sec + nsec * 1e-9, rpms


def parse_odom_full(data):
    """Parse Odometry CDR → (timestamp, roll, pitch, yaw, wx, wy, wz)."""
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    # frame_id
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    off = cdr_align(off, 4)
    # child_frame_id
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    off = cdr_align(off, 8)
    # position (3d) + quaternion (4d)
    struct.unpack_from('<3d', data, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', data, off); off += 32
    # pose covariance (36d)
    off += 288
    # twist: linear (3d) + angular (3d)
    struct.unpack_from('<3d', data, off); off += 24
    wx, wy, wz = struct.unpack_from('<3d', data, off)

    t = sec + nsec * 1e-9
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if norm < 1e-10:
        return t, 0.0, 0.0, 0.0, wx, wy, wz
    r = Rotation.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    return t, roll, pitch, yaw, wx, wy, wz


def load_bag(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name FROM topics')
    topics = {name: tid for tid, name in c.fetchall()}

    # ── cmd_raw → moments ──
    tid = topics['/uav/cmd_raw']
    c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    cmd_times, cmd_Mx, cmd_My = [], [], []
    for ts, data in c.fetchall():
        t, cmds = parse_cmd_raw(data)
        rpms = cmds * MaxRpm / MaxBit
        thrusts = C_T * rpms ** 2
        u = K_forward @ thrusts
        cmd_times.append(t)
        cmd_Mx.append(u[1])
        cmd_My.append(u[2])

    # ── actual_rpm → moments ──
    tid = topics['/uav/actual_rpm']
    c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    rpm_times, Mx_list, My_list, Mz_list = [], [], [], []
    for ts, data in c.fetchall():
        t, rpms = parse_actual_rpm(data)
        thrusts = C_T * rpms ** 2
        u = K_forward @ thrusts
        rpm_times.append(t)
        Mx_list.append(u[1])
        My_list.append(u[2])
        Mz_list.append(u[3])

    # ── odom → roll, pitch, angular velocity ──
    tid = topics['/mavros/local_position/odom']
    c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    odom_times, rolls, pitches = [], [], []
    wxs, wys, wzs = [], [], []
    for ts, data in c.fetchall():
        t, roll, pitch, yaw, wx, wy, wz = parse_odom_full(data)
        odom_times.append(t)
        rolls.append(roll)
        pitches.append(pitch)
        wxs.append(wx)
        wys.append(wy)
        wzs.append(wz)

    conn.close()

    cmd_times = np.array(cmd_times)
    rpm_times = np.array(rpm_times)
    odom_times = np.array(odom_times)
    t0 = rpm_times[0]  # reference time = actual_rpm start
    cmd_times -= t0
    rpm_times -= t0
    odom_times -= t0

    return (cmd_times, np.array(cmd_Mx), np.array(cmd_My),
            rpm_times, np.array(Mx_list), np.array(My_list), np.array(Mz_list),
            odom_times, np.array(rolls), np.array(pitches),
            np.array(wxs), np.array(wys), np.array(wzs))


# ── Load ──
db_path = '/home/user/drone_control_pkgs/bag_folder/2026_03_25_nmpc_01_1/2026_03_25_nmpc_01_1_0.db3'
(cmd_t, cmd_Mx, cmd_My,
 rpm_t, Mx, My, Mz,
 odom_t, roll, pitch, wx, wy, wz) = load_bag(db_path)

# ── Plot: 4 rows ──
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Row 1: Moments from cmd_raw
ax = axes[0]
ax.plot(cmd_t, cmd_Mx, color='tab:red', alpha=0.8, linewidth=0.8, label='Mx (roll)')
ax.plot(cmd_t, cmd_My, color='tab:blue', alpha=0.8, linewidth=0.8, label='My (pitch)')
ax.set_ylabel('Moment (Nm)', fontsize=12)
ax.set_title('Moments from cmd_raw (commanded)', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Row 2: Moments from actual RPM
ax = axes[1]
ax.plot(rpm_t, Mx, color='tab:red', alpha=0.8, linewidth=0.8, label='Mx (roll)')
ax.plot(rpm_t, My, color='tab:blue', alpha=0.8, linewidth=0.8, label='My (pitch)')
ax.set_ylabel('Moment (Nm)', fontsize=12)
ax.set_title('Moments from actual RPM', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Row 3: Roll & Pitch
ax = axes[2]
ax.plot(odom_t, roll, color='tab:red', alpha=0.8, label='Roll')
ax.plot(odom_t, pitch, color='tab:blue', alpha=0.8, label='Pitch')
ax.set_ylabel('Angle (deg)', fontsize=12)
ax.set_title('Roll & Pitch', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Row 4: Angular velocities (wx, wy, wz)
ax = axes[3]
ax.plot(odom_t, wx, color='tab:red', alpha=0.8, linewidth=0.8, label=r'$\omega_x$')
ax.plot(odom_t, wy, color='tab:blue', alpha=0.8, linewidth=0.8, label=r'$\omega_y$')
ax.plot(odom_t, wz, color='tab:green', alpha=0.8, linewidth=0.8, label=r'$\omega_z$')
ax.set_ylabel('Angular velocity (rad/s)', fontsize=12)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_title('Angular velocities', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = '/home/user/drone_control_pkgs/bag_folder/actual_rpm_moment_rp_omega.png'
plt.savefig(out_path, dpi=150)
plt.close()
print(f'Saved: {out_path}')
