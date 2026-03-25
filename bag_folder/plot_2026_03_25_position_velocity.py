#!/usr/bin/env python3
"""Plot position and world-frame linear velocity for 2026_03_25 bag."""

import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def cdr_align(off, alignment, cdr_start=4):
    rel = off - cdr_start
    rem = rel % alignment
    if rem:
        off += alignment - rem
    return off


db_path = '/home/user/drone_control_pkgs/bag_folder/2026_03_25_nmpc_01_1/2026_03_25_nmpc_01_1_0.db3'
conn = sqlite3.connect(db_path)
c = conn.cursor()

# odom (topic_id=5)
c.execute('SELECT data FROM messages WHERE topic_id=5 ORDER BY timestamp')
times, px_l, py_l, pz_l, vx_w, vy_w, vz_w = [], [], [], [], [], [], []
for data, in c.fetchall():
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen; off = cdr_align(off, 4)
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen; off = cdr_align(off, 8)
    px, py, pz = struct.unpack_from('<3d', data, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', data, off); off += 32
    off += 288  # pose covariance
    vx, vy, vz = struct.unpack_from('<3d', data, off)

    # Body → World
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    v_world = R @ np.array([vx, vy, vz])

    times.append(sec + nsec * 1e-9)
    px_l.append(px); py_l.append(py); pz_l.append(pz)
    vx_w.append(v_world[0]); vy_w.append(v_world[1]); vz_w.append(v_world[2])

conn.close()

times = np.array(times)
times -= times[0]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax = axes[0]
ax.plot(times, px_l, color='tab:red', linewidth=0.8, label='x')
ax.plot(times, py_l, color='tab:blue', linewidth=0.8, label='y')
ax.plot(times, pz_l, color='tab:green', linewidth=0.8, label='z')
ax.set_ylabel('Position (m)', fontsize=12)
ax.set_title('Position (2026_03_25)', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(times, vx_w, color='tab:red', linewidth=0.8, label='vx')
ax.plot(times, vy_w, color='tab:blue', linewidth=0.8, label='vy')
ax.plot(times, vz_w, color='tab:green', linewidth=0.8, label='vz')
ax.set_ylabel('Velocity (m/s)', fontsize=12)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_title('Linear velocity - world frame (R * v_body) (2026_03_25)', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = '/home/user/drone_control_pkgs/bag_folder/2026_03_25_position_velocity.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'Saved: {out}')
