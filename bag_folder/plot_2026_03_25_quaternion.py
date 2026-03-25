"""Plot quaternion components and Euler angles from 2026_03_25 bag.
CDR2 encoding: PoseWithCovariance has 4-byte DHEADER, shifting offsets by +4.
Corrected offsets: position=44, quaternion=68, twist.linear=388, twist.angular=412.
"""
import sqlite3, struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

db_path = 'bag_folder/2026_03_25_nmpc_01_1/2026_03_25_nmpc_01_1_0.db3'

conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute('SELECT id, name FROM topics')
topics = {name: tid for tid, name in c.fetchall()}

tid = topics['/mavros/local_position/odom']
c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))

times, qws, qxs, qys, qzs = [], [], [], [], []
rolls, pitches, yaws = [], [], []

for ts, data in c.fetchall():
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    t = sec + nsec * 1e-9

    # CDR2: PoseWithCovariance DHEADER at off=40, position at off=44, quaternion at off=68
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 68)

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-10:
        continue

    times.append(t)
    qws.append(qw)
    qxs.append(qx)
    qys.append(qy)
    qzs.append(qz)

    q_normed = q / norm
    r = Rotation.from_quat(q_normed)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    rolls.append(roll)
    pitches.append(pitch)
    yaws.append(yaw)

conn.close()

times = np.array(times)
t0 = times[0]
times -= t0

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle('2026_03_25 Quaternion & Euler Angles (corrected CDR2 offset)', fontsize=14)

# Row 1: Quaternion
ax = axes[0]
ax.plot(times, qws, label='qw', linewidth=1.0)
ax.plot(times, qxs, label='qx', linewidth=1.0)
ax.plot(times, qys, label='qy', linewidth=1.0)
ax.plot(times, qzs, label='qz', linewidth=1.0)
ax.set_ylabel('Quaternion')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_title('Quaternion Components')

# Row 2: Euler angles
ax = axes[1]
ax.plot(times, rolls, label='Roll', linewidth=1.0, color='tab:red')
ax.plot(times, pitches, label='Pitch', linewidth=1.0, color='tab:blue')
ax.plot(times, yaws, label='Yaw', linewidth=1.0, color='tab:green')
ax.set_ylabel('Angle [deg]')
ax.set_xlabel('Time [s]')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_title('Euler Angles (xyz intrinsic)')

plt.tight_layout()
plt.savefig('bag_folder/2026_03_25_quaternion.png', dpi=150)
plt.show()
print('Saved: bag_folder/2026_03_25_quaternion.png')
