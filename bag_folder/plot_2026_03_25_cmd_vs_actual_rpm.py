#!/usr/bin/env python3
"""Plot cmd_raw (RPM) vs actual RPM for 2026_03_25 bag."""

import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt

MaxBit = 8191
MaxRpm = 9800

db_path = '/home/user/drone_control_pkgs/bag_folder/2026_03_25_nmpc_01_1/2026_03_25_nmpc_01_1_0.db3'
conn = sqlite3.connect(db_path)
c = conn.cursor()

# cmd_raw (topic_id=1)
c.execute('SELECT data FROM messages WHERE topic_id=1 ORDER BY timestamp')
cmd_times, cmd_rpms = [], []
for data, in c.fetchall():
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    if off % 2 != 0: off += 1
    cmds = np.array(struct.unpack_from('<6h', data, off), dtype=np.float64)
    cmd_times.append(sec + nsec * 1e-9)
    cmd_rpms.append(cmds * MaxRpm / MaxBit)

# actual_rpm (topic_id=2)
c.execute('SELECT data FROM messages WHERE topic_id=2 ORDER BY timestamp')
rpm_times, act_rpms = [], []
for data, in c.fetchall():
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    off = (off + 3) & ~3
    rpms = np.array(struct.unpack_from('<6i', data, off), dtype=np.float64)
    rpm_times.append(sec + nsec * 1e-9)
    act_rpms.append(rpms)

conn.close()

cmd_times = np.array(cmd_times)
rpm_times = np.array(rpm_times)
cmd_rpms = np.array(cmd_rpms)
act_rpms = np.array(act_rpms)

t0 = cmd_times[0]
cmd_times -= t0
rpm_times -= t0

fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)

for i in range(6):
    ax = axes[i]
    ax.plot(cmd_times, cmd_rpms[:, i], color='tab:red', linewidth=0.8, alpha=0.8, label='cmd_raw (RPM)')
    ax.plot(rpm_times, act_rpms[:, i], color='tab:blue', linewidth=0.8, alpha=0.8, label='actual RPM')
    ax.set_ylabel(f'Rotor {i+1}', fontsize=11)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc='upper right', fontsize=9)

axes[0].set_title('cmd_raw (RPM) vs actual RPM (2026_03_25)', fontsize=13)
axes[5].set_xlabel('Time (s)', fontsize=12)

plt.tight_layout()
out = '/home/user/drone_control_pkgs/bag_folder/2026_03_25_cmd_vs_actual_rpm.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'Saved: {out}')
