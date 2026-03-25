#!/usr/bin/env python3
"""Plot per-rotor cmd_raw for 2026_03_25 bag."""

import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt

MaxBit = 8191
MaxRpm = 9800

db_path = '/home/user/drone_control_pkgs/bag_folder/2026_03_25_nmpc_01_1/2026_03_25_nmpc_01_1_0.db3'
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute('SELECT data FROM messages WHERE topic_id=1 ORDER BY timestamp')
times, all_cmds = [], []
for data, in c.fetchall():
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    if off % 2 != 0: off += 1
    cmds = struct.unpack_from('<6h', data, off)
    times.append(sec + nsec * 1e-9)
    all_cmds.append(cmds)

conn.close()

times = np.array(times)
times -= times[0]
cmds = np.array(all_cmds)

fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple', 'tab:brown']

for i in range(6):
    ax = axes[i]
    ax.plot(times, cmds[:, i], color=colors[i], linewidth=0.8)
    ax.set_ylabel(f'Rotor {i+1}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-100, 8500)

axes[0].set_title('cmd_raw (2026_03_25)', fontsize=13)
axes[5].set_xlabel('Time (s)', fontsize=12)

plt.tight_layout()
out = '/home/user/drone_control_pkgs/bag_folder/2026_03_25_cmd_raw.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'Saved: {out}')
