#!/usr/bin/env python3
"""Decompose the PD position-loop output into Kp·e_p and Kd·e_v terms,
and compare with the resulting desired tilt. Lets us tell whether the
"desired tilt noise" we see is measurement noise on velocity (would be
high-frequency) or just legitimate PD response to drone motion
(low-frequency, smooth).

Usage:  python3 pd_term_decomp.py [<bag_subdir> [<date_dir>]]
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

# Controller gains  (from drone_control/config/pd_nmpc_att_with_hgdo.yaml)
Kp = np.array([2.0, 2.0, 5.0])
Kd = np.array([2.0, 2.0, 3.5])
M, G = 3.188, 9.81

print(f'Analyzing: {DB}')
print(f'Kp = {Kp.tolist()},  Kd = {Kd.tolist()}')


def _align(off, n):
    return off + ((-(off - 4)) % n)


def parse_odom(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
    off += 36 * 8
    vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
    return np.array([px, py, pz, vx, vy, vz, qw, qx, qy, qz])


def quat_to_rotm(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx ** 2 + qy ** 2)]])


# ── Load odom ──
conn = sqlite3.connect(DB)
c = conn.cursor()
tid = {n: i for i, n in c.execute('SELECT id, name FROM topics').fetchall()}
rows = c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
                 (tid['/mavros/local_position/odom'],)).fetchall()
conn.close()

ts = np.array([r[0] for r in rows], dtype=np.float64)
od = np.array([parse_odom(bytes(r[1])) for r in rows])
t  = (ts - ts[0]) * 1e-9
p  = od[:, 0:3]
vb = od[:, 3:6]
q  = od[:, 6:10]

# World-frame velocity
vw = np.array([quat_to_rotm(q[i]) @ vb[i] for i in range(len(q))])

# Pick steady-state segment (drop first 5 s for takeoff)
mask = t > max(5.0, t[0])
t_a = t[mask]; p_a = p[mask]; vw_a = vw[mask]

# Reference position: use mean of airborne segment (we don't have /nmpc/ref reliably)
ref_p = p_a.mean(axis=0)
ref_v = np.zeros(3)

e_p = ref_p - p_a
e_v = ref_v - vw_a

Kp_term = e_p * Kp
Kd_term = e_v * Kd
a_des   = Kp_term + Kd_term
F_des   = M * (a_des + np.array([0, 0, G]))

# Desired tilt from F_des (yaw assumed 0 — we just want magnitude for noise check)
F_norm = np.linalg.norm(F_des, axis=1) + 1e-9
pitch_des = np.degrees(np.arctan2(F_des[:, 0], F_des[:, 2]))
roll_des  = np.degrees(-np.arctan2(F_des[:, 1], np.sqrt(F_des[:, 0] ** 2 + F_des[:, 2] ** 2)))

# Stats
def stat(name, x):
    print(f'  {name:24s} std={x.std():7.4f}  range [{x.min():+8.3f}, {x.max():+8.3f}]  mean={x.mean():+7.4f}')

print('\n== PD term decomposition (airborne segment) ==')
print('-- Position error e_p [m] --')
for k, ax in enumerate('xyz'):
    stat(f'e_p_{ax}', e_p[:, k])
print('-- World velocity error e_v [m/s] (= -v_world) --')
for k, ax in enumerate('xyz'):
    stat(f'e_v_{ax}', e_v[:, k])
print('-- Kp·e_p term [m/s²] --')
for k, ax in enumerate('xyz'):
    stat(f'Kp*e_p_{ax}', Kp_term[:, k])
print('-- Kd·e_v term [m/s²] --')
for k, ax in enumerate('xyz'):
    stat(f'Kd*e_v_{ax}', Kd_term[:, k])
print('-- a_des total [m/s²] --')
for k, ax in enumerate('xyz'):
    stat(f'a_des_{ax}', a_des[:, k])
print('-- Resulting desired tilt [deg] --')
stat('roll_des',  roll_des)
stat('pitch_des', pitch_des)


# ── Plots ──
fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
# Columns: x (left, → pitch), y (right, → roll)
for col, name in enumerate(['x', 'y']):
    axes[0, col].plot(t_a, e_p[:, col], 'k', lw=0.9)
    axes[0, col].set_ylabel(f'e_p_{name} [m]'); axes[0, col].grid(alpha=0.3)
    axes[0, col].set_title(f'Position error ({name})')

    axes[1, col].plot(t_a, e_v[:, col], 'k', lw=0.9)
    axes[1, col].set_ylabel(f'e_v_{name} [m/s]'); axes[1, col].grid(alpha=0.3)
    axes[1, col].set_title(f'World velocity error (−v_world_{name})')

    axes[2, col].plot(t_a, Kp_term[:, col], 'b', lw=0.9, label=f'Kp·e_p_{name}')
    axes[2, col].plot(t_a, Kd_term[:, col], 'r', lw=0.9, alpha=0.8, label=f'Kd·e_v_{name}')
    axes[2, col].plot(t_a, a_des [:, col], 'k', lw=1.1, label=f'sum a_des_{name}')
    axes[2, col].set_ylabel(f'a_des_{name} [m/s²]'); axes[2, col].grid(alpha=0.3)
    axes[2, col].legend(loc='upper right', fontsize=9)
    axes[2, col].set_title(f'PD output decomposition — {name}')

# Resulting desired tilt
axes[3, 0].plot(t_a, pitch_des, 'g', lw=0.9, label='Pitch des (from PD)')
axes[3, 0].set_ylabel('Pitch [deg]'); axes[3, 0].set_xlabel('Time [s]'); axes[3, 0].grid(alpha=0.3)
axes[3, 0].legend(loc='upper right', fontsize=9)
axes[3, 0].set_title('Desired pitch (from a_des_x)')

axes[3, 1].plot(t_a, roll_des,  'r', lw=0.9, label='Roll des (from PD)')
axes[3, 1].set_ylabel('Roll [deg]'); axes[3, 1].set_xlabel('Time [s]'); axes[3, 1].grid(alpha=0.3)
axes[3, 1].legend(loc='upper right', fontsize=9)
axes[3, 1].set_title('Desired roll (from −a_des_y)')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_pd_term_decomp.png'), dpi=120)
plt.close()

print(f'\nSaved: {TAG}_pd_term_decomp.png')
