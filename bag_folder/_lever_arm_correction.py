#!/usr/bin/env python3
"""Apply the CAD-given lever-arm correction and quantify its impact.

Assumes  CoM is r_z below the measurement point  →  r_offset_body = (rx, ry, +r_z)
The corrected CoM velocity is:
    v_CoM_world = v_meas_world − ω_world × (R(q) · r_offset_body)

Shows how much PD's Kd·e_v contribution changes — i.e., the practical
impact of the lever-arm correction on the desired tilt command.

Usage:
  python3 _lever_arm_correction.py <bag_subdir> <t_center> [<win_s>] \
                                   [<date>] [<tag>] [<rz_cm>] [<rx_cm>] [<ry_cm>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
T_CENTER = float(sys.argv[2])
WIN = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
DATE = sys.argv[4] if len(sys.argv) > 4 else '2026_05_14_free_flight'
TAG_OVR = sys.argv[5] if len(sys.argv) > 5 else None
R_Z_CM = float(sys.argv[6]) if len(sys.argv) > 6 else 5.0    # mocap 5 cm above CoM by default
R_X_CM = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
R_Y_CM = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
BAG_DIR = os.path.join(_HERE, DATE, BAG)
db = glob.glob(os.path.join(BAG_DIR, '*.db3'))[0]
parts = BAG.split('/')
OUT_DIR = os.path.join(_HERE, DATE, *parts[:-1])
TAG = TAG_OVR if TAG_OVR else parts[-1]

r_offset_body = np.array([R_X_CM, R_Y_CM, R_Z_CM]) * 0.01  # m
M = 3.146; G = 9.81
Kp = np.array([2.0, 2.0, 5.0]); Kd = np.array([2.0, 2.0, 3.5])


def _align(off, n):
    return off + (-(off - 4)) % n


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
    wx, wy, wz = struct.unpack_from('<3d', blob, off); off += 24
    return px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz


def parse_ref(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return struct.unpack_from('<8d', blob, off)


def quat_to_rotm(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ])


con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("SELECT id,name FROM topics")
tids = {n: i for i, n in cur.fetchall()}
tid = tids['/mavros/local_position/odom']
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tid}")
t0 = cur.fetchone()[0]


def fetch(topic, parser):
    tid = tids[topic]
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tid} ORDER BY timestamp")
    rows = cur.fetchall()
    t = np.array([(ts - t0) * 1e-9 for ts, _ in rows])
    d = np.array([parser(blob) for _, blob in rows])
    return t, d


ot, odom = fetch('/mavros/local_position/odom', parse_odom)
rt, ref = fetch('/nmpc/ref', parse_ref)
con.close()

m = (ot >= T_CENTER - WIN / 2) & (ot <= T_CENTER + WIN / 2)
t = ot[m]
p = odom[m, 0:3]
qs = odom[m, 3:7]
v_body = odom[m, 7:10]
w_body = odom[m, 10:13]

N = len(t)
v_world_meas = np.empty_like(v_body)
v_world_CoM  = np.empty_like(v_body)
w_world = np.empty_like(w_body)
lever_term_world = np.empty_like(v_body)

for i in range(N):
    R = quat_to_rotm(qs[i, 0], qs[i, 1], qs[i, 2], qs[i, 3])
    v_world_meas[i] = R @ v_body[i]
    w_w = R @ w_body[i]; w_world[i] = w_w
    r_w = R @ r_offset_body
    lever_term_world[i] = np.cross(w_w, r_w)
    v_world_CoM[i] = v_world_meas[i] - lever_term_world[i]

# Reference (ref_v)
mR = (rt >= T_CENTER - WIN / 2) & (rt <= T_CENTER + WIN / 2)
t_r = rt[mR]; rp_arr = ref[mR, 0:3]; rv_arr = ref[mR, 3:6]
if len(t_r) > 1:
    rx = np.interp(t, t_r, rp_arr[:, 0]); ry = np.interp(t, t_r, rp_arr[:, 1])
    rvx = np.interp(t, t_r, rv_arr[:, 0]); rvy = np.interp(t, t_r, rv_arr[:, 1])
else:
    rx = np.full(N, rp_arr[0, 0] if len(rp_arr) else 0)
    ry = np.full(N, rp_arr[0, 1] if len(rp_arr) else 0)
    rvx = np.zeros(N); rvy = np.zeros(N)

# PD e_v with measured velocity vs corrected
e_vx_meas = rvx - v_world_meas[:, 0]
e_vy_meas = rvy - v_world_meas[:, 1]
e_vx_corr = rvx - v_world_CoM[:, 0]
e_vy_corr = rvy - v_world_CoM[:, 1]

# Kd contribution to tilt (deg)
D_tilt_x_meas = np.degrees(np.arctan2(M * Kd[0] * e_vx_meas, M * G))
D_tilt_x_corr = np.degrees(np.arctan2(M * Kd[0] * e_vx_corr, M * G))
D_tilt_y_meas = np.degrees(np.arctan2(M * Kd[1] * e_vy_meas, M * G))
D_tilt_y_corr = np.degrees(np.arctan2(M * Kd[1] * e_vy_corr, M * G))

print(f'\nLever-arm correction with r_offset_body = ({R_X_CM:+.1f}, {R_Y_CM:+.1f}, {R_Z_CM:+.1f}) cm')
print(f'Window: {t[0]:.1f} → {t[-1]:.1f} s')
print(f'\n--- Velocity comparison ---')
print(f'  v_x: meas std = {v_world_meas[:,0].std():.3f}  →  CoM std = {v_world_CoM[:,0].std():.3f}  '
      f'({100*(v_world_CoM[:,0].std()-v_world_meas[:,0].std())/v_world_meas[:,0].std():+.1f}%)')
print(f'  v_y: meas std = {v_world_meas[:,1].std():.3f}  →  CoM std = {v_world_CoM[:,1].std():.3f}  '
      f'({100*(v_world_CoM[:,1].std()-v_world_meas[:,1].std())/v_world_meas[:,1].std():+.1f}%)')
print(f'  lever term magnitude: |ω×r| std_x = {lever_term_world[:,0].std()*100:.2f} cm/s, '
      f'std_y = {lever_term_world[:,1].std()*100:.2f} cm/s')
print(f'\n--- D-term tilt contribution (Kd·e_v) ---')
print(f'  D tilt X: meas std = {D_tilt_x_meas.std():.2f}°  →  corrected = {D_tilt_x_corr.std():.2f}°  '
      f'({100*(D_tilt_x_corr.std()-D_tilt_x_meas.std())/D_tilt_x_meas.std():+.1f}%)')
print(f'  D tilt Y: meas std = {D_tilt_y_meas.std():.2f}°  →  corrected = {D_tilt_y_corr.std():.2f}°  '
      f'({100*(D_tilt_y_corr.std()-D_tilt_y_meas.std())/D_tilt_y_meas.std():+.1f}%)')

fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

ax = axes[0]
ax.plot(t, v_world_meas[:, 0], 'r',  lw=1.5, label='v_x measured')
ax.plot(t, v_world_CoM[:, 0],  'g',  lw=1.2, label='v_x CoM-corrected')
ax.plot(t, lever_term_world[:, 0], 'b--', lw=1.0, alpha=0.7, label='ω × r_offset (subtracted)')
ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('v_x [m/s]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')
ax.set_title(f'{TAG} — lever-arm correction with r_offset = '
             f'({R_X_CM:+.1f}, {R_Y_CM:+.1f}, {R_Z_CM:+.1f}) cm')

ax = axes[1]
ax.plot(t, v_world_meas[:, 1], 'r',  lw=1.5, label='v_y measured')
ax.plot(t, v_world_CoM[:, 1],  'g',  lw=1.2, label='v_y CoM-corrected')
ax.plot(t, lever_term_world[:, 1], 'b--', lw=1.0, alpha=0.7, label='ω × r_offset (subtracted)')
ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('v_y [m/s]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')

ax = axes[2]
ax.plot(t, D_tilt_x_meas, 'r',  lw=1.5, label='D tilt X using v_meas')
ax.plot(t, D_tilt_x_corr, 'g',  lw=1.2, label='D tilt X using v_CoM')
ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('D tilt X [deg]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')

ax = axes[3]
ax.plot(t, D_tilt_y_meas, 'r',  lw=1.5, label='D tilt Y using v_meas')
ax.plot(t, D_tilt_y_corr, 'g',  lw=1.2, label='D tilt Y using v_CoM')
ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('D tilt Y [deg]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')
ax.set_xlabel('time [s]')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_lever_corr_rz{int(R_Z_CM)}_t{int(T_CENTER)}.png')
plt.savefig(out, dpi=120)
print(f'\nSaved: {out}')
