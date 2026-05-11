#!/usr/bin/env python3
"""Analyze a 2026_05_11_free_flight bag (default: eps_f_0p5).
Usage:  python3 analyze_2026_05_11_free_flight.py [<bag_subdir>]
Each output PNG is suffixed with the bag subdir name.
"""

import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG_SUBDIR = sys.argv[1] if len(sys.argv) > 1 else 'eps_f_0p5'
BAG_DIR = os.path.join(_HERE, '2026_05_11_free_flight', BAG_SUBDIR)
db_candidates = glob.glob(os.path.join(BAG_DIR, '*.db3'))
if not db_candidates:
    raise SystemExit(f'No .db3 found in {BAG_DIR}')
DB_PATH = db_candidates[0]
OUT_DIR = os.path.join(_HERE, '2026_05_11_free_flight')
TAG = BAG_SUBDIR
print(f'Analyzing: {DB_PATH}')


def _align(off, n):
    """XCDR1 alignment: relative offset (off - 4) must be multiple of n."""
    rel = off - 4
    pad = (-rel) % n
    return off + pad


def parse_odom(blob):
    off = 4 + 8  # CDR header + stamp(sec/nsec)
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4
    off += slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4
    off += slen2
    off = _align(off, 8)
    px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
    off += 36 * 8
    vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
    wx, wy, wz = struct.unpack_from('<3d', blob, off); off += 24
    return np.array([px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz])


def parse_wrench(blob):
    off = 4 + 8  # CDR header + stamp
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4
    off += slen
    off = _align(off, 8)
    return np.array(struct.unpack_from('<6d', blob, off))


def quat_to_rpy(q):
    qw, qx, qy, qz = q
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    return np.array([roll, pitch, yaw])


def quat_to_rotm(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ])


def force_to_rp(fx, fy, f_col, psi):
    """Reconstruct desired roll/pitch from world-frame (fx, fy) and collective thrust f_col.
    fz_world = sqrt(f_col^2 - fx^2 - fy^2) (assume positive); see force_to_attitude()."""
    fz_sq = f_col ** 2 - fx ** 2 - fy ** 2
    fz_world = np.sqrt(np.maximum(fz_sq, 0.0))
    n = np.sqrt(fx ** 2 + fy ** 2 + fz_world ** 2)
    if n < 1e-6:
        return 0.0, 0.0
    zbx, zby, zbz = fx / n, fy / n, fz_world / n
    xc = np.array([np.cos(psi), np.sin(psi), 0.0])
    yb = np.cross([zbx, zby, zbz], xc)
    yb_n = np.linalg.norm(yb)
    if yb_n < 1e-6:
        yb = np.array([-np.sin(psi), np.cos(psi), 0.0])
    else:
        yb = yb / yb_n
    xb = np.cross(yb, [zbx, zby, zbz])
    R = np.column_stack((xb, yb, [zbx, zby, zbz]))
    qw = 0.5 * np.sqrt(max(1 + R[0, 0] + R[1, 1] + R[2, 2], 0.0))
    if qw < 1e-6:
        # fall back via rpy from R
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(-R[2, 0])
        return roll, pitch
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    rpy = quat_to_rpy([qw, qx, qy, qz])
    return rpy[0], rpy[1]


# ── Load ──
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
tid = {name: i for i, name in c.execute('SELECT id, name FROM topics').fetchall()}


def fetch(topic, parser):
    ts, data = [], []
    for t, b in c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
                          (tid[topic],)).fetchall():
        ts.append(t); data.append(parser(bytes(b)))
    return np.array(ts, dtype=np.float64), np.array(data)


odom_ts, odom = fetch('/mavros/local_position/odom', parse_odom)
hgdo_ts, hgdo = fetch('/hgdo/wrench', parse_wrench)
ctrl_ts, ctrl = fetch('/nmpc/control', parse_wrench)
conn.close()

t0 = min(odom_ts[0], hgdo_ts[0], ctrl_ts[0])
odom_t = (odom_ts - t0) * 1e-9
hgdo_t = (hgdo_ts - t0) * 1e-9
ctrl_t = (ctrl_ts - t0) * 1e-9

# Position (subtract initial offset for readability)
pos = odom[:, 0:3] - odom[0, 0:3]

# RPY actual + world-frame velocity
rpy = np.array([quat_to_rpy(odom[i, 6:10]) for i in range(len(odom))])
v_world = np.array([quat_to_rotm(odom[i, 6:10]) @ odom[i, 3:6] for i in range(len(odom))])

# Desired roll/pitch from /nmpc/control using yaw interpolated from odom
psi_at_ctrl = np.interp(ctrl_t, odom_t, np.unwrap(rpy[:, 2]))
des_rp = np.zeros((len(ctrl_t), 2))
for i in range(len(ctrl_t)):
    des_rp[i] = force_to_rp(ctrl[i, 0], ctrl[i, 1], ctrl[i, 2], psi_at_ctrl[i])

# Actual roll/pitch interpolated onto ctrl_t for error stats
roll_act_at_ctrl = np.interp(ctrl_t, odom_t, rpy[:, 0])
pitch_act_at_ctrl = np.interp(ctrl_t, odom_t, rpy[:, 1])
roll_err = des_rp[:, 0] - roll_act_at_ctrl
pitch_err = des_rp[:, 1] - pitch_act_at_ctrl

# ─── HGDO-only equivalent desired roll/pitch ───
# Controller does:  F_des -= R_wb @ f_dist_body   (see pd_nmpc_att_with_dob.py:394)
# So the HGDO's contribution to F_des in world frame is  F_hgdo_w = -(R_wb @ f_dist).
# We reconstruct the desired tilt that this term alone would request, using the
# published collective f_col as the thrust magnitude.
hgdo_w = np.zeros((len(hgdo_t), 3))   # HGDO disturbance estimate in world frame
roll_hgdo_at_odom_t = np.zeros(len(odom_t))  # interp slots
# rebuild R_wb at each hgdo timestamp by interpolating quaternion → rpy then back
psi_at_hgdo  = np.interp(hgdo_t, odom_t, np.unwrap(rpy[:, 2]))
roll_at_hgdo = np.interp(hgdo_t, odom_t, rpy[:, 0])
pit_at_hgdo  = np.interp(hgdo_t, odom_t, rpy[:, 1])

def rpy_to_rotm(r, p, y):
    cr, sr = np.cos(r), np.sin(r); cp, sp = np.cos(p), np.sin(p); cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]])

for i in range(len(hgdo_t)):
    R_wb = rpy_to_rotm(roll_at_hgdo[i], pit_at_hgdo[i], psi_at_hgdo[i])
    hgdo_w[i] = R_wb @ hgdo[i, 0:3]   # body → world (matches controller)

# Use collective thrust from /nmpc/control interpolated to hgdo timestamps
fcol_at_hgdo = np.interp(hgdo_t, ctrl_t, ctrl[:, 2])

hgdo_rp = np.zeros((len(hgdo_t), 2))
for i in range(len(hgdo_t)):
    # F that controller injects on behalf of HGDO = -hgdo_w  (compensation)
    fx_w, fy_w, _ = -hgdo_w[i]
    hgdo_rp[i] = force_to_rp(fx_w, fy_w, fcol_at_hgdo[i], psi_at_hgdo[i])

# ─── PD-only (position/velocity) contribution to desired roll/pitch ───
# Controller:  F_des = F_pd - R_wb @ f_hgdo_body   →   F_pd = F_des + R_wb @ f_hgdo
# /nmpc/control publishes (fx, fy, f_col=||F_des||), so reconstruct F_des_z first.
hgdo_w_at_ctrl = np.zeros((len(ctrl_t), 3))
roll_at_ctrl = np.interp(ctrl_t, odom_t, rpy[:, 0])
pit_at_ctrl  = np.interp(ctrl_t, odom_t, rpy[:, 1])
hgdo_at_ctrl = np.column_stack([
    np.interp(ctrl_t, hgdo_t, hgdo[:, 0]),
    np.interp(ctrl_t, hgdo_t, hgdo[:, 1]),
    np.interp(ctrl_t, hgdo_t, hgdo[:, 2])])
for i in range(len(ctrl_t)):
    R_wb = rpy_to_rotm(roll_at_ctrl[i], pit_at_ctrl[i], psi_at_ctrl[i])
    hgdo_w_at_ctrl[i] = R_wb @ hgdo_at_ctrl[i]

# Full F_des in world frame from /nmpc/control
fz_sq = ctrl[:, 2] ** 2 - ctrl[:, 0] ** 2 - ctrl[:, 1] ** 2
Fdes_z = np.sqrt(np.maximum(fz_sq, 0.0))
Fdes_w = np.column_stack([ctrl[:, 0], ctrl[:, 1], Fdes_z])

# PD-only force (no HGDO compensation injected)
Fpd_w = Fdes_w + hgdo_w_at_ctrl
fcol_pd = np.linalg.norm(Fpd_w, axis=1)

pd_rp = np.zeros((len(ctrl_t), 2))
for i in range(len(ctrl_t)):
    pd_rp[i] = force_to_rp(Fpd_w[i, 0], Fpd_w[i, 1], fcol_pd[i], psi_at_ctrl[i])

# HGDO-only on ctrl timestamps (for matching plot grid)
hgdo_rp_at_ctrl = np.zeros((len(ctrl_t), 2))
for i in range(len(ctrl_t)):
    fx_w, fy_w, _ = -hgdo_w_at_ctrl[i]
    hgdo_rp_at_ctrl[i] = force_to_rp(fx_w, fy_w, ctrl[i, 2], psi_at_ctrl[i])


# ─────────── PLOT 1: HGDO fx, fy, fz ───────────
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
axes[0].plot(hgdo_t, hgdo[:, 0], 'r'); axes[0].set_ylabel('fx [N]'); axes[0].grid(alpha=0.3)
axes[0].set_title('HGDO Disturbance Force (/hgdo/wrench)')
axes[1].plot(hgdo_t, hgdo[:, 1], 'g'); axes[1].set_ylabel('fy [N]'); axes[1].grid(alpha=0.3)
axes[2].plot(hgdo_t, hgdo[:, 2], 'b'); axes[2].set_ylabel('fz [N]'); axes[2].grid(alpha=0.3)
axes[2].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'2026_05_11_{TAG}_hgdo_force.png'), dpi=120)
plt.close()

# ─────────── PLOT 2: position + world-frame velocity ───────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(odom_t, pos[:, 0], 'r', label='x')
axes[0].plot(odom_t, pos[:, 1], 'g', label='y')
axes[0].plot(odom_t, pos[:, 2], 'b', label='z')
axes[0].set_ylabel('Position [m]'); axes[0].grid(alpha=0.3); axes[0].legend(loc='upper right')
axes[0].set_title('Position (/mavros/local_position/odom)')

axes[1].plot(odom_t, v_world[:, 0], 'r', label='vx world')
axes[1].plot(odom_t, v_world[:, 1], 'g', label='vy world')
axes[1].plot(odom_t, v_world[:, 2], 'b', label='vz world')
axes[1].set_ylabel('Velocity [m/s]'); axes[1].set_xlabel('Time [s]')
axes[1].grid(alpha=0.3); axes[1].legend(loc='upper right')
axes[1].set_title('World-frame Linear Velocity  (R(q) @ v_body)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'2026_05_11_{TAG}_pos_vel_world.png'), dpi=120)
plt.close()

# ─────────── PLOT 3: desired vs actual roll/pitch ───────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(ctrl_t, np.degrees(des_rp[:, 0]), 'r--', label='Roll desired (from nmpc/control)')
axes[0].plot(odom_t, np.degrees(rpy[:, 0]), 'r',  alpha=0.8, label='Roll actual (odom)')
axes[0].set_ylabel('Roll [deg]'); axes[0].grid(alpha=0.3); axes[0].legend(loc='upper right')
axes[0].set_title('Desired (force→attitude) vs Actual Roll')

axes[1].plot(ctrl_t, np.degrees(des_rp[:, 1]), 'g--', label='Pitch desired (from nmpc/control)')
axes[1].plot(odom_t, np.degrees(rpy[:, 1]), 'g',  alpha=0.8, label='Pitch actual (odom)')
axes[1].set_ylabel('Pitch [deg]'); axes[1].set_xlabel('Time [s]')
axes[1].grid(alpha=0.3); axes[1].legend(loc='upper right')
axes[1].set_title('Desired (force→attitude) vs Actual Pitch')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'2026_05_11_{TAG}_des_vs_act_rp.png'), dpi=120)
plt.close()

# ─────────── PLOT 6: Torque decomposition (NMPC raw vs HGDO vs motor cmd) ───────────
#   /nmpc/control.torque = u_mpc  (NMPC raw, BEFORE HGDO compensation)
#   /hgdo/wrench.torque  = tau_dist
#   motor cmd torque     = u_mpc - tau_dist  (post-DOB, what actually drives rotors)
hgdo_tau_at_ctrl = np.column_stack([
    np.interp(ctrl_t, hgdo_t, hgdo[:, 3]),
    np.interp(ctrl_t, hgdo_t, hgdo[:, 4]),
    np.interp(ctrl_t, hgdo_t, hgdo[:, 5])])
motor_tau = ctrl[:, 3:6] - hgdo_tau_at_ctrl

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
labels_axis = ['Mx', 'My', 'Mz']
for k in range(3):
    axes[k].plot(ctrl_t, ctrl[:, 3 + k],  'k', lw=1.4, label=f'{labels_axis[k]}  /nmpc/control (NMPC raw, pre-DOB)')
    axes[k].plot(hgdo_t, hgdo[:, 3 + k],  'r', alpha=0.85, label=f'{labels_axis[k]}  /hgdo/wrench (HGDO)')
    axes[k].plot(ctrl_t, motor_tau[:, k], 'b', alpha=0.85, label=f'{labels_axis[k]}  motor cmd = nmpc − hgdo')
    axes[k].set_ylabel(f'{labels_axis[k]} [N·m]'); axes[k].grid(alpha=0.3)
    axes[k].legend(loc='upper right', fontsize=9)
axes[0].set_title('Torque decomposition: NMPC raw (/nmpc/control) vs HGDO vs motor command')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'2026_05_11_{TAG}_torque_decomp.png'), dpi=120)
plt.close()

# ─────────── PLOT 5: PD-only vs HGDO-only vs total desired vs actual (per axis) ───────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
axes[0].plot(ctrl_t, np.degrees(des_rp[:, 0]),       'k',  lw=1.6, label='Roll des total (/nmpc/control)')
axes[0].plot(ctrl_t, np.degrees(pd_rp[:, 0]),        'b',  lw=1.0, alpha=0.85,
             label='Roll des PD only (pos/vel)')
axes[0].plot(ctrl_t, np.degrees(hgdo_rp_at_ctrl[:, 0]), 'r', lw=1.0, alpha=0.85,
             label='Roll des HGDO only')
axes[0].plot(odom_t, np.degrees(rpy[:, 0]),          'g',  lw=0.9, alpha=0.6, label='Roll actual')
axes[0].set_ylabel('Roll [deg]'); axes[0].grid(alpha=0.3); axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_title('Roll: PD-only + HGDO-only = total desired (vs actual)')

axes[1].plot(ctrl_t, np.degrees(des_rp[:, 1]),       'k',  lw=1.6, label='Pitch des total (/nmpc/control)')
axes[1].plot(ctrl_t, np.degrees(pd_rp[:, 1]),        'b',  lw=1.0, alpha=0.85,
             label='Pitch des PD only (pos/vel)')
axes[1].plot(ctrl_t, np.degrees(hgdo_rp_at_ctrl[:, 1]), 'r', lw=1.0, alpha=0.85,
             label='Pitch des HGDO only')
axes[1].plot(odom_t, np.degrees(rpy[:, 1]),          'g',  lw=0.9, alpha=0.6, label='Pitch actual')
axes[1].set_ylabel('Pitch [deg]'); axes[1].set_xlabel('Time [s]')
axes[1].grid(alpha=0.3); axes[1].legend(loc='upper right', fontsize=9)
axes[1].set_title('Pitch: PD-only + HGDO-only = total desired (vs actual)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'2026_05_11_{TAG}_rp_decomp.png'), dpi=120)
plt.close()

# ─────────── PLOT 4: HGDO-only + PD-only desired roll/pitch + XY position overlay ───────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].plot(ctrl_t, np.degrees(des_rp[:, 0]),       'k',  lw=1.5, label='Roll des total (/nmpc/control)')
axes[0].plot(ctrl_t, np.degrees(pd_rp[:, 0]),        'b',  alpha=0.85, label='Roll des PD only (pos/vel)')
axes[0].plot(hgdo_t, np.degrees(hgdo_rp[:, 0]),      'r',  alpha=0.85, label='Roll des HGDO only')
axes[0].set_ylabel('Roll [deg]'); axes[0].grid(alpha=0.3); axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_title('Desired Roll decomposition: total = PD only + HGDO only')

axes[1].plot(ctrl_t, np.degrees(des_rp[:, 1]),       'k',  lw=1.5, label='Pitch des total (/nmpc/control)')
axes[1].plot(ctrl_t, np.degrees(pd_rp[:, 1]),        'b',  alpha=0.85, label='Pitch des PD only (pos/vel)')
axes[1].plot(hgdo_t, np.degrees(hgdo_rp[:, 1]),      'r',  alpha=0.85, label='Pitch des HGDO only')
axes[1].set_ylabel('Pitch [deg]'); axes[1].grid(alpha=0.3); axes[1].legend(loc='upper right', fontsize=9)
axes[1].set_title('Desired Pitch decomposition: total = PD only + HGDO only')

axes[2].plot(odom_t, pos[:, 0], 'r', label='x')
axes[2].plot(odom_t, pos[:, 1], 'g', label='y')
axes[2].set_ylabel('XY position [m]'); axes[2].set_xlabel('Time [s]')
axes[2].grid(alpha=0.3); axes[2].legend(loc='upper right')
axes[2].set_title('XY position drift (odom)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'2026_05_11_{TAG}_hgdo_des_rp.png'), dpi=120)
plt.close()

# ─────────── Stats ───────────
def stats(name, x):
    x_deg = np.degrees(x)
    print(f'  {name:18s}  std={np.std(x_deg):7.4f} deg   min={np.min(x_deg):8.4f} deg   max={np.max(x_deg):8.4f} deg')


print('\n=== Roll / Pitch statistics (deg) ===')
print('-- Desired (from /nmpc/control) --')
stats('roll_des',  des_rp[:, 0])
stats('pitch_des', des_rp[:, 1])
print('-- Actual (from odom) --')
stats('roll_act',  rpy[:, 0])
stats('pitch_act', rpy[:, 1])
print('-- Tracking error (des - act, on ctrl timestamps) --')
stats('roll_err',  roll_err)
stats('pitch_err', pitch_err)
print('-- HGDO-only equivalent desired tilt --')
stats('roll_hgdo',  hgdo_rp[:, 0])
stats('pitch_hgdo', hgdo_rp[:, 1])
print('-- PD-only (pos/vel) equivalent desired tilt --')
stats('roll_pd',  pd_rp[:, 0])
stats('pitch_pd', pd_rp[:, 1])

# Torque stats (N·m, not deg)
def stats_lin(name, x, unit='N·m'):
    print(f'  {name:18s}  std={np.std(x):7.4f} {unit}   min={np.min(x):8.4f}   max={np.max(x):8.4f}')

print('\n-- Torque (Mx, My, Mz) [N·m] --')
for k, ax in enumerate(['Mx', 'My', 'Mz']):
    stats_lin(f'{ax}_nmpc',  ctrl[:, 3 + k])
    stats_lin(f'{ax}_hgdo',  hgdo[:, 3 + k])
    stats_lin(f'{ax}_motor', motor_tau[:, k])

# Contribution share: std(hgdo_rp) / std(total_des_rp) on aligned time grid
des_roll_on_hgdo  = np.interp(hgdo_t, ctrl_t, des_rp[:, 0])
des_pitch_on_hgdo = np.interp(hgdo_t, ctrl_t, des_rp[:, 1])
mask = (hgdo_t >= ctrl_t[0]) & (hgdo_t <= ctrl_t[-1]) & (np.abs(hgdo[:, 0]) + np.abs(hgdo[:, 1]) > 1e-6)
if mask.sum() > 100:
    r_share = np.std(hgdo_rp[mask, 0]) / max(np.std(des_roll_on_hgdo[mask]),  1e-9)
    p_share = np.std(hgdo_rp[mask, 1]) / max(np.std(des_pitch_on_hgdo[mask]), 1e-9)
    corr_r = np.corrcoef(hgdo_rp[mask, 0], des_roll_on_hgdo[mask])[0, 1]
    corr_p = np.corrcoef(hgdo_rp[mask, 1], des_pitch_on_hgdo[mask])[0, 1]
    print(f'\n  std(hgdo_roll)  / std(total_des_roll)  = {r_share:.3f}   corr={corr_r:+.3f}')
    print(f'  std(hgdo_pitch) / std(total_des_pitch) = {p_share:.3f}   corr={corr_p:+.3f}')

print('\nFigures saved to:', OUT_DIR)
print(f'  - 2026_05_11_{TAG}_hgdo_force.png')
print(f'  - 2026_05_11_{TAG}_pos_vel_world.png')
print(f'  - 2026_05_11_{TAG}_des_vs_act_rp.png')
print(f'  - 2026_05_11_{TAG}_hgdo_des_rp.png')
print(f'  - 2026_05_11_{TAG}_rp_decomp.png')
print(f'  - 2026_05_11_{TAG}_torque_decomp.png')
