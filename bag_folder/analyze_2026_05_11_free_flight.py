#!/usr/bin/env python3
"""Analyze 2026_05_11_free_flight/eps_f_0p5:
  - /hgdo/wrench  : fx, fy, fz vs time
  - /mavros/local_position/odom : position + world-frame linear velocity (R @ v_body)
  - /nmpc/control : desired roll/pitch reconstructed from (fx, fy, f_col)
                    overlaid with actual roll/pitch from odom; print std / min / max.
"""

import os, sqlite3, struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, '2026_05_11_free_flight/eps_f_0p5/eps_f_0p5_0.db3')
OUT_DIR = os.path.join(_HERE, '2026_05_11_free_flight')


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


# ─────────── PLOT 1: HGDO fx, fy, fz ───────────
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
axes[0].plot(hgdo_t, hgdo[:, 0], 'r'); axes[0].set_ylabel('fx [N]'); axes[0].grid(alpha=0.3)
axes[0].set_title('HGDO Disturbance Force (/hgdo/wrench)')
axes[1].plot(hgdo_t, hgdo[:, 1], 'g'); axes[1].set_ylabel('fy [N]'); axes[1].grid(alpha=0.3)
axes[2].plot(hgdo_t, hgdo[:, 2], 'b'); axes[2].set_ylabel('fz [N]'); axes[2].grid(alpha=0.3)
axes[2].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '2026_05_11_hgdo_force.png'), dpi=120)
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
plt.savefig(os.path.join(OUT_DIR, '2026_05_11_pos_vel_world.png'), dpi=120)
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
plt.savefig(os.path.join(OUT_DIR, '2026_05_11_des_vs_act_rp.png'), dpi=120)
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

print('\nFigures saved to:', OUT_DIR)
print('  - 2026_05_11_hgdo_force.png')
print('  - 2026_05_11_pos_vel_world.png')
print('  - 2026_05_11_des_vs_act_rp.png')
