"""
Analyze 2026_04_21_unstable_yaw_sim bag:
  - RPY + yaw ref from /nmpc/ref
  - NMPC moments, HGDO moments, actual RPM moments
  - Effective moment (NMPC - HGDO) vs actual RPM moment
  - Individual rotor RPMs
"""

import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BAG_DIR = Path(__file__).parent / '2026_04_21_unstable_yaw_sim'
DB_PATH = BAG_DIR / '2026_04_21_unstable_yaw_sim_0.db3'

# Drone parameters
ARM_LENGTH = 0.265
K_M = 0.01569
C_T = 1.386e-7
S30 = np.sin(np.pi / 6)
C30 = np.cos(np.pi / 6)

CDR_START = 4

def cdr_align(offset, alignment):
    pos = offset - CDR_START
    r = pos % alignment
    return offset + (alignment - r) if r else offset

# ── Parsers ──────────────────────────────────────────────────

def parse_header(data, offset):
    sec = struct.unpack_from('<i', data, offset)[0]
    nsec = struct.unpack_from('<I', data, offset + 4)[0]
    t = sec + nsec * 1e-9
    str_len = struct.unpack_from('<I', data, offset + 8)[0]
    offset = offset + 12 + str_len
    return t, offset

def parse_wrench_stamped(data):
    off = CDR_START
    t, off = parse_header(data, off)
    off = cdr_align(off, 8)
    vals = struct.unpack_from('<6d', data, off)
    return t, np.array(vals)

def parse_odometry(data):
    off = CDR_START
    t, off = parse_header(data, off)
    # child_frame_id
    off = cdr_align(off, 4)
    str_len = struct.unpack_from('<I', data, off)[0]
    off += 4 + str_len
    # pose.pose (position + orientation)
    off = cdr_align(off, 8)
    px, py, pz = struct.unpack_from('<3d', data, off); off += 24
    ox, oy, oz, ow = struct.unpack_from('<4d', data, off); off += 32
    # pose.covariance (36 doubles)
    off += 36 * 8
    # twist.twist (linear + angular)
    vx, vy, vz = struct.unpack_from('<3d', data, off); off += 24
    wx, wy, wz = struct.unpack_from('<3d', data, off)
    q = np.array([ow, ox, oy, oz])
    return t, np.array([px, py, pz]), q, np.array([vx, vy, vz]), np.array([wx, wy, wz])

def parse_hexa_actual_rpm(data):
    off = CDR_START
    t, off = parse_header(data, off)
    off = cdr_align(off, 4)
    rpms = struct.unpack_from('<6i', data, off)
    return t, np.array(rpms, dtype=np.float64)

def parse_ref(data):
    off = CDR_START
    t, off = parse_header(data, off)
    off = cdr_align(off, 8)
    p = struct.unpack_from('<3d', data, off); off += 24
    v = struct.unpack_from('<3d', data, off); off += 24
    psi = struct.unpack_from('<d', data, off)[0]; off += 8
    psi_dot = struct.unpack_from('<d', data, off)[0]
    return t, np.array(p), np.array(v), psi, psi_dot

def quaternion_to_rpy(q):
    qw, qx, qy, qz = q
    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    sinp = np.clip(2*(qw*qy - qz*qx), -1, 1)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    return roll, pitch, yaw

def rpm_to_moments(rpms):
    thrusts = C_T * rpms**2
    signs = np.array([1, -1, -1, 1, 1, -1])
    Mx = ARM_LENGTH * (S30*(thrusts[0] - thrusts[1] - thrusts[2] + thrusts[3])
                       + thrusts[4] - thrusts[5])
    My = ARM_LENGTH * C30 * (-thrusts[0] - thrusts[1] + thrusts[2] + thrusts[3])
    Mz = K_M * np.sum(signs * thrusts)
    F = np.sum(thrusts)
    return F, Mx, My, Mz

# ── Read bag ─────────────────────────────────────────────────

def read_topic(db_path, topic_name):
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT m.timestamp, m.data FROM messages m "
        "JOIN topics t ON m.topic_id = t.id "
        "WHERE t.name = ? ORDER BY m.timestamp", (topic_name,))
    rows = cursor.fetchall()
    conn.close()
    return rows

print("Reading bag...")

odom_rows = read_topic(DB_PATH, '/mavros/local_position/odom')
nmpc_rows = read_topic(DB_PATH, '/nmpc/control')
hgdo_rows = read_topic(DB_PATH, '/hgdo/wrench')
rpm_rows = read_topic(DB_PATH, '/uav/actual_rpm')
ref_rows = read_topic(DB_PATH, '/nmpc/ref')
odom_sim_rows = read_topic(DB_PATH, '/mavros/local_position/odom_sim')

# ── Parse ────────────────────────────────────────────────────

print("Parsing...")

# Odom
odom_t, odom_pos, odom_rpy, odom_w = [], [], [], []
for ts, data in odom_rows:
    t = ts * 1e-9
    _, pos, q, v, w = parse_odometry(data)
    r, p, y = quaternion_to_rpy(q)
    odom_t.append(t); odom_pos.append(pos)
    odom_rpy.append([r, p, y]); odom_w.append(w)
odom_t = np.array(odom_t); odom_pos = np.array(odom_pos)
odom_rpy = np.array(odom_rpy); odom_w = np.array(odom_w)

# Odom sim
sim_t, sim_pos, sim_rpy = [], [], []
for ts, data in odom_sim_rows:
    t = ts * 1e-9
    _, pos, q, v, w = parse_odometry(data)
    r, p, y = quaternion_to_rpy(q)
    sim_t.append(t); sim_pos.append(pos); sim_rpy.append([r, p, y])
sim_t = np.array(sim_t); sim_pos = np.array(sim_pos); sim_rpy = np.array(sim_rpy)

# NMPC moments
nmpc_t, nmpc_F, nmpc_M = [], [], []
for ts, data in nmpc_rows:
    t = ts * 1e-9
    _, vals = parse_wrench_stamped(data)
    nmpc_t.append(t); nmpc_F.append(vals[2])
    nmpc_M.append(vals[3:6])
nmpc_t = np.array(nmpc_t); nmpc_F = np.array(nmpc_F)
nmpc_M = np.array(nmpc_M)

# HGDO wrench
hgdo_t, hgdo_M = [], []
for ts, data in hgdo_rows:
    t = ts * 1e-9
    _, vals = parse_wrench_stamped(data)
    hgdo_t.append(t); hgdo_M.append(vals[3:6])
hgdo_t = np.array(hgdo_t); hgdo_M = np.array(hgdo_M)

# Actual RPMs → moments
rpm_t, rpm_vals, rpm_F, rpm_M = [], [], [], []
for ts, data in rpm_rows:
    t = ts * 1e-9
    _, rpms = parse_hexa_actual_rpm(data)
    F, Mx, My, Mz = rpm_to_moments(rpms)
    rpm_t.append(t); rpm_vals.append(rpms)
    rpm_F.append(F); rpm_M.append([Mx, My, Mz])
rpm_t = np.array(rpm_t); rpm_vals = np.array(rpm_vals)
rpm_F = np.array(rpm_F); rpm_M = np.array(rpm_M)

# Reference
ref_t, ref_pos, ref_psi = [], [], []
for ts, data in ref_rows:
    t = ts * 1e-9
    _, p, v, psi, psi_dot = parse_ref(data)
    ref_t.append(t); ref_pos.append(p); ref_psi.append(psi)
ref_t = np.array(ref_t); ref_pos = np.array(ref_pos); ref_psi = np.array(ref_psi)

# ── Time normalization ───────────────────────────────────────

t0 = min(odom_t[0], nmpc_t[0], hgdo_t[0], rpm_t[0])
odom_t -= t0; sim_t -= t0; nmpc_t -= t0; hgdo_t -= t0; rpm_t -= t0; ref_t -= t0

print(f"Duration: {odom_t[-1]:.1f}s, Odom: {len(odom_t)}, NMPC: {len(nmpc_t)}, "
      f"HGDO: {len(hgdo_t)}, RPM: {len(rpm_t)}, Ref: {len(ref_t)}")

# ── Plot ─────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig.suptitle('Unstable Yaw Sim Analysis (PD+NMPC+HGDO, Free Flight)', fontsize=13)

# 1) RPY + yaw ref
ax = axes[0]
ax.plot(odom_t, np.degrees(odom_rpy[:, 0]), 'r', alpha=0.8, label='Roll (odom)')
ax.plot(odom_t, np.degrees(odom_rpy[:, 1]), 'g', alpha=0.8, label='Pitch (odom)')
ax.plot(odom_t, np.degrees(odom_rpy[:, 2]), 'b', alpha=0.8, label='Yaw (odom)')
if len(sim_t) > 0:
    ax.plot(sim_t, np.degrees(sim_rpy[:, 2]), 'b--', alpha=0.5, label='Yaw (sim)')
if len(ref_t) > 0:
    ax.step(ref_t, np.degrees(ref_psi), 'k--', linewidth=2, where='post', label='Yaw ref')
ax.set_ylabel('Angle [deg]')
ax.set_title('RPY from Odometry + Yaw Reference')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# 2) NMPC moments
ax = axes[1]
ax.plot(nmpc_t, nmpc_M[:, 0], 'r', alpha=0.8, label='Mx (NMPC)')
ax.plot(nmpc_t, nmpc_M[:, 1], 'g', alpha=0.8, label='My (NMPC)')
ax.plot(nmpc_t, nmpc_M[:, 2], 'b', alpha=0.8, label='Mz (NMPC)')
ax.set_ylabel('Moment [Nm]')
ax.set_title('NMPC Control Moments (u_mpc)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# 3) HGDO moments
ax = axes[2]
ax.plot(hgdo_t, hgdo_M[:, 0], 'r', alpha=0.8, label='τ_x (HGDO)')
ax.plot(hgdo_t, hgdo_M[:, 1], 'g', alpha=0.8, label='τ_y (HGDO)')
ax.plot(hgdo_t, hgdo_M[:, 2], 'b', alpha=0.8, label='τ_z (HGDO)')
ax.set_ylabel('Moment [Nm]')
ax.set_title('HGDO Disturbance Estimates')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# 4) Effective moment (NMPC - HGDO) vs actual RPM moment
hgdo_Mx_interp = np.interp(nmpc_t, hgdo_t, hgdo_M[:, 0])
hgdo_My_interp = np.interp(nmpc_t, hgdo_t, hgdo_M[:, 1])
hgdo_Mz_interp = np.interp(nmpc_t, hgdo_t, hgdo_M[:, 2])
eff_Mx = nmpc_M[:, 0] - hgdo_Mx_interp
eff_My = nmpc_M[:, 1] - hgdo_My_interp
eff_Mz = nmpc_M[:, 2] - hgdo_Mz_interp

ax = axes[3]
ax.plot(nmpc_t, eff_Mx, 'r', alpha=0.7, label='Mx eff (NMPC−HGDO)')
ax.plot(nmpc_t, eff_My, 'g', alpha=0.7, label='My eff (NMPC−HGDO)')
ax.plot(nmpc_t, eff_Mz, 'b', alpha=0.7, label='Mz eff (NMPC−HGDO)')
ax.plot(rpm_t, rpm_M[:, 0], 'r--', alpha=0.5, label='Mx (RPM)')
ax.plot(rpm_t, rpm_M[:, 1], 'g--', alpha=0.5, label='My (RPM)')
ax.plot(rpm_t, rpm_M[:, 2], 'b--', alpha=0.5, label='Mz (RPM)')
ax.set_ylabel('Moment [Nm]')
ax.set_xlabel('Time [s]')
ax.set_title('Effective Moment (NMPC−HGDO) vs Actual RPM Moment')
ax.legend(loc='upper right', fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(BAG_DIR.parent / 'unstable_yaw_analysis.png'), dpi=150)
print("Saved unstable_yaw_analysis.png")

# ── Figure 2: RPM + Position ────────────────────────────────

fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig2.suptitle('Rotor RPMs & Position (Unstable Yaw Sim)', fontsize=13)

ax = axes2[0]
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
for i in range(6):
    ax.plot(rpm_t, rpm_vals[:, i], color=colors[i], alpha=0.7, label=f'Rotor {i+1}')
ax.set_ylabel('RPM')
ax.set_title('Individual Rotor RPMs')
ax.legend(loc='upper right', fontsize=7, ncol=3)
ax.grid(True, alpha=0.3)

ax = axes2[1]
ax.plot(odom_t, odom_pos[:, 0], 'r', alpha=0.8, label='x')
ax.plot(odom_t, odom_pos[:, 1], 'g', alpha=0.8, label='y')
ax.plot(odom_t, odom_pos[:, 2], 'b', alpha=0.8, label='z')
if len(ref_t) > 0:
    ax.step(ref_t, ref_pos[:, 0], 'r--', alpha=0.5, where='post', label='x ref')
    ax.step(ref_t, ref_pos[:, 1], 'g--', alpha=0.5, where='post', label='y ref')
    ax.step(ref_t, ref_pos[:, 2], 'b--', alpha=0.5, where='post', label='z ref')
ax.set_ylabel('Position [m]')
ax.set_title('Position (odom) + Reference')
ax.legend(loc='upper right', fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

ax = axes2[2]
ax.plot(odom_t, np.degrees(odom_w[:, 0]), 'r', alpha=0.8, label='ωx')
ax.plot(odom_t, np.degrees(odom_w[:, 1]), 'g', alpha=0.8, label='ωy')
ax.plot(odom_t, np.degrees(odom_w[:, 2]), 'b', alpha=0.8, label='ωz')
ax.set_ylabel('Angular velocity [deg/s]')
ax.set_xlabel('Time [s]')
ax.set_title('Body Angular Velocity')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(BAG_DIR.parent / 'unstable_yaw_rpm_pos.png'), dpi=150)
print("Saved unstable_yaw_rpm_pos.png")

# ── Figure 3: Zoomed into instability onset ──────────────────

# Find when yaw diverges: first time |roll| or |pitch| > 15 deg
onset_idx = np.argmax(
    (np.abs(odom_rpy[:, 0]) > np.radians(15)) |
    (np.abs(odom_rpy[:, 1]) > np.radians(15))
)
t_onset = odom_t[onset_idx]
t_zoom_start = max(t_onset - 5.0, odom_t[0])
t_zoom_end = min(t_onset + 8.0, odom_t[-1])
print(f"Instability onset at t={t_onset:.2f}s (roll/pitch > 15 deg)")

fig3, axes3 = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
fig3.suptitle(f'Zoomed: Instability Onset (t={t_onset:.1f}s)', fontsize=13)

mask_o = (odom_t >= t_zoom_start) & (odom_t <= t_zoom_end)
mask_n = (nmpc_t >= t_zoom_start) & (nmpc_t <= t_zoom_end)
mask_h = (hgdo_t >= t_zoom_start) & (hgdo_t <= t_zoom_end)
mask_r = (rpm_t >= t_zoom_start) & (rpm_t <= t_zoom_end)
mask_ref = (ref_t >= t_zoom_start) & (ref_t <= t_zoom_end)

# 1) RPY zoomed
ax = axes3[0]
ax.plot(odom_t[mask_o], np.degrees(odom_rpy[mask_o, 0]), 'r', label='Roll')
ax.plot(odom_t[mask_o], np.degrees(odom_rpy[mask_o, 1]), 'g', label='Pitch')
ax.plot(odom_t[mask_o], np.degrees(odom_rpy[mask_o, 2]), 'b', label='Yaw')
if np.any(mask_ref):
    ax.step(ref_t[mask_ref], np.degrees(ref_psi[mask_ref]),
            'k--', linewidth=2, where='post', label='Yaw ref')
ax.set_ylabel('Angle [deg]')
ax.set_title('RPY')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# 2) NMPC moments zoomed
ax = axes3[1]
ax.plot(nmpc_t[mask_n], nmpc_M[mask_n, 0], 'r', label='Mx')
ax.plot(nmpc_t[mask_n], nmpc_M[mask_n, 1], 'g', label='My')
ax.plot(nmpc_t[mask_n], nmpc_M[mask_n, 2], 'b', label='Mz')
ax.set_ylabel('Moment [Nm]')
ax.set_title('NMPC Moments (u_mpc)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# 3) HGDO zoomed
ax = axes3[2]
ax.plot(hgdo_t[mask_h], hgdo_M[mask_h, 0], 'r', label='τ_x')
ax.plot(hgdo_t[mask_h], hgdo_M[mask_h, 1], 'g', label='τ_y')
ax.plot(hgdo_t[mask_h], hgdo_M[mask_h, 2], 'b', label='τ_z')
ax.set_ylabel('Moment [Nm]')
ax.set_title('HGDO Estimates')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# 4) Individual RPMs zoomed
ax = axes3[3]
for i in range(6):
    ax.plot(rpm_t[mask_r], rpm_vals[mask_r, i], color=colors[i], label=f'R{i+1}')
ax.axhline(y=7300, color='gray', linestyle=':', alpha=0.5, label='RPM max')
ax.axhline(y=2000, color='gray', linestyle='--', alpha=0.5, label='RPM min')
ax.set_ylabel('RPM')
ax.set_title('Rotor RPMs')
ax.legend(loc='upper right', fontsize=7, ncol=4)
ax.grid(True, alpha=0.3)

# 5) Angular velocity zoomed
ax = axes3[4]
ax.plot(odom_t[mask_o], np.degrees(odom_w[mask_o, 0]), 'r', label='ωx')
ax.plot(odom_t[mask_o], np.degrees(odom_w[mask_o, 1]), 'g', label='ωy')
ax.plot(odom_t[mask_o], np.degrees(odom_w[mask_o, 2]), 'b', label='ωz')
ax.set_ylabel('Angular vel [deg/s]')
ax.set_xlabel('Time [s]')
ax.set_title('Body Angular Velocity')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(BAG_DIR.parent / 'unstable_yaw_zoomed.png'), dpi=150)
print("Saved unstable_yaw_zoomed.png")

plt.show()
