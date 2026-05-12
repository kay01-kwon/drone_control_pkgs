#!/usr/bin/env python3
"""Implement HGDO offline from /mavros/local_position/odom + /uav/actual_rpm
and sweep eps_f, eps_tau ∈ {0.1, 0.3, 0.5}.  Compare to the actual
/hgdo/wrench in the bag.

  Dynamics integrated (matches drone_dob/include/models/hgdo_model/hgdo_model.cpp):
    γ̇_f = −(1/εf)(γ_f + v_w/εf) + (1/εf)(−R·[0,0,u_T/m] − g_vec)
    γ̇_t = −(1/εt)(γ_t + ω/εt)  + (1/εt)(−J⁻¹·M + J⁻¹·(ω×Jω))
  Output:
    d_force_body = Rᵀ·m·(γ_f + v_w/εf)
    d_torque     = J·(γ_t + ω/εt)

Usage:  python3 hgdo_offline_sweep.py [<bag_subdir> [<date_dir>]]
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
print(f'Analyzing: {DB}')

# Drone parameters (drone_control/config/pd_nmpc_att_with_hgdo.yaml)
M    = 3.146
J    = np.diag([0.06, 0.06, 0.08])
J_inv = np.linalg.inv(J)
L    = 0.265
C_T  = 1.2895e-7         # motor_const, thrust per RPM²
K_M  = 0.01569           # moment_const
G    = np.array([0.0, 0.0, -9.81])    # gravity vector (drone_dob: state_def.hpp)

# Allocation matrix (RPM thrust → [T, Mx, My, Mz]); see HexaRotorRpmToCmd
c60 = np.cos(np.pi / 3); s60 = np.sin(np.pi / 3)
ly = L * np.array([ c60,  1,  c60, -c60, -1, -c60])
lx = L * np.array([ s60,  0, -s60, -s60,  0,  s60])
B_alloc = np.array([
    [1, 1, 1, 1, 1, 1],
    ly,
    -lx,
    [-K_M, K_M, -K_M, K_M, -K_M, K_M]])


def _align(off, n):
    return off + ((-(off - 4)) % n)

def parse_odom(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    off += 24                              # skip position
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
    off += 36 * 8                          # skip pose covariance
    vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
    wx, wy, wz = struct.unpack_from('<3d', blob, off)
    return np.array([vx, vy, vz, wx, wy, wz, qw, qx, qy, qz])

def parse_actual_rpm(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    return np.array(struct.unpack_from('<6I', blob, off), dtype=np.float64)

def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return np.array(struct.unpack_from('<6d', blob, off))


def quat_to_rotm(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx ** 2 + qy ** 2)]])


# ── Load bag ──
conn = sqlite3.connect(DB)
c = conn.cursor()
tid = {n: i for i, n in c.execute('SELECT id, name FROM topics').fetchall()}

def fetch(topic, parser):
    rows = c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
                     (tid[topic],)).fetchall()
    ts = np.array([r[0] for r in rows], dtype=np.float64)
    dat = np.array([parser(bytes(r[1])) for r in rows])
    return ts, dat

odom_ts, odm = fetch('/mavros/local_position/odom', parse_odom)
rpm_ts,  rpm = fetch('/uav/actual_rpm',              parse_actual_rpm)
hg_ts,   hg  = fetch('/hgdo/wrench',                 parse_wrench)
conn.close()

t0 = min(odom_ts[0], rpm_ts[0], hg_ts[0])
odom_t = (odom_ts - t0) * 1e-9
rpm_t  = (rpm_ts  - t0) * 1e-9
hg_t   = (hg_ts   - t0) * 1e-9


# ── Common time grid ──
T_LO = max(odom_t[0], rpm_t[0]) + 1.0
T_HI = min(odom_t[-1], rpm_t[-1]) - 1.0
dt = 0.01
t_u = np.arange(T_LO, T_HI, dt)
n = len(t_u)

# Resample odom: body velocity, body ω, quaternion
v_b = np.column_stack([np.interp(t_u, odom_t, odm[:, k]) for k in range(3)])
w_b = np.column_stack([np.interp(t_u, odom_t, odm[:, 3 + k]) for k in range(3)])
qw_u = np.interp(t_u, odom_t, odm[:, 6])
qx_u = np.interp(t_u, odom_t, odm[:, 7])
qy_u = np.interp(t_u, odom_t, odm[:, 8])
qz_u = np.interp(t_u, odom_t, odm[:, 9])

# Resample actual RPM
rpm_u = np.column_stack([np.interp(t_u, rpm_t, rpm[:, k]) for k in range(6)])

# Compute u = (T, Mx, My, Mz) from rotor thrust = C_T·RPM²
rotor_thrust = C_T * rpm_u ** 2     # (n, 6)
u_cmd = rotor_thrust @ B_alloc.T    # (n, 4) — [T, Mx, My, Mz]

# v_world per timestep
v_w = np.zeros_like(v_b)
for i in range(n):
    R = quat_to_rotm([qw_u[i], qx_u[i], qy_u[i], qz_u[i]])
    v_w[i] = R @ v_b[i]


# ── HGDO simulator ──
def run_hgdo(eps_f, eps_tau):
    gamma_f = np.zeros((n, 3))
    gamma_t = np.zeros((n, 3))
    d_force_body = np.zeros((n, 3))
    d_torque     = np.zeros((n, 3))

    a_f, a_t = 1.0 / eps_f, 1.0 / eps_tau
    # Use Euler with substeps if needed
    n_sub = max(1, int(np.ceil(dt / (0.2 * min(eps_f, eps_tau)))))
    h = dt / n_sub

    for i in range(1, n):
        R = quat_to_rotm([qw_u[i - 1], qx_u[i - 1], qy_u[i - 1], qz_u[i - 1]])
        thrust_world = R @ np.array([0.0, 0.0, u_cmd[i - 1, 0] / M])
        M_cmd = u_cmd[i - 1, 1:4]
        w_prev = w_b[i - 1]
        v_prev = v_w[i - 1]
        Jw = J @ w_prev
        cross = np.cross(w_prev, Jw)

        gf = gamma_f[i - 1].copy()
        gt = gamma_t[i - 1].copy()
        for _ in range(n_sub):
            gf_dot = -a_f * (gf + a_f * v_prev) + a_f * (-thrust_world - G)
            gt_dot = -a_t * (gt + a_t * w_prev) + a_t * (-J_inv @ M_cmd + J_inv @ cross)
            gf += h * gf_dot
            gt += h * gt_dot
        gamma_f[i] = gf
        gamma_t[i] = gt

        R_curr = quat_to_rotm([qw_u[i], qx_u[i], qy_u[i], qz_u[i]])
        d_force_body[i] = R_curr.T @ (M * (gamma_f[i] + a_f * v_w[i]))
        d_torque[i]     = J @ (gamma_t[i] + a_t * w_b[i])

    return d_force_body, d_torque


eps_list = [0.1, 0.3, 0.5]
results = {}
for eps in eps_list:
    print(f'Running HGDO with eps_f = eps_tau = {eps} ...')
    results[eps] = run_hgdo(eps, eps)


# ── Plot — compare to bag /hgdo/wrench ──
colors = {0.1: 'tab:red', 0.3: 'tab:green', 0.5: 'tab:blue'}

def plot_panel(ax, t_bag, sig_bag, t_sim, sims_dict, title, ylabel):
    ax.plot(t_bag, sig_bag, 'k', lw=1.0, alpha=0.85, label='bag /hgdo/wrench (eps=0.15)')
    for eps, sig in sims_dict.items():
        ax.plot(t_sim, sig, color=colors[eps], lw=0.9, alpha=0.8, label=f'sim eps={eps}')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)


fig, axes = plt.subplots(6, 1, figsize=(15, 16), sharex=True)
labels_f = ['fx', 'fy', 'fz']
labels_t = ['Mx', 'My', 'Mz']

# Force panels (body frame, just like /hgdo/wrench)
for k in range(3):
    sims = {eps: results[eps][0][:, k] for eps in eps_list}
    plot_panel(axes[k], hg_t, hg[:, k], t_u, sims,
               f'HGDO force {labels_f[k]} (body frame)', f'{labels_f[k]} [N]')

# Torque panels
for k in range(3):
    sims = {eps: results[eps][1][:, k] for eps in eps_list}
    plot_panel(axes[3 + k], hg_t, hg[:, 3 + k], t_u, sims,
               f'HGDO torque {labels_t[k]} (body frame)', f'{labels_t[k]} [N·m]')

axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_hgdo_eps_sweep.png'), dpi=120)
plt.close()


# ── Stats ──
print(f'\n== HGDO output stats (airborne, 10s ≤ t ≤ 55s) ==')
# Exclude pre-takeoff and landing transients where ground reaction force
# shows up as a ~m·g spike in fz.
mask_h = (hg_t >= 10.0) & (hg_t <= 55.0)
mask_s = (t_u  >= 10.0) & (t_u  <= 55.0)
print('-- bag /hgdo/wrench (eps_f=0.15 currently in flight) --')
for k, name in enumerate(labels_f):
    print(f'  {name:4s}  std = {hg[mask_h, k].std():.4f}  peak±{max(abs(hg[mask_h, k].min()), abs(hg[mask_h, k].max())):.3f}')
for k, name in enumerate(labels_t):
    print(f'  {name:4s}  std = {hg[mask_h, 3+k].std():.4f}  peak±{max(abs(hg[mask_h, 3+k].min()), abs(hg[mask_h, 3+k].max())):.3f}')

for eps in eps_list:
    df, dt_ = results[eps]
    print(f'\n-- sim eps={eps} --')
    for k, name in enumerate(labels_f):
        print(f'  {name:4s}  std = {df[mask_s, k].std():.4f}  peak±{max(abs(df[mask_s, k].min()), abs(df[mask_s, k].max())):.3f}')
    for k, name in enumerate(labels_t):
        print(f'  {name:4s}  std = {dt_[mask_s, k].std():.4f}  peak±{max(abs(dt_[mask_s, k].min()), abs(dt_[mask_s, k].max())):.3f}')

print(f'\nSaved: {TAG}_hgdo_eps_sweep.png')
