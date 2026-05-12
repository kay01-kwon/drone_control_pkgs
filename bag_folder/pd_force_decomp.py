#!/usr/bin/env python3
"""Decompose F_des into Kp·e_p, Kd·e_v, and HGDO compensation components.

  F_des_xy_world = m·(Kp·e_p + Kd·e_v) − R(q)·f_hgdo_xy

For each axis (x, y) show on a single panel:
  • Kp term  (= m·Kp·e_p)
  • Kd term  (= m·Kd·e_v)
  • PD sum   (= m·a_pd)
  • HGDO comp (= −R·f_hgdo)
  • F_des published (= PD sum + HGDO comp)
  • Position deviation (right y-axis)

This makes visible cases where F_des is near zero even while position
error is large (Kp + Kd opposing each other, or HGDO cancelling PD).

Usage:  python3 pd_force_decomp.py [<bag_subdir> [<date_dir>]]
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
Kp = np.array([2.0, 2.0, 5.0])
Kd = np.array([2.0, 2.0, 3.5])
M = 3.188
print(f'Analyzing: {DB}')


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
    vx, vy, vz = struct.unpack_from('<3d', blob, off)
    return np.array([px, py, pz, vx, vy, vz, qw, qx, qy, qz])

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


conn = sqlite3.connect(DB)
c = conn.cursor()
tid = {n: i for i, n in c.execute('SELECT id, name FROM topics').fetchall()}

def fetch(topic, parser):
    rows = c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
                     (tid[topic],)).fetchall()
    ts = np.array([r[0] for r in rows], dtype=np.float64)
    dat = np.array([parser(bytes(r[1])) for r in rows])
    return ts, dat

odom_ts, od = fetch('/mavros/local_position/odom', parse_odom)
ctrl_ts, ct = fetch('/nmpc/control', parse_wrench)
hgdo_ts, hg = fetch('/hgdo/wrench',  parse_wrench)
conn.close()

t0 = min(odom_ts[0], ctrl_ts[0], hgdo_ts[0])
odom_t = (odom_ts - t0) * 1e-9
ctrl_t = (ctrl_ts - t0) * 1e-9
hgdo_t = (hgdo_ts - t0) * 1e-9

p = od[:, 0:3]
vb = od[:, 3:6]
q = od[:, 6:10]
v_w = np.array([quat_to_rotm(q[i]) @ vb[i] for i in range(len(q))])

# Reference: take mean of airborne segment (we don't have explicit ref)
mask = odom_t > 5.0
ref_p = p[mask].mean(axis=0)
print(f'ref_p (assumed = mean(p_airborne)) = {ref_p}')

e_p = ref_p - p
e_v = -v_w
Kp_term_a = e_p * Kp
Kd_term_a = e_v * Kd
a_pd = Kp_term_a + Kd_term_a

# In Newtons
Kp_F = M * Kp_term_a
Kd_F = M * Kd_term_a
F_pd = M * a_pd

# Interp Kp_F, Kd_F, F_pd to ctrl timestamps (odom is ~100Hz, ctrl too)
def to_ctrl(x, t_src=odom_t):
    return np.column_stack([np.interp(ctrl_t, t_src, x[:, k]) for k in range(3)])

Kp_F_c = to_ctrl(Kp_F)
Kd_F_c = to_ctrl(Kd_F)
F_pd_c = to_ctrl(F_pd)
pos_c  = to_ctrl(p)

# HGDO contribution in world frame: −R(q)·f_hgdo_body
hg_b_c = np.column_stack([np.interp(ctrl_t, hgdo_t, hg[:, k]) for k in range(3)])
roll_c  = np.interp(ctrl_t, odom_t, np.arctan2(2*(q[:,0]*q[:,1] + q[:,2]*q[:,3]),
                                                1 - 2*(q[:,1]**2 + q[:,2]**2)))
pit_c   = np.interp(ctrl_t, odom_t, np.arcsin(np.clip(2*(q[:,0]*q[:,2] - q[:,3]*q[:,1]), -1, 1)))
psi_c   = np.interp(ctrl_t, odom_t, np.unwrap(np.arctan2(2*(q[:,0]*q[:,3] + q[:,1]*q[:,2]),
                                                          1 - 2*(q[:,2]**2 + q[:,3]**2))))
def rpy_to_R(r, p_, y_):
    cr, sr = np.cos(r), np.sin(r); cp, sp = np.cos(p_), np.sin(p_); cy, sy = np.cos(y_), np.sin(y_)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])
hg_w_c = np.zeros_like(hg_b_c)
for i in range(len(ctrl_t)):
    hg_w_c[i] = rpy_to_R(roll_c[i], pit_c[i], psi_c[i]) @ hg_b_c[i]
HGDO_F = -hg_w_c

# F_des published
F_des = ct[:, 0:3]
# Recover Fz_world from f_col is NOT needed here — we only care about xy.

# Sanity: F_des_xy_published ≈ F_pd_xy + HGDO_F_xy
diff_x = F_des[:, 0] - (F_pd_c[:, 0] + HGDO_F[:, 0])
diff_y = F_des[:, 1] - (F_pd_c[:, 1] + HGDO_F[:, 1])
print(f'\nReconstruction error (F_des − (F_pd + HGDO)):')
print(f'  x  std = {diff_x.std():.3f} N (should be small if reconstruction is right)')
print(f'  y  std = {diff_y.std():.3f} N')


# ── Plot: 5 rows × 2 cols, each signal on its own panel ──
fig, axes = plt.subplots(5, 2, figsize=(15, 14), sharex=True)

mask_c = ctrl_t > 5.0
t_p = ctrl_t[mask_c]

for col, (axis, idx) in enumerate([('x', 0), ('y', 1)]):
    pos_dev = pos_c[mask_c, idx] - ref_p[idx]

    axes[0, col].plot(t_p, pos_dev, 'purple', lw=1.0)
    axes[0, col].axhline(0, color='gray', lw=0.5, alpha=0.5)
    axes[0, col].set_ylabel(f'{axis} − ref [m]')
    axes[0, col].grid(alpha=0.3)
    axes[0, col].set_title(f'{axis}-axis  —  position deviation  (ref = mean({axis}))')

    axes[1, col].plot(t_p, Kp_F_c[mask_c, idx], 'g', lw=1.0)
    axes[1, col].axhline(0, color='gray', lw=0.5, alpha=0.5)
    axes[1, col].set_ylabel(f'Kp·m·e_p_{axis} [N]')
    axes[1, col].grid(alpha=0.3)
    axes[1, col].set_title(f'Kp term (Kp={Kp[idx]})  →  proportional to position error')

    axes[2, col].plot(t_p, Kd_F_c[mask_c, idx], 'orange', lw=1.0)
    axes[2, col].axhline(0, color='gray', lw=0.5, alpha=0.5)
    axes[2, col].set_ylabel(f'Kd·m·e_v_{axis} [N]')
    axes[2, col].grid(alpha=0.3)
    axes[2, col].set_title(f'Kd term (Kd={Kd[idx]})  →  proportional to (−velocity)')

    axes[3, col].plot(t_p, F_pd_c[mask_c, idx], 'b', lw=1.0, label='PD sum')
    axes[3, col].plot(t_p, HGDO_F[mask_c, idx], 'r', lw=1.0, alpha=0.85, label='HGDO comp')
    axes[3, col].axhline(0, color='gray', lw=0.5, alpha=0.5)
    axes[3, col].set_ylabel(f'F_{axis} [N]')
    axes[3, col].grid(alpha=0.3)
    axes[3, col].legend(loc='upper right', fontsize=9)
    axes[3, col].set_title('PD sum  vs  HGDO compensation  (these add to F_des)')

    axes[4, col].plot(t_p, F_des[mask_c, idx], 'k', lw=1.0)
    axes[4, col].axhline(0, color='gray', lw=0.5, alpha=0.5)
    axes[4, col].set_ylabel(f'F_des_{axis} [N]')
    axes[4, col].grid(alpha=0.3)
    axes[4, col].set_title('F_des published  (= PD sum + HGDO comp)')
    axes[4, col].set_xlabel('Time [s]')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_pd_force_decomp.png'), dpi=120)
plt.close()

print(f'\nSaved: {TAG}_pd_force_decomp.png')
