#!/usr/bin/env python3
"""Position error + Desired roll/pitch overlay around a burst event.

Pulls /nmpc/ref to get reference, computes e_p = ref - act, derives desired
roll/pitch from /nmpc/control force (world xy + body z collective), and
overlays with actual roll/pitch.  Helps answer: did a large pos error drive
a large desired tilt, or did the desired tilt grow first and then push?

Usage:
  python3 _pos_err_des_rp.py <bag_subdir> <t_center> [<window_sec>] [<date_dir>] [<tag_override>]
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
BAG_DIR = os.path.join(_HERE, DATE, BAG)
db = glob.glob(os.path.join(BAG_DIR, '*.db3'))[0]
parts = BAG.split('/')
OUT_DIR = os.path.join(_HERE, DATE, *parts[:-1])
TAG = TAG_OVR if TAG_OVR else parts[-1]


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
    return px, py, pz, qw, qx, qy, qz, vx, vy, vz


def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return struct.unpack_from('<6d', blob, off)


def parse_ref(blob):
    """drone_msgs/Ref: header + p[3] + v[3] + psi + psi_dot"""
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    px, py, pz, vx, vy, vz, psi, psi_dot = struct.unpack_from('<8d', blob, off)
    return px, py, pz, vx, vy, vz, psi, psi_dot


def quat_to_rpy(qw, qx, qy, qz):
    r = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    sp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
    p = np.arcsin(sp)
    y = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    return r, p, y


def force_to_des_rp(fx_w, fy_w, fz_body, psi):
    """fx, fy world frame; fz body collective thrust.  Returns desired roll, pitch (rad)."""
    fz_sq = fz_body ** 2 - fx_w ** 2 - fy_w ** 2
    fz_w = np.sqrt(max(fz_sq, 0.0))
    n = np.sqrt(fx_w ** 2 + fy_w ** 2 + fz_w ** 2)
    if n < 1e-6: return 0.0, 0.0
    zb = np.array([fx_w, fy_w, fz_w]) / n
    xc = np.array([np.cos(psi), np.sin(psi), 0.0])
    yb = np.cross(zb, xc)
    yb_n = np.linalg.norm(yb)
    yb = yb / yb_n if yb_n > 1e-6 else np.array([-np.sin(psi), np.cos(psi), 0.0])
    xb = np.cross(yb, zb)
    R = np.column_stack((xb, yb, zb))
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-R[2, 0])
    return roll, pitch


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
ct, ctrl = fetch('/nmpc/control', parse_wrench)
rt, ref = fetch('/nmpc/ref', parse_ref)
ht, hg = fetch('/hgdo/wrench', parse_wrench)
con.close()

t_lo = T_CENTER - WIN / 2
t_hi = T_CENTER + WIN / 2
mO = (ot >= t_lo) & (ot <= t_hi)
mC = (ct >= t_lo) & (ct <= t_hi)
mR = (rt >= t_lo) & (rt <= t_hi)
mH = (ht >= t_lo) & (ht <= t_hi)

t_o = ot[mO]
p_x = odom[mO, 0]; p_y = odom[mO, 1]; p_z = odom[mO, 2]
v_x = odom[mO, 7]; v_y = odom[mO, 8]; v_z = odom[mO, 9]
rp_act = np.array([quat_to_rpy(odom[i, 3], odom[i, 4], odom[i, 5], odom[i, 6])
                   for i in np.where(mO)[0]])
roll_act = np.rad2deg(rp_act[:, 0]); pitch_act = np.rad2deg(rp_act[:, 1])
yaw_act = rp_act[:, 2]

# Reference (resample to odom timeline)
t_r = rt[mR]
ref_p = ref[mR, 0:3]; ref_v = ref[mR, 3:6]; ref_psi = ref[mR, 6]
if len(t_r) > 1:
    rx = np.interp(t_o, t_r, ref_p[:, 0])
    ry = np.interp(t_o, t_r, ref_p[:, 1])
    rz = np.interp(t_o, t_r, ref_p[:, 2])
    rvx = np.interp(t_o, t_r, ref_v[:, 0])
    rvy = np.interp(t_o, t_r, ref_v[:, 1])
    rvz = np.interp(t_o, t_r, ref_v[:, 2])
else:
    rx = np.full_like(p_x, ref_p[0, 0]) if len(ref_p) else np.zeros_like(p_x)
    ry = np.full_like(p_y, ref_p[0, 1]) if len(ref_p) else np.zeros_like(p_y)
    rz = np.full_like(p_z, ref_p[0, 2]) if len(ref_p) else np.zeros_like(p_z)
    rvx = rvy = rvz = np.zeros_like(p_x)

# Position error (ref - act) — what PD sees
e_x = rx - p_x; e_y = ry - p_y; e_z = rz - p_z
# Velocity error
e_vx = rvx - v_x; e_vy = rvy - v_y; e_vz = rvz - v_z

# Desired RP from /nmpc/control
t_c = ct[mC]
fx_w = ctrl[mC, 0]; fy_w = ctrl[mC, 1]; f_col = ctrl[mC, 2]
psi_c = np.interp(t_c, t_o, np.unwrap(yaw_act))
rp_des = np.array([force_to_des_rp(fx_w[i], fy_w[i], f_col[i], psi_c[i]) for i in range(len(t_c))])
roll_des = np.rad2deg(rp_des[:, 0]); pitch_des = np.rad2deg(rp_des[:, 1])

# Per-component "what-if" desired RP:
#   - If only PD acted (no HGDO):  fx,fy still in world, but the published values
#     already include HGDO=0 for xy because dob_force_xy=false; for fz the HGDO
#     IS applied to the collective.  So roll_des / pitch_des reflect the PD output.
# For an additional diagnostic, compute PD-only desired tilt from e_p,e_v
M = 3.146
Kp = np.array([2.0, 2.0, 5.0]); Kd = np.array([2.0, 2.0, 3.5])
a_pd_x = Kp[0] * e_x + Kd[0] * e_vx     # body-aligned simplification (e_v not rotated here)
a_pd_y = Kp[1] * e_y + Kd[1] * e_vy
F_pd_x = M * a_pd_x; F_pd_y = M * a_pd_y
F_pd_z = M * (Kp[2] * e_z + Kd[2] * e_vz + 9.81)
# Desired RP from PD only (no HGDO fz, treat collective = F_pd_z body z component)
rp_pd = np.array([force_to_des_rp(F_pd_x[i], F_pd_y[i], F_pd_z[i],
                                   np.unwrap(yaw_act)[i]) for i in range(len(t_o))])
roll_pd = np.rad2deg(rp_pd[:, 0]); pitch_pd = np.rad2deg(rp_pd[:, 1])

# HGDO force (for context)
t_h = ht[mH]
hg_fx = hg[mH, 0]; hg_fy = hg[mH, 1]; hg_fz = hg[mH, 2]

print(f'Window: {t_lo:.1f} → {t_hi:.1f} s')
print(f'  ref_p constant?  rx={rx[0]:.2f}→{rx[-1]:.2f}, ry={ry[0]:.2f}→{ry[-1]:.2f}, rz={rz[0]:.2f}→{rz[-1]:.2f}')
print(f'  pos err std:     e_x={e_x.std():.3f}, e_y={e_y.std():.3f}, e_z={e_z.std():.3f}')
print(f'  pos err max:     e_x={np.abs(e_x).max():.3f}, e_y={np.abs(e_y).max():.3f}, e_z={np.abs(e_z).max():.3f}')
print(f'  des RP std:      roll={roll_des.std():.2f}, pitch={pitch_des.std():.2f}')
print(f'  des RP max:      |roll|={np.abs(roll_des).max():.2f}, |pitch|={np.abs(pitch_des).max():.2f}')

fig, axes = plt.subplots(6, 1, figsize=(13, 15), sharex=True)

axes[0].plot(t_o, p_x, 'r-',  label='x act')
axes[0].plot(t_o, rx,  'r--', alpha=0.6, label='x ref')
axes[0].plot(t_o, p_y, 'g-',  label='y act')
axes[0].plot(t_o, ry,  'g--', alpha=0.6, label='y ref')
axes[0].plot(t_o, p_z, 'b-',  label='z act')
axes[0].plot(t_o, rz,  'b--', alpha=0.6, label='z ref')
axes[0].set_ylabel('pos [m]'); axes[0].grid(alpha=0.3); axes[0].legend(loc='upper right', ncol=3)
axes[0].set_title(f'{TAG} — pos err + des/act RP around t={T_CENTER:.1f} s')

axes[1].plot(t_o, e_x, 'r', label='e_x = ref−act')
axes[1].plot(t_o, e_y, 'g', label='e_y')
axes[1].plot(t_o, e_z, 'b', label='e_z')
axes[1].axhline(0, color='k', alpha=0.3)
axes[1].set_ylabel('pos err [m]'); axes[1].grid(alpha=0.3); axes[1].legend(loc='upper right')

axes[2].plot(t_o, v_x, 'r-', label='vx act')
axes[2].plot(t_o, v_y, 'g-', label='vy act')
axes[2].plot(t_o, v_z, 'b-', label='vz act')
axes[2].axhline(0, color='k', alpha=0.3)
axes[2].set_ylabel('vel [m/s]'); axes[2].grid(alpha=0.3); axes[2].legend(loc='upper right')

axes[3].plot(t_o, roll_act,  'r-', alpha=0.8, label='roll act')
axes[3].plot(t_c, roll_des,  'r--', alpha=0.9, label='roll des (cmd)')
axes[3].plot(t_o, roll_pd,   'r:',  alpha=0.6, label='roll PD-only')
axes[3].axhline(0, color='k', alpha=0.3)
axes[3].set_ylabel('roll [deg]'); axes[3].grid(alpha=0.3); axes[3].legend(loc='upper right')

axes[4].plot(t_o, pitch_act, 'g-', alpha=0.8, label='pitch act')
axes[4].plot(t_c, pitch_des, 'g--', alpha=0.9, label='pitch des (cmd)')
axes[4].plot(t_o, pitch_pd,  'g:',  alpha=0.6, label='pitch PD-only')
axes[4].axhline(0, color='k', alpha=0.3)
axes[4].set_ylabel('pitch [deg]'); axes[4].grid(alpha=0.3); axes[4].legend(loc='upper right')

axes[5].plot(t_h, hg_fx, 'r', label='HGDO fx')
axes[5].plot(t_h, hg_fy, 'g', label='HGDO fy')
axes[5].plot(t_h, hg_fz, 'b', label='HGDO fz')
axes[5].axhline(0, color='k', alpha=0.3)
axes[5].set_ylabel('HGDO force [N]'); axes[5].set_xlabel('time [s]')
axes[5].grid(alpha=0.3); axes[5].legend(loc='upper right')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_pos_err_des_rp_t{int(T_CENTER)}.png')
plt.savefig(out, dpi=120)
print(f'Saved: {out}')
