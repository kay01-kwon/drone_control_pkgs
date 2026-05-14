#!/usr/bin/env python3
"""Decompose PD into Kp·e_p (P) and Kd·e_v (D) contributions to show D-term
dominance at the limit-cycle frequency and the 90° lead.

For each axis (x, y):
  m·a_PD = m·(Kp·e_p + Kd·e_v)
  → equivalent tilt angle from the corresponding horizontal force component
    tilt ≈ atan2(F_xy, m·g)  (small-angle, near hover)

Panels (per axis):
  • e_p              [m]       — what PD sees
  • P-only tilt       [deg]    — Kp·e_p contribution
  • D-only tilt       [deg]    — Kd·e_v contribution
  • Sum P+D           [deg]    — PD prediction
  • Published des    [deg]     — actual /nmpc/control
The phase-lead of the D-only trace vs e_p, and its relative magnitude,
are the data behind "Kd dominates at limit-cycle freq".

Usage:
  python3 _pd_term_decomp_zoom.py <bag_subdir> <t_center> [<win_s>] [<date>] [<tag>]
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

M = 3.146
G = 9.81
Kp = np.array([2.0, 2.0, 5.0])
Kd = np.array([2.0, 2.0, 3.5])


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
ct, ctrl = fetch('/nmpc/control', parse_wrench)
rt, ref = fetch('/nmpc/ref', parse_ref)
con.close()

# Crop window
t_lo = T_CENTER - WIN / 2
t_hi = T_CENTER + WIN / 2
mO = (ot >= t_lo) & (ot <= t_hi)
mC = (ct >= t_lo) & (ct <= t_hi)
mR = (rt >= t_lo) & (rt <= t_hi)

t_o = ot[mO]
p_x = odom[mO, 0]; p_y = odom[mO, 1]
v_body = odom[mO, 7:10]  # body frame
qs = odom[mO, 3:7]       # [qw, qx, qy, qz]

# Transform v_body → v_world per sample (R(q) @ v_body)
v_world = np.empty_like(v_body)
for i in range(len(t_o)):
    R = quat_to_rotm(qs[i, 0], qs[i, 1], qs[i, 2], qs[i, 3])
    v_world[i] = R @ v_body[i]

# Reference
t_r = rt[mR]
rp_arr = ref[mR, 0:3]; rv_arr = ref[mR, 3:6]
if len(t_r) > 1:
    rx = np.interp(t_o, t_r, rp_arr[:, 0]); ry = np.interp(t_o, t_r, rp_arr[:, 1])
    rvx = np.interp(t_o, t_r, rv_arr[:, 0]); rvy = np.interp(t_o, t_r, rv_arr[:, 1])
else:
    rx = np.full_like(p_x, rp_arr[0, 0] if len(rp_arr) else 0.0)
    ry = np.full_like(p_y, rp_arr[0, 1] if len(rp_arr) else 0.0)
    rvx = np.zeros_like(p_x); rvy = np.zeros_like(p_y)

e_x = rx - p_x; e_y = ry - p_y
e_vx = rvx - v_world[:, 0]; e_vy = rvy - v_world[:, 1]

# Force contributions per axis (world frame)
F_Kp_x = M * Kp[0] * e_x;   F_Kp_y = M * Kp[1] * e_y
F_Kd_x = M * Kd[0] * e_vx;  F_Kd_y = M * Kd[1] * e_vy
F_PD_x = F_Kp_x + F_Kd_x;   F_PD_y = F_Kp_y + F_Kd_y

# Convert horizontal force to equivalent tilt angle (small-angle near hover)
# Body z thrust ≈ m·g, so atan2(F_xy, m·g) gives tilt.
tilt_Kp_x = np.degrees(np.arctan2(F_Kp_x, M * G))
tilt_Kp_y = np.degrees(np.arctan2(F_Kp_y, M * G))
tilt_Kd_x = np.degrees(np.arctan2(F_Kd_x, M * G))
tilt_Kd_y = np.degrees(np.arctan2(F_Kd_y, M * G))
tilt_PD_x = np.degrees(np.arctan2(F_PD_x, M * G))
tilt_PD_y = np.degrees(np.arctan2(F_PD_y, M * G))

# Published F_des → tilt
t_c = ct[mC]
fx_w = ctrl[mC, 0]; fy_w = ctrl[mC, 1]; f_col = ctrl[mC, 2]
# Tilt magnitude from world fx,fy and collective body thrust (approx)
tilt_pub_x = np.degrees(np.arctan2(fx_w, f_col))
tilt_pub_y = np.degrees(np.arctan2(fy_w, f_col))


def xcorr(a, b, dt, max_lag_s=1.0):
    a = a - a.mean(); b = b - b.mean()
    n = len(a); ml = int(max_lag_s / dt)
    lags = np.arange(-ml, ml + 1)
    c = np.zeros_like(lags, dtype=float)
    for i, k in enumerate(lags):
        if k >= 0:
            x, y = a[:n - k], b[k:]
        else:
            x, y = a[-k:], b[:n + k]
        if len(x) > 5:
            c[i] = np.corrcoef(x, y)[0, 1]
    j = np.argmax(np.abs(c))
    return c[j], lags[j] * dt * 1000


dt = np.median(np.diff(t_o))
print(f'Window: {t_lo:.1f} → {t_hi:.1f} s  (dt={dt*1000:.1f} ms)')
print('\n--- AXIS X ---')
print(f'  e_x:        std={e_x.std():.3f} m   max={np.abs(e_x).max():.3f}')
print(f'  P tilt:     std={tilt_Kp_x.std():.2f}° max={np.abs(tilt_Kp_x).max():.2f}°')
print(f'  D tilt:     std={tilt_Kd_x.std():.2f}° max={np.abs(tilt_Kd_x).max():.2f}°')
print(f'  D/P std ratio: {tilt_Kd_x.std()/tilt_Kp_x.std():.2f}×')
r, lag = xcorr(e_x, tilt_Kd_x, dt)
print(f'  xcorr(e_x, D tilt):  corr={r:+.3f} lag={lag:+.0f} ms (90° lead ≈ T/4)')
r, lag = xcorr(e_x, tilt_Kp_x, dt)
print(f'  xcorr(e_x, P tilt):  corr={r:+.3f} lag={lag:+.0f} ms (in phase or 180°)')

print('\n--- AXIS Y ---')
print(f'  e_y:        std={e_y.std():.3f} m   max={np.abs(e_y).max():.3f}')
print(f'  P tilt:     std={tilt_Kp_y.std():.2f}° max={np.abs(tilt_Kp_y).max():.2f}°')
print(f'  D tilt:     std={tilt_Kd_y.std():.2f}° max={np.abs(tilt_Kd_y).max():.2f}°')
print(f'  D/P std ratio: {tilt_Kd_y.std()/tilt_Kp_y.std():.2f}×')
r, lag = xcorr(e_y, tilt_Kd_y, dt)
print(f'  xcorr(e_y, D tilt):  corr={r:+.3f} lag={lag:+.0f} ms')
r, lag = xcorr(e_y, tilt_Kp_y, dt)
print(f'  xcorr(e_y, P tilt):  corr={r:+.3f} lag={lag:+.0f} ms')


def align_zero(ax1, ax2):
    A = max(abs(ax1.get_ylim()[0]), abs(ax1.get_ylim()[1]))
    B = max(abs(ax2.get_ylim()[0]), abs(ax2.get_ylim()[1]))
    ax1.set_ylim(-A, A); ax2.set_ylim(-B, B)


fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

ax = axes[0]
ax.plot(t_o, e_x, 'k', lw=1.8, label='e_x [m]')
ax.set_ylabel('e_x [m]'); ax.axhline(0, color='k', alpha=0.3, lw=0.7); ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(t_o, tilt_Kp_x, 'r-',  lw=1.2, label='P only:  m·Kp·e_p tilt')
ax2.plot(t_o, tilt_Kd_x, 'b-',  lw=1.2, label='D only:  m·Kd·e_v tilt')
ax2.plot(t_o, tilt_PD_x, 'g--', lw=1.2, label='P + D sum')
ax2.plot(t_c, tilt_pub_x, color='gray', alpha=0.6, lw=1.2, label='published des')
ax2.set_ylabel('tilt [deg]')
align_zero(ax, ax2)
ax2.legend(loc='upper right', fontsize=8, ncol=2)
ax.set_title(f'{TAG} — AXIS X:  e_x vs P-only / D-only / sum / published   '
             f'(D/P std ratio = {tilt_Kd_x.std()/tilt_Kp_x.std():.2f}×)')

ax = axes[1]
ax.plot(t_o, e_y, 'k', lw=1.8, label='e_y [m]')
ax.set_ylabel('e_y [m]'); ax.axhline(0, color='k', alpha=0.3, lw=0.7); ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(t_o, tilt_Kp_y, 'r-',  lw=1.2, label='P only')
ax2.plot(t_o, tilt_Kd_y, 'b-',  lw=1.2, label='D only')
ax2.plot(t_o, tilt_PD_y, 'g--', lw=1.2, label='P + D sum')
ax2.plot(t_c, tilt_pub_y, color='gray', alpha=0.6, lw=1.2, label='published des')
ax2.set_ylabel('tilt [deg]')
align_zero(ax, ax2)
ax2.legend(loc='upper right', fontsize=8, ncol=2)
ax.set_title(f'AXIS Y:  e_y vs P-only / D-only / sum / published   '
             f'(D/P std ratio = {tilt_Kd_y.std()/tilt_Kp_y.std():.2f}×)')
ax.set_xlabel('time [s]')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_pd_decomp_t{int(T_CENTER)}.png')
plt.savefig(out, dpi=120)
print(f'\nSaved: {out}')
