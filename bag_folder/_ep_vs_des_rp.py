#!/usr/bin/env python3
"""Position error vs Desired roll/pitch (twin-axis) — causality at a glance.

Four panels:
  1. e_x  vs  des_roll
  2. e_x  vs  des_pitch
  3. e_y  vs  des_roll
  4. e_y  vs  des_pitch
The "correct" coupling (e_x↔des_pitch and e_y↔des_roll in ENU body axes)
should show strong correlation; the cross ones should be weaker.  This
lets us read off whether the PD term, the HGDO, or the attitude lag
is shaping des_RP.

Usage:
  python3 _ep_vs_des_rp.py <bag_subdir> <t_center> [<window_sec>] [<date_dir>] [<tag>]
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
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return struct.unpack_from('<8d', blob, off)


def quat_to_rpy(qw, qx, qy, qz):
    r = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    sp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
    p = np.arcsin(sp)
    y = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    return r, p, y


def force_to_des_rp(fx_w, fy_w, fz_body, psi):
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
con.close()

t_lo = T_CENTER - WIN / 2
t_hi = T_CENTER + WIN / 2
mO = (ot >= t_lo) & (ot <= t_hi)
mC = (ct >= t_lo) & (ct <= t_hi)
mR = (rt >= t_lo) & (rt <= t_hi)

t_o = ot[mO]
p_x = odom[mO, 0]; p_y = odom[mO, 1]
rp_act = np.array([quat_to_rpy(odom[i, 3], odom[i, 4], odom[i, 5], odom[i, 6])
                   for i in np.where(mO)[0]])
yaw_act = rp_act[:, 2]

t_r = rt[mR]
rp = ref[mR, 0:3]
rx = np.interp(t_o, t_r, rp[:, 0]) if len(t_r) > 1 else np.full_like(p_x, rp[0, 0])
ry = np.interp(t_o, t_r, rp[:, 1]) if len(t_r) > 1 else np.full_like(p_y, rp[0, 1])
e_x = rx - p_x
e_y = ry - p_y

t_c = ct[mC]
fx_w = ctrl[mC, 0]; fy_w = ctrl[mC, 1]; f_col = ctrl[mC, 2]
psi_c = np.interp(t_c, t_o, np.unwrap(yaw_act))
rp_des = np.array([force_to_des_rp(fx_w[i], fy_w[i], f_col[i], psi_c[i]) for i in range(len(t_c))])
roll_des = np.rad2deg(rp_des[:, 0])
pitch_des = np.rad2deg(rp_des[:, 1])

# Cross-correlations (sign-aware): max |corr| and lag
e_x_c = np.interp(t_c, t_o, e_x)
e_y_c = np.interp(t_c, t_o, e_y)


def xcorr_lag(a, b, dt, max_lag_s=1.0):
    a = a - a.mean(); b = b - b.mean()
    n = len(a); maxlag = int(max_lag_s / dt)
    lags = np.arange(-maxlag, maxlag + 1)
    c = np.zeros_like(lags, dtype=float)
    for i, k in enumerate(lags):
        if k >= 0:
            x, y = a[:n - k], b[k:]
        else:
            x, y = a[-k:], b[:n + k]
        if len(x) > 5:
            c[i] = np.corrcoef(x, y)[0, 1]
    j = np.argmax(np.abs(c))
    return c[j], lags[j] * dt * 1000  # ms; positive means b lags a


dt = np.median(np.diff(t_c)) if len(t_c) > 1 else 0.01
print(f'Window: {t_lo:.1f} → {t_hi:.1f} s')
print(f'Cross-correlation (b lags a if positive lag):')
for an, av in [('e_x', e_x_c), ('e_y', e_y_c)]:
    for bn, bv in [('roll_des', roll_des), ('pitch_des', pitch_des)]:
        r, lag = xcorr_lag(av, bv, dt)
        print(f'  {an} → {bn}:  corr={r:+.3f}  lag={lag:+.0f} ms')

fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
pairs = [
    ('e_x', e_x_c, 'roll_des',  roll_des,  'r', 'C0'),
    ('e_x', e_x_c, 'pitch_des', pitch_des, 'r', 'C2'),
    ('e_y', e_y_c, 'roll_des',  roll_des,  'g', 'C0'),
    ('e_y', e_y_c, 'pitch_des', pitch_des, 'g', 'C2'),
]
for ax, (an, av, bn, bv, ac, bc) in zip(axes, pairs):
    r, lag = xcorr_lag(av, bv, dt)
    ax.plot(t_c, av, color=ac, label=f'{an} [m]')
    ax.set_ylabel(f'{an} [m]', color=ac); ax.tick_params(axis='y', labelcolor=ac)
    ax.axhline(0, color='k', alpha=0.2)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t_c, bv, color=bc, label=f'{bn} [deg]')
    ax2.set_ylabel(f'{bn} [deg]', color=bc); ax2.tick_params(axis='y', labelcolor=bc)
    ax2.axhline(0, color='k', alpha=0.2, ls=':')
    ax.set_title(f'{an}  ↔  {bn}     corr={r:+.3f}, lag={lag:+.0f} ms')
axes[-1].set_xlabel('time [s]')
fig.suptitle(f'{TAG} — pos-error vs desired RP (t={T_CENTER:.0f}±{WIN/2:.0f} s)', y=1.00)
plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_ep_vs_des_rp_t{int(T_CENTER)}.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f'Saved: {out}')
