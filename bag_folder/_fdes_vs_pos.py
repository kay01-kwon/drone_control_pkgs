#!/usr/bin/env python3
"""F_des_xy (world) overlaid with position (world) — direct PD verification.

Both signals are in world frame, so axis coupling and yaw rotation do not
muddle the picture.  PD law in world frame:
    F_des_x_world = m·(Kp·e_x + Kd·e_v_x)   (HGDO fxy disabled in this exp)
    F_des_y_world = m·(Kp·e_y + Kd·e_v_y)

Expected behavior:
  • corr(e_p, F_des) > 0   (PD correctly opposing position error)
  • lag < 0                (F_des leads e_p because Kd·e_v term has 90° lead
                            at limit-cycle frequency)
  • |lag| ≈ 90° / ω        (signature of D-term dominance)

Usage:
  python3 _fdes_vs_pos.py <bag_subdir> <t_center> [<win_s>] [<date>] [<tag>]
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
    px, py, pz = struct.unpack_from('<3d', blob, off)
    return px, py, pz


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


ot, p = fetch('/mavros/local_position/odom', parse_odom)
ct, ctrl = fetch('/nmpc/control', parse_wrench)
rt, ref = fetch('/nmpc/ref', parse_ref)
con.close()

# Window
mO = (ot >= T_CENTER - WIN / 2) & (ot <= T_CENTER + WIN / 2)
mC = (ct >= T_CENTER - WIN / 2) & (ct <= T_CENTER + WIN / 2)
mR = (rt >= T_CENTER - WIN / 2) & (rt <= T_CENTER + WIN / 2)

t_o = ot[mO]
p_x = p[mO, 0]; p_y = p[mO, 1]

t_r = rt[mR]
rp = ref[mR, 0:3]
if len(t_r) > 1:
    rx = np.interp(t_o, t_r, rp[:, 0]); ry = np.interp(t_o, t_r, rp[:, 1])
else:
    rx = np.full_like(p_x, rp[0, 0] if len(rp) else 0)
    ry = np.full_like(p_y, rp[0, 1] if len(rp) else 0)

e_x = rx - p_x
e_y = ry - p_y

# F_des world frame (force.x, force.y are world; force.z is body collective)
t_c = ct[mC]
F_x = ctrl[mC, 0]
F_y = ctrl[mC, 1]

# Resample F_des to odom timeline for direct comparison
F_x_o = np.interp(t_o, t_c, F_x)
F_y_o = np.interp(t_o, t_c, F_y)


def xcorr_lag(a, b, dt, max_lag_s=1.0):
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

r_xx, lag_xx = xcorr_lag(e_x, F_x_o, dt)
r_yy, lag_yy = xcorr_lag(e_y, F_y_o, dt)
r_xy, lag_xy = xcorr_lag(e_x, F_y_o, dt)
r_yx, lag_yx = xcorr_lag(e_y, F_x_o, dt)

print(f'Window: {t_o[0]:.1f} → {t_o[-1]:.1f} s')
print(f'\nDirect (world frame, same axis):')
print(f'  e_x → F_x:  corr={r_xx:+.3f}  lag={lag_xx:+.0f} ms   '
      f'(lead ≈ {-lag_xx*360/1000:.0f}° at 1 Hz)')
print(f'  e_y → F_y:  corr={r_yy:+.3f}  lag={lag_yy:+.0f} ms')
print(f'\nCross axis (should be small if no yaw / coupling):')
print(f'  e_x → F_y:  corr={r_xy:+.3f}  lag={lag_xy:+.0f} ms')
print(f'  e_y → F_x:  corr={r_yx:+.3f}  lag={lag_yx:+.0f} ms')


def align_zero(ax1, ax2):
    A = max(abs(ax1.get_ylim()[0]), abs(ax1.get_ylim()[1]))
    B = max(abs(ax2.get_ylim()[0]), abs(ax2.get_ylim()[1]))
    ax1.set_ylim(-A, A); ax2.set_ylim(-B, B)


fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

# Panel 1: pos_x and ref_x + F_des_x
ax = axes[0]
ax.plot(t_o, p_x, 'r-',  lw=1.5, label='p_x [m]')
ax.plot(t_o, rx,  'r--', lw=0.8, alpha=0.6, label='ref_x')
ax.set_ylabel('p_x [m]', color='r'); ax.tick_params(axis='y', labelcolor='r')
ax.axhline(0, color='k', alpha=0.3, lw=0.7); ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(t_o, F_x_o, 'b-', lw=1.2, label='F_des_x [N] (world)')
ax2.set_ylabel('F_des_x [N]', color='b'); ax2.tick_params(axis='y', labelcolor='b')
align_zero(ax, ax2)
ax.set_title(f'{TAG} — p_x vs F_des_x (world)   '
             f'e_x↔F_x: corr={r_xx:+.3f}, lag={lag_xx:+.0f} ms')
ax2.legend(loc='upper right')

# Panel 2: e_x vs F_des_x (same as above but plotting e_x for direct check)
ax = axes[1]
ax.plot(t_o, e_x, 'r', lw=1.5, label='e_x = ref-p [m]')
ax.set_ylabel('e_x [m]', color='r'); ax.tick_params(axis='y', labelcolor='r')
ax.axhline(0, color='k', alpha=0.3, lw=0.7); ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(t_o, F_x_o, 'b', lw=1.2, label='F_des_x [N]')
ax2.set_ylabel('F_des_x [N]', color='b'); ax2.tick_params(axis='y', labelcolor='b')
align_zero(ax, ax2)
ax.set_title(f'e_x ↔ F_des_x   (positive corr = PD damping working;  '
             f'lag<0 = D-term lead)')
ax2.legend(loc='upper right')

# Panel 3: p_y and ref_y + F_des_y
ax = axes[2]
ax.plot(t_o, p_y, 'g-',  lw=1.5, label='p_y [m]')
ax.plot(t_o, ry,  'g--', lw=0.8, alpha=0.6, label='ref_y')
ax.set_ylabel('p_y [m]', color='g'); ax.tick_params(axis='y', labelcolor='g')
ax.axhline(0, color='k', alpha=0.3, lw=0.7); ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(t_o, F_y_o, 'm-', lw=1.2, label='F_des_y [N] (world)')
ax2.set_ylabel('F_des_y [N]', color='m'); ax2.tick_params(axis='y', labelcolor='m')
align_zero(ax, ax2)
ax.set_title(f'p_y vs F_des_y (world)   '
            f'e_y↔F_y: corr={r_yy:+.3f}, lag={lag_yy:+.0f} ms')
ax2.legend(loc='upper right')

# Panel 4: e_y vs F_des_y
ax = axes[3]
ax.plot(t_o, e_y, 'g', lw=1.5, label='e_y [m]')
ax.set_ylabel('e_y [m]', color='g'); ax.tick_params(axis='y', labelcolor='g')
ax.axhline(0, color='k', alpha=0.3, lw=0.7); ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(t_o, F_y_o, 'm', lw=1.2, label='F_des_y [N]')
ax2.set_ylabel('F_des_y [N]', color='m'); ax2.tick_params(axis='y', labelcolor='m')
align_zero(ax, ax2)
ax.set_title(f'e_y ↔ F_des_y   '
            f'Cross check  e_x↔F_y: {r_xy:+.3f}  e_y↔F_x: {r_yx:+.3f}')
ax.set_xlabel('time [s]')
ax2.legend(loc='upper right')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_fdes_vs_pos_t{int(T_CENTER)}.png')
plt.savefig(out, dpi=120)
print(f'\nSaved: {out}')
