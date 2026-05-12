#!/usr/bin/env python3
"""Overlay XYZ position with F_des (post-DOB, world frame) per axis.

  /nmpc/control.force.x = F_des_x (world)
  /nmpc/control.force.y = F_des_y (world)
  /nmpc/control.force.z = f_col   (= ||F_des||, magnitude — NOT F_des_z)
  → F_des_z_world = sqrt(f_col² − F_des_x² − F_des_y²)

We also subtract m·g from F_des_z_world to highlight the control effort
above hover thrust.

Usage:  python3 pos_vs_fdes.py [<bag_subdir> [<date_dir>]]
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
M, G = 3.188, 9.81
print(f'Analyzing: {DB}')


def _align(off, n):
    return off + ((-(off - 4)) % n)

def parse_odom_p(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    return np.array(struct.unpack_from('<3d', blob, off))

def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return np.array(struct.unpack_from('<6d', blob, off))


conn = sqlite3.connect(DB)
c = conn.cursor()
tid = {n: i for i, n in c.execute('SELECT id, name FROM topics').fetchall()}

def fetch(topic, parser):
    rows = c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
                     (tid[topic],)).fetchall()
    ts = np.array([r[0] for r in rows], dtype=np.float64)
    dat = np.array([parser(bytes(r[1])) for r in rows])
    return ts, dat

odom_ts, p = fetch('/mavros/local_position/odom', parse_odom_p)
ctrl_ts, ctrl = fetch('/nmpc/control', parse_wrench)
conn.close()

t0 = min(odom_ts[0], ctrl_ts[0])
odom_t = (odom_ts - t0) * 1e-9
ctrl_t = (ctrl_ts - t0) * 1e-9

# Reconstruct F_des_z_world = sqrt(f_col^2 − Fx^2 − Fy^2)
fz_sq = ctrl[:, 2] ** 2 - ctrl[:, 0] ** 2 - ctrl[:, 1] ** 2
F_des_z_w = np.sqrt(np.maximum(fz_sq, 0.0))
F_des_x = ctrl[:, 0]
F_des_y = ctrl[:, 1]

# Position with mean-removed offset for visualization
mask = odom_t > 5.0
p_mean = p[mask].mean(axis=0)
p_zero = p - p_mean

# Stats
print('\n== Position and F_des stats (airborne) ==')
mask_c = ctrl_t > 5.0
for k, ax in enumerate('xyz'):
    print(f'  pos_{ax}     mean={p[mask, k].mean():+7.4f} m   std={p[mask, k].std():6.4f} m   '
          f'range [{p[mask, k].min():+7.3f}, {p[mask, k].max():+7.3f}]')
print('-- F_des in world frame --')
print(f'  Fx_des     mean={F_des_x[mask_c].mean():+7.4f} N   std={F_des_x[mask_c].std():6.4f}   '
      f'peak±{max(abs(F_des_x[mask_c].min()), abs(F_des_x[mask_c].max())):6.3f}')
print(f'  Fy_des     mean={F_des_y[mask_c].mean():+7.4f} N   std={F_des_y[mask_c].std():6.4f}   '
      f'peak±{max(abs(F_des_y[mask_c].min()), abs(F_des_y[mask_c].max())):6.3f}')
print(f'  Fz_des_w   mean={F_des_z_w[mask_c].mean():+7.4f} N   std={F_des_z_w[mask_c].std():6.4f}   '
      f'(hover m·g = {M*G:.2f} N)')
print(f'  Fz_des−mg  mean={(F_des_z_w[mask_c]-M*G).mean():+7.4f} N   std={(F_des_z_w[mask_c]-M*G).std():6.4f}')


# ── Plots — 3 rows × twin y-axes ──
fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
labels = [('x', 0, F_des_x),
          ('y', 1, F_des_y),
          ('z', 2, F_des_z_w - M * G)]

for k, (axis, idx, F) in enumerate(labels):
    ax_p = axes[k]
    ax_F = ax_p.twinx()
    l1, = ax_p.plot(odom_t, p_zero[:, idx], 'b', lw=1.0, label=f'pos_{axis} (mean removed)')
    if axis == 'z':
        l2, = ax_F.plot(ctrl_t, F, 'r', lw=0.9, alpha=0.85, label=f'F_des_z_world − m·g')
        ax_F.set_ylabel(f'F_des_z − m·g [N]', color='r')
    else:
        l2, = ax_F.plot(ctrl_t, F, 'r', lw=0.9, alpha=0.85, label=f'F_des_{axis} (world)')
        ax_F.set_ylabel(f'F_des_{axis} [N]', color='r')
    ax_p.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax_F.axhline(0, color='gray', lw=0.5, alpha=0.5, ls='--')
    ax_p.set_ylabel(f'pos_{axis} [m]', color='b')
    ax_p.tick_params(axis='y', labelcolor='b')
    ax_F.tick_params(axis='y', labelcolor='r')
    ax_p.grid(alpha=0.3)
    ax_p.legend([l1, l2], [l1.get_label(), l2.get_label()], loc='upper right', fontsize=9)
    ax_p.set_title(f'pos_{axis}  vs  F_des_{axis} (world)')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_pos_vs_Fdes.png'), dpi=120)
plt.close()

print(f'\nSaved: {TAG}_pos_vs_Fdes.png')
