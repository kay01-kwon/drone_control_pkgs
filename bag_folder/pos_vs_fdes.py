#!/usr/bin/env python3
"""Overlay XYZ position with /nmpc/control force per axis.

  /nmpc/control.force.x = F_des_x in WORLD frame (post-DOB)
  /nmpc/control.force.y = F_des_y in WORLD frame (post-DOB)
  /nmpc/control.force.z = f_col, collective thrust along BODY z (NOT world z)

Position is plotted as p_axis − p_axis(0) so each curve starts at 0.
We also subtract m·g from f_col to show control effort above hover.

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

# /nmpc/control: force.x, force.y are WORLD frame; force.z is BODY z (collective)
F_des_x = ctrl[:, 0]    # world
F_des_y = ctrl[:, 1]    # world
f_col   = ctrl[:, 2]    # body z (collective thrust)

# Position with INITIAL position subtracted per axis
p_init = p[0].copy()
p_zero = p - p_init
mask = odom_t > 5.0
print(f'\nInitial position p[0] = {p_init.tolist()}')

# Stats
print('\n== Position and F_des stats (airborne) ==')
mask_c = ctrl_t > 5.0
for k, ax in enumerate('xyz'):
    print(f'  pos_{ax}     mean={p[mask, k].mean():+7.4f} m   std={p[mask, k].std():6.4f} m   '
          f'range [{p[mask, k].min():+7.3f}, {p[mask, k].max():+7.3f}]')
print('-- /nmpc/control force (fx, fy in world; fz in body) --')
print(f'  Fx_des (world)  mean={F_des_x[mask_c].mean():+7.4f} N   std={F_des_x[mask_c].std():6.4f}   '
      f'peak±{max(abs(F_des_x[mask_c].min()), abs(F_des_x[mask_c].max())):6.3f}')
print(f'  Fy_des (world)  mean={F_des_y[mask_c].mean():+7.4f} N   std={F_des_y[mask_c].std():6.4f}   '
      f'peak±{max(abs(F_des_y[mask_c].min()), abs(F_des_y[mask_c].max())):6.3f}')
print(f'  f_col  (body z) mean={f_col[mask_c].mean():+7.4f} N   std={f_col[mask_c].std():6.4f}   '
      f'(hover m·g = {M*G:.2f} N)')
print(f'  f_col − m·g     mean={(f_col[mask_c]-M*G).mean():+7.4f} N   std={(f_col[mask_c]-M*G).std():6.4f}')


# ── Plots — 3 rows × twin y-axes ──
fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
labels = [('x', 0, F_des_x,       'F_des_x (world)'),
          ('y', 1, F_des_y,       'F_des_y (world)'),
          ('z', 2, f_col - M * G, 'f_col − m·g (body z)')]

for k, (axis, idx, F, F_label) in enumerate(labels):
    ax_p = axes[k]
    ax_F = ax_p.twinx()
    l1, = ax_p.plot(odom_t, p_zero[:, idx], 'b', lw=1.0, label=f'pos_{axis} − p_{axis}(0)')
    l2, = ax_F.plot(ctrl_t, F,              'r', lw=0.9, alpha=0.85, label=F_label)
    ax_p.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax_F.axhline(0, color='gray', lw=0.5, alpha=0.5, ls='--')
    ax_p.set_ylabel(f'pos_{axis} [m]', color='b')
    ax_F.set_ylabel(f'{F_label} [N]', color='r')
    ax_p.tick_params(axis='y', labelcolor='b')
    ax_F.tick_params(axis='y', labelcolor='r')
    ax_p.grid(alpha=0.3)
    ax_p.legend([l1, l2], [l1.get_label(), l2.get_label()], loc='upper right', fontsize=9)
    title_frame = 'world' if axis != 'z' else 'body z'
    ax_p.set_title(f'pos_{axis}  vs  {F_label}')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_pos_vs_Fdes.png'), dpi=120)
plt.close()

print(f'\nSaved: {TAG}_pos_vs_Fdes.png')
