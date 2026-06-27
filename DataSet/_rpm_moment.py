#!/usr/bin/env python3
"""Compare body moments computed from actual RPM vs commanded RPM (cmd_raw).

cmd_raw (uint16) -> rpm_cmd = cmd_raw * 9800/8191   (inverse of Rpm_to_cmd_raw)
thrust_i = C_T * rpm_i^2
[F, Mx, My, Mz] = K_forward @ thrust   (hexa allocation)

Plots Mx/My/Mz (cmd vs actual) over time, plus the actual attitude (roll/pitch)
so you can see how the cmd->actual moment gap correlates with attitude error.

Usage:
  python3 _rpm_moment.py <bag_subdir> [<date_root>] [<tag>]
  e.g. python3 _rpm_moment.py 01/hgdo/wo_ff DataSet wo_ff
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
ROOT = sys.argv[2] if len(sys.argv) > 2 else 'DataSet'
TAG = sys.argv[3] if len(sys.argv) > 3 else BAG.replace('/', '_')
db = glob.glob(os.path.join(_HERE, ROOT, BAG, '*.db3'))[0]
OUT_DIR = os.path.join(_HERE, ROOT, os.path.dirname(BAG))

# Drone params (from yaml)
L = 0.265
C_T = 1.3175e-07
K_M = 0.01569
MAX_BIT = 8191
MAX_RPM = 9800

# Hexa allocation (matches control_allocator.py)
c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)
ly = np.array([L * c, L, L * c, -L * c, -L, -L * c])
lx = np.array([L * s, 0, -L * s, -L * s, 0, L * s])
km = np.array([-K_M, K_M, -K_M, K_M, -K_M, K_M])
K_forward = np.vstack([np.ones(6), ly, -lx, km])   # rows: F, Mx, My, Mz


def _align(off, n):
    return off + (-(off - 4)) % n


def parse_cmd_raw(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 2)
    return np.array(struct.unpack_from('<6H', blob, off), dtype=float)


def parse_actual_rpm(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    return np.array(struct.unpack_from('<6I', blob, off), dtype=float)


def parse_odom(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off)
    return pz, qw, qx, qy, qz


def quat_rp(qw, qx, qy, qz):
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    sp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
    return roll, np.arcsin(sp)


def moments(rpm):
    """rpm: (N,6) -> (N,3) Mx,My,Mz"""
    thrust = C_T * rpm**2
    u = thrust @ K_forward.T   # (N,4)
    return u[:, 1:4]


con = sqlite3.connect(db); cur = con.cursor()
cur.execute("SELECT id,name FROM topics"); tids = {n: i for i, n in cur.fetchall()}
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tids['/mavros/local_position/odom']}")
t0 = cur.fetchone()[0]


def fetch(topic, parser):
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tids[topic]} ORDER BY timestamp")
    r = cur.fetchall()
    return np.array([(t - t0) * 1e-9 for t, _ in r]), np.array([parser(b) for _, b in r])


at, arpm = fetch('/uav/actual_rpm', parse_actual_rpm)
ct, craw = fetch('/uav/cmd_raw', parse_cmd_raw)
ot, odom = fetch('/mavros/local_position/odom', parse_odom)
con.close()

rpm_cmd = craw * MAX_RPM / MAX_BIT
M_act = moments(arpm)
M_cmd = moments(rpm_cmd)

# Airborne window
pz = odom[:, 0]
z_rel = pz - pz[ot < 5].mean() if (ot < 5).any() else pz
ab = z_rel > 0.05
t_to = ot[np.argmax(ab)]; t_land = ot[len(ab) - 1 - np.argmax(ab[::-1])]

# Attitude
rp = np.array([quat_rp(*odom[i, 1:5]) for i in range(len(ot))])
roll = np.degrees(rp[:, 0]); pitch = np.degrees(rp[:, 1])

# Resample cmd moment onto actual-rpm timeline for difference
M_cmd_i = np.vstack([np.interp(at, ct, M_cmd[:, k]) for k in range(3)]).T
dM = M_act - M_cmd_i

# Stats (airborne)
m = (at >= t_to + 2) & (at <= t_land - 2)
print(f"{TAG}  airborne {t_to:.1f}-{t_land:.1f}s")
names = ['Mx (roll)', 'My (pitch)', 'Mz (yaw)']
for k in range(3):
    print(f"  {names[k]}: cmd std={M_cmd_i[m,k].std():.4f}  act std={M_act[m,k].std():.4f}  "
          f"diff std={dM[m,k].std():.4f}  corr={np.corrcoef(M_cmd_i[m,k],M_act[m,k])[0,1]:.3f}")

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
for k, (ax, nm) in enumerate(zip(axes[:3], names)):
    ax.plot(ct, M_cmd[:, k], 'b', lw=0.7, alpha=0.8, label='cmd (rpm_cmd)')
    ax.plot(at, M_act[:, k], 'r', lw=0.7, alpha=0.8, label='actual rpm')
    ax.axvspan(t_to, t_land, alpha=0.05, color='g')
    ax.set_ylabel(f'{nm} [N·m]'); ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)
axes[0].set_title(f'{TAG} — body moment: commanded vs actual RPM')

axes[3].plot(ot, roll, 'r', lw=0.8, label='roll act')
axes[3].plot(ot, pitch, 'g', lw=0.8, label='pitch act')
axes[3].axvspan(t_to, t_land, alpha=0.05, color='g')
axes[3].axhline(0, color='k', alpha=0.3, lw=0.7)
axes[3].set_ylabel('attitude [deg]'); axes[3].set_xlabel('time [s]')
axes[3].grid(alpha=0.3); axes[3].legend(loc='upper right', fontsize=8)

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_rpm_moment.png')
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
