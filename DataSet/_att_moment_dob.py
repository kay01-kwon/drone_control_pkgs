#!/usr/bin/env python3
"""Overlay actual-RPM body moment with attitude (roll<->Mx, pitch<->My),
and plot DOB moment over time.

Panels:
  1. roll (left axis)  vs  Mx_actual (right axis)
  2. pitch (left axis) vs  My_actual (right axis)
  3. DOB torque Mx, My, Mz over time

Usage:
  python3 _att_moment_dob.py <bag_subdir> [<root>] [<tag>]
  e.g. python3 _att_moment_dob.py 01/hgdo/wo_ff . hgdo_wo_ff
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
ROOT = sys.argv[2] if len(sys.argv) > 2 else '.'
TAG = sys.argv[3] if len(sys.argv) > 3 else BAG.replace('/', '_')
db = glob.glob(os.path.join(_HERE, ROOT, BAG, '*.db3'))[0]
OUT_DIR = os.path.join(_HERE, ROOT, os.path.dirname(BAG))

L, C_T, K_M = 0.265, 1.3175e-07, 0.01569
c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)
ly = np.array([L * c, L, L * c, -L * c, -L, -L * c])
lx = np.array([L * s, 0, -L * s, -L * s, 0, L * s])
km = np.array([-K_M, K_M, -K_M, K_M, -K_M, K_M])
K_forward = np.vstack([np.ones(6), ly, -lx, km])


def _align(o, n):
    return o + (-(o - 4)) % n


def parse_actual_rpm(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 4)
    return np.array(struct.unpack_from('<6I', b, o), dtype=float)


def parse_odom(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 4)
    s2 = struct.unpack_from('<I', b, o)[0]; o += 4 + s2; o = _align(o, 8)
    pz = struct.unpack_from('<d', b, o + 16)[0]
    qx, qy, qz, qw = struct.unpack_from('<4d', b, o + 24)
    return pz, qw, qx, qy, qz


def parse_wrench(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 8)
    return struct.unpack_from('<6d', b, o)   # fx,fy,fz,mx,my,mz


def quat_rp(qw, qx, qy, qz):
    r = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    p = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    return np.degrees(r), np.degrees(p)


con = sqlite3.connect(db); cur = con.cursor()
cur.execute("SELECT id,name FROM topics"); tids = {n: i for i, n in cur.fetchall()}
wrench_topic = '/hgdo/wrench' if '/hgdo/wrench' in tids else '/l1_adaptive/wrench'
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tids['/mavros/local_position/odom']}")
t0 = cur.fetchone()[0]


def fetch(topic, parser):
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tids[topic]} ORDER BY timestamp")
    r = cur.fetchall()
    return np.array([(t - t0) * 1e-9 for t, _ in r]), np.array([parser(b) for _, b in r])


at, arpm = fetch('/uav/actual_rpm', parse_actual_rpm)
ot, odom = fetch('/mavros/local_position/odom', parse_odom)
wt, wr = fetch(wrench_topic, parse_wrench)
con.close()

thrust = C_T * arpm**2
M_act = thrust @ K_forward.T   # (N,4): F,Mx,My,Mz
Mx, My = M_act[:, 1], M_act[:, 2]

rp = np.array([quat_rp(*odom[i, 1:5]) for i in range(len(ot))])
roll, pitch = rp[:, 0], rp[:, 1]

dob_Mx, dob_My, dob_Mz = wr[:, 3], wr[:, 4], wr[:, 5]

# Airborne window
pz = odom[:, 0]
z = pz - pz[ot < 5].mean() if (ot < 5).any() else pz
ab = z > 0.05
t_to = ot[np.argmax(ab)]; t_land = ot[len(ab) - 1 - np.argmax(ab[::-1])]


def align_zero(a1, a2):
    A = max(abs(a1.get_ylim()[0]), abs(a1.get_ylim()[1]))
    B = max(abs(a2.get_ylim()[0]), abs(a2.get_ylim()[1]))
    a1.set_ylim(-A, A); a2.set_ylim(-B, B)


fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

# Panel 1: roll vs Mx_act
ax = axes[0]
ax.plot(ot, roll, 'r', lw=0.9, label='roll act [deg]')
ax.set_ylabel('roll [deg]', color='r'); ax.tick_params(axis='y', labelcolor='r')
ax.axhline(0, color='k', alpha=0.3, lw=0.7); ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(at, Mx, 'b', lw=0.7, alpha=0.8, label='Mx actual [N·m]')
ax2.set_ylabel('Mx actual [N·m]', color='b'); ax2.tick_params(axis='y', labelcolor='b')
align_zero(ax, ax2)
ax.axvspan(t_to, t_land, alpha=0.05, color='g')
ax.set_title(f'{TAG} — roll vs actual Mx')

# Panel 2: pitch vs My_act
ax = axes[1]
ax.plot(ot, pitch, 'g', lw=0.9, label='pitch act [deg]')
ax.set_ylabel('pitch [deg]', color='g'); ax.tick_params(axis='y', labelcolor='g')
ax.axhline(0, color='k', alpha=0.3, lw=0.7); ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(at, My, 'b', lw=0.7, alpha=0.8, label='My actual [N·m]')
ax2.set_ylabel('My actual [N·m]', color='b'); ax2.tick_params(axis='y', labelcolor='b')
align_zero(ax, ax2)
ax.axvspan(t_to, t_land, alpha=0.05, color='g')
ax.set_title('pitch vs actual My')

# Panel 3: DOB torque over time
ax = axes[2]
ax.plot(wt, dob_Mx, 'r', lw=0.8, label='DOB Mx')
ax.plot(wt, dob_My, 'g', lw=0.8, label='DOB My')
ax.plot(wt, dob_Mz, 'b', lw=0.8, label='DOB Mz')
ax.axhline(0, color='k', alpha=0.3, lw=0.7); ax.grid(alpha=0.3)
ax.axvspan(t_to, t_land, alpha=0.05, color='g')
ax.set_ylabel('DOB torque [N·m]'); ax.set_xlabel('time [s]')
ax.legend(loc='upper right', fontsize=8)
ax.set_title(f'DOB moment ({wrench_topic})')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_att_moment_dob.png')
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
