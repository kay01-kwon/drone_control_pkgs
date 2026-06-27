#!/usr/bin/env python3
"""Plot desired vs actual roll and pitch.

desired roll/pitch reconstructed from /nmpc/control wrench:
  force.x, force.y are world-frame; force.z is body collective thrust.
actual roll/pitch from odom quaternion.

Usage:
  python3 _des_act_rp.py <bag_subdir> [<root>] [<tag>]
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


def _align(o, n):
    return o + (-(o - 4)) % n


def parse_odom(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 4)
    s2 = struct.unpack_from('<I', b, o)[0]; o += 4 + s2; o = _align(o, 8)
    pz = struct.unpack_from('<d', b, o + 16)[0]
    qx, qy, qz, qw = struct.unpack_from('<4d', b, o + 24)
    return pz, qw, qx, qy, qz


def parse_wrench(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 8)
    return struct.unpack_from('<6d', b, o)   # fx,fy,fz,mx,my,mz


def quat_rpy(qw, qx, qy, qz):
    r = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    p = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    y = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return r, p, y


def force_to_des_rp(fx_w, fy_w, fz_body, psi):
    fz_sq = fz_body**2 - fx_w**2 - fy_w**2
    fz_w = np.sqrt(max(fz_sq, 0.0))
    n = np.sqrt(fx_w**2 + fy_w**2 + fz_w**2)
    if n < 1e-6:
        return 0.0, 0.0
    zb = np.array([fx_w, fy_w, fz_w]) / n
    xc = np.array([np.cos(psi), np.sin(psi), 0.0])
    yb = np.cross(zb, xc); yn = np.linalg.norm(yb)
    yb = yb / yn if yn > 1e-6 else np.array([-np.sin(psi), np.cos(psi), 0.0])
    xb = np.cross(yb, zb)
    R = np.column_stack((xb, yb, zb))
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-R[2, 0])
    return roll, pitch


con = sqlite3.connect(db); cur = con.cursor()
cur.execute("SELECT id,name FROM topics"); tids = {n: i for i, n in cur.fetchall()}
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tids['/mavros/local_position/odom']}")
t0 = cur.fetchone()[0]


def fetch(topic, parser):
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tids[topic]} ORDER BY timestamp")
    r = cur.fetchall()
    return np.array([(t - t0) * 1e-9 for t, _ in r]), np.array([parser(b) for _, b in r])


ot, od = fetch('/mavros/local_position/odom', parse_odom)
ct, cw = fetch('/nmpc/control', parse_wrench)
con.close()

rpy = np.array([quat_rpy(*od[i, 1:5]) for i in range(len(ot))])
roll_act = np.degrees(rpy[:, 0]); pitch_act = np.degrees(rpy[:, 1])
yaw = rpy[:, 2]

# desired rp on ctrl timeline
psi_c = np.interp(ct, ot, np.unwrap(yaw))
des = np.array([force_to_des_rp(cw[i, 0], cw[i, 1], cw[i, 2], psi_c[i]) for i in range(len(ct))])
roll_des = np.degrees(des[:, 0]); pitch_des = np.degrees(des[:, 1])

# airborne
pz = od[:, 0]; z = pz - pz[ot < 5].mean() if (ot < 5).any() else pz
ab = z > 0.05
t_to = ot[np.argmax(ab)]; t_land = ot[len(ab) - 1 - np.argmax(ab[::-1])]

# stats airborne
ma = (ot >= t_to + 2) & (ot <= t_land - 2)
mc = (ct >= t_to + 2) & (ct <= t_land - 2)
print(f"{TAG}  airborne {t_to:.1f}-{t_land:.1f}s")
print(f"  roll : des std={roll_des[mc].std():.2f}  act std={roll_act[ma].std():.2f}  "
      f"act peak={np.abs(roll_act[ma]-roll_act[ma].mean()).max():.2f}")
print(f"  pitch: des std={pitch_des[mc].std():.2f}  act std={pitch_act[ma].std():.2f}  "
      f"act peak={np.abs(pitch_act[ma]-pitch_act[ma].mean()).max():.2f}")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(ot, roll_act, 'r-', lw=0.9, label='roll actual')
axes[0].plot(ct, roll_des, 'k--', lw=0.9, alpha=0.8, label='roll desired')
axes[0].axvspan(t_to, t_land, alpha=0.05, color='g'); axes[0].axhline(0, color='k', alpha=0.3, lw=0.7)
axes[0].set_ylabel('roll [deg]'); axes[0].grid(alpha=0.3); axes[0].legend(loc='upper right')
axes[0].set_title(f'{TAG} — desired vs actual roll / pitch')

axes[1].plot(ot, pitch_act, 'g-', lw=0.9, label='pitch actual')
axes[1].plot(ct, pitch_des, 'k--', lw=0.9, alpha=0.8, label='pitch desired')
axes[1].axvspan(t_to, t_land, alpha=0.05, color='g'); axes[1].axhline(0, color='k', alpha=0.3, lw=0.7)
axes[1].set_ylabel('pitch [deg]'); axes[1].set_xlabel('time [s]'); axes[1].grid(alpha=0.3)
axes[1].legend(loc='upper right')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_des_act_rp.png')
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
