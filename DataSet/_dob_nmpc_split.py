#!/usr/bin/env python3
"""Separate DOB vs pure-NMPC torque contribution and relate each to attitude.

/nmpc/control torque = motor command (post-DOB) = NMPC_pure + DOB
  => NMPC_pure = published - DOB
/hgdo (or l1) /wrench torque = DOB

For roll(Mx) and pitch(My) we check whether each torque component acts to
*reduce* the attitude error (negative corr with attitude => restoring) or
*increase* it.  Also reports std of each component.

Usage:
  python3 _dob_nmpc_split.py <bag_subdir> [<root>] [<tag>]
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


def parse_wrench(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 8)
    return struct.unpack_from('<6d', b, o)   # fx,fy,fz,mx,my,mz


def parse_odom(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 4)
    s2 = struct.unpack_from('<I', b, o)[0]; o += 4 + s2; o = _align(o, 8)
    pz = struct.unpack_from('<d', b, o + 16)[0]
    qx, qy, qz, qw = struct.unpack_from('<4d', b, o + 24)
    off = o + 24 + 32 + 36 * 8
    vx, vy, vz, wx, wy, wz = struct.unpack_from('<6d', b, off)
    return pz, qw, qx, qy, qz, wx, wy, wz


def quat_rp(qw, qx, qy, qz):
    r = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    p = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    return np.degrees(r), np.degrees(p)


con = sqlite3.connect(db); cur = con.cursor()
cur.execute("SELECT id,name FROM topics"); tids = {n: i for i, n in cur.fetchall()}
wtopic = '/hgdo/wrench' if '/hgdo/wrench' in tids else '/l1_adaptive/wrench'
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tids['/mavros/local_position/odom']}")
t0 = cur.fetchone()[0]


def fetch(topic, parser):
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tids[topic]} ORDER BY timestamp")
    r = cur.fetchall()
    return np.array([(t - t0) * 1e-9 for t, _ in r]), np.array([parser(b) for _, b in r])


ct, pub = fetch('/nmpc/control', parse_wrench)
dt_, dob = fetch(wtopic, parse_wrench)
ot, od = fetch('/mavros/local_position/odom', parse_odom)
con.close()

# common ctrl timeline
Mx_pub, My_pub = pub[:, 3], pub[:, 4]
dob_Mx = np.interp(ct, dt_, dob[:, 3]); dob_My = np.interp(ct, dt_, dob[:, 4])
# pure NMPC = published - DOB
nmpc_Mx = Mx_pub - dob_Mx; nmpc_My = My_pub - dob_My

rp = np.array([quat_rp(*od[i, 1:5]) for i in range(len(ot))])
roll, pitch = rp[:, 0], rp[:, 1]
wx, wy = od[:, 5], od[:, 6]
roll_c = np.interp(ct, ot, roll - roll.mean())
pitch_c = np.interp(ct, ot, pitch - pitch.mean())

pz = od[:, 0]; z = pz - pz[ot < 5].mean() if (ot < 5).any() else pz
ab = z > 0.05
t_to = ot[np.argmax(ab)]; t_land = ot[len(ab) - 1 - np.argmax(ab[::-1])]
m = (ct >= t_to + 2) & (ct <= t_land - 2)


def corr(a, b):
    return np.corrcoef(a[m], b[m])[0, 1]


print(f"{TAG}  airborne {t_to:.1f}-{t_land:.1f}s   ({wtopic})")
print("  torque std [N·m]:")
print(f"    roll : published={Mx_pub[m].std():.4f}  NMPC_pure={nmpc_Mx[m].std():.4f}  DOB={dob_Mx[m].std():.4f}")
print(f"    pitch: published={My_pub[m].std():.4f}  NMPC_pure={nmpc_My[m].std():.4f}  DOB={dob_My[m].std():.4f}")
print("  corr with attitude (neg => restoring / opposes tilt):")
print(f"    roll : NMPC_pure vs roll={corr(nmpc_Mx,roll_c):+.3f}   DOB vs roll={corr(dob_Mx,roll_c):+.3f}")
print(f"    pitch: NMPC_pure vs pitch={corr(nmpc_My,pitch_c):+.3f}  DOB vs pitch={corr(dob_My,pitch_c):+.3f}")
print("  DOB mean (bias) [N·m]:")
print(f"    roll DOB mean={dob_Mx[m].mean():+.4f}   pitch DOB mean={dob_My[m].mean():+.4f}")

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
# Panel 1: roll torque split
ax = axes[0]
ax.plot(ct, Mx_pub, 'k', lw=0.7, alpha=0.6, label='published Mx (motor cmd)')
ax.plot(ct, nmpc_Mx, 'b', lw=0.7, alpha=0.8, label='NMPC pure Mx')
ax.plot(ct, dob_Mx, 'r', lw=0.8, label='DOB Mx')
ax.axvspan(t_to, t_land, alpha=0.05, color='g'); ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('roll torque [N·m]'); ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)
ax.set_title(f'{TAG} — torque split (published = NMPC_pure + DOB)')
# Panel 2: pitch torque split
ax = axes[1]
ax.plot(ct, My_pub, 'k', lw=0.7, alpha=0.6, label='published My')
ax.plot(ct, nmpc_My, 'b', lw=0.7, alpha=0.8, label='NMPC pure My')
ax.plot(ct, dob_My, 'r', lw=0.8, label='DOB My')
ax.axvspan(t_to, t_land, alpha=0.05, color='g'); ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('pitch torque [N·m]'); ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)
# Panel 3: attitude
ax = axes[2]
ax.plot(ot, roll - roll.mean(), 'r', lw=0.8, label='roll (mean-sub)')
ax.plot(ot, pitch - pitch.mean(), 'g', lw=0.8, label='pitch (mean-sub)')
ax.axvspan(t_to, t_land, alpha=0.05, color='g'); ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('attitude [deg]'); ax.set_xlabel('time [s]'); ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_dob_nmpc_split.png')
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
