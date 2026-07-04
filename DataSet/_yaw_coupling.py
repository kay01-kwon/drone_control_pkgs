#!/usr/bin/env python3
"""Does yaw (Mz control) degrade position control?

Two coupling paths tested:
  P1  yaw error -> world/body misalignment -> position force projects wrong
      => check corr between |yaw error| and |xy position error| (rolling),
         and whether cross-axis F coupling grows when yaw error is large.
  P2  Mz allocation eats roll/pitch authority (hexa yaw ~12x weaker)
      => check corr between |Mz| and roll/pitch tracking error.

Usage:
  python3 _yaw_coupling.py <bag_subdir> [<root>] [<tag>]
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
    px, py, pz = struct.unpack_from('<3d', b, o)
    qx, qy, qz, qw = struct.unpack_from('<4d', b, o + 24)
    off = o + 24 + 32 + 36 * 8
    vx, vy, vz, wx, wy, wz = struct.unpack_from('<6d', b, off)
    return px, py, pz, qw, qx, qy, qz, wx, wy, wz


def parse_wrench(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 8)
    return struct.unpack_from('<6d', b, o)


def parse_ref(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 8)
    return struct.unpack_from('<8d', b, o)   # p3 v3 psi psidot


def quat_rpy(qw, qx, qy, qz):
    r = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    p = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    y = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return r, p, y


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
rt, ref = fetch('/nmpc/ref', parse_ref)
con.close()

rpy = np.array([quat_rpy(*od[i, 3:7]) for i in range(len(ot))])
roll, pitch, yaw = np.degrees(rpy[:, 0]), np.degrees(rpy[:, 1]), np.degrees(rpy[:, 2])
px, py = od[:, 0], od[:, 1]
Mz = cw[:, 5]
Fx, Fy = cw[:, 0], cw[:, 1]

# refs
ref_x = np.interp(ot, rt, ref[:, 0]); ref_y = np.interp(ot, rt, ref[:, 1])
ref_psi = np.degrees(np.interp(ot, rt, ref[:, 6]))
e_x = ref_x - px; e_y = ref_y - py
yaw_err = yaw - ref_psi

# airborne
pz = od[:, 2]; z = pz - pz[ot < 5].mean() if (ot < 5).any() else pz
ab = z > 0.05
t_to = ot[np.argmax(ab)]; t_land = ot[len(ab) - 1 - np.argmax(ab[::-1])]
m = (ot >= t_to + 2) & (ot <= t_land - 2)
mc = (ct >= t_to + 2) & (ct <= t_land - 2)

# --- P1: yaw error vs position error ---
# rolling 2s std
fs = 1 / np.median(np.diff(ot)); w = int(2 * fs)
def rstd(x):
    out = np.zeros_like(x)
    for i in range(len(x)):
        a, b = max(0, i - w // 2), min(len(x), i + w // 2)
        out[i] = x[a:b].std()
    return out
ye_rs = rstd(yaw_err); exy_rs = np.sqrt(rstd(e_x)**2 + rstd(e_y)**2)
c_p1 = np.corrcoef(ye_rs[m], exy_rs[m])[0, 1]

# cross-axis F coupling split by yaw-error magnitude
ex_c = np.interp(ct, ot, e_x); ey_c = np.interp(ct, ot, e_y)
yerr_c = np.interp(ct, ot, np.abs(yaw_err))
hi = yerr_c > np.median(yerr_c[mc]); lo = ~hi
def safe_corr(a, b, msk):
    msk = msk & mc
    return np.corrcoef(a[msk], b[msk])[0, 1] if msk.sum() > 20 else np.nan
# cross-axis: e_x vs F_y and e_y vs F_x
cxy_hi = safe_corr(ex_c, Fy, hi); cxy_lo = safe_corr(ex_c, Fy, lo)
cyx_hi = safe_corr(ey_c, Fx, hi); cyx_lo = safe_corr(ey_c, Fx, lo)

# --- P2: |Mz| vs roll/pitch tracking (proxy: attitude std in windows) ---
Mz_c = np.abs(Mz)
roll_c = np.interp(ct, ot, np.abs(roll - roll[m].mean()))
pitch_c = np.interp(ct, ot, np.abs(pitch - pitch[m].mean()))
c_mz_roll = safe_corr(Mz_c, roll_c, np.ones(len(ct), bool))
c_mz_pitch = safe_corr(Mz_c, pitch_c, np.ones(len(ct), bool))

print(f"{TAG}  airborne {t_to:.1f}-{t_land:.1f}s")
print(f"  yaw: std={yaw[m].std():.2f}deg  err std={yaw_err[m].std():.2f}  |err|max={np.abs(yaw_err[m]).max():.2f}")
print(f"  Mz: std={Mz[mc].std():.4f}  |Mz|max={np.abs(Mz[mc]).max():.4f} N·m")
print(f"\n  P1 yaw->position:")
print(f"    corr(|yaw_err| rolling, |e_xy| rolling) = {c_p1:+.3f}")
print(f"    cross F coupling  e_x~F_y: hi-yaw={cxy_hi:+.3f} lo-yaw={cxy_lo:+.3f}")
print(f"                      e_y~F_x: hi-yaw={cyx_hi:+.3f} lo-yaw={cyx_lo:+.3f}")
print(f"  P2 Mz->attitude:")
print(f"    corr(|Mz|, |roll|)={c_mz_roll:+.3f}  corr(|Mz|, |pitch|)={c_mz_pitch:+.3f}")

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
ax = axes[0]
ax.plot(ot, yaw - ref_psi, 'purple', lw=0.9, label='yaw error [deg]')
ax.axhline(0, color='k', alpha=0.3, lw=0.7); ax.axvspan(t_to, t_land, alpha=0.05, color='g')
ax.set_ylabel('yaw err [deg]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')
ax.set_title(f'{TAG} — yaw coupling  (P1 corr={c_p1:+.2f}; cross e_y~F_x hi/lo yaw={cyx_hi:+.2f}/{cyx_lo:+.2f})')

ax = axes[1]
ax.plot(ot, e_x, 'r', lw=0.8, label='e_x'); ax.plot(ot, e_y, 'g', lw=0.8, label='e_y')
ax.axhline(0, color='k', alpha=0.3, lw=0.7); ax.axvspan(t_to, t_land, alpha=0.05, color='g')
ax.set_ylabel('pos err [m]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')

ax = axes[2]
ax.plot(ct, Mz, 'b', lw=0.7, label='Mz (yaw torque)')
ax.axhline(0, color='k', alpha=0.3, lw=0.7); ax.axvspan(t_to, t_land, alpha=0.05, color='g')
ax.set_ylabel('Mz [N·m]'); ax.set_xlabel('time [s]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_yaw_coupling.png')
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
