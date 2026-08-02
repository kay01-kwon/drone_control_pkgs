#!/usr/bin/env python3
"""Ground-effect (GE) pivot moment from the dataset, per the image-source +
fountain model (ge_derivation.pdf).

For each sample uses actual roll, pitch, relative altitude h (= z - z0, with
h=0.315 m at level), and per-rotor thrust Ti = C_T * rpm_i^2, then computes
the roll- and pitch-axis GE moment about the pivot:

  zi        = h*cos(phi) + (b_i + lp)*sin(phi)      # b_i = ly_i (roll) / lx_i (pitch)
  dvi/vinf  = (R^2/4) * sum_j (zi+zj) / (dij^2+(zi+zj)^2)^{3/2}
  k(zc)     = S9rot(zc) / dvi_level(zc)
  dvf/vinf  = 2 R^2 Jk zc / (rd^2+4 zc^2)^{3/2},  rd = 2d - c
  grot_i    = 1/(1 - k dvi)
  gfull_i   = 1/(1 - k dvi - dvf)
  dMGE      = sum_i b_i (grot_i-1) Ti  +  [sum_i (gfull_i-1) Ti] * lp

Usage:
  python3 _ge_moment.py <bag_subdir> [<tag>] [<h_level>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else BAG.replace('/', '_')
H_LEVEL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.315
db = glob.glob(os.path.join(_HERE, BAG, '*.db3'))[0]
OUT_DIR = os.path.join(_HERE, os.path.dirname(BAG))

# platform params (PDF p.5)
R = 0.127; d = 0.265; c = 0.22; lp = 0.140; Jk = 2.2
rd = 2 * d - c
C_T = 1.3175e-07

# rotor geometry (same as control_allocator)
cs, sn = np.cos(np.pi / 3), np.sin(np.pi / 3)
ly = d * np.array([cs, 1, cs, -cs, -1, -cs])
lx = d * np.array([sn, 0, -sn, -sn, 0, sn])
# pairwise horizontal separation
xy = np.column_stack([lx, ly])
dij = np.sqrt(((xy[:, None, :] - xy[None, :, :])**2).sum(-1))   # 6x6


def S9rot(z):
    return (R**2 * z / (d**2 / 4 + 4 * z**2)**1.5
            + R**2 / 2 * z / (21 * d**2 / 4 + 4 * z**2)**1.5
            + R**2 * z / (13 * d**2 / 4 + 4 * z**2)**1.5
            + R**2 / 2 * z / (7 * d**2 / 4 + 4 * z**2)**1.5)


def dvi_level(zc):
    # reference rotor 0, all rotors at zc
    s = 0.0
    for j in range(6):
        s += 2 * zc / (dij[0, j]**2 + 4 * zc**2)**1.5
    return R**2 / 4 * s


def ge_moment_axis(phi, h, Ti, arm):
    """arm = ly (roll) or lx (pitch); b_i = arm for roll, -arm for pitch handled by caller sign."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    zi = h * cphi + (arm + lp) * sphi            # (6,)
    zi = np.maximum(zi, 1e-3)
    zc = h * cphi + lp * sphi
    zc = max(zc, 1e-3)
    # dvi per rotor
    dvi = np.zeros(6)
    for i in range(6):
        zz = zi[i] + zi
        dvi[i] = (R**2 / 4) * np.sum(zz / (dij[i]**2 + zz**2)**1.5)
    k = S9rot(zc) / dvi_level(zc)
    dvf = 2 * R**2 * Jk * zc / (rd**2 + 4 * zc**2)**1.5
    grot = 1.0 / (1.0 - k * dvi)
    gfull = 1.0 / (1.0 - k * dvi - dvf)
    return grot, gfull


def dM_roll(phi_r, h, Ti):
    grot, gfull = ge_moment_axis(phi_r, h, Ti, ly)
    return np.sum(ly * (grot - 1) * Ti) + np.sum((gfull - 1) * Ti) * lp


def dM_pitch(phi_p, h, Ti):
    grot, gfull = ge_moment_axis(phi_p, h, Ti, lx)
    return np.sum((-lx) * (grot - 1) * Ti) + np.sum((gfull - 1) * Ti) * lp


# ---- self-check against PDF table (phi=0, h=0.315, equal thrust, f=0.7W) ----
for mm in [3.0]:
    W = mm * 9.81; f = 0.70 * W; Ti0 = np.full(6, f / 6)
    grot, gfull = ge_moment_axis(0.0, 0.315, Ti0, ly)
    a_check = np.sum((gfull - 1) * Ti0) * lp   # roll a at phi=0 (interference sum=0 by symmetry)
    print(f"[self-check] phi=0,h=0.315,f=0.7W(m={mm}): "
          f"a/(f*lp)={100*a_check/(f*lp):.3f}%  (PDF 11.666%)  gfull-1={gfull[0]-1:.5f}")


def _align(o, n):
    return o + (-(o - 4)) % n


def parse_odom(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 4)
    s2 = struct.unpack_from('<I', b, o)[0]; o += 4 + s2; o = _align(o, 8)
    pz = struct.unpack_from('<d', b, o + 16)[0]
    qx, qy, qz, qw = struct.unpack_from('<4d', b, o + 24)
    return pz, qw, qx, qy, qz


def parse_rpm(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 4)
    return np.array(struct.unpack_from('<6I', b, o), dtype=float)


def quat_rp(qw, qx, qy, qz):
    r = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    p = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    return r, p


con = sqlite3.connect(db); cur = con.cursor()
cur.execute("SELECT id,name FROM topics"); tids = {n: i for i, n in cur.fetchall()}
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tids['/mavros/local_position/odom']}")
t0 = cur.fetchone()[0]


def fetch(topic, parser):
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tids[topic]} ORDER BY timestamp")
    r = cur.fetchall()
    return np.array([(t - t0) * 1e-9 for t, _ in r]), [parser(b) for _, b in r]


ot, od = fetch('/mavros/local_position/odom', parse_odom)
rt, rpm = fetch('/uav/actual_rpm', parse_rpm)
con.close()
od = np.array(od)
pz = od[:, 0]; z0 = pz[ot < 5].mean() if (ot < 5).any() else 0.0
z_rel = pz - z0                                    # measured relative altitude
# rotor ground clearance = z_rel + offset, calibrated so that level hover -> H_LEVEL
_ab0 = z_rel > 0.05
if _ab0.any():
    _hover = np.median(z_rel[_ab0])
else:
    _hover = H_LEVEL
offset = H_LEVEL - _hover
print(f"  z_rel hover median={_hover:.3f}m -> rotor-height offset={offset:+.3f}m "
      f"(gear/mount), level rotor height={H_LEVEL}m")
h = np.maximum(z_rel + offset, 0.02)
rp = np.array([quat_rp(*od[i, 1:5]) for i in range(len(ot))])
roll, pitch = rp[:, 0], rp[:, 1]

# thrust per rotor interpolated to odom timeline
rpm = np.array(rpm)
Ti_all = C_T * rpm**2
Ti_i = np.array([np.interp(ot, rt, Ti_all[:, k]) for k in range(6)]).T  # (N,6)

# airborne
ab = (pz - z0) > 0.05
t_to = ot[np.argmax(ab)]; t_land = ot[len(ab) - 1 - np.argmax(ab[::-1])]

dMr = np.array([dM_roll(roll[i], h[i], Ti_i[i]) for i in range(len(ot))])
dMp = np.array([dM_pitch(pitch[i], h[i], Ti_i[i]) for i in range(len(ot))])

m = (ot >= t_to + 2) & (ot <= t_land - 2)
print(f"\n{TAG}  airborne {t_to:.1f}-{t_land:.1f}s")
print(f"  GE moment roll : mean={dMr[m].mean():+.4f}  std={dMr[m].std():.4f}  range[{dMr[m].min():+.3f},{dMr[m].max():+.3f}] N·m")
print(f"  GE moment pitch: mean={dMp[m].mean():+.4f}  std={dMp[m].std():.4f}  range[{dMp[m].min():+.3f},{dMp[m].max():+.3f}] N·m")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
ax = axes[0]
ax.plot(ot, np.degrees(roll), 'r', lw=0.8, label='roll'); ax.plot(ot, np.degrees(pitch), 'g', lw=0.8, label='pitch')
ax.plot(ot, (h) * 100, 'b', lw=0.8, alpha=0.6, label='rotor height h [cm]')
ax.axvspan(t_to, t_land, alpha=0.05, color='g'); ax.grid(alpha=0.3); ax.legend(loc='upper right')
ax.set_ylabel('att[deg]/h[cm]'); ax.set_title(f'{TAG} — ground-effect pivot moment (h_level={H_LEVEL}m)')

ax = axes[1]
ax.plot(ot, dMr, 'r', lw=0.9, label='GE moment roll [N·m]')
ax.axvspan(t_to, t_land, alpha=0.05, color='g'); ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('ΔM_GE roll'); ax.grid(alpha=0.3); ax.legend(loc='upper right')

ax = axes[2]
ax.plot(ot, dMp, 'g', lw=0.9, label='GE moment pitch [N·m]')
ax.axvspan(t_to, t_land, alpha=0.05, color='g'); ax.axhline(0, color='k', alpha=0.3, lw=0.7)
ax.set_ylabel('ΔM_GE pitch'); ax.set_xlabel('time [s]'); ax.grid(alpha=0.3); ax.legend(loc='upper right')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_ge_moment.png')
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
