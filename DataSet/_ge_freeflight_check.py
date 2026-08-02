#!/usr/bin/env python3
"""Feasibility check: can the GE model be validated in FREE FLIGHT (ascent)?

Key correction vs the pivot derivation: in free flight there is no pivot, so
the parallel-axis transfer  [sum (gfull-1)Ti] * lp  does NOT apply.  The
free-flight GE moment about the CoM is only

  dM_free = sum_i b'_i (grot_i - 1) Ti   +   F_GE * r_cg      (fountain acts at
            (interference asymmetry)          geometric centre; arm = CoM offset ~1 cm)

with b'_i arms about the CoM and F_GE = sum_i (gfull_i - 1) Ti.

This script quantifies, from a real bag:
  (S)  predicted free-flight GE moment during ascent  (signal)
  (N)  empirically measured residual-moment noise floor in hover:
         std( J*wdot - M_act(actual rpm) )  (mean-subtracted)
  (G)  gyroscopic moment estimate  Jr * |sum(+-Omega_i)| * |omega|
  (F)  predicted GE force decay during ascent vs measured vertical-force
       residual  m*(vzdot+g) - T_z  (both referenced to hover mean)

Verdict: moment-level validation feasible iff S >> N.  Force-level check is
plotted directly (does the measured residual track the model decay?).

Usage: python3 _ge_freeflight_check.py <bag_subdir> <tag> <mass>
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sig

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else BAG.replace('/', '_')
MASS = float(sys.argv[3]) if len(sys.argv) > 3 else (3.322 if BAG.startswith('02') else 3.05)
H_LEVEL = 0.315
db = glob.glob(os.path.join(_HERE, BAG, '*.db3'))[0]
OUT_DIR = os.path.join(_HERE, os.path.dirname(BAG))

# platform
R = 0.127; d = 0.265; c = 0.22; lp = 0.140; Jk = 2.2; rd = 2 * d - c
C_T = 1.3175e-07; G = 9.81
Jxx = 0.06; Jyy = 0.06
Jr = 9e-5                     # prop(15g rod 2R) + motor bell rotor inertia est.
# CoM offset from identified moment_offset [W*x_off, W*y_off] (pivot-free)
W = MASS * G
x_cg = 0.3032 / W; y_cg = -0.1474 / W          # ~ +9mm, -4.5mm

cs, sn = np.cos(np.pi / 3), np.sin(np.pi / 3)
ly = d * np.array([cs, 1, cs, -cs, -1, -cs])
lx = d * np.array([sn, 0, -sn, -sn, 0, sn])
km_sign = np.array([-1, 1, -1, 1, -1, 1])
xy = np.column_stack([lx, ly])
dij = np.sqrt(((xy[:, None, :] - xy[None, :, :])**2).sum(-1))


def S9rot(z):
    return (R**2 * z / (d**2 / 4 + 4 * z**2)**1.5
            + R**2 / 2 * z / (21 * d**2 / 4 + 4 * z**2)**1.5
            + R**2 * z / (13 * d**2 / 4 + 4 * z**2)**1.5
            + R**2 / 2 * z / (7 * d**2 / 4 + 4 * z**2)**1.5)


def dvi_level(zc):
    s = 0.0
    for j in range(6):
        s += 2 * zc / (dij[0, j]**2 + 4 * zc**2)**1.5
    return R**2 / 4 * s


def ge_gains(roll, pitch, h):
    """Per-rotor gains at attitude (roll,pitch) and rotor-plane height h."""
    # rotor heights: z_i = h + tilt-induced offset (small angles)
    zi = h + ly * np.sin(roll) - lx * np.sin(pitch)
    zi = np.maximum(zi, 0.02)
    zc = max(h, 0.02)
    dvi = np.zeros(6)
    for i in range(6):
        zz = zi[i] + zi
        dvi[i] = (R**2 / 4) * np.sum(zz / (dij[i]**2 + zz**2)**1.5)
    k = S9rot(zc) / dvi_level(zc)
    dvf = 2 * R**2 * Jk * zc / (rd**2 + 4 * zc**2)**1.5
    grot = 1.0 / (1.0 - k * dvi)
    gfull = 1.0 / (1.0 - k * dvi - dvf)
    return grot, gfull


def _align(o, n):
    return o + (-(o - 4)) % n


def parse_odom(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 4)
    s2 = struct.unpack_from('<I', b, o)[0]; o += 4 + s2; o = _align(o, 8)
    px, py, pz = struct.unpack_from('<3d', b, o)
    qx, qy, qz, qw = struct.unpack_from('<4d', b, o + 24)
    off = o + 24 + 32 + 36 * 8
    vx, vy, vz, wx, wy, wz = struct.unpack_from('<6d', b, off)
    return pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz


def parse_rpm(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 4)
    return np.array(struct.unpack_from('<6I', b, o), dtype=float)


con = sqlite3.connect(db); cur = con.cursor()
cur.execute("SELECT id,name FROM topics"); tids = {n: i for i, n in cur.fetchall()}
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tids['/mavros/local_position/odom']}")
t0 = cur.fetchone()[0]


def fetch(topic, parser):
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tids[topic]} ORDER BY timestamp")
    r = cur.fetchall()
    return np.array([(t - t0) * 1e-9 for t, _ in r]), np.array([parser(b) for _, b in r])


ot, od = fetch('/mavros/local_position/odom', parse_odom)
rt, rpm = fetch('/uav/actual_rpm', parse_rpm)
con.close()

pz = od[:, 0]; qw, qx, qy, qz_ = od[:, 1], od[:, 2], od[:, 3], od[:, 4]
vb = od[:, 5:8]; wb = od[:, 8:11]
roll = np.arctan2(2 * (qw * qx + qy * qz_), 1 - 2 * (qx**2 + qy**2))
pitch = np.arcsin(np.clip(2 * (qw * qy - qz_ * qx), -1, 1))

z0 = pz[ot < 5].mean(); z_rel = pz - z0
ab = z_rel > 0.05
t_to = ot[np.argmax(ab)]; t_land = ot[len(ab) - 1 - np.argmax(ab[::-1])]
hov = (ot >= t_to + 6) & (ot <= t_land - 3)
# rotor-height offset so hover rotor height = H_LEVEL
offset = H_LEVEL - np.median(z_rel[hov])
h_rot = np.maximum(z_rel + offset, 0.02)

# world vz
vz_w = np.empty(len(ot))
for i in range(len(ot)):
    qwi, qxi, qyi, qzi = qw[i], qx[i], qy[i], qz_[i]
    # third row of R(q) dot v_body
    vz_w[i] = (2 * (qxi * qzi - qyi * qwi) * vb[i, 0]
               + 2 * (qyi * qzi + qxi * qwi) * vb[i, 1]
               + (1 - 2 * (qxi**2 + qyi**2)) * vb[i, 2])

fs = 1 / np.median(np.diff(ot))
b2, a2 = sig.butter(2, 2.0 / (fs / 2))
wx_f = sig.filtfilt(b2, a2, wb[:, 0]); wy_f = sig.filtfilt(b2, a2, wb[:, 1])
vz_f = sig.filtfilt(b2, a2, vz_w)
wdx = np.gradient(wx_f, 1 / fs); wdy = np.gradient(wy_f, 1 / fs)
az = np.gradient(vz_f, 1 / fs)

# thrust per rotor on odom timeline
Ti = np.array([np.interp(ot, rt, C_T * rpm[:, k]**2) for k in range(6)]).T
Ttot = Ti.sum(1)
# actual-rpm body moment
Mx_act = (Ti * ly).sum(1)
My_act = (Ti * (-lx)).sum(1)

# ---- (N) hover residual-moment noise floor (mean-subtracted) ----
rx = Jxx * wdx - Mx_act; ry = Jyy * wdy - My_act
Nx = rx[hov].std(); Ny = ry[hov].std()

# ---- (G) gyroscopic estimate ----
Om = np.interp(ot, rt, (rpm * km_sign).sum(1)) * 2 * np.pi / 60  # signed sum
Mgyro = Jr * np.abs(Om[hov]).mean() * np.abs(np.hypot(wx_f, wy_f)[hov]).max()

# ---- (S,F) model prediction over ascent+hover ----
win = (ot >= t_to - 0.5) & (ot <= min(t_to + 8, t_land))
idx = np.where(win | hov)[0][::2]     # subsample for speed
dM_roll = np.full(len(ot), np.nan); dM_pitch = np.full(len(ot), np.nan)
F_GE = np.full(len(ot), np.nan)
for i in idx:
    grot, gfull = ge_gains(roll[i], pitch[i], h_rot[i])
    Fge = np.sum((gfull - 1) * Ti[i])
    F_GE[i] = Fge
    # arms about CoM
    dM_roll[i] = np.sum((ly - y_cg) * (grot - 1) * Ti[i]) + Fge * (-y_cg)
    dM_pitch[i] = np.sum(-(lx - x_cg) * (grot - 1) * Ti[i]) + Fge * (x_cg)

# signal = change of GE moment from hover level during ascent
mask_a = win & ~np.isnan(dM_roll)
mask_h = hov & ~np.isnan(dM_roll)
S_roll = np.nanmax(np.abs(dM_roll[mask_a] - np.nanmean(dM_roll[mask_h])))
S_pitch = np.nanmax(np.abs(dM_pitch[mask_a] - np.nanmean(dM_pitch[mask_h])))

# force: measured residual vs model, hover-referenced
tilt = np.cos(roll) * np.cos(pitch)
F_res = MASS * (az + G) - Ttot * tilt
F_res_ref = F_res - F_res[hov].mean()
F_GE_ref = F_GE - np.nanmean(F_GE[mask_h])
NF = F_res[hov].std()
SF = np.nanmax(np.abs(F_GE_ref[mask_a]))

print(f"{TAG}  (m={MASS}kg, hover rotor height={H_LEVEL}m, gear offset={offset:+.3f}m)")
print(f"  [S] free-flight GE moment signal (ascent vs hover): roll {S_roll*1000:.1f}  pitch {S_pitch*1000:.1f}  mN·m")
print(f"      absolute GE moment in hover: roll {np.nanmean(dM_roll[mask_h])*1000:+.1f}  pitch {np.nanmean(dM_pitch[mask_h])*1000:+.1f} mN·m")
print(f"  [N] hover residual-moment noise floor: roll {Nx*1000:.1f}  pitch {Ny*1000:.1f}  mN·m")
print(f"  [G] gyroscopic moment estimate: {Mgyro*1000:.2f} mN·m  (Jr={Jr:g})")
print(f"  --> moment S/N: roll {S_roll/Nx:.2f}, pitch {S_pitch/Ny:.2f}")
print(f"  [F] GE force signal (ascent-hover): {SF:.2f} N   vs force noise floor {NF:.2f} N  -> S/N {SF/NF:.2f}")

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
tw = (ot >= t_to - 1) & (ot <= min(t_to + 8, t_land))
ax = axes[0]
ax.plot(ot[tw], h_rot[tw] * 100, 'b', label='rotor height h [cm]')
ax.plot(ot[tw], np.degrees(roll[tw]), 'r', lw=0.8, label='roll [deg]')
ax.plot(ot[tw], np.degrees(pitch[tw]), 'g', lw=0.8, label='pitch [deg]')
ax.axvline(t_to, color='k', ls=':', alpha=0.5)
ax.set_ylabel('h[cm] / att[deg]'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title(f'{TAG} — free-flight GE feasibility (ascent)')

ax = axes[1]
ax.plot(ot[tw], np.where(tw, dM_roll, np.nan)[tw] * 1000, 'r.-', ms=2, lw=0.7, label='pred GE roll [mN·m]')
ax.plot(ot[tw], np.where(tw, dM_pitch, np.nan)[tw] * 1000, 'g.-', ms=2, lw=0.7, label='pred GE pitch [mN·m]')
ax.axhspan(-Nx * 1000, Nx * 1000, color='gray', alpha=0.25, label=f'residual noise ±1σ ({Nx*1000:.0f} mN·m)')
ax.axvline(t_to, color='k', ls=':', alpha=0.5)
ax.set_ylabel('GE moment [mN·m]'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[2]
ax.plot(ot[tw], F_res_ref[tw], 'k', lw=0.8, alpha=0.7, label='measured m(az+g)−T (hover-ref)')
ax.plot(ot[tw], np.where(tw, F_GE_ref, np.nan)[tw], 'm.-', ms=2, lw=0.9, label='model GE force (hover-ref)')
ax.axhspan(-NF, NF, color='gray', alpha=0.2, label=f'force noise ±1σ ({NF:.1f} N)')
ax.axvline(t_to, color='k', ls=':', alpha=0.5)
ax.set_ylabel('ΔF vertical [N]'); ax.set_xlabel('time [s]'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_ge_freeflight.png')
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
