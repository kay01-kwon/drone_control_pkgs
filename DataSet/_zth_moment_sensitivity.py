#!/usr/bin/env python3
"""Chattering sensitivity of the moment-mode altitude threshold z_th
(pd_nmpc_att_with_dob.py lines 539-556).

Reproduces the exact mode logic offline from recorded thrust and z, then
sweeps z_th and counts how many times the compensation moment source
switches between M_ff (ground) and DOB (flight) — the chattering metric
the reviewer asks about.  Also compares against a hysteresis variant.

Mode logic reproduced:
    airborne   = thrust >= W
    was_airborne latched; cleared when in_flight becomes False
    in_flight  = airborne OR (was_airborne AND z > z_th)
    if in_flight            -> DOB moment
    elif z < z_th (& ff)    -> M_ff   (ground reaction FF)
    else                    -> DOB moment (yaw zeroed)
  => the M_ff <-> DOB switch is the chattering of interest.

Usage:
  python3 _zth_moment_sensitivity.py <bag_subdir> [<tag>] [<m_mass>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else BAG.replace('/', '_')
M = float(sys.argv[3]) if len(sys.argv) > 3 else (3.322 if BAG.startswith('02') else 3.0)
G = 9.81
W = M * G
C_T = 1.3175e-07
db = glob.glob(os.path.join(_HERE, BAG, '*.db3'))[0]
OUT_DIR = os.path.join(_HERE, os.path.dirname(BAG))


def _align(o, n):
    return o + (-(o - 4)) % n


def parse_oz(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 4)
    s2 = struct.unpack_from('<I', b, o)[0]; o += 4 + s2; o = _align(o, 8)
    return struct.unpack_from('<d', b, o + 16)[0]


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
    return np.array([(t - t0) * 1e-9 for t, _ in r]), [parser(b) for _, b in r]


ot, oz = fetch('/mavros/local_position/odom', parse_oz)
rt, rpm = fetch('/uav/actual_rpm', parse_rpm)
con.close()
z = np.array(oz)
z = z - z[ot < 5].mean() if (ot < 5).any() else z
thrust_r = np.array([C_T * (r**2).sum() for r in rpm])
thrust = np.interp(ot, rt, thrust_r)   # align thrust to odom timeline

# ground z noise
gm = ot < (ot[np.argmax(z > 0.05)] - 1) if (z > 0.05).any() else (ot < 5)
z_noise = z[gm].std() if gm.sum() > 10 else 0.0


def run_single(z_th):
    """Exact logic; returns number of M_ff<->DOB source switches."""
    was_airborne = False
    prev_src = None
    switches = 0
    for zz, th in zip(z, thrust):
        airborne = th >= W
        if airborne:
            was_airborne = True
        in_flight = airborne or (was_airborne and zz > z_th)
        if in_flight:
            src = 'DOB'
        else:
            was_airborne = False
            src = 'Mff' if zz < z_th else 'DOB'
        if prev_src is not None and src != prev_src:
            switches += 1
        prev_src = src
    return switches


def run_hyst(z_th, band, dwell_n):
    """Hysteresis on the z gate: enter DOB(flight) at z_th+band, return to
    ground at z_th-band, with dwell; count M_ff<->DOB switches."""
    was_airborne = False
    inflight_z = False; ec = 0; xc = 0
    prev_src = None; switches = 0
    enter = z_th + band; exit_ = max(z_th - band, 0.0)
    for zz, th in zip(z, thrust):
        airborne = th >= W
        if airborne:
            was_airborne = True
        ec = ec + 1 if zz > enter else 0
        xc = xc + 1 if zz < exit_ else 0
        if not inflight_z and ec > dwell_n:
            inflight_z = True
        if inflight_z and xc > dwell_n:
            inflight_z = False
        in_flight = airborne or (was_airborne and inflight_z)
        if in_flight:
            src = 'DOB'
        else:
            was_airborne = False
            src = 'Mff' if not inflight_z else 'DOB'
        if prev_src is not None and src != prev_src:
            switches += 1
        prev_src = src
    return switches


fs = 1 / np.median(np.diff(ot))
zth_sweep = np.arange(0.004, 0.041, 0.002)
single = [run_single(z_) for z_ in zth_sweep]
hyst = [run_hyst(z_, 0.008, int(0.3 * fs)) for z_ in zth_sweep]

print(f"{TAG}  (W={W:.1f}N, ground z noise={z_noise*1000:.2f}mm)")
print(f"  z_th=10mm: single switches={run_single(0.010)}, hyst switches={run_hyst(0.010,0.008,int(0.3*fs))}")
print(f"  ideal = 2 (one takeoff Mff->DOB, one landing DOB->Mff)")
print(f"\n  z_th[mm] single hyst")
for zt, s_, h_ in zip(zth_sweep, single, hyst):
    print(f"    {zt*1000:5.1f}  {s_:4d}  {h_:3d}")

fig, axes = plt.subplots(2, 1, figsize=(12, 9))
ax = axes[0]
ax.plot(zth_sweep * 1000, single, 'r-o', ms=4, label='single threshold (current)')
ax.plot(zth_sweep * 1000, hyst, 'b-s', ms=4, label='hysteresis (±8mm, 0.3s dwell)')
ax.axhline(2, color='k', ls='--', alpha=0.5, label='ideal (2 switches)')
ax.axvline(10, color='g', ls=':', alpha=0.7, label='z_th=10mm (paper)')
if z_noise > 0:
    ax.axvspan(0, z_noise * 1000, color='r', alpha=0.08, label=f'ground noise band ({z_noise*1000:.1f}mm)')
ax.set_xlabel('moment threshold z_th [mm]'); ax.set_ylabel('M_ff<->DOB switches')
ax.set_yscale('symlog'); ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
ax.set_title(f'{TAG} — moment-mode threshold chattering sensitivity')

ax = axes[1]
ax.plot(ot, z * 1000, 'b', lw=0.8, label='z rel [mm]')
ax.axhline(10, color='g', ls='--', alpha=0.7, label='z_th=10mm')
ax.set_xlabel('time [s]'); ax.set_ylabel('z [mm]'); ax.set_ylim(-30, 100)
ax.grid(alpha=0.3); ax.legend(loc='upper right')
ax.set_title('z(t)')
plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_zth_moment_sens.png')
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
