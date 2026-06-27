#!/usr/bin/env python3
"""Where does the residual omega noise come from, AFTER damper mount + PX4
dynamic notch?

Compares the PSD of body angular velocity at three pipeline stages:
  (1) /mavros/imu/data_raw  -- raw gyro (pre PX4 filtering)
  (2) /mavros/imu/data      -- PX4 notch+LPF applied
  (3) odom twist.angular    -- EKF2 output (what the controller consumes)
overlaid with the rotor fundamental band (from mean actual RPM).

Goal: tell whether residual noise is (a) a narrow peak (damper resonance /
leftover harmonic -> more notching) or (b) broadband (-> RPM-aided EKF).

Usage:
  python3 _omega_residual.py <bag_subdir> [<root>] [<tag>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
ROOT = sys.argv[2] if len(sys.argv) > 2 else '.'
TAG = sys.argv[3] if len(sys.argv) > 3 else BAG.replace('/', '_')
db = glob.glob(os.path.join(_HERE, ROOT, BAG, '*.db3'))[0]
OUT_DIR = os.path.join(_HERE, ROOT, os.path.dirname(BAG))


def _align(o, n):
    return o + (-(o - 4)) % n


def parse_imu(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 8)
    o += 4 * 8 + 9 * 8           # quat + cov
    wx, wy, wz = struct.unpack_from('<3d', b, o)
    return wx, wy, wz


def parse_odom(b):
    o = 4 + 8; sl = struct.unpack_from('<I', b, o)[0]; o += 4 + sl; o = _align(o, 4)
    s2 = struct.unpack_from('<I', b, o)[0]; o += 4 + s2; o = _align(o, 8)
    pz = struct.unpack_from('<d', b, o + 16)[0]
    o += 3 * 8 + 4 * 8 + 36 * 8 + 3 * 8
    wx, wy, wz = struct.unpack_from('<3d', b, o)
    return pz, wx, wy, wz


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


rt_, raw = fetch('/mavros/imu/data_raw', parse_imu)
ft_, fil = fetch('/mavros/imu/data', parse_imu)
ot_, od = fetch('/mavros/local_position/odom', parse_odom)
at_, rpm = fetch('/uav/actual_rpm', parse_rpm)
con.close()

# Airborne window
pz = od[:, 0]
z = pz - pz[ot_ < 5].mean() if (ot_ < 5).any() else pz
ab = z > 0.05
t_to = ot_[np.argmax(ab)]; t_land = ot_[len(ab) - 1 - np.argmax(ab[::-1])]
lo, hi = t_to + 2, t_land - 2

mean_rpm = rpm[(at_ >= lo) & (at_ <= hi)].mean()
f_rotor = mean_rpm / 60.0

# IMU effective rate
fs_imu = 1.0 / np.median(np.diff(rt_))
fs_odom = 1.0 / np.median(np.diff(ot_))


def psd_axis(t, sig, comp, fs):
    m = (t >= lo) & (t <= hi)
    x = sig[m, comp] if sig.ndim > 1 else sig[m]
    n = int((t[m][-1] - t[m][0]) * fs)
    tu = t[m][0] + np.arange(n) / fs
    xu = np.interp(tu, t[m], x) - np.interp(tu, t[m], x).mean()
    f, P = signal.welch(xu, fs=fs, nperseg=min(4096, n // 4))
    return f, P


# odom omega columns: 1,2,3 = wx,wy,wz ; imu columns 0,1,2
fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
labels = ['wx (roll rate)', 'wy (pitch rate)', 'wz (yaw rate)']
print(f"{TAG}  airborne {t_to:.1f}-{t_land:.1f}s")
print(f"IMU rate {fs_imu:.0f}Hz (Nyq {fs_imu/2:.0f}), odom rate {fs_odom:.0f}Hz (Nyq {fs_odom/2:.0f})")
print(f"mean rotor RPM {mean_rpm:.0f} -> fundamental {f_rotor:.1f} Hz")
print("band-RMS [rad/s]  (odom = controller input):")

for k in range(3):
    fr, Pr = psd_axis(rt_, raw, k, fs_imu)
    ff, Pf = psd_axis(ft_, fil, k, fs_imu)
    fo, Po = psd_axis(ot_, od, k + 1, fs_odom)

    ax = axes[k]
    ax.loglog(fr, Pr, 'gray', alpha=0.6, lw=0.8, label='raw IMU')
    ax.loglog(ff, Pf, 'b', alpha=0.8, lw=0.9, label='PX4 notch+LPF')
    ax.loglog(fo, Po, 'r', alpha=0.9, lw=1.0, label='EKF2 (odom, ctrl input)')
    ax.axvline(f_rotor, color='orange', ls='--', alpha=0.6, label=f'rotor fund {f_rotor:.0f}Hz')
    ax.axvline(2 * f_rotor, color='orange', ls=':', alpha=0.4)
    # alias of rotor fundamental into odom Nyquist
    f_alias = abs(((f_rotor + fs_odom / 2) % fs_odom) - fs_odom / 2)
    ax.axvline(f_alias, color='purple', ls='--', alpha=0.5, label=f'rotor alias {f_alias:.0f}Hz')
    ax.set_ylabel(f'PSD {labels[k]}'); ax.grid(alpha=0.3, which='both')
    ax.legend(loc='lower left', fontsize=8)

    # band rms on odom (controller input)
    def bnd(f, P, a, b):
        mm = (f >= a) & (f < b); return np.sqrt(np.trapezoid(P[mm], f[mm]))
    tot = bnd(fo, Po, 0.05, fs_odom / 2)
    print(f"  {labels[k]:16s} tot={tot:.3f}  <2Hz={bnd(fo,Po,0.05,2):.3f}  "
          f"2-10Hz={bnd(fo,Po,2,10):.3f}  >10Hz={bnd(fo,Po,10,fs_odom/2):.3f}")

axes[0].set_title(f'{TAG} — omega PSD through pipeline (raw -> PX4 filter -> EKF2)')
axes[-1].set_xlabel('frequency [Hz]')
plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_omega_residual.png')
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
