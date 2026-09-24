#!/usr/bin/env python3
"""Rotor-model identification from flight logs (reproducible companion to
docs/rotor_model_identification.pdf).

Identifies / checks:
  (A) thrust coefficient C_T from the steady-hover force balance
        sum_i C_T w_i^2 * cos(tilt) = m g   ->   C_T = m g / < sum_i w_i^2 cos(tilt) >
  (B) same, keeping the measured vertical acceleration  m (az + g)
  (M) motor/ESC command -> actual-RPM response by a Welch ETFE
        H(f) = S_xy(f) / S_xx(f),  x = cmd_raw*9800/8191,  y = actual_rpm
      reporting DC gain, -3 dB frequency and the equivalent delay from the
      low-frequency phase slope.

Usage:  python3 _rotor_model_id.py <bag_subdir> <mass_kg>
        e.g.  python3 _rotor_model_id.py 01/hgdo/wo_ff 3.050
"""
import os, sys, sqlite3, struct, glob
import numpy as np
from scipy import signal

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
M = float(sys.argv[2])
G = 9.81
MAX_BIT, MAX_RPM = 8191, 9800          # cmd_converter.py


def _al(o, n):
    return o + (-(o - 4)) % n


def p_rpm(b):
    o = 4 + 8; s = struct.unpack_from('<I', b, o)[0]; o += 4 + s; o = _al(o, 4)
    return np.array(struct.unpack_from('<6I', b, o), float)


def p_cmd(b):
    o = 4 + 8; s = struct.unpack_from('<I', b, o)[0]; o += 4 + s; o = _al(o, 2)
    return np.array(struct.unpack_from('<6H', b, o), float)


def p_odom(b):
    o = 4 + 8; s = struct.unpack_from('<I', b, o)[0]; o += 4 + s; o = _al(o, 4)
    s2 = struct.unpack_from('<I', b, o)[0]; o += 4 + s2; o = _al(o, 8)
    pz = struct.unpack_from('<d', b, o + 16)[0]
    qx, qy, qz, qw = struct.unpack_from('<4d', b, o + 24)
    vx, vy, vz = struct.unpack_from('<3d', b, o + 24 + 32 + 36 * 8)
    return pz, qw, qx, qy, qz, vx, vy, vz


db = glob.glob(os.path.join(_HERE, BAG, '*.db3'))[0]
con = sqlite3.connect(db); cur = con.cursor()
cur.execute("SELECT id,name FROM topics"); tid = {n: i for i, n in cur.fetchall()}
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tid['/mavros/local_position/odom']}")
t0 = cur.fetchone()[0]


def fetch(topic, parser):
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tid[topic]} ORDER BY timestamp")
    r = cur.fetchall()
    return np.array([(t - t0) * 1e-9 for t, _ in r]), np.array([parser(b) for _, b in r])


rt, rpm = fetch('/uav/actual_rpm', p_rpm)
ct, craw = fetch('/uav/cmd_raw', p_cmd)
ot, od = fetch('/mavros/local_position/odom', p_odom)
con.close()

# ---- steady-hover window from mocap altitude (independent of the controller) ----
pz = od[:, 0]; z0 = pz[ot < 5].mean(); zr = pz - z0
ab = zr > 0.05
i0 = int(np.argmax(ab)); i1 = len(ab) - 1 - int(np.argmax(ab[::-1]))
t_lo, t_hi = ot[i0] + 3.0, ot[i1] - 3.0          # trim take-off / landing transients
wr = (rt > t_lo) & (rt < t_hi)

# ---- tilt: R33 = cos(total tilt) ----
qw, qx, qy, qz = od[:, 1], od[:, 2], od[:, 3], od[:, 4]
c33 = 1 - 2 * (qx**2 + qy**2)
c33i = np.interp(rt, ot, c33)

S = (rpm**2).sum(1)                              # sum_i w_i^2  [rpm^2]

# (A) static balance, with and without tilt correction
CT_A0 = M * G / S[wr].mean()
CT_A = M * G / (S[wr] * c33i[wr]).mean()

# (B) with measured vertical acceleration (world-frame vz -> az, 2 Hz LPF)
vb = od[:, 5:8]
vz_w = (2 * (qx * qz - qy * qw)) * vb[:, 0] + (2 * (qy * qz + qx * qw)) * vb[:, 1] + c33 * vb[:, 2]
fs_o = 1 / np.median(np.diff(ot))
b, a = signal.butter(2, 2.0 / (fs_o / 2))
az = np.gradient(signal.filtfilt(b, a, vz_w), 1 / fs_o)
azi = np.interp(rt, ot, az)
CT_B = (M * (azi[wr] + G)).mean() / (S[wr] * c33i[wr]).mean()

# (M) motor command -> actual rpm ETFE, rotor-averaged
cmd_rpm = craw * MAX_RPM / MAX_BIT
cmi = np.array([np.interp(rt, ct, cmd_rpm[:, k]) for k in range(6)]).T
fs_r = 1 / np.median(np.diff(rt))
Hs = []
for k in range(6):
    x = cmi[wr, k] - cmi[wr, k].mean(); y = rpm[wr, k] - rpm[wr, k].mean()
    f, Pxy = signal.csd(x, y, fs=fs_r, nperseg=512)
    _, Pxx = signal.welch(x, fs=fs_r, nperseg=512)
    Hs.append(Pxy / Pxx)
H = np.mean(Hs, 0); mag = np.abs(H); ph = np.unwrap(np.angle(H))
g0 = mag[(f > 0.2) & (f < 0.8)].mean()
f3 = f[int(np.argmax(mag < g0 / np.sqrt(2)))]
sel = (f > 0.3) & (f < 2.0)
tau = -np.polyfit(2 * np.pi * f[sel], ph[sel], 1)[0]

print(f"{BAG}  m = {M} kg   hover window {t_lo:.1f}-{t_hi:.1f} s  ({wr.sum()} samples @ {fs_r:.0f} Hz)")
print(f"  mean rotor speed        : {np.sqrt(S[wr].mean()/6):.0f} rpm   mean tilt cos = {c33i[wr].mean():.4f}")
print(f"  (A)  C_T static         : {CT_A0:.4e}  N/rpm^2   (no tilt correction)")
print(f"  (A') C_T static, tilt   : {CT_A:.4e}  N/rpm^2")
print(f"  (B)  C_T with az        : {CT_B:.4e}  N/rpm^2")
print(f"  (M)  motor cmd->rpm     : DC gain {g0:.3f}, f_-3dB {f3:.2f} Hz, eq. delay {tau*1e3:.0f} ms")
