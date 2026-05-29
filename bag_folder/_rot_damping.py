#!/usr/bin/env python3
"""Estimate rotational viscous damping b from flight data.

Rigid-body rotational dynamics (roll/pitch, gyroscopic negligible near hover):
    tau_cmd = J*omega_dot + b*omega
=>  (tau_cmd - J*omega_dot) = b*omega       (regress slope = b)

tau_cmd = motor-realised torque = /nmpc/control torque (post-DOB).
omega from odom (LPF'd before differentiating).

Usage:
  python3 _rot_damping.py <bag_subdir> [<date>] [<tag>] [<J>] [<omega_lpf_hz>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
DATE = sys.argv[2] if len(sys.argv) > 2 else '2026_05_29'
TAG_OVR = sys.argv[3] if len(sys.argv) > 3 else None
J = float(sys.argv[4]) if len(sys.argv) > 4 else 0.06
LPF_HZ = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
db = glob.glob(os.path.join(_HERE, DATE, BAG, '*.db3'))[0]
parts = BAG.split('/')
OUT_DIR = os.path.join(_HERE, DATE, *parts[:-1])
TAG = TAG_OVR if TAG_OVR else parts[-1]


def _align(off, n):
    return off + (-(off - 4)) % n


def parse_odom(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    off += 3 * 8 + 4 * 8 + 36 * 8 + 3 * 8   # pos, quat, posecov, lin vel
    wx, wy, wz = struct.unpack_from('<3d', blob, off)
    pz = struct.unpack_from('<d', blob, 4 + 8 + 4 + slen + _pad(slen) + 4 + slen2 + _pad2(slen2) + 16)[0] if False else 0
    return wx, wy, wz


def parse_odom_full(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    pz = struct.unpack_from('<d', blob, off + 16)[0]
    off += 3 * 8 + 4 * 8 + 36 * 8 + 3 * 8
    wx, wy, wz = struct.unpack_from('<3d', blob, off)
    return pz, wx, wy, wz


def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return struct.unpack_from('<6d', blob, off)


con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("SELECT id,name FROM topics")
tids = {n: i for i, n in cur.fetchall()}
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tids['/mavros/local_position/odom']}")
t0 = cur.fetchone()[0]


def fetch(topic, parser):
    tid = tids[topic]
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tid} ORDER BY timestamp")
    rows = cur.fetchall()
    t = np.array([(ts - t0) * 1e-9 for ts, _ in rows])
    d = np.array([parser(b) for _, b in rows])
    return t, d


ot, odom = fetch('/mavros/local_position/odom', parse_odom_full)
ct, ctrl = fetch('/nmpc/control', parse_wrench)
con.close()

pz = odom[:, 0]
z_rel = pz - pz[ot < 5.0].mean() if (ot < 5.0).any() else pz
ab = z_rel > 0.05
t_to = ot[np.argmax(ab)]; t_land = ot[len(ab) - 1 - np.argmax(ab[::-1])]

# Common 100 Hz grid in airborne window
fs = 100.0
tu = np.arange(t_to + 3.0, t_land - 3.0, 1 / fs)
wx = np.interp(tu, ot, odom[:, 1]); wy = np.interp(tu, ot, odom[:, 2])
# torque cmd (roll=Mx, pitch=My)
Mx = np.interp(tu, ct, ctrl[:, 3]); My = np.interp(tu, ct, ctrl[:, 4])

# LPF omega then differentiate
b_lp, a_lp = signal.butter(2, LPF_HZ / (fs / 2))
wx_f = signal.filtfilt(b_lp, a_lp, wx)
wy_f = signal.filtfilt(b_lp, a_lp, wy)
Mx_f = signal.filtfilt(b_lp, a_lp, Mx)
My_f = signal.filtfilt(b_lp, a_lp, My)
wxd = np.gradient(wx_f, 1 / fs)
wyd = np.gradient(wy_f, 1 / fs)


def fit_b(tau, wd, w):
    """tau - J*wd = b*w  → slope b (and intercept for offset)."""
    y = tau - J * wd
    A = np.vstack([w, np.ones_like(w)]).T
    (b, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = b * w + c
    R2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, c, R2


bx, cx, R2x = fit_b(Mx_f, wxd, wx_f)
by, cy, R2y = fit_b(My_f, wyd, wy_f)

print(f'{TAG}  airborne {t_to:.1f}-{t_land:.1f}s,  J={J}, omega LPF={LPF_HZ}Hz')
print(f'\nRotational damping  tau = J*omega_dot + b*omega:')
print(f'  roll  (Mx vs wx):  b = {bx:+.4f} N·m/(rad/s)   R²={R2x:.3f}')
print(f'  pitch (My vs wy):  b = {by:+.4f} N·m/(rad/s)   R²={R2y:.3f}')
print(f'\n  omega std: wx={wx_f.std():.3f}  wy={wy_f.std():.3f} rad/s')
print(f'  For reference: sim used b=0.1 (≈flight) and b=0.3')

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, w, tau, wd, b, c, R2, name in [
        (axes[0], wx_f, Mx_f, wxd, bx, cx, R2x, 'roll (Mx, wx)'),
        (axes[1], wy_f, My_f, wyd, by, cy, R2y, 'pitch (My, wy)')]:
    y = tau - J * wd
    ax.scatter(w, y, s=4, alpha=0.25, c='steelblue')
    xs = np.linspace(w.min(), w.max(), 50)
    ax.plot(xs, b * xs + c, 'r-', lw=2, label=f'b={b:+.3f} N·m/(rad/s)\nR²={R2:.3f}')
    ax.plot(xs, 0.1 * xs + c, 'g--', lw=1.2, alpha=0.7, label='sim b=0.1')
    ax.plot(xs, 0.3 * xs + c, 'm:', lw=1.2, alpha=0.7, label='sim b=0.3')
    ax.set_xlabel(f'omega [rad/s]'); ax.set_ylabel('tau - J*omega_dot [N·m]')
    ax.grid(alpha=0.3); ax.legend(); ax.set_title(f'{name}')
fig.suptitle(f'{TAG} — rotational damping estimate (slope = b)', y=1.02)
plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_rot_damping.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f'\nSaved: {out}')
