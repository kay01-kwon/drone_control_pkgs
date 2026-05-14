#!/usr/bin/env python3
"""PSD analysis of body angular velocity and motor RPM.
Usage: python3 _psd_vibration.py <bag_subdir> [<date_dir>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
DATE = sys.argv[2] if len(sys.argv) > 2 else '2026_05_14_free_flight'
TAG_OVR = sys.argv[3] if len(sys.argv) > 3 else None
BAG_DIR = os.path.join(_HERE, DATE, BAG)
db = glob.glob(os.path.join(BAG_DIR, '*.db3'))[0]
parts = BAG.split('/')
OUT_DIR = os.path.join(_HERE, DATE, *parts[:-1])
TAG = TAG_OVR if TAG_OVR else parts[-1]


def _align(off, n):
    rel = off - 4
    pad = (-rel) % n
    return off + pad


def parse_odom(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4
    off += slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4
    off += slen2
    off = _align(off, 8)
    off += 3 * 8        # position
    off += 4 * 8        # orientation
    off += 36 * 8       # pose covariance
    vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
    wx, wy, wz = struct.unpack_from('<3d', blob, off); off += 24
    return vx, vy, vz, wx, wy, wz


def parse_rpm(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    return list(struct.unpack_from('<6I', blob, off))


con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("SELECT id,name FROM topics")
tids = {n: i for i, n in cur.fetchall()}
tid = tids['/mavros/local_position/odom']
cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tid}")
t0 = cur.fetchone()[0]


def fetch(topic, parser):
    tid = tids[topic]
    cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tid} ORDER BY timestamp")
    rows = cur.fetchall()
    t = np.array([(ts - t0) * 1e-9 for ts, _ in rows])
    d = np.array([parser(blob) for _, blob in rows])
    return t, d


ot, odom = fetch('/mavros/local_position/odom', parse_odom)
rt, rrpm = fetch('/uav/actual_rpm', parse_rpm)
con.close()

wx = odom[:, 3]; wy = odom[:, 4]; wz = odom[:, 5]
mean_rpm = rrpm.mean(axis=1)

# Detect airborne window (motors > 5500 RPM)
above = mean_rpm > 5500
diff = np.diff(above.astype(int))
rising = np.where(diff == 1)[0]
falling = np.where(diff == -1)[0]
if len(rising) == 0 or len(falling) == 0:
    raise SystemExit('Could not detect airborne window')
t_to = rt[rising[0] + 1]
# pick longest sustained airborne stretch
t_land = rt[falling[-1] + 1] if falling[-1] > rising[0] else rt[-1]
print(f'Airborne window: t={t_to:.2f} → {t_land:.2f} s  ({t_land-t_to:.1f} s)')

# Crop ω to airborne (skip first/last 2 s for transient)
t_lo = t_to + 2.0
t_hi = t_land - 1.0
mO = (ot >= t_lo) & (ot <= t_hi)
mR = (rt >= t_lo) & (rt <= t_hi)

t_o = ot[mO]; wx_a = wx[mO]; wy_a = wy[mO]; wz_a = wz[mO]
t_r = rt[mR]; rpm_a = mean_rpm[mR]

# Resample to uniform 100 Hz for Welch
fs = 100.0
T = t_o[-1] - t_o[0]
N = int(T * fs)
if N < 256:
    raise SystemExit(f'Not enough samples ({N})')
t_u = t_o[0] + np.arange(N) / fs
wx_u = np.interp(t_u, t_o, wx_a)
wy_u = np.interp(t_u, t_o, wy_a)
wz_u = np.interp(t_u, t_o, wz_a)
rpm_u = np.interp(t_u, t_r, rpm_a)

# Remove mean
wx_u -= wx_u.mean(); wy_u -= wy_u.mean(); wz_u -= wz_u.mean()
rpm_u -= rpm_u.mean()

# Welch PSD
nperseg = min(2048, N // 4)
fx, Px = signal.welch(wx_u, fs=fs, nperseg=nperseg)
fy, Py = signal.welch(wy_u, fs=fs, nperseg=nperseg)
fz, Pz = signal.welch(wz_u, fs=fs, nperseg=nperseg)
fr, Pr = signal.welch(rpm_u, fs=fs, nperseg=nperseg)
# Coherence wz vs rpm (yaw most likely coupled with rotor)
fc, Cz = signal.coherence(wz_u, rpm_u, fs=fs, nperseg=nperseg)
_, Cx = signal.coherence(wx_u, rpm_u, fs=fs, nperseg=nperseg)
_, Cy = signal.coherence(wy_u, rpm_u, fs=fs, nperseg=nperseg)

# Identify peaks > 5x median for each ω
def report_peaks(f, P, name):
    med = np.median(P[1:])  # skip DC
    peaks_idx = np.where(P[1:] > 10 * med)[0] + 1
    if len(peaks_idx) == 0:
        print(f'  {name}: no sharp peaks (>10× median)')
        return
    # Cluster adjacent peaks (within 1 Hz)
    peaks_f = f[peaks_idx]
    peaks_p = P[peaks_idx]
    print(f'  {name}: sharp peaks at')
    last_f = -999
    for pf, pp in zip(peaks_f, peaks_p):
        if pf - last_f > 0.5:
            print(f'      f={pf:.2f} Hz   P={pp:.3e}   ratio={pp/med:.1f}×')
            last_f = pf


print(f'\nPSD peaks (vs median, sampled at {fs} Hz, Nyquist {fs/2} Hz):')
report_peaks(fx, Px, 'ω_x (roll rate)')
report_peaks(fy, Py, 'ω_y (pitch rate)')
report_peaks(fz, Pz, 'ω_z (yaw rate)')
report_peaks(fr, Pr, 'mean RPM')

# Rotor blade-pass info (mean RPM in airborne)
rpm_mean = rpm_a.mean()
print(f'\nMean rotor RPM = {rpm_mean:.0f}  =  {rpm_mean/60:.2f} Hz  (above Nyquist {fs/2} Hz, only ~aliasing)')


fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
axes[0].loglog(fx, Px, 'r', label='ω_x')
axes[0].loglog(fy, Py, 'g', label='ω_y')
axes[0].loglog(fz, Pz, 'b', label='ω_z')
axes[0].set_ylabel('PSD ω [(rad/s)²/Hz]'); axes[0].grid(True, alpha=0.3, which='both')
axes[0].legend(); axes[0].set_title(f'{TAG} — Body angular velocity PSD (airborne, {fs} Hz)')

axes[1].loglog(fr, Pr, 'm')
axes[1].set_ylabel('PSD mean RPM [(rpm)²/Hz]'); axes[1].grid(True, alpha=0.3, which='both')
axes[1].set_title('Mean motor RPM PSD')

axes[2].semilogx(fc, Cx, 'r', label='Cxy(ω_x, RPM)')
axes[2].semilogx(fc, Cy, 'g', label='Cxy(ω_y, RPM)')
axes[2].semilogx(fc, Cz, 'b', label='Cxy(ω_z, RPM)')
axes[2].set_ylabel('Coherence'); axes[2].set_ylim(0, 1)
axes[2].grid(True, alpha=0.3, which='both'); axes[2].legend()
axes[2].set_title('Coherence: ω vs mean RPM')

# Linear-freq zoom 0–25 Hz of ω PSDs (where structural vibration usually lives)
mZ = fx <= 25
axes[3].plot(fx[mZ], Px[mZ], 'r', label='ω_x')
axes[3].plot(fy[mZ], Py[mZ], 'g', label='ω_y')
axes[3].plot(fz[mZ], Pz[mZ], 'b', label='ω_z')
axes[3].set_xlabel('Frequency [Hz]'); axes[3].set_ylabel('PSD ω')
axes[3].grid(True, alpha=0.3); axes[3].legend(); axes[3].set_title('Zoom 0–25 Hz (linear y)')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_psd_vibration.png')
plt.savefig(out, dpi=120)
print(f'\nSaved: {out}')

# Print ω std for context
print(f'\nω std (airborne): wx={wx_a.std():.3f}  wy={wy_a.std():.3f}  wz={wz_a.std():.3f}  rad/s')
print(f'RPM std (airborne): {rpm_a.std():.1f}  (mean={rpm_mean:.0f})')
