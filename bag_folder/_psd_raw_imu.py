#!/usr/bin/env python3
"""Raw IMU (un-EKF2-filtered) PSD analysis to detect structural / rotor vibration
that gets aliased into the 100 Hz odom ω stream.
Usage: python3 _psd_raw_imu.py <bag_subdir> [<date_dir>]
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
BAG_DIR = os.path.join(_HERE, DATE, BAG)
db = glob.glob(os.path.join(BAG_DIR, '*.db3'))[0]
parts = BAG.split('/')
OUT_DIR = os.path.join(_HERE, DATE, *parts[:-1])
TAG = parts[-1]


def _align(off, n):
    rel = off - 4
    pad = (-rel) % n
    return off + pad


def parse_imu(blob):
    """sensor_msgs/Imu after stamp+frame: quaternion(4 d), cov[9], ang_vel(3 d), cov[9], lin_acc(3 d), cov[9]"""
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    off += 4 * 8           # orientation quaternion
    off += 9 * 8           # orientation cov
    wx, wy, wz = struct.unpack_from('<3d', blob, off); off += 24
    off += 9 * 8           # ang_vel cov
    ax, ay, az = struct.unpack_from('<3d', blob, off); off += 24
    return wx, wy, wz, ax, ay, az


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


it, imu = fetch('/mavros/imu/data_raw', parse_imu)
rt, rrpm = fetch('/uav/actual_rpm', parse_rpm)
con.close()

wx = imu[:, 0]; wy = imu[:, 1]; wz = imu[:, 2]
ax = imu[:, 3]; ay = imu[:, 4]; az = imu[:, 5]

mean_rpm = rrpm.mean(axis=1)
above = mean_rpm > 5500
diff = np.diff(above.astype(int))
rising = np.where(diff == 1)[0]
falling = np.where(diff == -1)[0]
t_to = rt[rising[0] + 1]
t_land = rt[falling[-1] + 1] if len(falling) and falling[-1] > rising[0] else rt[-1]
print(f'Airborne window: t={t_to:.2f} → {t_land:.2f} s  ({t_land-t_to:.1f} s)')

# IMU effective sample rate
dt_med = np.median(np.diff(it))
fs_imu = 1.0 / dt_med
print(f'IMU effective rate: {fs_imu:.1f} Hz (Nyquist {fs_imu/2:.1f} Hz)')

# Crop to airborne (skip 2 s margin)
mI = (it >= t_to + 2.0) & (it <= t_land - 1.0)
t_a = it[mI]
wx_a = wx[mI]; wy_a = wy[mI]; wz_a = wz[mI]
ax_a = ax[mI]; ay_a = ay[mI]; az_a = az[mI]

# Resample to uniform fs_imu
fs = round(fs_imu)
N = int((t_a[-1] - t_a[0]) * fs)
t_u = t_a[0] + np.arange(N) / fs
wx_u = np.interp(t_u, t_a, wx_a) - wx_a.mean()
wy_u = np.interp(t_u, t_a, wy_a) - wy_a.mean()
wz_u = np.interp(t_u, t_a, wz_a) - wz_a.mean()
ax_u = np.interp(t_u, t_a, ax_a) - ax_a.mean()
ay_u = np.interp(t_u, t_a, ay_a) - ay_a.mean()
az_u = np.interp(t_u, t_a, az_a) - az_a.mean()

nperseg = min(4096, N // 4)
fw, Pwx = signal.welch(wx_u, fs=fs, nperseg=nperseg)
_,  Pwy = signal.welch(wy_u, fs=fs, nperseg=nperseg)
_,  Pwz = signal.welch(wz_u, fs=fs, nperseg=nperseg)
fa, Pax = signal.welch(ax_u, fs=fs, nperseg=nperseg)
_,  Pay = signal.welch(ay_u, fs=fs, nperseg=nperseg)
_,  Paz = signal.welch(az_u, fs=fs, nperseg=nperseg)

# Power in bands
def band(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return np.trapezoid(P[m], f[m])


print(f'\nω band power [(rad/s)²]:')
for name, P in [('wx', Pwx), ('wy', Pwy), ('wz', Pwz)]:
    p_lo  = band(fw, P, 0, 2)
    p_mid = band(fw, P, 2, 20)
    p_hi  = band(fw, P, 20, fs/2)
    p_tot = p_lo + p_mid + p_hi
    print(f'  {name}: <2Hz={p_lo:.5f}  2–20Hz={p_mid:.5f}  >20Hz={p_hi:.5f}   (tot={p_tot:.5f}, hi%={100*p_hi/p_tot:.1f})')

print(f'\nacc band power [(m/s²)²]:')
for name, P in [('ax', Pax), ('ay', Pay), ('az', Paz)]:
    p_lo  = band(fa, P, 0, 2)
    p_mid = band(fa, P, 2, 20)
    p_hi  = band(fa, P, 20, fs/2)
    p_tot = p_lo + p_mid + p_hi
    print(f'  {name}: <2Hz={p_lo:.4f}  2–20Hz={p_mid:.4f}  >20Hz={p_hi:.4f}   (tot={p_tot:.4f}, hi%={100*p_hi/p_tot:.1f})')

# Look for peaks above 10 Hz (structural / rotor)
def find_peaks_above(f, P, fmin=10.0, ratio=10.0):
    m = f >= fmin
    if not m.any(): return []
    med = np.median(P[m])
    idx = np.where((P > ratio * med) & m)[0]
    out = []
    last = -10
    for i in idx:
        if f[i] - last > 1.0:
            out.append((f[i], P[i], P[i] / med))
            last = f[i]
    return out


print('\nω peaks >10 Hz (sharp, >10× median):')
for name, P in [('wx', Pwx), ('wy', Pwy), ('wz', Pwz)]:
    peaks = find_peaks_above(fw, P, fmin=10.0)
    if not peaks:
        print(f'  {name}: none')
    else:
        for f, p, r in peaks[:6]:
            print(f'  {name}: f={f:6.1f} Hz  P={p:.3e}  {r:5.1f}× median')

print('\nacc peaks >10 Hz:')
for name, P in [('ax', Pax), ('ay', Pay), ('az', Paz)]:
    peaks = find_peaks_above(fa, P, fmin=10.0)
    if not peaks:
        print(f'  {name}: none')
    else:
        for f, p, r in peaks[:6]:
            print(f'  {name}: f={f:6.1f} Hz  P={p:.3e}  {r:5.1f}× median')

rpm_mean = mean_rpm[(rt >= t_to + 2.0) & (rt <= t_land - 1.0)].mean()
print(f'\nMean rotor RPM: {rpm_mean:.0f}  =  {rpm_mean/60:.2f} Hz  (blade-pass for 2-blade: {2*rpm_mean/60:.1f} Hz)')

fig, axes = plt.subplots(2, 1, figsize=(11, 9))
axes[0].loglog(fw, Pwx, 'r', alpha=0.8, label='ω_x')
axes[0].loglog(fw, Pwy, 'g', alpha=0.8, label='ω_y')
axes[0].loglog(fw, Pwz, 'b', alpha=0.8, label='ω_z')
axes[0].axvline(rpm_mean / 60, color='k', alpha=0.3, ls='--', label=f'rotor fund {rpm_mean/60:.0f} Hz')
axes[0].axvline(2 * rpm_mean / 60, color='k', alpha=0.3, ls=':', label='2× blade-pass')
axes[0].set_ylabel('PSD ω [(rad/s)²/Hz]'); axes[0].grid(True, alpha=0.3, which='both')
axes[0].legend(); axes[0].set_title(f'{TAG} — Raw IMU ω PSD (airborne, fs={fs} Hz)')

axes[1].loglog(fa, Pax, 'r', alpha=0.8, label='a_x')
axes[1].loglog(fa, Pay, 'g', alpha=0.8, label='a_y')
axes[1].loglog(fa, Paz, 'b', alpha=0.8, label='a_z')
axes[1].axvline(rpm_mean / 60, color='k', alpha=0.3, ls='--')
axes[1].axvline(2 * rpm_mean / 60, color='k', alpha=0.3, ls=':')
axes[1].set_xlabel('Frequency [Hz]'); axes[1].set_ylabel('PSD a [(m/s²)²/Hz]')
axes[1].grid(True, alpha=0.3, which='both'); axes[1].legend()
axes[1].set_title('Raw IMU linear accel PSD')

plt.tight_layout()
out = os.path.join(OUT_DIR, f'{TAG}_psd_raw_imu.png')
plt.savefig(out, dpi=120)
print(f'\nSaved: {out}')
