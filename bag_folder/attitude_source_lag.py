#!/usr/bin/env python3
"""Compare attitude sources (mocap, IMU AHRS, EKF2 odom) to quantify
relative latency.

  /S550/pose                  ← mocap pose, recorded on laptop (ground truth)
  /mavros/imu/data            ← IMU AHRS quaternion (fast complementary filter)
  /mavros/local_position/odom ← PX4 EKF2 fused state (suspect: slow)

For each source we compute roll/pitch from the orientation quaternion
and use the rosbag receive timestamp as the common time axis.  Then
cross-correlate pairs to find the lag of EKF2 vs mocap and EKF2 vs IMU
(positive lag ⇒ EKF2 is delayed).

Usage:  python3 attitude_source_lag.py [<bag_subdir> [<date_dir>]]
"""

import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG_SUBDIR = sys.argv[1] if len(sys.argv) > 1 else '02_ct_1p255'
DATE_DIR   = sys.argv[2] if len(sys.argv) > 2 else '2026_05_05_free_flight'
DB = glob.glob(os.path.join(_HERE, DATE_DIR, BAG_SUBDIR, '*.db3'))[0]
OUT_DIR = os.path.join(_HERE, DATE_DIR)
TAG = BAG_SUBDIR
print(f'Analyzing: {DB}')


def _align(off, n):
    return off + ((-(off - 4)) % n)


def parse_imu_quat(blob):
    """sensor_msgs/Imu → quaternion (qw, qx, qy, qz)."""
    off = 4 + 8                                # CDR + stamp
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off)
    return np.array([qw, qx, qy, qz])


def parse_pose_quat(blob):
    """geometry_msgs/PoseStamped → quaternion (qw, qx, qy, qz)."""
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    px, py, pz, qx, qy, qz, qw = struct.unpack_from('<7d', blob, off)
    return np.array([qw, qx, qy, qz])


def parse_odom_quat(blob):
    """nav_msgs/Odometry → quaternion (qw, qx, qy, qz)."""
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    off += 24                                  # skip position
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off)
    return np.array([qw, qx, qy, qz])


def quat_to_rp(q):
    qw, qx, qy, qz = q
    roll  = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    return roll, pitch


# ── Load ──
conn = sqlite3.connect(DB)
c = conn.cursor()
tid = {n: i for i, n in c.execute('SELECT id, name FROM topics').fetchall()}

def fetch(topic, parser):
    ts, dat = [], []
    for t, b in c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
                          (tid[topic],)).fetchall():
        ts.append(t); dat.append(parser(bytes(b)))
    return np.array(ts, dtype=np.float64), np.array(dat)

imu_ts,  imu_q  = fetch('/mavros/imu/data',            parse_imu_quat)
moc_ts,  moc_q  = fetch('/S550/pose',                  parse_pose_quat)
odom_ts, odom_q = fetch('/mavros/local_position/odom', parse_odom_quat)
conn.close()

print(f'rates  IMU:  {len(imu_ts)/((imu_ts[-1]-imu_ts[0])*1e-9):.1f} Hz')
print(f'rates  mocap:{len(moc_ts)/((moc_ts[-1]-moc_ts[0])*1e-9):.1f} Hz')
print(f'rates  odom: {len(odom_ts)/((odom_ts[-1]-odom_ts[0])*1e-9):.1f} Hz')

t0 = min(imu_ts[0], moc_ts[0], odom_ts[0])
imu_t  = (imu_ts  - t0) * 1e-9
moc_t  = (moc_ts  - t0) * 1e-9
odom_t = (odom_ts - t0) * 1e-9

# Roll/pitch arrays
def rp_arr(q): return np.array([quat_to_rp(q[i]) for i in range(len(q))])
imu_rp  = rp_arr(imu_q)
moc_rp  = rp_arr(moc_q)
odom_rp = rp_arr(odom_q)


# ── Mocap quaternion sometimes published as (-q) of the IMU one; align signs. ──
# Use a few overlapping samples to figure out global sign.
def align_sign(t_ref, rp_ref, t_x, rp_x):
    # rp can flip sign if quaternion was negated by mocap stream
    rp_ref_u = np.interp(t_x, t_ref, rp_ref)
    return -1.0 if np.mean(rp_ref_u * rp_x) < 0 else 1.0

# Use IMU as reference for sign — but actually angles shouldn't flip with quaternion sign.
# Just plot and trust.

# ── Cross-correlation pairs ──
dt_u = 0.005                                            # 200 Hz grid
T_LO = max(imu_t[0], moc_t[0], odom_t[0]) + 2.0
T_HI = min(imu_t[-1], moc_t[-1], odom_t[-1]) - 2.0
t_u = np.arange(T_LO, T_HI, dt_u)

def resamp_pair(rt, ra, ta, tb, idx):
    a = np.interp(rt, ta, ra[:, idx])
    b = np.interp(rt, tb, ta_b[:, idx]) if False else None
    return a

def res(ta, ra, idx):
    return np.interp(t_u, ta, ra[:, idx])

imu_R, imu_P  = res(imu_t,  imu_rp, 0),  res(imu_t,  imu_rp, 1)
moc_R, moc_P  = res(moc_t,  moc_rp, 0),  res(moc_t,  moc_rp, 1)
odom_R, odom_P = res(odom_t, odom_rp, 0), res(odom_t, odom_rp, 1)


def xcorr_lag(a, b, dt, max_lag_s=0.5):
    """Lag at peak xcorr.  Positive lag ⇒ a lags b (a is delayed)."""
    a = a - a.mean(); b = b - b.mean()
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9: return 0.0, 0.0, np.array([0.0]), np.array([0.0])
    n = len(a); max_k = int(max_lag_s / dt)
    lags = np.arange(-max_k, max_k + 1)
    c = np.zeros_like(lags, dtype=float)
    for j, k in enumerate(lags):
        if k >= 0: c[j] = np.mean(a[k:] * b[:n - k])
        else:      c[j] = np.mean(a[:n + k] * b[-k:])
    c /= (sa * sb)
    j_pk = np.argmax(c)
    return lags[j_pk] * dt, c[j_pk], lags * dt, c


def report(name, a, b):
    lag_r, peak_r, lt_r, cv_r = xcorr_lag(a[0], b[0], dt_u)
    lag_p, peak_p, lt_p, cv_p = xcorr_lag(a[1], b[1], dt_u)
    print(f'  {name:30s}  roll lag={lag_r*1000:+5.0f} ms (pk={peak_r:.3f})   '
          f'pitch lag={lag_p*1000:+5.0f} ms (pk={peak_p:.3f})')
    return (lt_r, cv_r, lag_r), (lt_p, cv_p, lag_p)


print('\n== Attitude source latency (positive lag ⇒ left source delayed vs right) ==')
ekf_vs_imu  = report('EKF2 vs IMU',         (odom_R, odom_P), (imu_R, imu_P))
ekf_vs_moc  = report('EKF2 vs mocap',       (odom_R, odom_P), (moc_R, moc_P))
imu_vs_moc  = report('IMU vs mocap',        (imu_R,  imu_P),  (moc_R, moc_P))


# ── Plots ──
# 1) Overlay all three roll/pitch on a short window for visual sanity
t_win0, t_win1 = T_LO + 2.0, T_LO + 12.0      # 10-second slice
mask = (t_u >= t_win0) & (t_u <= t_win1)

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(t_u[mask], np.degrees(moc_R[mask]),  'k',  lw=1.6, label='mocap (truth)')
axes[0].plot(t_u[mask], np.degrees(imu_R[mask]),  'b',  lw=1.0, alpha=0.85, label='IMU AHRS')
axes[0].plot(t_u[mask], np.degrees(odom_R[mask]), 'r',  lw=1.0, alpha=0.85, label='EKF2 odom')
axes[0].set_ylabel('Roll [deg]'); axes[0].grid(alpha=0.3); axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_title(f'{TAG}  —  Roll/Pitch sources comparison (10 s window)')

axes[1].plot(t_u[mask], np.degrees(moc_P[mask]),  'k',  lw=1.6, label='mocap (truth)')
axes[1].plot(t_u[mask], np.degrees(imu_P[mask]),  'b',  lw=1.0, alpha=0.85, label='IMU AHRS')
axes[1].plot(t_u[mask], np.degrees(odom_P[mask]), 'r',  lw=1.0, alpha=0.85, label='EKF2 odom')
axes[1].set_ylabel('Pitch [deg]'); axes[1].set_xlabel('Time [s]')
axes[1].grid(alpha=0.3); axes[1].legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_attitude_source_overlay.png'), dpi=120)
plt.close()

# 2) xcorr curves
fig, axes = plt.subplots(3, 2, figsize=(13, 9))
for i, (name, (rdat, pdat)) in enumerate([
        ('EKF2 vs IMU',   ekf_vs_imu),
        ('EKF2 vs mocap', ekf_vs_moc),
        ('IMU vs mocap',  imu_vs_moc)]):
    lt, cv, lag = rdat
    axes[i, 0].plot(lt * 1000, cv, 'r')
    axes[i, 0].axvline(lag * 1000, color='k', ls='--', alpha=0.5)
    axes[i, 0].set_title(f'{name}  roll  lag={lag*1000:+.0f} ms')
    axes[i, 0].set_xlabel('lag [ms]'); axes[i, 0].set_ylabel('xcorr'); axes[i, 0].grid(alpha=0.3)
    lt, cv, lag = pdat
    axes[i, 1].plot(lt * 1000, cv, 'g')
    axes[i, 1].axvline(lag * 1000, color='k', ls='--', alpha=0.5)
    axes[i, 1].set_title(f'{name}  pitch  lag={lag*1000:+.0f} ms')
    axes[i, 1].set_xlabel('lag [ms]'); axes[i, 1].set_ylabel('xcorr'); axes[i, 1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_attitude_source_lag.png'), dpi=120)
plt.close()

print('\nSaved:')
print(f'  - {TAG}_attitude_source_overlay.png')
print(f'  - {TAG}_attitude_source_lag.png')
