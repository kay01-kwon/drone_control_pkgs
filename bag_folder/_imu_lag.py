#!/usr/bin/env python3
"""Measure actual delay introduced by the IMU LPF + EKF2 chain.

Cross-correlates three angular-velocity sources:
  (1) /mavros/imu/data_raw  — closest to raw gyro
  (2) /mavros/imu/data      — PX4 IMU-filtered (the 15 Hz cutoff)
  (3) odom twist.angular    — EKF2 output (what the controller consumes)

Lag(raw → filtered)  ≈ IMU LPF group delay
Lag(raw → odom)      ≈ IMU LPF + EKF2 delay (full sensing chain)

Usage:
  python3 _imu_lag.py <bag_subdir> [<date>]
"""
import os, sys, sqlite3, struct, glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG = sys.argv[1]
DATE = sys.argv[2] if len(sys.argv) > 2 else '2026_05_15_free_flight'
db = glob.glob(os.path.join(_HERE, DATE, BAG, '*.db3'))[0]


def _align(off, n):
    return off + (-(off - 4)) % n


def parse_imu(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    off += 4 * 8           # orientation quat
    off += 9 * 8           # orientation cov
    wx, wy, wz = struct.unpack_from('<3d', blob, off); off += 24
    return wx, wy, wz


def parse_odom_w(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    off += 3 * 8 + 4 * 8 + 36 * 8 + 3 * 8   # pos, quat, posecov, lin vel
    wx, wy, wz = struct.unpack_from('<3d', blob, off)
    return wx, wy, wz


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


rt, raw = fetch('/mavros/imu/data_raw', parse_imu)
ft, fil = fetch('/mavros/imu/data', parse_imu)
ot, odm = fetch('/mavros/local_position/odom', parse_odom_w)
con.close()

# Airborne crop using raw signal energy as proxy (skip likely-ground first part)
t_lo = max(rt[0], ft[0], ot[0]) + 30.0
t_hi = min(rt[-1], ft[-1], ot[-1]) - 5.0

# Resample all to common 200 Hz grid
fs = 200.0
tu = np.arange(t_lo, t_hi, 1 / fs)


def resample(t, d, comp):
    return np.interp(tu, t, d[:, comp])


def xcorr_lag(a, b, fs, max_lag_s=0.2):
    a = a - a.mean(); b = b - b.mean()
    n = len(a); ml = int(max_lag_s * fs)
    lags = np.arange(-ml, ml + 1)
    c = np.zeros(len(lags))
    for i, k in enumerate(lags):
        if k >= 0:
            x, y = a[:n - k], b[k:]
        else:
            x, y = a[-k:], b[:n + k]
        if len(x) > 10:
            c[i] = np.corrcoef(x, y)[0, 1]
    j = np.argmax(c)
    return lags[j] / fs * 1000, c[j]   # ms, peak corr


print(f'Window: {t_lo:.1f} → {t_hi:.1f} s  @ {fs} Hz')
print(f'\nLag measured per axis (positive = 2nd signal lags 1st):\n')
print(f'{"axis":6s} {"raw→filtered (IMU LPF)":28s} {"raw→odom (LPF+EKF2)":24s}')
for comp, axis in enumerate('xyz'):
    raw_u = resample(rt, raw, comp)
    fil_u = resample(ft, fil, comp)
    odm_u = resample(ot, odm, comp)
    lag_rf, c_rf = xcorr_lag(raw_u, fil_u, fs)
    lag_ro, c_ro = xcorr_lag(raw_u, odm_u, fs)
    print(f'  ω_{axis}   {lag_rf:+6.1f} ms (corr {c_rf:.3f})        '
          f'{lag_ro:+6.1f} ms (corr {c_ro:.3f})')
