#!/usr/bin/env python3
"""Check the time evolution of u_nmpc = cmd + tau_hat and the drift pattern.

If HGDO were correctly compensating a SMALL constant disturbance:
  - τ̂ would settle to ≈ -5 mNm within ~1s (5*ε_tau)
  - u_nmpc would settle to ≈ 0
  - cmd would settle to ≈ -τ̂ ≈ +5 mNm

What we observe: cmd and τ̂ drift monotonically over 30+ seconds.
This script checks whether the drift is in u_nmpc (NMPC integrator),
in τ̂ (observer bias), or in both."""

import sqlite3, struct
import numpy as np
import matplotlib.pyplot as plt

C_T = 1.386e-07
k_m = 0.01569
l = 0.265
MaxBit = 8191
MaxRpm = 9800

def p_cmd(data):
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    if off % 2 != 0: off += 1
    cmds = np.array(struct.unpack_from('<6h', data, off), dtype=np.float64)
    return sec + nsec*1e-9, cmds

def p_wrench(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    tx, ty, tz = struct.unpack_from('<3d', data, 52)
    return sec + nsec*1e-9, tx, ty, tz

lx1 = l*np.sin(np.pi/3); ly1 = l*np.cos(np.pi/3)
lx2 = 0.0;                ly2 = l
lx3 = -l*np.sin(np.pi/3); ly3 = l*np.cos(np.pi/3)
lx4 = -l*np.sin(np.pi/3); ly4 = -l*np.cos(np.pi/3)
lx5 = 0.0;                ly5 = -l
lx6 = l*np.sin(np.pi/3);  ly6 = -l*np.cos(np.pi/3)
K_forward = np.array([
    [1, 1, 1, 1, 1, 1],
    [ly1, ly2, ly3, ly4, ly5, ly6],
    [-lx1, -lx2, -lx3, -lx4, -lx5, -lx6],
    [-k_m, k_m, -k_m, k_m, -k_m, k_m],
])

fig, axes = plt.subplots(3, 2, figsize=(16, 11))

for col, name in enumerate(['roll_test_hgdo_eps_01', 'roll_test_hgdo_eps_02']):
    db = f'{name}/{name}_0.db3'
    conn = sqlite3.connect(db); c = conn.cursor()
    c.execute('SELECT id, name FROM topics'); topics = {n: tid for tid, n in c.fetchall()}

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/uav/cmd_raw'],))
    TC, Mxc, Myc, Mzc, Fc = [], [], [], [], []
    for data, in c.fetchall():
        t, cmds = p_cmd(data)
        rpms = cmds * MaxRpm / MaxBit
        thr = C_T * rpms**2
        u = K_forward @ thr
        TC.append(t); Fc.append(u[0]); Mxc.append(u[1]); Myc.append(u[2]); Mzc.append(u[3])
    TC = np.array(TC); Fc = np.array(Fc); Mxc = np.array(Mxc); Myc = np.array(Myc); Mzc = np.array(Mzc)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/hgdo/wrench'],))
    TW, HTX, HTY, HTZ = [], [], [], []
    for data, in c.fetchall():
        t, tx, ty, tz = p_wrench(data)
        TW.append(t); HTX.append(tx); HTY.append(ty); HTZ.append(tz)
    TW = np.array(TW); HTX = np.array(HTX); HTY = np.array(HTY); HTZ = np.array(HTZ)
    conn.close()

    t0 = min(TC[0], TW[0])
    TC -= t0; TW -= t0

    # Interpolate HGDO onto cmd timestamps
    Htx_c = np.interp(TC, TW, HTX)
    Hty_c = np.interp(TC, TW, HTY)
    Htz_c = np.interp(TC, TW, HTZ)

    # u_nmpc = cmd + tau_hat  (NMPC's raw moment, before DOB subtraction)
    U_nmpc_x = Mxc + Htx_c
    U_nmpc_y = Myc + Hty_c
    U_nmpc_z = Mzc + Htz_c

    # Smooth u_nmpc (LPF at 1 Hz to see the slow trend)
    from scipy.signal import butter, filtfilt
    fs = 1.0 / np.mean(np.diff(TC))
    b, a = butter(2, 1.0/(fs/2), btype='low')
    U_nmpc_y_lp = filtfilt(b, a, U_nmpc_y)
    U_nmpc_z_lp = filtfilt(b, a, U_nmpc_z)
    Myc_lp = filtfilt(b, a, Myc)
    Mzc_lp = filtfilt(b, a, Mzc)
    Hty_c_lp = filtfilt(b, a, Hty_c)
    Htz_c_lp = filtfilt(b, a, Htz_c)

    eps_str = '0.1' if '01' in name else '0.2'

    ax = axes[0, col]
    ax.plot(TC, Myc*1000, 'tab:blue', lw=0.3, alpha=0.3, label='cmd My')
    ax.plot(TC, -Hty_c*1000, 'tab:red', lw=0.3, alpha=0.3, label='-τ̂_y')
    ax.plot(TC, U_nmpc_y*1000, 'tab:green', lw=0.3, alpha=0.3, label='u_nmpc,y = cmd + τ̂')
    ax.plot(TC, Myc_lp*1000, 'tab:blue', lw=1.5, label='cmd My (1Hz LPF)')
    ax.plot(TC, -Hty_c_lp*1000, 'tab:red', lw=1.5, label='-τ̂_y (LPF)')
    ax.plot(TC, U_nmpc_y_lp*1000, 'tab:green', lw=2.0, label='u_nmpc,y (LPF)')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('mNm'); ax.set_title(f'{name} (ε={eps_str})  Pitch-axis: is u_nmpc drifting?')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    ax = axes[1, col]
    ax.plot(TC, Mzc*1000, 'tab:blue', lw=0.3, alpha=0.3, label='cmd Mz')
    ax.plot(TC, -Htz_c*1000, 'tab:red', lw=0.3, alpha=0.3, label='-τ̂_z')
    ax.plot(TC, U_nmpc_z*1000, 'tab:green', lw=0.3, alpha=0.3, label='u_nmpc,z')
    ax.plot(TC, Mzc_lp*1000, 'tab:blue', lw=1.5, label='cmd Mz (LPF)')
    ax.plot(TC, -Htz_c_lp*1000, 'tab:red', lw=1.5, label='-τ̂_z (LPF)')
    ax.plot(TC, U_nmpc_z_lp*1000, 'tab:green', lw=2.0, label='u_nmpc,z (LPF)')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('mNm'); ax.set_title(f'{name}  Yaw-axis: is u_nmpc drifting?')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    # residual: cmd + tau_hat  (= u_nmpc).  If ≈0 throughout, DOB is doing exactly the trim.
    #                                   If growing, NMPC integrator is fighting the observer.
    ax = axes[2, col]
    ax.plot(TC, U_nmpc_x*1000, 'tab:red',  lw=0.3, alpha=0.5)
    ax.plot(TC, U_nmpc_y*1000, 'tab:blue', lw=0.3, alpha=0.5)
    ax.plot(TC, U_nmpc_z*1000, 'tab:green',lw=0.3, alpha=0.5)
    Uxlp = filtfilt(b, a, U_nmpc_x)
    ax.plot(TC, Uxlp*1000,    'tab:red',  lw=1.5, label='u_nmpc,x (LPF)')
    ax.plot(TC, U_nmpc_y_lp*1000, 'tab:blue', lw=1.5, label='u_nmpc,y (LPF)')
    ax.plot(TC, U_nmpc_z_lp*1000, 'tab:green',lw=1.5, label='u_nmpc,z (LPF)')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('mNm'); ax.set_xlabel('s')
    ax.set_title(f'{name}  Implied NMPC moment u_nmpc = cmd + τ̂   (should be ~0 if DOB is perfect)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Print slope over flight window
    flt = Fc > 20
    if flt.any():
        t0f = TC[np.argmax(flt)]
        mask = (TC > t0f + 3) & (TC < TC[-1] - 1)
        if mask.sum() > 100:
            ty_drift = (Hty_c[mask][-1] - Hty_c[mask][0]) / (TC[mask][-1] - TC[mask][0])
            tz_drift = (Htz_c[mask][-1] - Htz_c[mask][0]) / (TC[mask][-1] - TC[mask][0])
            un_y_mean = U_nmpc_y[mask].mean()
            un_z_mean = U_nmpc_z[mask].mean()
            print(f'{name} (ε={eps_str}):  τ̂_y drift rate = {ty_drift*1000:+.2f} mNm/s | τ̂_z drift rate = {tz_drift*1000:+.2f} mNm/s')
            print(f'   u_nmpc,y mean = {un_y_mean*1000:+.2f} mNm | u_nmpc,z mean = {un_z_mean*1000:+.2f} mNm')
            # first 2s value vs last 2s value
            early = (TC > t0f) & (TC < t0f + 2.0)
            late  = (TC > TC[-1] - 3.0) & (TC < TC[-1] - 0.5)
            if early.any() and late.any():
                print(f'   cmd_My: first 2s = {Myc[early].mean()*1000:+.2f} mNm | last 2.5s = {Myc[late].mean()*1000:+.2f} mNm')
                print(f'   τ̂_y  : first 2s = {Hty_c[early].mean()*1000:+.2f} mNm | last 2.5s = {Hty_c[late].mean()*1000:+.2f} mNm')
                print(f'   u_nmpc,y: first 2s = {U_nmpc_y[early].mean()*1000:+.2f} mNm | last 2.5s = {U_nmpc_y[late].mean()*1000:+.2f} mNm')
            print()

plt.tight_layout()
out = 'roll_tests_hgdo_unmpc_drift.png'
plt.savefig(out, dpi=110); plt.close()
print(f'[saved] {out}')
