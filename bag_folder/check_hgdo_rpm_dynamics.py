#!/usr/bin/env python3
"""Time-series look at HGDO runs: does cmd_My drift, oscillate, or sit at +280 mNm?

Also: compare the 6 RPMs over time to see if there's a slow wind-up pattern."""

import sqlite3, struct
import numpy as np
import matplotlib.pyplot as plt

C_T = 1.386e-07
k_m = 0.01569
l = 0.265
MaxBit = 8191
MaxRpm = 9800

def p_rpm(data):
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    off = (off + 3) & ~3
    rpms = np.array(struct.unpack_from('<6i', data, off), dtype=np.float64)
    return sec + nsec*1e-9, rpms

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
    fx, fy, fz = struct.unpack_from('<3d', data, 28)
    tx, ty, tz = struct.unpack_from('<3d', data, 52)
    return sec + nsec*1e-9, fx, fy, fz, tx, ty, tz


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

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/uav/actual_rpm'],))
    TR, RPM = [], []
    for data, in c.fetchall():
        t, r = p_rpm(data)
        TR.append(t); RPM.append(r)
    TR = np.array(TR); RPM = np.array(RPM)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/uav/cmd_raw'],))
    TC, CRPM = [], []
    for data, in c.fetchall():
        t, cmds = p_cmd(data)
        rpms = cmds * MaxRpm / MaxBit
        TC.append(t); CRPM.append(rpms)
    TC = np.array(TC); CRPM = np.array(CRPM)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/hgdo/wrench'],))
    TW, HTX, HTY, HTZ = [], [], [], []
    for data, in c.fetchall():
        t, _, _, _, tx, ty, tz = p_wrench(data)
        TW.append(t); HTX.append(tx); HTY.append(ty); HTZ.append(tz)
    TW = np.array(TW); HTX = np.array(HTX); HTY = np.array(HTY); HTZ = np.array(HTZ)
    conn.close()

    t0 = min(TR[0], TC[0], TW[0])
    TR -= t0; TC -= t0; TW -= t0

    # compute actual moments
    Thr = C_T * RPM**2
    U_act = (K_forward @ Thr.T).T
    Thr_c = C_T * CRPM**2
    U_cmd = (K_forward @ Thr_c.T).T

    # panel 1: per-motor RPM
    ax = axes[0, col]
    motor_colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple', 'tab:brown']
    for i in range(6):
        ax.plot(TR, RPM[:, i], color=motor_colors[i], lw=0.5, label=f'M{i+1}')
    ax.set_ylabel('actual RPM'); ax.set_title(f'{name}  per-motor actual RPM')
    ax.legend(ncol=6, fontsize=8); ax.grid(alpha=0.3)

    # panel 2: cmd moments + tau_hat
    ax = axes[1, col]
    ax.plot(TC, U_cmd[:, 1]*1000, 'tab:red', lw=0.5, label='cmd Mx')
    ax.plot(TC, U_cmd[:, 2]*1000, 'tab:blue', lw=0.5, label='cmd My')
    ax.plot(TC, U_cmd[:, 3]*1000, 'tab:green', lw=0.5, label='cmd Mz')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('mNm'); ax.set_title(f'{name}  cmd_raw moments')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # panel 3: HGDO tau_hat
    ax = axes[2, col]
    ax.plot(TW, HTX*1000, 'tab:red', lw=0.5, label='HGDO τ_x')
    ax.plot(TW, HTY*1000, 'tab:blue', lw=0.5, label='HGDO τ_y')
    ax.plot(TW, HTZ*1000, 'tab:green', lw=0.5, label='HGDO τ_z')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('mNm'); ax.set_xlabel('s'); ax.set_title(f'{name}  HGDO disturbance estimate')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Print time ranges where tau_hat_y crosses key thresholds
    print(f'--- {name} (eps_tau={"0.1" if "01" in name else "0.2"}) ---')
    # Find when HGDO first crosses -50 mNm in Y
    # Also check: what is tau_hat at t=arming (thrust up)?
    # Proxy: first big jump in U_cmd[0]
    F_cmd = U_cmd[:, 0]
    above = F_cmd > 20.0
    if above.any():
        t_arm = TC[np.argmax(above)]
        print(f'  thrust>20N at t={t_arm:.2f}s (arming proxy)')
        # HGDO near arming
        near = (TW >= t_arm - 1.0) & (TW <= t_arm + 3.0)
        if near.any():
            ty_near = HTY[near]*1000
            print(f'  τ_hat_y during first 3s of flight:  min={ty_near.min():+.2f} max={ty_near.max():+.2f} end={ty_near[-1]:+.2f} mNm')
    # Final tau_hat
    if len(TW) > 10:
        ty_final = HTY[-100:]*1000
        print(f'  τ_hat_y last 100 samples: mean={ty_final.mean():+.2f}  std={ty_final.std():.3f} mNm')

plt.tight_layout()
out = 'roll_tests_hgdo_cmd_dynamics.png'
plt.savefig(out, dpi=110); plt.close()
print(f'[saved] {out}')
