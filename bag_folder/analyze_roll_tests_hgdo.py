#!/usr/bin/env python3
"""Analyze HGDO-enabled tests: eps_tau 0.1 (bag _01) vs 0.2 (bag _02).

NMPC + HGDO (use_dob_compensation: true), Q_rate=0.5 (sweet spot from earlier tests).
User observation: eps_tau=0.2 (bag _02) looked more stable.

Checks:
  - HGDO tau_x convergence DC value (should match observed trim bias)
  - HGDO tau_x noise std
  - Roll steady-state mean (should be ~0 with DOB)
  - cmd Mx noise amplification
  - Residual Mx (actual vs HGDO estimate)
"""

import sqlite3, struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import welch

# allocation constants
C_T = 1.386e-07
k_m = 0.01569
l   = 0.265
MaxBit = 8191
MaxRpm = 9800

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
    [-k_m, k_m, -k_m, k_m, -k_m, k_m]
])

LABEL = {
    'roll_test_hgdo_eps_01': 'eps_tau=0.1  (BW ~1.6 Hz)',
    'roll_test_hgdo_eps_02': 'eps_tau=0.2  (BW ~0.8 Hz)',
}
COLOR = {
    'roll_test_hgdo_eps_01': 'tab:red',
    'roll_test_hgdo_eps_02': 'tab:blue',
}


def p_cmd(data):
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    if off % 2 != 0: off += 1
    cmds = np.array(struct.unpack_from('<6h', data, off), dtype=np.float64)
    return sec + nsec*1e-9, cmds

def p_rpm(data):
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    off = (off + 3) & ~3
    rpms = np.array(struct.unpack_from('<6i', data, off), dtype=np.float64)
    return sec + nsec*1e-9, rpms

def p_odom(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, 44)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 68)
    vx, vy, vz = struct.unpack_from('<3d', data, 388)
    wx, wy, wz = struct.unpack_from('<3d', data, 412)
    return sec + nsec*1e-9, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz

def p_pose(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, 28)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 52)
    return sec + nsec*1e-9, px, py, pz, qx, qy, qz, qw

def p_wrench(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    fx, fy, fz = struct.unpack_from('<3d', data, 28)
    tx, ty, tz = struct.unpack_from('<3d', data, 52)
    return sec + nsec*1e-9, fx, fy, fz, tx, ty, tz


def load(db):
    conn = sqlite3.connect(db); c = conn.cursor()
    c.execute('SELECT id, name FROM topics')
    topics = {n: tid for tid, n in c.fetchall()}

    # odom
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/mavros/local_position/odom'],))
    T, PX, PY, PZ, R_, P_, Y_, WX, WY, WZ = [], [], [], [], [], [], [], [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = p_odom(data)
        q = np.array([qx,qy,qz,qw]); nr = np.linalg.norm(q)
        if not np.isfinite(nr) or nr < 1e-10: continue
        r, p, y = Rotation.from_quat(q/nr).as_euler('xyz', degrees=True)
        T.append(t); PX.append(px); PY.append(py); PZ.append(pz)
        R_.append(r); P_.append(p); Y_.append(y)
        WX.append(wx); WY.append(wy); WZ.append(wz)
    T = np.array(T); PX=np.array(PX); PY=np.array(PY); PZ=np.array(PZ)
    R_ = np.array(R_); P_ = np.array(P_); Y_ = np.array(Y_)
    WX = np.array(WX); WY = np.array(WY); WZ = np.array(WZ)

    # mocap
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/S550/pose'],))
    Tm, Rm, Pm, Ym, Mpx, Mpy, Mpz = [], [], [], [], [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw = p_pose(data)
        q = np.array([qx,qy,qz,qw]); nr = np.linalg.norm(q)
        if not np.isfinite(nr) or nr < 1e-10: continue
        r, p, y = Rotation.from_quat(q/nr).as_euler('xyz', degrees=True)
        Tm.append(t); Rm.append(r); Pm.append(p); Ym.append(y)
        Mpx.append(px); Mpy.append(py); Mpz.append(pz)
    Tm = np.array(Tm); Rm = np.array(Rm); Pm = np.array(Pm); Ym = np.array(Ym)
    Mpx = np.array(Mpx); Mpy = np.array(Mpy); Mpz = np.array(Mpz)

    # cmd_raw -> moments
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/uav/cmd_raw'],))
    Tc, Fc, Mxc, Myc, Mzc = [], [], [], [], []
    for data, in c.fetchall():
        t, cmds = p_cmd(data)
        rpms = cmds * MaxRpm / MaxBit
        thr = C_T * rpms**2
        u = K_forward @ thr
        Tc.append(t); Fc.append(u[0]); Mxc.append(u[1]); Myc.append(u[2]); Mzc.append(u[3])
    Tc = np.array(Tc); Fc = np.array(Fc); Mxc = np.array(Mxc); Myc = np.array(Myc); Mzc = np.array(Mzc)

    # actual_rpm -> moments
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/uav/actual_rpm'],))
    Tr, Fr, Mxr, Myr, Mzr = [], [], [], [], []
    for data, in c.fetchall():
        t, rpms = p_rpm(data)
        thr = C_T * rpms**2
        u = K_forward @ thr
        Tr.append(t); Fr.append(u[0]); Mxr.append(u[1]); Myr.append(u[2]); Mzr.append(u[3])
    Tr = np.array(Tr); Fr = np.array(Fr); Mxr = np.array(Mxr); Myr = np.array(Myr); Mzr = np.array(Mzr)

    # hgdo wrench
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/hgdo/wrench'],))
    Th, Hfx, Hfy, Hfz, Htx, Hty, Htz = [], [], [], [], [], [], []
    for data, in c.fetchall():
        t, fx, fy, fz, tx, ty, tz = p_wrench(data)
        Th.append(t); Hfx.append(fx); Hfy.append(fy); Hfz.append(fz)
        Htx.append(tx); Hty.append(ty); Htz.append(tz)
    Th = np.array(Th); Hfx = np.array(Hfx); Hfy = np.array(Hfy); Hfz = np.array(Hfz)
    Htx = np.array(Htx); Hty = np.array(Hty); Htz = np.array(Htz)

    conn.close()

    t0 = min(T[0], Tm[0], Tc[0], Tr[0], Th[0])
    T -= t0; Tm -= t0; Tc -= t0; Tr -= t0; Th -= t0

    Mpz0 = Mpz[0]; Mpz = Mpz - Mpz0

    return dict(T=T, PX=PX, PY=PY, PZ=PZ, R=R_, P=P_, Y=Y_, WX=WX, WY=WY, WZ=WZ,
                Tm=Tm, Rm=Rm, Pm=Pm, Ym=Ym, Mpx=Mpx, Mpy=Mpy, Mpz=Mpz,
                Tc=Tc, Fc=Fc, Mxc=Mxc, Myc=Myc, Mzc=Mzc,
                Tr=Tr, Fr=Fr, Mxr=Mxr, Myr=Myr, Mzr=Mzr,
                Th=Th, Htx=Htx, Hty=Hty, Htz=Htz, Hfz=Hfz)


def detect_flight(d, thrust_thresh=6.0):
    Fc = d['Fc']; Tc = d['Tc']
    above = Fc > thrust_thresh
    if not above.any():
        return (Tc[0], Tc[-1])
    i0 = np.argmax(above); i1 = len(above)-1 - np.argmax(above[::-1])
    return (Tc[i0], Tc[i1])


def per_bag_plot(name, d, out):
    t0, t1 = detect_flight(d)
    fig, ax = plt.subplots(7, 1, figsize=(14, 20), sharex=True)

    ax[0].plot(d['Tm'], d['Rm'], 'tab:red', lw=1.0, label='roll (mocap)')
    ax[0].plot(d['Tm'], d['Pm'], 'tab:blue', lw=1.0, label='pitch (mocap)')
    ax[0].plot(d['Tm'], d['Ym'], 'tab:green', lw=1.0, label='yaw (mocap)')
    ax[0].axvspan(t0, t1, color='yellow', alpha=0.1, label=f'flight {t0:.1f}-{t1:.1f}s')
    ax[0].axhline(0, color='k', lw=0.4)
    ax[0].set_ylabel('deg')
    ax[0].set_title(f'{name}  Attitude   [{LABEL[name]}]')
    ax[0].legend(ncol=4, fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(d['T'], d['WX'], 'tab:red', lw=0.6, label='wx')
    ax[1].plot(d['T'], d['WY'], 'tab:blue', lw=0.6, label='wy')
    ax[1].plot(d['T'], d['WZ'], 'tab:green', lw=0.6, label='wz')
    ax[1].axhline(0, color='k', lw=0.4)
    ax[1].set_ylabel('rad/s'); ax[1].set_title('Body rates')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    ax[2].plot(d['Th'], d['Htx'], 'tab:red',   lw=0.7, label='HGDO tau_x')
    ax[2].plot(d['Th'], d['Hty'], 'tab:blue',  lw=0.7, label='HGDO tau_y')
    ax[2].plot(d['Th'], d['Htz'], 'tab:green', lw=0.7, label='HGDO tau_z')
    ax[2].axhline(0, color='k', lw=0.4)
    ax[2].set_ylabel('Nm'); ax[2].set_title('HGDO disturbance torque estimate')
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

    ax[3].plot(d['Tc'], d['Mxc'], 'tab:red', lw=0.6, label='cmd Mx')
    ax[3].plot(d['Tc'], d['Myc'], 'tab:blue', lw=0.6, label='cmd My')
    ax[3].plot(d['Tc'], d['Mzc'], 'tab:green', lw=0.6, label='cmd Mz')
    ax[3].axhline(0, color='k', lw=0.4)
    ax[3].set_ylabel('Nm'); ax[3].set_title('Commanded moments (cmd_raw allocation)')
    ax[3].legend(fontsize=9); ax[3].grid(alpha=0.3)

    ax[4].plot(d['Tr'], d['Mxr'], 'tab:red', lw=0.6, label='act Mx')
    ax[4].plot(d['Tr'], d['Myr'], 'tab:blue', lw=0.6, label='act My')
    ax[4].plot(d['Tr'], d['Mzr'], 'tab:green', lw=0.6, label='act Mz')
    ax[4].axhline(0, color='k', lw=0.4)
    ax[4].set_ylabel('Nm'); ax[4].set_title('Actual moments')
    ax[4].legend(fontsize=9); ax[4].grid(alpha=0.3)

    ax[5].plot(d['Tc'], d['Fc'], 'tab:purple', lw=0.8, label='cmd F')
    ax[5].plot(d['Tr'], d['Fr'], 'tab:orange', lw=0.8, alpha=0.7, label='act F')
    ax[5].plot(d['Th'], d['Hfz'], 'tab:cyan', lw=0.6, alpha=0.7, label='HGDO F_z disturbance')
    ax[5].set_ylabel('N'); ax[5].set_title('Thrust + HGDO Fz estimate')
    ax[5].legend(fontsize=9); ax[5].grid(alpha=0.3)

    ax[6].plot(d['Tm'], d['Mpx'], 'tab:red', lw=0.8, label='x')
    ax[6].plot(d['Tm'], d['Mpy'], 'tab:blue', lw=0.8, label='y')
    ax[6].plot(d['Tm'], d['Mpz'], 'tab:green', lw=0.8, label='z (off sub)')
    ax[6].set_ylabel('m'); ax[6].set_xlabel('s'); ax[6].set_title('Mocap position')
    ax[6].legend(fontsize=9); ax[6].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()
    print(f'[saved] {out}')


def window_stats(d, frac_end=0.5):
    t0, t1 = detect_flight(d)
    span = t1 - t0
    t_a = t0 + (1 - frac_end) * span
    t_b = t1 - 0.5
    res = {}
    for key, tname in [('Rm','Tm'), ('Pm','Tm'), ('Ym','Tm'),
                        ('WX','T'), ('WY','T'), ('WZ','T'),
                        ('Mxc','Tc'), ('Myc','Tc'),
                        ('Mxr','Tr'), ('Myr','Tr'),
                        ('Htx','Th'), ('Hty','Th'), ('Htz','Th')]:
        T = d[tname]; V = d[key]
        m = (T >= t_a) & (T <= t_b)
        if m.sum() < 10:
            res[key] = (np.nan, np.nan)
        else:
            v = V[m]
            res[key] = (float(v.mean()), float(v.std()))
    res['win'] = (t_a, t_b)
    return res


def compare_plot(data, out):
    fig, ax = plt.subplots(6, 1, figsize=(14, 18), sharex=False)

    # 1. roll
    ax[0].set_title('Roll (mocap)')
    for n, d in data.items():
        ax[0].plot(d['Tm'], d['Rm'], color=COLOR[n], lw=0.8, label=f'{n}  [{LABEL[n]}]')
    ax[0].axhline(0, color='k', lw=0.4); ax[0].set_ylabel('roll (deg)')
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    # 2. wx
    ax[1].set_title('Body roll rate wx')
    for n, d in data.items():
        ax[1].plot(d['T'], d['WX'], color=COLOR[n], lw=0.5, label=n)
    ax[1].axhline(0, color='k', lw=0.4); ax[1].set_ylabel('wx (rad/s)')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    # 3. HGDO tau_x
    ax[2].set_title('HGDO disturbance torque estimate  tau_x  (should converge to ~trim bias)')
    for n, d in data.items():
        ax[2].plot(d['Th'], d['Htx'], color=COLOR[n], lw=0.7, label=n)
    ax[2].axhline(0, color='k', lw=0.4); ax[2].set_ylabel('tau_x (Nm)')
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

    # 4. cmd Mx
    ax[3].set_title('Commanded Mx (what NMPC puts out, with DOB subtraction if enabled)')
    for n, d in data.items():
        ax[3].plot(d['Tc'], d['Mxc'], color=COLOR[n], lw=0.5, label=n)
    ax[3].axhline(0, color='k', lw=0.4); ax[3].set_ylabel('cmd Mx (Nm)')
    ax[3].legend(fontsize=9); ax[3].grid(alpha=0.3)

    # 5. actual Mx vs HGDO tau_x
    ax[4].set_title('Actual Mx (allocation)  — note HGDO is the disturbance estimate, actual is total applied moment')
    for n, d in data.items():
        ax[4].plot(d['Tr'], d['Mxr'], color=COLOR[n], lw=0.5, label=f'{n} act Mx')
    ax[4].axhline(0, color='k', lw=0.4); ax[4].set_ylabel('act Mx (Nm)')
    ax[4].legend(fontsize=9); ax[4].grid(alpha=0.3)

    # 6. PSD of wx, tau_x, cmdMx
    ax[5].set_title('PSD during flight window (resampled 100 Hz)')
    for n, d in data.items():
        t0, t1 = detect_flight(d)
        fs = 100.0
        tg = np.arange(t0 + 1.0, t1 - 0.5, 1/fs)
        if len(tg) < 256: continue
        wxg  = np.interp(tg, d['T'],  d['WX'])
        htxg = np.interp(tg, d['Th'], d['Htx'])
        mxcg = np.interp(tg, d['Tc'], d['Mxc'])
        f_wx, P_wx   = welch(wxg,  fs=fs, nperseg=min(1024, len(tg)//2))
        f_tx, P_tx   = welch(htxg, fs=fs, nperseg=min(1024, len(tg)//2))
        f_mxc, P_mxc = welch(mxcg, fs=fs, nperseg=min(1024, len(tg)//2))
        ax[5].semilogy(f_wx,  P_wx,  color=COLOR[n], lw=1.0, ls='-',  label=f'{n} wx')
        ax[5].semilogy(f_tx,  P_tx,  color=COLOR[n], lw=1.0, ls='--', alpha=0.8, label=f'{n} tau_x (HGDO)')
        ax[5].semilogy(f_mxc, P_mxc, color=COLOR[n], lw=1.0, ls=':',  alpha=0.8, label=f'{n} cmd Mx')
    ax[5].set_xlim(0, 50); ax[5].set_xlabel('Hz'); ax[5].set_ylabel('PSD')
    ax[5].legend(fontsize=8, ncol=2); ax[5].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()
    print(f'[saved] {out}')


def initial_convergence(d, out_prefix, name):
    """Plot arming transient of tau_x."""
    t0, t1 = detect_flight(d)
    mask = (d['Th'] >= t0 - 0.5) & (d['Th'] <= t0 + 3.0)
    T = d['Th'][mask]; Tx = d['Htx'][mask]
    if len(T) < 10: return None
    # Final value estimate from last half of flight
    t_final_a = t0 + (t1 - t0) * 0.5
    t_final_b = t1 - 0.5
    mask_f = (d['Th'] >= t_final_a) & (d['Th'] <= t_final_b)
    tx_final = float(np.mean(d['Htx'][mask_f])) if mask_f.sum() > 10 else float('nan')
    # time to within 10% of final
    thresh = 0.1 * abs(tx_final) if abs(tx_final) > 1e-4 else 0.001
    within = np.where(np.abs(Tx - tx_final) < thresh)[0]
    t_settle = float(T[within[0]] - t0) if len(within) > 0 else float('nan')
    return dict(t_settle=t_settle, tx_final=tx_final, t_arm=t0)


def main():
    base = '/home/user/drone_control_pkgs/bag_folder'
    bags = ['roll_test_hgdo_eps_01', 'roll_test_hgdo_eps_02']
    data = {}
    for n in bags:
        db = f'{base}/{n}/{n}_0.db3'
        print(f'load {n}')
        data[n] = load(db)
        per_bag_plot(n, data[n], f'{base}/{n}_analysis.png')
    compare_plot(data, f'{base}/roll_tests_hgdo_compare.png')

    # Stats
    print()
    print('=== Steady-state stats (last 50% of flight window) ===')
    hdr = f'{"bag":<28} | {"win":<13} | {"roll_μ":>7} {"roll_σ":>7} {"wx_σ":>6} | {"τx_μ(mNm)":>10} {"τx_σ(mNm)":>10} | {"cmdMx_μ(mNm)":>13} {"cmdMx_σ(mNm)":>13} | {"actMx_μ(mNm)":>13} {"actMx_σ(mNm)":>13}'
    print(hdr); print('-'*len(hdr))
    for n, d in data.items():
        s = window_stats(d)
        ta, tb = s['win']
        print(f'{n:<28} | {ta:5.1f}-{tb:5.1f}s | {s["Rm"][0]:7.3f} {s["Rm"][1]:7.3f} {s["WX"][1]:6.3f} | {s["Htx"][0]*1000:10.3f} {s["Htx"][1]*1000:10.3f} | {s["Mxc"][0]*1000:13.3f} {s["Mxc"][1]*1000:13.3f} | {s["Mxr"][0]*1000:13.3f} {s["Mxr"][1]*1000:13.3f}')

    # Convergence
    print()
    print('=== HGDO tau_x arming transient ===')
    for n, d in data.items():
        r = initial_convergence(d, None, n)
        if r is None: continue
        print(f'{n:<28} | final tau_x = {r["tx_final"]*1000:7.3f} mNm | settle(<10% of final) = {r["t_settle"]:.3f}s')


if __name__ == '__main__':
    main()
