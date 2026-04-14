#!/usr/bin/env python3
"""Analyze the pitch-axis HGDO test (dob_axis_mode='y').

Stand orientation flipped to free the pitch axis; roll & yaw constrained.
Q tuning has also been softened (Q_att 20 → 2, Q_rate 5 → 0.5, R 10 → 5).

Mask identity expected:
    cmd_Mx = u_nmpc_x            (mask=0)
    cmd_My = u_nmpc_y - τ̂_y      (mask=1)   ← active axis
    cmd_Mz = u_nmpc_z            (mask=0)

User noted that the airframe has a known pitch-axis eccentricity, so
τ̂_y should settle to a clearly nonzero bias (much larger than the
roll trim of ~-10 mNm). No hand-disturbance was applied this time.
"""

import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import welch

# ----------------------------------------------------------------- consts
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
    [-k_m, k_m, -k_m, k_m, -k_m, k_m],
])

# ----------------------------------------------------------------- parsers
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


def p_wrench_stamped(data):
    """CDR alignment is relative to payload start (offset 4)."""
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    payload_off = off - 4
    pad = (-payload_off) & 7
    off += pad
    fx, fy, fz = struct.unpack_from('<3d', data, off); off += 24
    tx, ty, tz = struct.unpack_from('<3d', data, off)
    return sec + nsec*1e-9, fx, fy, fz, tx, ty, tz


# ----------------------------------------------------------------- loader
def load(db):
    conn = sqlite3.connect(db); c = conn.cursor()
    c.execute('SELECT id, name FROM topics')
    topics = {n: tid for tid, n in c.fetchall()}

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/mavros/local_position/odom'],))
    T, PX, PY, PZ, R_, P_, Y_, WX, WY, WZ = ([] for _ in range(10))
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = p_odom(data)
        q = np.array([qx, qy, qz, qw]); nr = np.linalg.norm(q)
        if not np.isfinite(nr) or nr < 1e-10:
            continue
        r, p, y = Rotation.from_quat(q/nr).as_euler('xyz', degrees=True)
        T.append(t); PX.append(px); PY.append(py); PZ.append(pz)
        R_.append(r); P_.append(p); Y_.append(y)
        WX.append(wx); WY.append(wy); WZ.append(wz)
    T = np.array(T); R_ = np.array(R_); P_ = np.array(P_); Y_ = np.array(Y_)
    WX = np.array(WX); WY = np.array(WY); WZ = np.array(WZ)
    PX = np.array(PX); PY = np.array(PY); PZ = np.array(PZ)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/S550/pose'],))
    Tm, Rm, Pm, Ym, Mpx, Mpy, Mpz = ([] for _ in range(7))
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw = p_pose(data)
        q = np.array([qx, qy, qz, qw]); nr = np.linalg.norm(q)
        if not np.isfinite(nr) or nr < 1e-10:
            continue
        r, p, y = Rotation.from_quat(q/nr).as_euler('xyz', degrees=True)
        Tm.append(t); Rm.append(r); Pm.append(p); Ym.append(y)
        Mpx.append(px); Mpy.append(py); Mpz.append(pz)
    Tm = np.array(Tm); Rm = np.array(Rm); Pm = np.array(Pm); Ym = np.array(Ym)
    Mpx = np.array(Mpx); Mpy = np.array(Mpy); Mpz = np.array(Mpz)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/uav/cmd_raw'],))
    Tc, Fc, Mxc, Myc, Mzc = ([] for _ in range(5))
    for data, in c.fetchall():
        t, cmds = p_cmd(data)
        rpms = cmds * MaxRpm / MaxBit
        thr = C_T * rpms**2
        u = K_forward @ thr
        Tc.append(t); Fc.append(u[0]); Mxc.append(u[1]); Myc.append(u[2]); Mzc.append(u[3])
    Tc = np.array(Tc); Fc = np.array(Fc)
    Mxc = np.array(Mxc); Myc = np.array(Myc); Mzc = np.array(Mzc)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/uav/actual_rpm'],))
    Tr, Fr, Mxr, Myr, Mzr = ([] for _ in range(5))
    for data, in c.fetchall():
        t, rpms = p_rpm(data)
        thr = C_T * rpms**2
        u = K_forward @ thr
        Tr.append(t); Fr.append(u[0]); Mxr.append(u[1]); Myr.append(u[2]); Mzr.append(u[3])
    Tr = np.array(Tr); Fr = np.array(Fr)
    Mxr = np.array(Mxr); Myr = np.array(Myr); Mzr = np.array(Mzr)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/nmpc/control'],))
    Tn, Nfz, Nmx, Nmy, Nmz = ([] for _ in range(5))
    for data, in c.fetchall():
        t, fx, fy, fz, tx, ty, tz = p_wrench_stamped(data)
        Tn.append(t); Nfz.append(fz); Nmx.append(tx); Nmy.append(ty); Nmz.append(tz)
    Tn = np.array(Tn); Nfz = np.array(Nfz)
    Nmx = np.array(Nmx); Nmy = np.array(Nmy); Nmz = np.array(Nmz)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/hgdo/wrench'],))
    Th, Hfx, Hfy, Hfz, Htx, Hty, Htz = ([] for _ in range(7))
    for data, in c.fetchall():
        t, fx, fy, fz, tx, ty, tz = p_wrench_stamped(data)
        Th.append(t); Hfx.append(fx); Hfy.append(fy); Hfz.append(fz)
        Htx.append(tx); Hty.append(ty); Htz.append(tz)
    Th = np.array(Th); Hfz = np.array(Hfz)
    Htx = np.array(Htx); Hty = np.array(Hty); Htz = np.array(Htz)

    conn.close()

    t0 = min(T[0], Tm[0], Tc[0], Tr[0], Tn[0], Th[0])
    T -= t0; Tm -= t0; Tc -= t0; Tr -= t0; Tn -= t0; Th -= t0
    Mpz = Mpz - Mpz[0]

    return dict(T=T, PX=PX, PY=PY, PZ=PZ, R=R_, P=P_, Y=Y_,
                WX=WX, WY=WY, WZ=WZ,
                Tm=Tm, Rm=Rm, Pm=Pm, Ym=Ym,
                Mpx=Mpx, Mpy=Mpy, Mpz=Mpz,
                Tc=Tc, Fc=Fc, Mxc=Mxc, Myc=Myc, Mzc=Mzc,
                Tr=Tr, Fr=Fr, Mxr=Mxr, Myr=Myr, Mzr=Mzr,
                Tn=Tn, Nfz=Nfz, Nmx=Nmx, Nmy=Nmy, Nmz=Nmz,
                Th=Th, Htx=Htx, Hty=Hty, Htz=Htz, Hfz=Hfz)


def detect_flight(d, thrust_thresh=6.0):
    Fc = d['Fc']; Tc = d['Tc']
    above = Fc > thrust_thresh
    if not above.any():
        return (Tc[0], Tc[-1])
    i0 = np.argmax(above)
    i1 = len(above) - 1 - np.argmax(above[::-1])
    return (Tc[i0], Tc[i1])


# -------------------------------------------------------------- per-bag plot
def per_bag_plot(name, label, d, out):
    t0, t1 = detect_flight(d)

    Htx_c = np.interp(d['Tc'], d['Th'], d['Htx'])
    Hty_c = np.interp(d['Tc'], d['Th'], d['Hty'])
    Htz_c = np.interp(d['Tc'], d['Th'], d['Htz'])
    Nmx_c = np.interp(d['Tc'], d['Tn'], d['Nmx'])
    Nmy_c = np.interp(d['Tc'], d['Tn'], d['Nmy'])
    Nmz_c = np.interp(d['Tc'], d['Tn'], d['Nmz'])

    # Expected for axis-mode='y':
    #   cmd_Mx = u_nmpc_x            (mask=0)
    #   cmd_My = u_nmpc_y - τ̂_y      (mask=1, active)
    #   cmd_Mz = u_nmpc_z            (mask=0)
    resid_x = d['Mxc'] - Nmx_c                   # ≈ 0
    resid_y = d['Myc'] - (Nmy_c - Hty_c)         # ≈ 0
    resid_z = d['Mzc'] - Nmz_c                   # ≈ 0

    fig, ax = plt.subplots(8, 1, figsize=(15, 22), sharex=True)

    ax[0].plot(d['T'],  d['R'], 'tab:red',   lw=1.0, label='roll  (EKF2)')
    ax[0].plot(d['T'],  d['P'], 'tab:blue',  lw=1.0, label='pitch (EKF2)')
    ax[0].plot(d['T'],  d['Y'], 'tab:green', lw=1.0, label='yaw   (EKF2)')
    ax[0].plot(d['Tm'], d['Rm'], 'tab:red',   lw=0.6, ls='--', alpha=0.45, label='roll  (mocap)')
    ax[0].plot(d['Tm'], d['Pm'], 'tab:blue',  lw=0.6, ls='--', alpha=0.45, label='pitch (mocap)')
    ax[0].plot(d['Tm'], d['Ym'], 'tab:green', lw=0.6, ls='--', alpha=0.45, label='yaw   (mocap)')
    ax[0].axvspan(t0, t1, color='yellow', alpha=0.1, label=f'flight {t0:.1f}-{t1:.1f}s')
    ax[0].axhline(0, color='k', lw=0.4)
    ax[0].set_ylabel('deg')
    ax[0].set_title(f'{name}   Attitude (EKF2 solid / mocap dashed)   [{label}]   (axis-mode=y)')
    ax[0].legend(ncol=4, fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(d['T'], d['WX'], 'tab:red',   lw=0.6, label='wx')
    ax[1].plot(d['T'], d['WY'], 'tab:blue',  lw=0.6, label='wy (active axis)')
    ax[1].plot(d['T'], d['WZ'], 'tab:green', lw=0.6, label='wz')
    ax[1].axhline(0, color='k', lw=0.4)
    ax[1].set_ylabel('rad/s'); ax[1].set_title('Body rates')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    # τ̂: pitch is the only axis fed back; roll/yaw are spectators
    ax[2].plot(d['Th'], d['Htx']*1000, 'tab:red',   lw=0.7, label='τ̂_x  (bypassed)')
    ax[2].plot(d['Th'], d['Hty']*1000, 'tab:blue',  lw=0.9, label='τ̂_y  (FED BACK)')
    ax[2].plot(d['Th'], d['Htz']*1000, 'tab:green', lw=0.7, label='τ̂_z  (bypassed)')
    ax[2].axhline(0, color='k', lw=0.4)
    ax[2].set_ylabel('mNm'); ax[2].set_title('HGDO τ̂  —  only τ̂_y enters the NMPC command')
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

    # Pitch axis (active): cmd_My = u_nmpc_y - τ̂_y
    ax[3].plot(d['Tn'], d['Nmy']*1000, 'tab:purple', lw=0.7, label='u_nmpc_y')
    ax[3].plot(d['Tc'], d['Myc']*1000, 'tab:blue',   lw=0.7, label='cmd_My')
    ax[3].plot(d['Th'], -d['Hty']*1000, 'k',         lw=0.5, alpha=0.5, label='-τ̂_y  (eccentricity comp.)')
    ax[3].axhline(0, color='k', lw=0.4)
    ax[3].set_ylabel('mNm'); ax[3].set_title('Pitch axis (active): cmd_My = u_nmpc_y - τ̂_y')
    ax[3].legend(fontsize=9); ax[3].grid(alpha=0.3)

    # Roll: mask=0, cmd_Mx should equal u_nmpc_x
    ax[4].plot(d['Tn'], d['Nmx']*1000, 'tab:purple', lw=0.7, label='u_nmpc_x')
    ax[4].plot(d['Tc'], d['Mxc']*1000, 'tab:red',    lw=0.7, label='cmd_Mx')
    ax[4].plot(d['Th'], -d['Htx']*1000, 'k',         lw=0.4, alpha=0.4, label='-τ̂_x (NOT subtracted)')
    ax[4].axhline(0, color='k', lw=0.4)
    ax[4].set_ylabel('mNm'); ax[4].set_title('Roll axis: cmd_Mx should equal u_nmpc_x (mask=0)')
    ax[4].legend(fontsize=9); ax[4].grid(alpha=0.3)

    # Yaw: mask=0, cmd_Mz should equal u_nmpc_z
    ax[5].plot(d['Tn'], d['Nmz']*1000, 'tab:purple', lw=0.7, label='u_nmpc_z')
    ax[5].plot(d['Tc'], d['Mzc']*1000, 'tab:green',  lw=0.7, label='cmd_Mz')
    ax[5].plot(d['Th'], -d['Htz']*1000, 'k',         lw=0.4, alpha=0.4, label='-τ̂_z (NOT subtracted)')
    ax[5].axhline(0, color='k', lw=0.4)
    ax[5].set_ylabel('mNm'); ax[5].set_title('Yaw axis: cmd_Mz should equal u_nmpc_z (mask=0)')
    ax[5].legend(fontsize=9); ax[5].grid(alpha=0.3)

    ax[6].plot(d['Tc'], resid_x*1000, 'tab:red',   lw=0.5, label='cmd_Mx - u_nmpc_x          (mask=0)')
    ax[6].plot(d['Tc'], resid_y*1000, 'tab:blue',  lw=0.5, label='cmd_My - (u_nmpc_y - τ̂_y)  (mask=1)')
    ax[6].plot(d['Tc'], resid_z*1000, 'tab:green', lw=0.5, label='cmd_Mz - u_nmpc_z          (mask=0)')
    ax[6].axhline(0, color='k', lw=0.4)
    ax[6].set_ylabel('mNm')
    ax[6].set_title('Mask-identity residuals  (≈ 0 if axis-mode=y is active)')
    ax[6].legend(fontsize=9); ax[6].grid(alpha=0.3)

    ax[7].plot(d['Tc'], d['Fc'], 'tab:purple', lw=0.8, label='cmd F')
    ax[7].plot(d['Tr'], d['Fr'], 'tab:orange', lw=0.8, alpha=0.7, label='actual F')
    ax[7].plot(d['Tm'], d['Mpz']*50 + 50, 'tab:gray', lw=0.5, alpha=0.6, label='mocap Δz×50+50')
    ax[7].set_ylabel('N'); ax[7].set_xlabel('s')
    ax[7].set_title('Thrust (mocap Δz overlaid)')
    ax[7].legend(fontsize=9); ax[7].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()
    print(f'[saved] {out}')


# -------------------------------------------------------------- stats
def window_stats(d, frac_end=0.5):
    t0, t1 = detect_flight(d)
    span = t1 - t0
    t_a = t0 + (1 - frac_end) * span
    t_b = t1 - 0.5
    res = {}
    pairs = [
        ('R',  'T'),  ('P',  'T'),  ('Y',  'T'),
        ('Rm', 'Tm'), ('Pm', 'Tm'), ('Ym', 'Tm'),
        ('WX', 'T'),  ('WY', 'T'),  ('WZ', 'T'),
        ('Mxc', 'Tc'), ('Myc', 'Tc'), ('Mzc', 'Tc'),
        ('Mxr', 'Tr'), ('Myr', 'Tr'), ('Mzr', 'Tr'),
        ('Nmx', 'Tn'), ('Nmy', 'Tn'), ('Nmz', 'Tn'),
        ('Htx', 'Th'), ('Hty', 'Th'), ('Htz', 'Th'),
    ]
    for key, tname in pairs:
        T = d[tname]; V = d[key]
        m = (T >= t_a) & (T <= t_b)
        if m.sum() < 10:
            res[key] = (np.nan, np.nan)
        else:
            v = V[m]
            res[key] = (float(v.mean()), float(v.std()))
    res['win'] = (t_a, t_b)
    return res


def drift_stats(d):
    t0, t1 = detect_flight(d)
    mask = (d['Th'] >= t0 + 3.0) & (d['Th'] <= t1 - 1.0)
    if mask.sum() < 100:
        return dict(dx=np.nan, dy=np.nan, dz=np.nan)
    Tw = d['Th'][mask]
    dx = (d['Htx'][mask][-1] - d['Htx'][mask][0]) / (Tw[-1] - Tw[0])
    dy = (d['Hty'][mask][-1] - d['Hty'][mask][0]) / (Tw[-1] - Tw[0])
    dz = (d['Htz'][mask][-1] - d['Htz'][mask][0]) / (Tw[-1] - Tw[0])
    return dict(dx=dx, dy=dy, dz=dz)


def initial_convergence_y(d):
    t0, t1 = detect_flight(d)
    mask = (d['Th'] >= t0 - 0.5) & (d['Th'] <= t0 + 3.0)
    T = d['Th'][mask]; Ty = d['Hty'][mask]
    if len(T) < 10:
        return None
    t_final_a = t0 + (t1 - t0) * 0.5
    t_final_b = t1 - 0.5
    mask_f = (d['Th'] >= t_final_a) & (d['Th'] <= t_final_b)
    ty_final = float(np.mean(d['Hty'][mask_f])) if mask_f.sum() > 10 else float('nan')
    thresh = 0.1 * abs(ty_final) if abs(ty_final) > 1e-4 else 0.001
    within = np.where(np.abs(Ty - ty_final) < thresh)[0]
    t_settle = float(T[within[0]] - t0) if len(within) > 0 else float('nan')
    return dict(t_settle=t_settle, ty_final=ty_final, t_arm=t0)


def mask_identity_check(d):
    t0, t1 = detect_flight(d)
    mask = (d['Tc'] >= t0) & (d['Tc'] <= t1 - 0.5)
    if mask.sum() < 10:
        return None
    Hty_c = np.interp(d['Tc'], d['Th'], d['Hty'])
    Nmx_c = np.interp(d['Tc'], d['Tn'], d['Nmx'])
    Nmy_c = np.interp(d['Tc'], d['Tn'], d['Nmy'])
    Nmz_c = np.interp(d['Tc'], d['Tn'], d['Nmz'])
    rx = (d['Mxc'] - Nmx_c)[mask]
    ry = (d['Myc'] - (Nmy_c - Hty_c))[mask]
    rz = (d['Mzc'] - Nmz_c)[mask]
    return dict(
        rx_rms=float(np.sqrt(np.mean(rx**2))), rx_max=float(np.max(np.abs(rx))),
        ry_rms=float(np.sqrt(np.mean(ry**2))), ry_max=float(np.max(np.abs(ry))),
        rz_rms=float(np.sqrt(np.mean(rz**2))), rz_max=float(np.max(np.abs(rz))),
    )


def eccentricity_estimate(ty_final_Nm, mass_kg=3.188, g=9.81):
    """If τ̂_y is purely from CoG offset along x-body, |τ| = m·g·|dx|."""
    return abs(ty_final_Nm) / (mass_kg * g) * 1000.0   # mm


# -------------------------------------------------------------- main
def main():
    base = '/home/user/drone_control_pkgs/bag_folder'
    bags = [('pitch_test_hgdo_eps_01', 'eps_tau=0.1  (BW ~1.6 Hz)')]
    for n, label in bags:
        db = f'{base}/{n}/{n}_0.db3'
        print(f'load {n}')
        d = load(db)
        per_bag_plot(n, label, d, f'{base}/{n}_analysis.png')

        s = window_stats(d)
        ta, tb = s['win']
        print()
        print('=== Steady-state stats (last 50% of flight window) ===')
        print(f'   window: {ta:.2f}s – {tb:.2f}s')
        print(f'   pitch (EKF2)  μ={s["P"][0]:+.3f}°  σ={s["P"][1]:.3f}°')
        print(f'   pitch (mocap) μ={s["Pm"][0]:+.3f}°  σ={s["Pm"][1]:.3f}°')
        print(f'   roll  (EKF2)  μ={s["R"][0]:+.3f}°  σ={s["R"][1]:.3f}°  (constrained)')
        print(f'   yaw   (EKF2)  μ={s["Y"][0]:+.3f}°  σ={s["Y"][1]:.3f}°  (constrained)')
        print(f'   wy σ = {s["WY"][1]:.4f} rad/s   wx σ = {s["WX"][1]:.4f}   wz σ = {s["WZ"][1]:.4f}')
        print(f'   τ̂_y     μ = {s["Hty"][0]*1000:+.3f} mNm   σ = {s["Hty"][1]*1000:.3f} mNm   (active)')
        print(f'   τ̂_x     μ = {s["Htx"][0]*1000:+.3f} mNm   σ = {s["Htx"][1]*1000:.3f} mNm   (bypass)')
        print(f'   τ̂_z     μ = {s["Htz"][0]*1000:+.3f} mNm   σ = {s["Htz"][1]*1000:.3f} mNm   (bypass)')
        print(f'   cmd_My  μ = {s["Myc"][0]*1000:+.3f} mNm   σ = {s["Myc"][1]*1000:.3f} mNm')
        print(f'   u_nmpc_y μ = {s["Nmy"][0]*1000:+.3f} mNm  (should be ≈ 0 if pitch tracked perfectly)')

        r = initial_convergence_y(d)
        if r is not None:
            print()
            print('=== τ̂_y arming transient ===')
            print(f'   final τ̂_y = {r["ty_final"]*1000:+.3f} mNm  '
                  f'| settle (<10% of final) = {r["t_settle"]:.3f} s')
            print(f'   implied CoG offset (along x_body) ≈ '
                  f'{eccentricity_estimate(r["ty_final"]):.2f} mm')

        dr = drift_stats(d)
        print()
        print('=== τ̂ drift rate over flight window (mNm/s) ===')
        print(f'   dτ̂_x/dt = {dr["dx"]*1000:+.3f}  (bypass)')
        print(f'   dτ̂_y/dt = {dr["dy"]*1000:+.3f}  (active — should be ≈ 0)')
        print(f'   dτ̂_z/dt = {dr["dz"]*1000:+.3f}  (bypass)')

        mc = mask_identity_check(d)
        print()
        print('=== Mask-identity residuals ===')
        print(f'   rx (cmd_Mx - u_nmpc_x)         rms = {mc["rx_rms"]*1000:6.3f} mNm   max = {mc["rx_max"]*1000:6.3f}')
        print(f'   ry (cmd_My - (u_nmpc_y-τ̂_y))   rms = {mc["ry_rms"]*1000:6.3f} mNm   max = {mc["ry_max"]*1000:6.3f}')
        print(f'   rz (cmd_Mz - u_nmpc_z)         rms = {mc["rz_rms"]*1000:6.3f} mNm   max = {mc["rz_max"]*1000:6.3f}')


if __name__ == '__main__':
    main()
