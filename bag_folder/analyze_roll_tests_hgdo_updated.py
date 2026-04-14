#!/usr/bin/env python3
"""Analyze the *updated* HGDO roll-tests: dob_axis_mode='x' (roll only).

Compared to the pre-update runs:
  * τ̂_y, τ̂_z are still computed by HGDO and published on /hgdo/wrench,
    but are NOT subtracted from cmd → no more positive-feedback drift on
    the mechanically-constrained pitch/yaw axes.
  * τ̂_x IS subtracted → roll-axis disturbance (incl. trim bias) gets
    compensated exactly as before.

Now that /nmpc/control is logged, we can directly check the axis-mode
mask identity:

      cmd_Mx = u_nmpc_x - τ̂_x     (mask=1 on x)
      cmd_My = u_nmpc_y            (mask=0 on y)   ← pitch bypasses DOB
      cmd_Mz = u_nmpc_z            (mask=0 on z)   ← yaw   bypasses DOB

If the implementation is correct, (cmd_My - u_nmpc_y) and
(cmd_Mz - u_nmpc_z) must be ≈ 0 throughout the flight, while
(cmd_Mx - u_nmpc_x) ≈ -τ̂_x.
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

LABEL = {
    'roll_test_hgdo_eps_01_updated': 'eps_tau=0.1  (BW ~1.6 Hz)',
    'roll_test_hgdo_eps_02_updated': 'eps_tau=0.2  (BW ~0.8 Hz)',
}
COLOR = {
    'roll_test_hgdo_eps_01_updated': 'tab:red',
    'roll_test_hgdo_eps_02_updated': 'tab:blue',
}


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
    """Robust WrenchStamped parser (handles variable frame_id length).

    CDR alignment is relative to the payload start (i.e., offset 4),
    not the absolute buffer offset.
    """
    off = 4                                       # CDR encap. header
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    # Align next field (f64) to 8 within the CDR payload.
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

    # odom (filtered)
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

    # mocap
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

    # cmd_raw  → allocation → cmd moments (post-DOB)
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

    # actual_rpm → actual moments
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

    # /nmpc/control  (baseline NMPC moment, before DOB subtraction)
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/nmpc/control'],))
    Tn, Nfz, Nmx, Nmy, Nmz = ([] for _ in range(5))
    for data, in c.fetchall():
        t, fx, fy, fz, tx, ty, tz = p_wrench_stamped(data)
        Tn.append(t); Nfz.append(fz); Nmx.append(tx); Nmy.append(ty); Nmz.append(tz)
    Tn = np.array(Tn); Nfz = np.array(Nfz)
    Nmx = np.array(Nmx); Nmy = np.array(Nmy); Nmz = np.array(Nmz)

    # /hgdo/wrench
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

    Mpz = Mpz - Mpz[0]          # subtract hover-start offset for display

    return dict(T=T, PX=PX, PY=PY, PZ=PZ, R=R_, P=P_, Y=Y_,
                WX=WX, WY=WY, WZ=WZ,
                Tm=Tm, Rm=Rm, Pm=Pm, Ym=Ym,
                Mpx=Mpx, Mpy=Mpy, Mpz=Mpz,
                Tc=Tc, Fc=Fc, Mxc=Mxc, Myc=Myc, Mzc=Mzc,
                Tr=Tr, Fr=Fr, Mxr=Mxr, Myr=Myr, Mzr=Mzr,
                Tn=Tn, Nfz=Nfz, Nmx=Nmx, Nmy=Nmy, Nmz=Nmz,
                Th=Th, Htx=Htx, Hty=Hty, Htz=Htz, Hfz=Hfz)


# ------------------------------------------------------------ flight window
def detect_flight(d, thrust_thresh=6.0):
    Fc = d['Fc']; Tc = d['Tc']
    above = Fc > thrust_thresh
    if not above.any():
        return (Tc[0], Tc[-1])
    i0 = np.argmax(above)
    i1 = len(above) - 1 - np.argmax(above[::-1])
    return (Tc[i0], Tc[i1])


# ------------------------------------------------------------ per-bag plot
def per_bag_plot(name, d, out):
    t0, t1 = detect_flight(d)

    # Interpolate τ̂, u_nmpc onto cmd timebase to check mask identity
    Htx_c = np.interp(d['Tc'], d['Th'], d['Htx'])
    Hty_c = np.interp(d['Tc'], d['Th'], d['Hty'])
    Htz_c = np.interp(d['Tc'], d['Th'], d['Htz'])
    Nmx_c = np.interp(d['Tc'], d['Tn'], d['Nmx'])
    Nmy_c = np.interp(d['Tc'], d['Tn'], d['Nmy'])
    Nmz_c = np.interp(d['Tc'], d['Tn'], d['Nmz'])

    # Expected axis-mode='x' behavior:
    #    cmd_Mx = u_nmpc_x - τ̂_x     (mask=1)
    #    cmd_My = u_nmpc_y            (mask=0)
    #    cmd_Mz = u_nmpc_z            (mask=0)
    resid_x = d['Mxc'] - (Nmx_c - Htx_c)        # should be ~0
    resid_y = d['Myc'] - Nmy_c                   # should be ~0
    resid_z = d['Mzc'] - Nmz_c                   # should be ~0

    fig, ax = plt.subplots(8, 1, figsize=(15, 22), sharex=True)

    # Primary: EKF2 (mavros/local_position/odom) since that is what the
    # NMPC actually closes the loop on. Mocap overlaid (dashed, lighter)
    # for ground-truth reference.
    ax[0].plot(d['T'],  d['R'], 'tab:red',   lw=1.0, label='roll  (EKF2)')
    ax[0].plot(d['T'],  d['P'], 'tab:blue',  lw=1.0, label='pitch (EKF2)')
    ax[0].plot(d['T'],  d['Y'], 'tab:green', lw=1.0, label='yaw   (EKF2)')
    ax[0].plot(d['Tm'], d['Rm'], 'tab:red',   lw=0.6, ls='--', alpha=0.45, label='roll  (mocap)')
    ax[0].plot(d['Tm'], d['Pm'], 'tab:blue',  lw=0.6, ls='--', alpha=0.45, label='pitch (mocap)')
    ax[0].plot(d['Tm'], d['Ym'], 'tab:green', lw=0.6, ls='--', alpha=0.45, label='yaw   (mocap)')
    ax[0].axvspan(t0, t1, color='yellow', alpha=0.1, label=f'flight {t0:.1f}-{t1:.1f}s')
    ax[0].axhline(0, color='k', lw=0.4)
    ax[0].set_ylabel('deg')
    ax[0].set_title(f'{name}   Attitude  (EKF2 solid / mocap dashed)   [{LABEL[name]}]   (axis-mode=x)')
    ax[0].legend(ncol=4, fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(d['T'], d['WX'], 'tab:red',   lw=0.6, label='wx')
    ax[1].plot(d['T'], d['WY'], 'tab:blue',  lw=0.6, label='wy')
    ax[1].plot(d['T'], d['WZ'], 'tab:green', lw=0.6, label='wz')
    ax[1].axhline(0, color='k', lw=0.4)
    ax[1].set_ylabel('rad/s'); ax[1].set_title('Body rates')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    # HGDO τ̂: roll is the only axis fed back; pitch/yaw are spectators
    ax[2].plot(d['Th'], d['Htx']*1000, 'tab:red',   lw=0.7, label='τ̂_x  (fed back)')
    ax[2].plot(d['Th'], d['Hty']*1000, 'tab:blue',  lw=0.7, label='τ̂_y  (bypassed)')
    ax[2].plot(d['Th'], d['Htz']*1000, 'tab:green', lw=0.7, label='τ̂_z  (bypassed)')
    ax[2].axhline(0, color='k', lw=0.4)
    ax[2].set_ylabel('mNm'); ax[2].set_title('HGDO τ̂  —  only τ̂_x enters the NMPC command')
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

    # NMPC baseline vs cmd (roll axis)
    ax[3].plot(d['Tn'], d['Nmx']*1000, 'tab:purple', lw=0.7, label='u_nmpc_x')
    ax[3].plot(d['Tc'], d['Mxc']*1000, 'tab:red',    lw=0.7, label='cmd_Mx')
    ax[3].plot(d['Th'], -d['Htx']*1000, 'k',         lw=0.5, alpha=0.5, label='-τ̂_x')
    ax[3].axhline(0, color='k', lw=0.4)
    ax[3].set_ylabel('mNm'); ax[3].set_title('Roll axis: cmd_Mx = u_nmpc_x - τ̂_x')
    ax[3].legend(fontsize=9); ax[3].grid(alpha=0.3)

    # Pitch: mask=0, so cmd_My should equal u_nmpc_y
    ax[4].plot(d['Tn'], d['Nmy']*1000, 'tab:purple', lw=0.7, label='u_nmpc_y')
    ax[4].plot(d['Tc'], d['Myc']*1000, 'tab:blue',   lw=0.7, label='cmd_My')
    ax[4].plot(d['Th'], -d['Hty']*1000, 'k',         lw=0.4, alpha=0.4, label='-τ̂_y (NOT subtracted)')
    ax[4].axhline(0, color='k', lw=0.4)
    ax[4].set_ylabel('mNm'); ax[4].set_title('Pitch axis: cmd_My should equal u_nmpc_y (mask=0)')
    ax[4].legend(fontsize=9); ax[4].grid(alpha=0.3)

    # Yaw: mask=0, so cmd_Mz should equal u_nmpc_z
    ax[5].plot(d['Tn'], d['Nmz']*1000, 'tab:purple', lw=0.7, label='u_nmpc_z')
    ax[5].plot(d['Tc'], d['Mzc']*1000, 'tab:green',  lw=0.7, label='cmd_Mz')
    ax[5].plot(d['Th'], -d['Htz']*1000, 'k',         lw=0.4, alpha=0.4, label='-τ̂_z (NOT subtracted)')
    ax[5].axhline(0, color='k', lw=0.4)
    ax[5].set_ylabel('mNm'); ax[5].set_title('Yaw axis: cmd_Mz should equal u_nmpc_z (mask=0)')
    ax[5].legend(fontsize=9); ax[5].grid(alpha=0.3)

    # Mask identity residuals (should be ≈ 0 throughout)
    ax[6].plot(d['Tc'], resid_x*1000, 'tab:red',   lw=0.5, label='cmd_Mx - (u_nmpc_x - τ̂_x)')
    ax[6].plot(d['Tc'], resid_y*1000, 'tab:blue',  lw=0.5, label='cmd_My - u_nmpc_y')
    ax[6].plot(d['Tc'], resid_z*1000, 'tab:green', lw=0.5, label='cmd_Mz - u_nmpc_z')
    ax[6].axhline(0, color='k', lw=0.4)
    ax[6].set_ylabel('mNm')
    ax[6].set_title('Mask-identity residuals   (should be ≈ 0 if axis-mode=x is active)')
    ax[6].legend(fontsize=9); ax[6].grid(alpha=0.3)

    # thrust & mocap position
    ax[7].plot(d['Tc'], d['Fc'], 'tab:purple', lw=0.8, label='cmd F')
    ax[7].plot(d['Tr'], d['Fr'], 'tab:orange', lw=0.8, alpha=0.7, label='actual F')
    ax[7].plot(d['Tm'], d['Mpz']*50 + 50, 'tab:gray', lw=0.5, alpha=0.6, label='mocap Δz×50 + 50 (N-scale)')
    ax[7].set_ylabel('N'); ax[7].set_xlabel('s')
    ax[7].set_title('Thrust (and mocap Δz overlaid)')
    ax[7].legend(fontsize=9); ax[7].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()
    print(f'[saved] {out}')


# ---------------------------------------------------------------- stats
def window_stats(d, frac_end=0.5):
    t0, t1 = detect_flight(d)
    span = t1 - t0
    t_a = t0 + (1 - frac_end) * span
    t_b = t1 - 0.5
    res = {}
    pairs = [
        ('R',  'T'),  ('P',  'T'),  ('Y',  'T'),     # EKF2 (what NMPC sees)
        ('Rm', 'Tm'), ('Pm', 'Tm'), ('Ym', 'Tm'),    # mocap (ground truth)
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
    """Rate of change of τ̂ on each axis through the flight window."""
    t0, t1 = detect_flight(d)
    mask = (d['Th'] >= t0 + 3.0) & (d['Th'] <= t1 - 1.0)
    if mask.sum() < 100:
        return dict(dx=np.nan, dy=np.nan, dz=np.nan)
    Tw = d['Th'][mask]
    dx = (d['Htx'][mask][-1] - d['Htx'][mask][0]) / (Tw[-1] - Tw[0])
    dy = (d['Hty'][mask][-1] - d['Hty'][mask][0]) / (Tw[-1] - Tw[0])
    dz = (d['Htz'][mask][-1] - d['Htz'][mask][0]) / (Tw[-1] - Tw[0])
    return dict(dx=dx, dy=dy, dz=dz)


def initial_convergence(d):
    t0, t1 = detect_flight(d)
    mask = (d['Th'] >= t0 - 0.5) & (d['Th'] <= t0 + 3.0)
    T = d['Th'][mask]; Tx = d['Htx'][mask]
    if len(T) < 10:
        return None
    t_final_a = t0 + (t1 - t0) * 0.5
    t_final_b = t1 - 0.5
    mask_f = (d['Th'] >= t_final_a) & (d['Th'] <= t_final_b)
    tx_final = float(np.mean(d['Htx'][mask_f])) if mask_f.sum() > 10 else float('nan')
    thresh = 0.1 * abs(tx_final) if abs(tx_final) > 1e-4 else 0.001
    within = np.where(np.abs(Tx - tx_final) < thresh)[0]
    t_settle = float(T[within[0]] - t0) if len(within) > 0 else float('nan')
    return dict(t_settle=t_settle, tx_final=tx_final, t_arm=t0)


def mask_identity_check(d):
    """Return max and RMS of the three residuals over the flight window."""
    t0, t1 = detect_flight(d)
    mask = (d['Tc'] >= t0) & (d['Tc'] <= t1 - 0.5)
    if mask.sum() < 10:
        return None
    Htx_c = np.interp(d['Tc'], d['Th'], d['Htx'])
    Nmx_c = np.interp(d['Tc'], d['Tn'], d['Nmx'])
    Nmy_c = np.interp(d['Tc'], d['Tn'], d['Nmy'])
    Nmz_c = np.interp(d['Tc'], d['Tn'], d['Nmz'])
    rx = (d['Mxc'] - (Nmx_c - Htx_c))[mask]
    ry = (d['Myc'] - Nmy_c)[mask]
    rz = (d['Mzc'] - Nmz_c)[mask]
    return dict(
        rx_rms=float(np.sqrt(np.mean(rx**2))), rx_max=float(np.max(np.abs(rx))),
        ry_rms=float(np.sqrt(np.mean(ry**2))), ry_max=float(np.max(np.abs(ry))),
        rz_rms=float(np.sqrt(np.mean(rz**2))), rz_max=float(np.max(np.abs(rz))),
    )


# ------------------------------------------------------------- compare plot
def compare_plot(data, out):
    fig, ax = plt.subplots(6, 1, figsize=(15, 20), sharex=False)

    ax[0].set_title('Roll (EKF2 solid / mocap dashed)    (axis-mode=x for both)')
    for n, d in data.items():
        ax[0].plot(d['T'],  d['R'],  color=COLOR[n], lw=0.9, label=f'{n}  [{LABEL[n]}]  EKF2')
        ax[0].plot(d['Tm'], d['Rm'], color=COLOR[n], lw=0.6, ls='--', alpha=0.45, label=f'{n} mocap')
    ax[0].axhline(0, color='k', lw=0.4); ax[0].set_ylabel('roll (deg)')
    ax[0].legend(fontsize=8, ncol=2); ax[0].grid(alpha=0.3)

    ax[1].set_title('Body roll rate wx')
    for n, d in data.items():
        ax[1].plot(d['T'], d['WX'], color=COLOR[n], lw=0.5, label=n)
    ax[1].axhline(0, color='k', lw=0.4); ax[1].set_ylabel('wx (rad/s)')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    ax[2].set_title('HGDO τ̂_x  (only axis fed back to cmd)')
    for n, d in data.items():
        ax[2].plot(d['Th'], d['Htx']*1000, color=COLOR[n], lw=0.7, label=n)
    ax[2].axhline(0, color='k', lw=0.4); ax[2].set_ylabel('τ̂_x (mNm)')
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

    ax[3].set_title('HGDO τ̂_y, τ̂_z  (NOT fed back — should still drift since pitch/yaw constrained)')
    for n, d in data.items():
        ax[3].plot(d['Th'], d['Hty']*1000, color=COLOR[n], lw=0.5, ls='-',  label=f'{n} τ̂_y')
        ax[3].plot(d['Th'], d['Htz']*1000, color=COLOR[n], lw=0.5, ls='--', label=f'{n} τ̂_z')
    ax[3].axhline(0, color='k', lw=0.4); ax[3].set_ylabel('mNm')
    ax[3].legend(fontsize=8, ncol=2); ax[3].grid(alpha=0.3)

    ax[4].set_title('Commanded Mx  (= u_nmpc_x - τ̂_x)')
    for n, d in data.items():
        ax[4].plot(d['Tc'], d['Mxc']*1000, color=COLOR[n], lw=0.5, label=n)
    ax[4].axhline(0, color='k', lw=0.4); ax[4].set_ylabel('cmd Mx (mNm)')
    ax[4].legend(fontsize=9); ax[4].grid(alpha=0.3)

    ax[5].set_title('PSD during flight (resampled 100 Hz)')
    for n, d in data.items():
        t0, t1 = detect_flight(d)
        fs = 100.0
        tg = np.arange(t0 + 1.0, t1 - 0.5, 1/fs)
        if len(tg) < 256: continue
        wxg  = np.interp(tg, d['T'],  d['WX'])
        htxg = np.interp(tg, d['Th'], d['Htx'])
        mxcg = np.interp(tg, d['Tc'], d['Mxc'])
        nperseg = min(1024, len(tg)//2)
        f_wx, P_wx   = welch(wxg,  fs=fs, nperseg=nperseg)
        f_tx, P_tx   = welch(htxg, fs=fs, nperseg=nperseg)
        f_mxc, P_mxc = welch(mxcg, fs=fs, nperseg=nperseg)
        ax[5].semilogy(f_wx,  P_wx,  color=COLOR[n], lw=1.0, ls='-',  label=f'{n} wx')
        ax[5].semilogy(f_tx,  P_tx,  color=COLOR[n], lw=1.0, ls='--', alpha=0.8, label=f'{n} τ̂_x')
        ax[5].semilogy(f_mxc, P_mxc, color=COLOR[n], lw=1.0, ls=':',  alpha=0.8, label=f'{n} cmd Mx')
    ax[5].set_xlim(0, 50); ax[5].set_xlabel('Hz'); ax[5].set_ylabel('PSD')
    ax[5].legend(fontsize=8, ncol=2); ax[5].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()
    print(f'[saved] {out}')


# ------------------------------------------------------------------ main
def main():
    base = '/home/user/drone_control_pkgs/bag_folder'
    bags = ['roll_test_hgdo_eps_01_updated', 'roll_test_hgdo_eps_02_updated']
    data = {}
    for n in bags:
        db = f'{base}/{n}/{n}_0.db3'
        print(f'load {n}')
        data[n] = load(db)
        per_bag_plot(n, data[n], f'{base}/{n}_analysis.png')
    compare_plot(data, f'{base}/roll_tests_hgdo_updated_compare.png')

    # Steady-state stats
    print()
    print('=== Steady-state stats (last 50% of flight window) ===')
    print('   ekf2 = roll from /mavros/local_position/odom (closed-loop feedback)')
    print('   mocap = /S550/pose (ground truth reference)')
    hdr = (f'{"bag":<34} | {"win":<13} | '
           f'{"rollE2_μ":>9} {"rollE2_σ":>9} | '
           f'{"rollMC_μ":>9} {"rollMC_σ":>9} | '
           f'{"wx_σ":>6} | '
           f'{"τ̂x_μ(mNm)":>10} {"τ̂x_σ(mNm)":>10} | '
           f'{"cmdMx_μ":>9} {"cmdMx_σ":>9}')
    print(hdr); print('-' * len(hdr))
    for n, d in data.items():
        s = window_stats(d)
        ta, tb = s['win']
        print(f'{n:<34} | {ta:5.1f}-{tb:5.1f}s | '
              f'{s["R"][0]:9.3f} {s["R"][1]:9.3f} | '
              f'{s["Rm"][0]:9.3f} {s["Rm"][1]:9.3f} | '
              f'{s["WX"][1]:6.3f} | '
              f'{s["Htx"][0]*1000:10.3f} {s["Htx"][1]*1000:10.3f} | '
              f'{s["Mxc"][0]*1000:9.3f} {s["Mxc"][1]*1000:9.3f}')

    # Convergence
    print()
    print('=== τ̂_x arming transient ===')
    for n, d in data.items():
        r = initial_convergence(d)
        if r is None: continue
        print(f'{n:<34} | final τ̂_x = {r["tx_final"]*1000:7.3f} mNm | '
              f'settle(<10% of final) = {r["t_settle"]:.3f}s')

    # Drift rates (key claim: τ̂_x is stable; τ̂_y,τ̂_z may still drift
    # because pitch/yaw are still constrained, but they no longer
    # contaminate the closed loop.)
    print()
    print('=== τ̂ drift rate over flight window (mNm/s) ===')
    for n, d in data.items():
        r = drift_stats(d)
        print(f'{n:<34} | dτ̂_x/dt = {r["dx"]*1000:+.3f} | '
              f'dτ̂_y/dt = {r["dy"]*1000:+.3f} | dτ̂_z/dt = {r["dz"]*1000:+.3f}')

    # Mask-identity check
    print()
    print('=== Mask-identity residuals over flight window ===')
    print('( cmd - expected , where expected uses mask=[1,0,0]; should be ~0 )')
    for n, d in data.items():
        r = mask_identity_check(d)
        if r is None: continue
        print(f'{n:<34} | '
              f'rx rms/max = {r["rx_rms"]*1000:6.3f}/{r["rx_max"]*1000:6.3f} mNm | '
              f'ry rms/max = {r["ry_rms"]*1000:6.3f}/{r["ry_max"]*1000:6.3f} mNm | '
              f'rz rms/max = {r["rz_rms"]*1000:6.3f}/{r["rz_max"]*1000:6.3f} mNm')


if __name__ == '__main__':
    main()
