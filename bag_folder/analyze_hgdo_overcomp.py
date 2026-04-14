#!/usr/bin/env python3
"""Investigate HGDO over-compensation of disturbance torques.

User observation: "보니까 hgdo에서 Tau z를 과하게 보상하고 있습니다..."
                 (HGDO is over-compensating Tau_z)

Hypothesis tree
---------------
The DOB compensation is: cmd = u_nmpc - tau_hat
If HGDO is correct (tau_hat = tau_dist), in hovered steady state:
   actual_moment = cmd = -tau_hat = -tau_dist
So the commanded/actual moments should match the -tau_hat value.

From earlier no-DOB baselines (tests 04-06, true trim moments):
   Mx ≈ -15 mNm, My ≈ +5 mNm, Mz ≈ +5 mNm
=> true tau_dist  ≈ [+15, -5, -5] mNm  (negatives since drone needs
   opposing control to stay level).

Thus a properly-tuned HGDO should converge to tau_hat ≈ [+15, -5, -5] mNm.
Anything substantially larger means the observer itself is biased.

This script:
  1. Parses /hgdo/wrench to get tau_hat directly.
  2. Parses actual RPM -> actual moments.
  3. Parses cmd_raw -> commanded moments.
  4. Computes the implied u_nmpc = cmd + tau_hat  (what NMPC wanted).
  5. Reports steady-state means + identifies over-compensation.
  6. Checks gyro bias contribution (J*w_bias/eps_tau term in tau_hat).
"""

import sqlite3, struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Allocation constants (same as other scripts)
C_T = 1.386e-07
k_m = 0.01569
l = 0.265
MaxBit = 8191
MaxRpm = 9800

# Drone MoI (from hgdo_config.yaml)
J = np.diag([0.060, 0.060, 0.080])
# HGDO eps_tau per bag
EPS_TAU = {
    'roll_test_hgdo_eps_01': 0.1,
    'roll_test_hgdo_eps_02': 0.2,
}

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
    wx, wy, wz = struct.unpack_from('<3d', data, 412)
    return sec + nsec*1e-9, wx, wy, wz

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

    T, WX, WY, WZ = [], [], [], []
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/mavros/local_position/odom'],))
    for data, in c.fetchall():
        t, wx, wy, wz = p_odom(data)
        T.append(t); WX.append(wx); WY.append(wy); WZ.append(wz)

    Tc, Fc, Mxc, Myc, Mzc = [], [], [], [], []
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/uav/cmd_raw'],))
    for data, in c.fetchall():
        t, cmds = p_cmd(data)
        rpms = cmds * MaxRpm / MaxBit
        thr = C_T * rpms**2
        u = K_forward @ thr
        Tc.append(t); Fc.append(u[0]); Mxc.append(u[1]); Myc.append(u[2]); Mzc.append(u[3])

    Tr, Fr, Mxr, Myr, Mzr = [], [], [], [], []
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/uav/actual_rpm'],))
    for data, in c.fetchall():
        t, rpms = p_rpm(data)
        thr = C_T * rpms**2
        u = K_forward @ thr
        Tr.append(t); Fr.append(u[0]); Mxr.append(u[1]); Myr.append(u[2]); Mzr.append(u[3])

    Th, Hfx, Hfy, Hfz, Htx, Hty, Htz = [], [], [], [], [], [], []
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/hgdo/wrench'],))
    for data, in c.fetchall():
        t, fx, fy, fz, tx, ty, tz = p_wrench(data)
        Th.append(t); Hfx.append(fx); Hfy.append(fy); Hfz.append(fz)
        Htx.append(tx); Hty.append(ty); Htz.append(tz)

    conn.close()

    T  = np.asarray(T);  WX = np.asarray(WX);  WY = np.asarray(WY);  WZ = np.asarray(WZ)
    Tc = np.asarray(Tc); Fc = np.asarray(Fc); Mxc= np.asarray(Mxc); Myc= np.asarray(Myc); Mzc= np.asarray(Mzc)
    Tr = np.asarray(Tr); Fr = np.asarray(Fr); Mxr= np.asarray(Mxr); Myr= np.asarray(Myr); Mzr= np.asarray(Mzr)
    Th = np.asarray(Th); Htx= np.asarray(Htx); Hty= np.asarray(Hty); Htz= np.asarray(Htz)
    Hfx= np.asarray(Hfx); Hfy= np.asarray(Hfy); Hfz= np.asarray(Hfz)

    t0 = min(T[0], Tc[0], Tr[0], Th[0])
    T -= t0; Tc -= t0; Tr -= t0; Th -= t0
    return dict(T=T, WX=WX, WY=WY, WZ=WZ,
                Tc=Tc, Fc=Fc, Mxc=Mxc, Myc=Myc, Mzc=Mzc,
                Tr=Tr, Fr=Fr, Mxr=Mxr, Myr=Myr, Mzr=Mzr,
                Th=Th, Htx=Htx, Hty=Hty, Htz=Htz,
                Hfx=Hfx, Hfy=Hfy, Hfz=Hfz)


def detect_flight(d, thrust_thresh=6.0):
    Fc = d['Fc']; Tc = d['Tc']
    above = Fc > thrust_thresh
    if not above.any():
        return (Tc[0], Tc[-1])
    i0 = np.argmax(above); i1 = len(above)-1 - np.argmax(above[::-1])
    return (Tc[i0], Tc[i1])


def window_mean_std(T, V, ta, tb):
    m = (T >= ta) & (T <= tb)
    if m.sum() < 10:
        return (np.nan, np.nan, 0)
    v = V[m]
    return float(v.mean()), float(v.std()), int(m.sum())


def main():
    base = '/home/user/drone_control_pkgs/bag_folder'
    bags = ['roll_test_hgdo_eps_01', 'roll_test_hgdo_eps_02']

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))

    print('=== HGDO over-compensation diagnostic ===')
    print()
    print(f'{"bag":<28} | eps | {"tauHat_x(mNm)":>13} {"tauHat_y(mNm)":>13} {"tauHat_z(mNm)":>13} | {"actMx(mNm)":>11} {"actMy(mNm)":>11} {"actMz(mNm)":>11} | {"cmdMx(mNm)":>11} {"cmdMy(mNm)":>11} {"cmdMz(mNm)":>11} | {"u_nmpc_y(mNm)":>13} {"u_nmpc_z(mNm)":>13}')
    print('-' * 220)

    table_rows = []

    for col, name in enumerate(bags):
        d = load(f'{base}/{name}/{name}_0.db3')
        t0, t1 = detect_flight(d)
        # pick middle third of flight (past arming transient, before disarm)
        span = t1 - t0
        ta = t0 + 0.35 * span
        tb = t0 + 0.90 * span
        eps = EPS_TAU[name]

        htx_m, htx_s, _ = window_mean_std(d['Th'], d['Htx'], ta, tb)
        hty_m, hty_s, _ = window_mean_std(d['Th'], d['Hty'], ta, tb)
        htz_m, htz_s, _ = window_mean_std(d['Th'], d['Htz'], ta, tb)
        mxr_m, mxr_s, _ = window_mean_std(d['Tr'], d['Mxr'], ta, tb)
        myr_m, myr_s, _ = window_mean_std(d['Tr'], d['Myr'], ta, tb)
        mzr_m, mzr_s, _ = window_mean_std(d['Tr'], d['Mzr'], ta, tb)
        mxc_m, mxc_s, _ = window_mean_std(d['Tc'], d['Mxc'], ta, tb)
        myc_m, myc_s, _ = window_mean_std(d['Tc'], d['Myc'], ta, tb)
        mzc_m, mzc_s, _ = window_mean_std(d['Tc'], d['Mzc'], ta, tb)
        wxm, wxs, _ = window_mean_std(d['T'], d['WX'], ta, tb)
        wym, wys, _ = window_mean_std(d['T'], d['WY'], ta, tb)
        wzm, wzs, _ = window_mean_std(d['T'], d['WZ'], ta, tb)

        # u_nmpc = cmd + tau_hat  (cmd = u_nmpc - tau_hat => u_nmpc = cmd + tau_hat)
        u_nmpc_y = myc_m + hty_m
        u_nmpc_z = mzc_m + htz_m

        print(f'{name:<28} | {eps:.1f} | {htx_m*1000:13.3f} {hty_m*1000:13.3f} {htz_m*1000:13.3f} | '
              f'{mxr_m*1000:11.3f} {myr_m*1000:11.3f} {mzr_m*1000:11.3f} | '
              f'{mxc_m*1000:11.3f} {myc_m*1000:11.3f} {mzc_m*1000:11.3f} | '
              f'{u_nmpc_y*1000:13.3f} {u_nmpc_z*1000:13.3f}')

        table_rows.append(dict(name=name, eps=eps,
                               htx=htx_m, hty=hty_m, htz=htz_m,
                               mxr=mxr_m, myr=myr_m, mzr=mzr_m,
                               mxc=mxc_m, myc=myc_m, mzc=mzc_m,
                               wxm=wxm, wym=wym, wzm=wzm,
                               wxs=wxs, wys=wys, wzs=wzs,
                               htx_s=htx_s, hty_s=hty_s, htz_s=htz_s,
                               ta=ta, tb=tb))

        # Plot per-bag tau_hat vs actual moments
        ax = axes[0, col]
        ax.plot(d['Th'], d['Htx']*1000, 'tab:red', lw=0.7, label='τ̂_x  (HGDO)')
        ax.plot(d['Tr'], d['Mxr']*1000, 'tab:red', lw=0.4, alpha=0.5, ls='--', label='actual Mx')
        ax.plot(d['Tr'], -d['Mxr']*1000, 'k', lw=0.4, alpha=0.3, ls=':', label='-actual Mx  (ideal τ̂)')
        ax.axvspan(ta, tb, color='yellow', alpha=0.1)
        ax.axhline(0, color='k', lw=0.4)
        ax.set_ylabel('mNm'); ax.set_title(f'{name}  τ̂_x vs applied Mx'); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        ax.plot(d['Th'], d['Hty']*1000, 'tab:blue', lw=0.7, label='τ̂_y (HGDO)')
        ax.plot(d['Tr'], d['Myr']*1000, 'tab:blue', lw=0.4, alpha=0.5, ls='--', label='actual My')
        ax.plot(d['Tr'], -d['Myr']*1000, 'k', lw=0.4, alpha=0.3, ls=':', label='-actual My (ideal τ̂)')
        ax.axvspan(ta, tb, color='yellow', alpha=0.1)
        ax.axhline(0, color='k', lw=0.4)
        ax.set_ylabel('mNm'); ax.set_title(f'{name}  τ̂_y vs applied My'); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2, col]
        ax.plot(d['Th'], d['Htz']*1000, 'tab:green', lw=0.7, label='τ̂_z (HGDO)')
        ax.plot(d['Tr'], d['Mzr']*1000, 'tab:green', lw=0.4, alpha=0.5, ls='--', label='actual Mz')
        ax.plot(d['Tr'], -d['Mzr']*1000, 'k', lw=0.4, alpha=0.3, ls=':', label='-actual Mz (ideal τ̂)')
        ax.axvspan(ta, tb, color='yellow', alpha=0.1)
        ax.axhline(0, color='k', lw=0.4)
        ax.set_ylabel('mNm'); ax.set_title(f'{name}  τ̂_z vs applied Mz'); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # gyro bias contribution: J*w/eps_tau
        ax = axes[3, col]
        wy_contrib = J[1,1] * d['WY'] / eps * 1000
        wz_contrib = J[2,2] * d['WZ'] / eps * 1000
        wx_contrib = J[0,0] * d['WX'] / eps * 1000
        ax.plot(d['T'], wx_contrib, 'tab:red', lw=0.5, alpha=0.6, label=f'Jxx·wx/ε (max={np.max(np.abs(wx_contrib)):.1f})')
        ax.plot(d['T'], wy_contrib, 'tab:blue', lw=0.5, alpha=0.6, label=f'Jyy·wy/ε (max={np.max(np.abs(wy_contrib)):.1f})')
        ax.plot(d['T'], wz_contrib, 'tab:green', lw=0.5, alpha=0.6, label=f'Jzz·wz/ε (max={np.max(np.abs(wz_contrib)):.1f})')
        ax.axhline(0, color='k', lw=0.4)
        ax.set_ylabel('mNm'); ax.set_xlabel('s')
        ax.set_title(f'{name}  J·w/ε_tau component of τ̂ (gain = {1.0/eps:.0f})')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = f'{base}/roll_tests_hgdo_overcomp.png'
    plt.savefig(out, dpi=110)
    plt.close()
    print(f'\n[saved] {out}')

    # Print interpretation
    print()
    print('=== Interpretation ===')
    print('If HGDO worked correctly (unbiased observer), steady-state τ̂ should satisfy:')
    print('   τ̂ ≈ -(actual moment)     (cmd = u_nmpc - τ̂ ≈ -τ̂ when u_nmpc→0 at zero attitude error)')
    print()
    print('True trim from no-DOB runs (tests 04/05/06, middle of flight):')
    print('   actual Mx ≈ -10..-20 mNm → true τ_dist_x ≈ +10..+20 mNm')
    print('   actual My ≈   +4..+5 mNm → true τ_dist_y ≈ -4..-5 mNm')
    print('   actual Mz ≈   +2..+5 mNm → true τ_dist_z ≈ -2..-5 mNm')
    print()
    for r in table_rows:
        print(f'  {r["name"]}  (eps_tau={r["eps"]:.1f}):')
        print(f'    τ̂_x   = {r["htx"]*1000:+8.2f} mNm   | -actMx = {-r["mxr"]*1000:+8.2f} mNm  | residual = {(r["htx"]+r["mxr"])*1000:+8.2f} mNm')
        print(f'    τ̂_y   = {r["hty"]*1000:+8.2f} mNm   | -actMy = {-r["myr"]*1000:+8.2f} mNm  | residual = {(r["hty"]+r["myr"])*1000:+8.2f} mNm')
        print(f'    τ̂_z   = {r["htz"]*1000:+8.2f} mNm   | -actMz = {-r["mzr"]*1000:+8.2f} mNm  | residual = {(r["htz"]+r["mzr"])*1000:+8.2f} mNm')
        # implied u_nmpc = cmd + tau_hat
        u_nmpc_x = r["mxc"] + r["htx"]
        u_nmpc_y = r["myc"] + r["hty"]
        u_nmpc_z = r["mzc"] + r["htz"]
        print(f'    Implied u_nmpc = cmd + τ̂   :  x={u_nmpc_x*1000:+7.2f}  y={u_nmpc_y*1000:+7.2f}  z={u_nmpc_z*1000:+7.2f}  mNm  (should be ~0 in perfect steady state)')
        # gyro-bias contribution to tau_hat = J*w_mean/eps_tau  (DC component only)
        bias_x = J[0,0] * r["wxm"] / r["eps"] * 1000
        bias_y = J[1,1] * r["wym"] / r["eps"] * 1000
        bias_z = J[2,2] * r["wzm"] / r["eps"] * 1000
        print(f'    J*w_mean/ε_tau (DC drift term):  x={bias_x:+7.2f}  y={bias_y:+7.2f}  z={bias_z:+7.2f}  mNm   (w_mean=[{r["wxm"]:+.4f},{r["wym"]:+.4f},{r["wzm"]:+.4f}] rad/s)')
        print()


if __name__ == '__main__':
    main()
