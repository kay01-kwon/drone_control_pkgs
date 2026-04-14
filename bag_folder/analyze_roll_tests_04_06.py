#!/usr/bin/env python3
"""Analyze roll_test_04/05/06: NMPC only, NO external disturbance.

User reports steady-state convergence is poor.

Weights:
  test_04: Q_att=[3.0,3.0,1.0], Q_rate=[0.2,0.2,0.25], R=5.0
  test_05: Q_att=[2.0,2.0,1.0], Q_rate=[0.8,0.8,0.25], R=5.0
  test_06: Q_att=[2.0,2.0,1.0], Q_rate=[0.01,0.01,0.025], R=5.0
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

WEIGHTS = {
    'roll_test_04': 'Qatt=[3,3,1]  Qrate=[0.2,0.2,0.25]  R=5',
    'roll_test_05': 'Qatt=[2,2,1]  Qrate=[0.8,0.8,0.25]  R=5',
    'roll_test_06': 'Qatt=[2,2,1]  Qrate=[0.01,0.01,0.025]  R=5',
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


def load(db):
    conn = sqlite3.connect(db); c = conn.cursor()
    c.execute('SELECT id, name FROM topics')
    topics = {n: tid for tid, n in c.fetchall()}

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/mavros/local_position/odom'],))
    T, PX, PY, PZ, R_, P_, Y_, WX, WY, WZ = [], [], [], [], [], [], [], [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = p_odom(data)
        q = np.array([qx,qy,qz,qw]); nn = np.linalg.norm(q)
        if not np.isfinite(nn) or nn < 1e-10: continue
        r, p, y = Rotation.from_quat(q/nn).as_euler('xyz', degrees=True)
        T.append(t); PX.append(px); PY.append(py); PZ.append(pz)
        R_.append(r); P_.append(p); Y_.append(y)
        WX.append(wx); WY.append(wy); WZ.append(wz)
    T = np.array(T); PX = np.array(PX); PY = np.array(PY); PZ = np.array(PZ)
    R_ = np.array(R_); P_ = np.array(P_); Y_ = np.array(Y_)
    WX = np.array(WX); WY = np.array(WY); WZ = np.array(WZ)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/S550/pose'],))
    Tm, Rm, Pm, Ym, Mpx, Mpy, Mpz = [], [], [], [], [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw = p_pose(data)
        q = np.array([qx,qy,qz,qw]); nn = np.linalg.norm(q)
        if not np.isfinite(nn) or nn < 1e-10: continue
        r, p, y = Rotation.from_quat(q/nn).as_euler('xyz', degrees=True)
        Tm.append(t); Rm.append(r); Pm.append(p); Ym.append(y)
        Mpx.append(px); Mpy.append(py); Mpz.append(pz)
    Tm = np.array(Tm); Rm = np.array(Rm); Pm = np.array(Pm); Ym = np.array(Ym)
    Mpx = np.array(Mpx); Mpy = np.array(Mpy); Mpz = np.array(Mpz)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/uav/cmd_raw'],))
    Tc, Fc, Mxc, Myc, Mzc = [], [], [], [], []
    for data, in c.fetchall():
        t, cmds = p_cmd(data)
        rpms = cmds * MaxRpm / MaxBit
        thr = C_T * rpms**2
        u = K_forward @ thr
        Tc.append(t); Fc.append(u[0]); Mxc.append(u[1]); Myc.append(u[2]); Mzc.append(u[3])
    Tc = np.array(Tc); Fc = np.array(Fc); Mxc = np.array(Mxc); Myc = np.array(Myc); Mzc = np.array(Mzc)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/uav/actual_rpm'],))
    Tr, Fr, Mxr, Myr, Mzr = [], [], [], [], []
    for data, in c.fetchall():
        t, rpms = p_rpm(data)
        thr = C_T * rpms**2
        u = K_forward @ thr
        Tr.append(t); Fr.append(u[0]); Mxr.append(u[1]); Myr.append(u[2]); Mzr.append(u[3])
    Tr = np.array(Tr); Fr = np.array(Fr); Mxr = np.array(Mxr); Myr = np.array(Myr); Mzr = np.array(Mzr)

    conn.close()

    t0 = min(T[0], Tm[0], Tc[0], Tr[0])
    T -= t0; Tm -= t0; Tc -= t0; Tr -= t0

    Mpz0 = Mpz[0]; Mpz = Mpz - Mpz0
    return dict(T=T, PX=PX, PY=PY, PZ=PZ, R=R_, P=P_, Y=Y_, WX=WX, WY=WY, WZ=WZ,
                Tm=Tm, Rm=Rm, Pm=Pm, Ym=Ym, Mpx=Mpx, Mpy=Mpy, Mpz=Mpz,
                Tc=Tc, Fc=Fc, Mxc=Mxc, Myc=Myc, Mzc=Mzc,
                Tr=Tr, Fr=Fr, Mxr=Mxr, Myr=Myr, Mzr=Mzr)


def detect_flight(d, thrust_thresh=6.0):
    """Return (t_start, t_end) for hover flight (cmd_F > thresh)."""
    Fc = d['Fc']; Tc = d['Tc']
    above = Fc > thrust_thresh
    if not above.any():
        return (Tc[0], Tc[-1])
    i0 = np.argmax(above); i1 = len(above) - 1 - np.argmax(above[::-1])
    return (Tc[i0], Tc[i1])


def per_bag_plot(name, d, out):
    t0, t1 = detect_flight(d)
    # 6-row per bag
    fig, ax = plt.subplots(6, 1, figsize=(14, 18), sharex=True)

    ax[0].plot(d['Tm'], d['Rm'], 'tab:red', lw=1.0, label='roll (mocap)')
    ax[0].plot(d['Tm'], d['Pm'], 'tab:blue', lw=1.0, label='pitch (mocap)')
    ax[0].plot(d['Tm'], d['Ym'], 'tab:green', lw=1.0, label='yaw (mocap)')
    ax[0].axvspan(t0, t1, color='yellow', alpha=0.1, label=f'flight {t0:.1f}-{t1:.1f}s')
    ax[0].axhline(0, color='k', lw=0.4)
    ax[0].set_ylabel('deg')
    ax[0].set_title(f'{name}  Attitude (mocap)   [{WEIGHTS[name]}]')
    ax[0].legend(ncol=4, fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(d['T'], d['WX'], 'tab:red', lw=0.6, label='wx')
    ax[1].plot(d['T'], d['WY'], 'tab:blue', lw=0.6, label='wy')
    ax[1].plot(d['T'], d['WZ'], 'tab:green', lw=0.6, label='wz')
    ax[1].axhline(0, color='k', lw=0.4)
    ax[1].set_ylabel('rad/s'); ax[1].set_title('Body rates')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    ax[2].plot(d['Tc'], d['Mxc'], 'tab:red', lw=0.6, label='cmd Mx')
    ax[2].plot(d['Tc'], d['Myc'], 'tab:blue', lw=0.6, label='cmd My')
    ax[2].plot(d['Tc'], d['Mzc'], 'tab:green', lw=0.6, label='cmd Mz')
    ax[2].axhline(0, color='k', lw=0.4)
    ax[2].set_ylabel('Nm'); ax[2].set_title('Commanded moments')
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

    ax[3].plot(d['Tr'], d['Mxr'], 'tab:red', lw=0.6, label='act Mx')
    ax[3].plot(d['Tr'], d['Myr'], 'tab:blue', lw=0.6, label='act My')
    ax[3].plot(d['Tr'], d['Mzr'], 'tab:green', lw=0.6, label='act Mz')
    ax[3].axhline(0, color='k', lw=0.4)
    ax[3].set_ylabel('Nm'); ax[3].set_title('Actual moments')
    ax[3].legend(fontsize=9); ax[3].grid(alpha=0.3)

    ax[4].plot(d['Tc'], d['Fc'], 'tab:purple', lw=0.8, label='cmd F')
    ax[4].plot(d['Tr'], d['Fr'], 'tab:orange', lw=0.8, alpha=0.7, label='act F')
    ax[4].set_ylabel('N'); ax[4].set_title('Thrust')
    ax[4].legend(fontsize=9); ax[4].grid(alpha=0.3)

    ax[5].plot(d['Tm'], d['Mpx'], 'tab:red', lw=0.8, label='x')
    ax[5].plot(d['Tm'], d['Mpy'], 'tab:blue', lw=0.8, label='y')
    ax[5].plot(d['Tm'], d['Mpz'], 'tab:green', lw=0.8, label='z')
    ax[5].set_ylabel('m'); ax[5].set_xlabel('s'); ax[5].set_title('Mocap position (z offset removed)')
    ax[5].legend(fontsize=9); ax[5].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()
    print(f'[saved] {out}')


def stationary_stats(d, frac_end=0.5):
    """Compute stats over last frac_end of flight window."""
    t0, t1 = detect_flight(d)
    span = t1 - t0
    t_a = t0 + (1 - frac_end) * span
    t_b = t1 - 0.5  # avoid land transient
    res = {}
    for key, tname in [('Rm','Tm'), ('Pm','Tm'), ('Ym','Tm'),
                        ('WX','T'), ('WY','T'), ('WZ','T'),
                        ('Mxc','Tc'), ('Myc','Tc'), ('Mzc','Tc'),
                        ('Mxr','Tr'), ('Myr','Tr'), ('Mzr','Tr')]:
        T = d[tname]; V = d[key]
        m = (T >= t_a) & (T <= t_b)
        if m.sum() < 10:
            res[key] = (np.nan, np.nan, np.nan)
        else:
            v = V[m]
            res[key] = (float(v.mean()), float(v.std()), float(np.sqrt(np.mean(v**2))))
    res['win'] = (t_a, t_b)
    return res


def compare_plot(data, out):
    colors = {'roll_test_04':'tab:red','roll_test_05':'tab:blue','roll_test_06':'tab:green'}
    fig, axes = plt.subplots(5, 1, figsize=(14, 17), sharex=False)

    axes[0].set_title('Roll (mocap) - no external disturbance')
    for n, d in data.items():
        axes[0].plot(d['Tm'], d['Rm'], color=colors[n], lw=0.8, label=f'{n}  [{WEIGHTS[n]}]')
    axes[0].axhline(0, color='k', lw=0.4); axes[0].set_ylabel('roll (deg)')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_title('Pitch (mocap)')
    for n, d in data.items():
        axes[1].plot(d['Tm'], d['Pm'], color=colors[n], lw=0.8, label=n)
    axes[1].axhline(0, color='k', lw=0.4); axes[1].set_ylabel('pitch (deg)')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    axes[2].set_title('Body roll rate wx')
    for n, d in data.items():
        axes[2].plot(d['T'], d['WX'], color=colors[n], lw=0.6, label=n)
    axes[2].axhline(0, color='k', lw=0.4); axes[2].set_ylabel('wx (rad/s)')
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    axes[3].set_title('Commanded Mx')
    for n, d in data.items():
        axes[3].plot(d['Tc'], d['Mxc'], color=colors[n], lw=0.6, label=n)
    axes[3].axhline(0, color='k', lw=0.4); axes[3].set_ylabel('cmd Mx (Nm)')
    axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)

    # PSD of wx during flight only, resampled
    axes[4].set_title('PSD of wx (flight window only, resampled 100 Hz)')
    for n, d in data.items():
        t0, t1 = detect_flight(d)
        fs = 100.0
        tg = np.arange(t0, t1, 1/fs)
        if len(tg) < 256: continue
        wxg = np.interp(tg, d['T'], d['WX'])
        f, P = welch(wxg, fs=fs, nperseg=min(1024, len(tg)//2))
        axes[4].semilogy(f, P, color=colors[n], lw=1.0, label=n)
    axes[4].set_xlim(0, 50); axes[4].set_xlabel('Hz'); axes[4].set_ylabel('PSD')
    axes[4].legend(fontsize=8); axes[4].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()
    print(f'[saved] {out}')


def main():
    base = '/home/user/drone_control_pkgs/bag_folder'
    bags = ['roll_test_04', 'roll_test_05', 'roll_test_06']
    data = {}
    for n in bags:
        db = f'{base}/{n}/{n}_0.db3'
        print(f'load {n}')
        data[n] = load(db)
        per_bag_plot(n, data[n], f'{base}/{n}_analysis.png')
    compare_plot(data, f'{base}/roll_tests_04_06_compare.png')

    # Stationary stats
    print()
    print('=== Stationary stats (last 50% of flight, excluding last 0.5s) ===')
    print(f'{"bag":<14} | {"weights":<44} | {"win(s)":<14}')
    print(f'{"":<14} | {"roll_mean":>9} {"roll_std":>8} {"wx_std":>7} | {"Mx_mean":>8} {"Mx_std":>8} | {"My_mean":>8} {"My_std":>8}')
    print('-'*130)
    for n, d in data.items():
        s = stationary_stats(d)
        ta, tb = s['win']
        print(f'{n:<14} | {WEIGHTS[n]:<44} | {ta:5.1f}-{tb:5.1f}s')
        print(f'{"":<14} | {s["Rm"][0]:9.3f} {s["Rm"][1]:8.3f} {s["WX"][1]:7.3f} | {s["Mxc"][0]:8.5f} {s["Mxc"][1]:8.5f} | {s["Myc"][0]:8.5f} {s["Myc"][1]:8.5f}')


if __name__ == '__main__':
    main()
