#!/usr/bin/env python3
"""Deeper look: angular rate noise amplification and Mx convergence.

Context: Q_att=[2,2,1], Q_rate=[0.5,0.5,0.25], R=5.0, NMPC only.
Question from user:
  - wx noise seems amplified?
  - Mx does not converge to zero?
"""

import sqlite3, struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import welch

# same constants
C_T = 1.386e-07
k_m = 0.01569
l   = 0.265
MaxBit = 8191
MaxRpm = 9800

lx1 = l * np.sin(np.pi / 3); ly1 = l * np.cos(np.pi / 3)
lx2 = 0.0;                    ly2 = l
lx3 = -l * np.sin(np.pi / 3); ly3 = l * np.cos(np.pi / 3)
lx4 = -l * np.sin(np.pi / 3); ly4 = -l * np.cos(np.pi / 3)
lx5 = 0.0;                    ly5 = -l
lx6 = l * np.sin(np.pi / 3);  ly6 = -l * np.cos(np.pi / 3)
K_forward = np.array([
    [1, 1, 1, 1, 1, 1],
    [ly1, ly2, ly3, ly4, ly5, ly6],
    [-lx1, -lx2, -lx3, -lx4, -lx5, -lx6],
    [-k_m, k_m, -k_m, k_m, -k_m, k_m]
])


def parse_cmd_raw(data):
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    if off % 2 != 0: off += 1
    cmds = np.array(struct.unpack_from('<6h', data, off), dtype=np.float64)
    return sec + nsec * 1e-9, cmds


def parse_actual_rpm(data):
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    off = (off + 3) & ~3
    rpms = np.array(struct.unpack_from('<6i', data, off), dtype=np.float64)
    return sec + nsec * 1e-9, rpms


def parse_odom(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 68)
    wx, wy, wz = struct.unpack_from('<3d', data, 412)
    return sec + nsec * 1e-9, qx, qy, qz, qw, wx, wy, wz


def parse_pose(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 52)
    return sec + nsec * 1e-9, qx, qy, qz, qw


def load(db_path):
    conn = sqlite3.connect(db_path); c = conn.cursor()
    c.execute('SELECT id, name FROM topics')
    topics = {n: tid for tid, n in c.fetchall()}

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/mavros/local_position/odom'],))
    T, WX, WY, WZ, R_, P_, Y_ = [], [], [], [], [], [], []
    for data, in c.fetchall():
        t, qx, qy, qz, qw, wx, wy, wz = parse_odom(data)
        q = np.array([qx, qy, qz, qw]); nr = np.linalg.norm(q)
        if not np.isfinite(nr) or nr < 1e-10: continue
        roll, pitch, yaw = Rotation.from_quat(q / nr).as_euler('xyz', degrees=True)
        T.append(t); WX.append(wx); WY.append(wy); WZ.append(wz)
        R_.append(roll); P_.append(pitch); Y_.append(yaw)
    T = np.array(T); WX = np.array(WX); WY = np.array(WY); WZ = np.array(WZ)
    R_ = np.array(R_); P_ = np.array(P_); Y_ = np.array(Y_)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/S550/pose'],))
    Tm, Rm = [], []
    for data, in c.fetchall():
        t, qx, qy, qz, qw = parse_pose(data)
        q = np.array([qx, qy, qz, qw]); nr = np.linalg.norm(q)
        if not np.isfinite(nr) or nr < 1e-10: continue
        roll, _, _ = Rotation.from_quat(q / nr).as_euler('xyz', degrees=True)
        Tm.append(t); Rm.append(roll)
    Tm = np.array(Tm); Rm = np.array(Rm)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/uav/cmd_raw'],))
    Tc, Mx_cmd = [], []
    for data, in c.fetchall():
        t, cmds = parse_cmd_raw(data)
        rpms = cmds * MaxRpm / MaxBit
        thrusts = C_T * rpms ** 2
        u = K_forward @ thrusts
        Tc.append(t); Mx_cmd.append(u[1])
    Tc = np.array(Tc); Mx_cmd = np.array(Mx_cmd)

    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/uav/actual_rpm'],))
    Tr, Mx_act = [], []
    for data, in c.fetchall():
        t, rpms = parse_actual_rpm(data)
        thrusts = C_T * rpms ** 2
        u = K_forward @ thrusts
        Tr.append(t); Mx_act.append(u[1])
    Tr = np.array(Tr); Mx_act = np.array(Mx_act)

    conn.close()

    t0 = min(T[0], Tm[0], Tc[0], Tr[0])
    T -= t0; Tm -= t0; Tc -= t0; Tr -= t0
    return dict(T=T, WX=WX, WY=WY, WZ=WZ, R=R_, P=P_, Y=Y_,
                Tm=Tm, Rm=Rm, Tc=Tc, Mx_cmd=Mx_cmd, Tr=Tr, Mx_act=Mx_act)


def find_quiet_window(T, R, min_len=3.0, roll_thresh=2.0):
    """Find a quiet window (|roll| < thresh) of at least min_len seconds after start."""
    # resample on uniform grid
    dt = np.median(np.diff(T))
    mask = np.abs(R) < roll_thresh
    # find first long run of True after t>5 (skip arming)
    start = None; best_run = (0, 0, 0)
    k = 0
    while k < len(T):
        if T[k] < 5.0 or not mask[k]:
            k += 1; continue
        j = k
        while j < len(T) and mask[j] and T[j] - T[k] < 100:
            j += 1
        if T[j-1] - T[k] > best_run[2]:
            best_run = (k, j-1, T[j-1] - T[k])
        k = j
    return best_run  # idx_start, idx_end, duration


def main():
    base = '/home/user/drone_control_pkgs/bag_folder'
    bags = ['roll_test_01', 'roll_test_02', 'roll_test_03']
    colors = {'roll_test_01':'tab:red','roll_test_02':'tab:blue','roll_test_03':'tab:green'}

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))

    summary_rows = []

    for col, name in enumerate(bags):
        d = load(f'{base}/{name}/{name}_0.db3')

        # Resample wx and Mx_cmd to uniform 100 Hz for spectral analysis
        fs_target = 100.0
        t_grid = np.arange(0.0, d['T'][-1], 1.0/fs_target)
        wx_u   = np.interp(t_grid, d['T'],  d['WX'])
        mx_u   = np.interp(t_grid, d['Tc'], d['Mx_cmd'])
        mxa_u  = np.interp(t_grid, d['Tr'], d['Mx_act'])

        # PSD — full run
        f_wx,  Pwx  = welch(wx_u,  fs=fs_target, nperseg=1024)
        f_mx,  Pmx  = welch(mx_u,  fs=fs_target, nperseg=1024)
        f_mxa, Pmxa = welch(mxa_u, fs=fs_target, nperseg=1024)

        # Quiet window: last 3+ seconds where |roll|<1 deg
        i0, i1, dur = find_quiet_window(d['T'], d['R'], min_len=2.0, roll_thresh=1.5)
        if dur < 1.0:
            i0, i1, dur = find_quiet_window(d['T'], d['R'], min_len=1.0, roll_thresh=3.0)
        t_q0, t_q1 = d['T'][i0], d['T'][i1]

        # clip uniform signals to quiet window
        mask_q = (t_grid >= t_q0) & (t_grid <= t_q1)
        wx_q = wx_u[mask_q]
        mx_q = mx_u[mask_q]
        mxa_q = mxa_u[mask_q]

        ax = axes[0, col]
        ax.plot(d['T'], d['WX'], color=colors[name], lw=0.6)
        ax.axvspan(t_q0, t_q1, color='gray', alpha=0.25, label=f'quiet {dur:.1f}s')
        ax.set_title(f'{name}  wx (rad/s)')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.plot(d['Tc'], d['Mx_cmd'], color='tab:purple', lw=0.6, label='cmd Mx')
        ax.plot(d['Tr'], d['Mx_act'], color='tab:orange', lw=0.6, alpha=0.7, label='act Mx')
        ax.axvspan(t_q0, t_q1, color='gray', alpha=0.25)
        ax.axhline(0, color='k', lw=0.4)
        ax.set_title(f'{name}  Mx  (zoomed on quiet window ↓)')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        # zoom on quiet window
        ax = axes[2, col]
        ax.plot(t_grid[mask_q], mx_q, color='tab:purple', lw=0.8, label=f'cmd Mx  mean={mx_q.mean():.4f}, rms={np.sqrt(np.mean(mx_q**2)):.4f}')
        ax.plot(t_grid[mask_q], mxa_u[mask_q], color='tab:orange', lw=0.8, alpha=0.7,
                label=f'act Mx  mean={mxa_q.mean():.4f}, rms={np.sqrt(np.mean(mxa_q**2)):.4f}')
        ax.axhline(0, color='k', lw=0.4)
        ax.set_title(f'{name}  Mx quiet window  ({t_q0:.1f}–{t_q1:.1f}s, {dur:.1f}s)')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        # PSD
        ax = axes[3, col]
        ax.semilogy(f_wx, Pwx, color='tab:red',    lw=1.0, label='wx')
        ax.semilogy(f_mx, Pmx, color='tab:purple', lw=1.0, label='cmd Mx')
        ax.semilogy(f_mxa, Pmxa, color='tab:orange', lw=1.0, alpha=0.7, label='act Mx')
        ax.set_xlim(0, 50)
        ax.set_xlabel('Hz'); ax.set_ylabel('PSD')
        ax.set_title(f'{name}  PSD (100 Hz resampled)')
        ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)

        # dominant freq / peak power in quiet window
        f_wx_q, Pwx_q = welch(wx_q, fs=fs_target, nperseg=min(512, len(wx_q)//2)) if len(wx_q) > 128 else (np.zeros(2), np.zeros(2))
        f_mx_q, Pmx_q = welch(mx_q, fs=fs_target, nperseg=min(512, len(mx_q)//2)) if len(mx_q) > 128 else (np.zeros(2), np.zeros(2))
        peak_f_wx = f_wx_q[np.argmax(Pwx_q)] if len(Pwx_q) > 2 else np.nan
        peak_f_mx = f_mx_q[np.argmax(Pmx_q)] if len(Pmx_q) > 2 else np.nan
        wx_std = wx_q.std()
        mx_std = mx_q.std()
        mxa_std = mxa_q.std()
        mx_mean = mx_q.mean()
        mxa_mean = mxa_q.mean()

        summary_rows.append(dict(
            name=name, dur=dur,
            wx_std=wx_std, wx_pkf=peak_f_wx,
            mx_mean=mx_mean, mx_std=mx_std, mx_pkf=peak_f_mx,
            mxa_mean=mxa_mean, mxa_std=mxa_std,
        ))

    plt.tight_layout()
    out = f'{base}/roll_tests_noise_analysis.png'
    plt.savefig(out, dpi=110)
    plt.close()
    print(f'[saved] {out}')

    # Print table
    print()
    print('=== Quiet-window noise / bias analysis (|roll|<~1.5°) ===')
    hdr = f'{"bag":<14} | {"dur(s)":>6} | {"wx_std":>7} | {"wx_peakHz":>9} | {"cmdMx_mean":>10} | {"cmdMx_std":>9} | {"cmdMx_pkHz":>10} | {"actMx_mean":>10} | {"actMx_std":>9}'
    print(hdr); print('-'*len(hdr))
    for r in summary_rows:
        print(f'{r["name"]:<14} | {r["dur"]:6.2f} | {r["wx_std"]:7.4f} | {r["wx_pkf"]:9.2f} | {r["mx_mean"]:10.5f} | {r["mx_std"]:9.5f} | {r["mx_pkf"]:10.2f} | {r["mxa_mean"]:10.5f} | {r["mxa_std"]:9.5f}')


if __name__ == '__main__':
    main()
