#!/usr/bin/env python3
"""Compare 0401_sim vs 0407_sim at lift moment.

Actual-RPM derived moments (Mx/My/Mz) with roll/pitch/yaw and wx/wy/wz,
in the same 6-row layout as *_mpc_moments_rpy.png.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_2026_04_01_sim import load_bag, K_forward, C_T, W


def actual_rpm_wrench(d):
    """Compute (F_total, Mx, My, Mz) from actual RPM using K_forward."""
    rpms = d['rpm_raw']  # (N, 6)
    thrusts = C_T * rpms ** 2
    u = thrusts @ K_forward.T  # (N, 4) -> F,Mx,My,Mz
    return u[:, 0], u[:, 1], u[:, 2], u[:, 3]


def find_lift_time(d):
    """First time when actual-RPM total thrust >= weight W."""
    F = d['rpm_F']
    t = d['rpm_t']
    mask = F >= W
    if not np.any(mask):
        return t[0]
    return t[np.argmax(mask)]


def load(bag):
    db = f'/home/user/drone_control_pkgs/bag_folder/{bag}/{bag}_0.db3'
    d = load_bag(db)
    F, Mx, My, Mz = actual_rpm_wrench(d)
    d['rpm_F'] = F
    d['rpm_Mx'] = Mx
    d['rpm_My'] = My
    d['rpm_Mz'] = Mz
    d['t_lift'] = find_lift_time(d)
    return d


def shift(d, t0):
    """Return copies of arrays shifted by -t0 so lift is at 0."""
    return dict(
        rpm_t=d['rpm_t'] - t0,
        mpc_t=d['mpc_t'] - t0,
        odom_t=d['odom_t'] - t0,
        rpm_Mx=d['rpm_Mx'], rpm_My=d['rpm_My'], rpm_Mz=d['rpm_Mz'],
        mpc_Mx=d['mpc_Mx'], mpc_My=d['mpc_My'], mpc_Mz=d['mpc_Mz'],
        odom_roll=d['odom_roll'] - d['odom_roll'][0],
        odom_pitch=d['odom_pitch'],
        odom_yaw=d['odom_yaw'],
        odom_wx=d['odom_wx'], odom_wy=d['odom_wy'], odom_wz=d['odom_wz'],
    )


def main():
    d01 = load('2026_04_01_sim')
    d07 = load('2026_04_07_sim')

    print(f"0401 lift @ {d01['t_lift']:.2f}s, 0407 lift @ {d07['t_lift']:.2f}s")

    s01 = shift(d01, d01['t_lift'])
    s07 = shift(d07, d07['t_lift'])

    # Window around lift: [-1, +5] s
    t_min, t_max = -1.0, 5.0

    rows = [
        ('Mx', 'rpm_Mx', 'mpc_Mx', 'Roll',  'odom_roll',  'wx', 'odom_wx'),
        ('My', 'rpm_My', 'mpc_My', 'Pitch', 'odom_pitch', 'wy', 'odom_wy'),
        ('Mz', 'rpm_Mz', 'mpc_Mz', 'Yaw',   'odom_yaw',   'wz', 'odom_wz'),
    ]

    fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True)

    for i, (m_lbl, m_key, mpc_key, a_lbl, a_key, w_lbl, w_key) in enumerate(rows):
        # Moment + angle
        ax1 = axes[i * 2]
        ln1 = ax1.plot(s01['rpm_t'], s01[m_key], 'tab:blue', lw=0.9, label=f'0401 rpm {m_lbl}')
        ln2 = ax1.plot(s07['rpm_t'], s07[m_key], 'tab:cyan', lw=0.9, label=f'0407 rpm {m_lbl}')
        lnM1 = ax1.plot(s01['mpc_t'], s01[mpc_key], 'tab:purple', lw=0.8, ls='--', label=f'0401 MPC {m_lbl}')
        lnM2 = ax1.plot(s07['mpc_t'], s07[mpc_key], 'tab:pink',   lw=0.8, ls='--', label=f'0407 MPC {m_lbl}')
        ax1.set_ylabel(f'{m_lbl} (Nm)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(0, color='k', ls=':', lw=0.8, alpha=0.5)
        axR = ax1.twinx()
        ln3 = axR.plot(s01['odom_t'], s01[a_key], 'tab:red',    lw=0.9, label=f'0401 {a_lbl}')
        ln4 = axR.plot(s07['odom_t'], s07[a_key], 'tab:orange', lw=0.9, label=f'0407 {a_lbl}')
        axR.set_ylabel(f'{a_lbl} (deg)', color='tab:red')
        axR.tick_params(axis='y', labelcolor='tab:red')
        m_max = max(abs(ax1.get_ylim()[0]), abs(ax1.get_ylim()[1]), 0.001)
        ax1.set_ylim(-m_max, m_max)
        a_max = max(abs(axR.get_ylim()[0]), abs(axR.get_ylim()[1]), 0.001)
        axR.set_ylim(-a_max, a_max)
        lns = ln1 + ln2 + lnM1 + lnM2 + ln3 + ln4
        ax1.legend(lns, [l.get_label() for l in lns], loc='upper right', fontsize=7, ncol=3)
        ax1.set_title(f'{m_lbl} (rpm & MPC) + {a_lbl}  (0401 vs 0407, lift-aligned)')

        # Moment + angular velocity
        ax3 = axes[i * 2 + 1]
        ln5 = ax3.plot(s01['rpm_t'], s01[m_key], 'tab:blue', lw=0.9, label=f'0401 rpm {m_lbl}')
        ln6 = ax3.plot(s07['rpm_t'], s07[m_key], 'tab:cyan', lw=0.9, label=f'0407 rpm {m_lbl}')
        lnM3 = ax3.plot(s01['mpc_t'], s01[mpc_key], 'tab:purple', lw=0.8, ls='--', label=f'0401 MPC {m_lbl}')
        lnM4 = ax3.plot(s07['mpc_t'], s07[mpc_key], 'tab:pink',   lw=0.8, ls='--', label=f'0407 MPC {m_lbl}')
        ax3.set_ylabel(f'{m_lbl} (Nm)', color='tab:blue')
        ax3.tick_params(axis='y', labelcolor='tab:blue')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(0, color='k', ls=':', lw=0.8, alpha=0.5)
        axW = ax3.twinx()
        ln7 = axW.plot(s01['odom_t'], s01[w_key], 'tab:green', lw=0.9, label=f'0401 {w_lbl}')
        ln8 = axW.plot(s07['odom_t'], s07[w_key], 'tab:olive', lw=0.9, label=f'0407 {w_lbl}')
        axW.set_ylabel(f'{w_lbl} (rad/s)', color='tab:green')
        axW.tick_params(axis='y', labelcolor='tab:green')
        m_max = max(abs(ax3.get_ylim()[0]), abs(ax3.get_ylim()[1]), 0.001)
        ax3.set_ylim(-m_max, m_max)
        w_max = max(abs(axW.get_ylim()[0]), abs(axW.get_ylim()[1]), 0.001)
        axW.set_ylim(-w_max, w_max)
        lns_w = ln5 + ln6 + lnM3 + lnM4 + ln7 + ln8
        ax3.legend(lns_w, [l.get_label() for l in lns_w], loc='upper right', fontsize=7, ncol=3)
        ax3.set_title(f'{m_lbl} (rpm & MPC) + {w_lbl}  (0401 vs 0407, lift-aligned)')

    axes[-1].set_xlabel('Time from lift (s)')
    axes[-1].set_xlim(t_min, t_max)

    plt.tight_layout()
    out = '/home/user/drone_control_pkgs/bag_folder/compare_0401_0407_lift_actual_rpm_moments_rpy.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
