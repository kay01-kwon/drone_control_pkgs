#!/usr/bin/env python3
"""Analyze roll_test_01/02/03 bags.

Context:
 - NMPC only (no disturbance observer)
 - Attitude weights: [2.0, 2.0, 1.0]  (roll, pitch, yaw)
 - Angular rate weights: [0.5, 0.5, 0.25]
 - Manual (hand) disturbance applied — roll axis only.

Topics in each bag:
  /S550/pose                  (mocap pose)
  /mavros/local_position/odom (EKF2 odom)
  /uav/actual_rpm             (measured motor RPM)
  /uav/cmd_raw                (commanded motor cmd bits)
  /mavros/rc/in
"""

import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ── Drone / allocation constants (same as other scripts in repo) ──
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


# ── CDR parsers ──
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
    px, py, pz = struct.unpack_from('<3d', data, 44)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 68)
    vx, vy, vz = struct.unpack_from('<3d', data, 388)
    wx, wy, wz = struct.unpack_from('<3d', data, 412)
    t = sec + nsec * 1e-9
    return t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz


def parse_pose_stamped(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, 28)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 52)
    t = sec + nsec * 1e-9
    return t, px, py, pz, qx, qy, qz, qw


def load_bag(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name FROM topics')
    topics = {name: tid for tid, name in c.fetchall()}

    # odom (EKF2)
    tid = topics['/mavros/local_position/odom']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    odom_t = []
    ekf_px, ekf_py, ekf_pz = [], [], []
    ekf_vx_w, ekf_vy_w, ekf_vz_w = [], [], []
    ekf_roll, ekf_pitch, ekf_yaw = [], [], []
    ekf_wx, ekf_wy, ekf_wz = [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = parse_odom(data)
        q = np.array([qx, qy, qz, qw])
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < 1e-10:
            continue
        R = Rotation.from_quat(q / n)
        roll, pitch, yaw = R.as_euler('xyz', degrees=True)
        v_world = R.as_matrix() @ np.array([vx, vy, vz])
        odom_t.append(t)
        ekf_px.append(px); ekf_py.append(py); ekf_pz.append(pz)
        ekf_vx_w.append(v_world[0]); ekf_vy_w.append(v_world[1]); ekf_vz_w.append(v_world[2])
        ekf_roll.append(roll); ekf_pitch.append(pitch); ekf_yaw.append(yaw)
        ekf_wx.append(wx); ekf_wy.append(wy); ekf_wz.append(wz)

    # mocap
    tid = topics['/S550/pose']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    mocap_t = []
    mocap_px, mocap_py, mocap_pz = [], [], []
    mocap_roll, mocap_pitch, mocap_yaw = [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw = parse_pose_stamped(data)
        q = np.array([qx, qy, qz, qw])
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < 1e-10:
            continue
        R = Rotation.from_quat(q / n)
        roll, pitch, yaw = R.as_euler('xyz', degrees=True)
        mocap_t.append(t)
        mocap_px.append(px); mocap_py.append(py); mocap_pz.append(pz)
        mocap_roll.append(roll); mocap_pitch.append(pitch); mocap_yaw.append(yaw)

    # cmd_raw
    tid = topics['/uav/cmd_raw']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    cmd_t, cmd_F, cmd_Mx, cmd_My, cmd_Mz = [], [], [], [], []
    for data, in c.fetchall():
        t, cmds = parse_cmd_raw(data)
        rpms = cmds * MaxRpm / MaxBit
        thrusts = C_T * rpms ** 2
        u = K_forward @ thrusts
        cmd_t.append(t)
        cmd_F.append(u[0]); cmd_Mx.append(u[1]); cmd_My.append(u[2]); cmd_Mz.append(u[3])

    # actual rpm
    tid = topics['/uav/actual_rpm']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    rpm_t, rpm_F, rpm_Mx, rpm_My, rpm_Mz = [], [], [], [], []
    rpm_raw = []
    for data, in c.fetchall():
        t, rpms = parse_actual_rpm(data)
        thrusts = C_T * rpms ** 2
        u = K_forward @ thrusts
        rpm_t.append(t)
        rpm_F.append(u[0]); rpm_Mx.append(u[1]); rpm_My.append(u[2]); rpm_Mz.append(u[3])
        rpm_raw.append(rpms.copy())

    conn.close()

    odom_t = np.array(odom_t); mocap_t = np.array(mocap_t)
    cmd_t = np.array(cmd_t); rpm_t = np.array(rpm_t)

    # Start time offset: use first mocap sample as AUTO start is not known exactly
    t0 = min(odom_t[0], mocap_t[0], cmd_t[0], rpm_t[0])
    odom_t -= t0; mocap_t -= t0; cmd_t -= t0; rpm_t -= t0

    mocap_pz = np.array(mocap_pz)
    mocap_pz_offset = mocap_pz[0]
    mocap_pz = mocap_pz - mocap_pz_offset

    ekf_pz = np.array(ekf_pz)
    ekf_pz_offset = ekf_pz[0]
    ekf_pz = ekf_pz - ekf_pz_offset

    return dict(
        odom_t=odom_t,
        ekf_px=np.array(ekf_px), ekf_py=np.array(ekf_py), ekf_pz=ekf_pz,
        ekf_vx=np.array(ekf_vx_w), ekf_vy=np.array(ekf_vy_w), ekf_vz=np.array(ekf_vz_w),
        ekf_roll=np.array(ekf_roll), ekf_pitch=np.array(ekf_pitch), ekf_yaw=np.array(ekf_yaw),
        ekf_wx=np.array(ekf_wx), ekf_wy=np.array(ekf_wy), ekf_wz=np.array(ekf_wz),
        mocap_t=mocap_t,
        mocap_px=np.array(mocap_px), mocap_py=np.array(mocap_py), mocap_pz=mocap_pz,
        mocap_roll=np.array(mocap_roll), mocap_pitch=np.array(mocap_pitch), mocap_yaw=np.array(mocap_yaw),
        cmd_t=cmd_t, cmd_F=np.array(cmd_F),
        cmd_Mx=np.array(cmd_Mx), cmd_My=np.array(cmd_My), cmd_Mz=np.array(cmd_Mz),
        rpm_t=rpm_t, rpm_F=np.array(rpm_F),
        rpm_Mx=np.array(rpm_Mx), rpm_My=np.array(rpm_My), rpm_Mz=np.array(rpm_Mz),
        rpm_raw=np.array(rpm_raw),
    )


def stats(arr):
    if len(arr) == 0:
        return dict(rms=np.nan, peak=np.nan, mean=np.nan, std=np.nan)
    return dict(
        rms=float(np.sqrt(np.mean(arr ** 2))),
        peak=float(np.max(np.abs(arr))),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
    )


def plot_one(bag_name, d, out_base):
    """Per-bag 6-row plot."""
    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)

    # 1. Roll / Pitch / Yaw (mocap vs EKF2)
    ax = axes[0]
    ax.plot(d['mocap_t'], d['mocap_roll'],  'tab:red',   lw=1.0, label='roll  (mocap)')
    ax.plot(d['mocap_t'], d['mocap_pitch'], 'tab:blue',  lw=1.0, label='pitch (mocap)')
    ax.plot(d['mocap_t'], d['mocap_yaw'],   'tab:green', lw=1.0, label='yaw   (mocap)')
    ax.plot(d['odom_t'],  d['ekf_roll'],  'tab:red',   lw=0.6, ls='--', alpha=0.6, label='roll  (EKF2)')
    ax.plot(d['odom_t'],  d['ekf_pitch'], 'tab:blue',  lw=0.6, ls='--', alpha=0.6, label='pitch (EKF2)')
    ax.plot(d['odom_t'],  d['ekf_yaw'],   'tab:green', lw=0.6, ls='--', alpha=0.6, label='yaw   (EKF2)')
    ax.set_ylabel('Attitude (deg)')
    ax.set_title(f'{bag_name} — Attitude (mocap solid, EKF2 dashed)')
    ax.legend(ncol=3, fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # 2. Angular velocity
    ax = axes[1]
    ax.plot(d['odom_t'], d['ekf_wx'], 'tab:red',   lw=0.8, label='wx')
    ax.plot(d['odom_t'], d['ekf_wy'], 'tab:blue',  lw=0.8, label='wy')
    ax.plot(d['odom_t'], d['ekf_wz'], 'tab:green', lw=0.8, label='wz')
    ax.set_ylabel('omega (rad/s)')
    ax.set_title('Body angular velocity (EKF2)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    # 3. Commanded moments (from cmd_raw)
    ax = axes[2]
    ax.plot(d['cmd_t'], d['cmd_Mx'], 'tab:red',   lw=0.8, label='cmd Mx')
    ax.plot(d['cmd_t'], d['cmd_My'], 'tab:blue',  lw=0.8, label='cmd My')
    ax.plot(d['cmd_t'], d['cmd_Mz'], 'tab:green', lw=0.8, label='cmd Mz')
    ax.set_ylabel('Moment (Nm)')
    ax.set_title('Commanded moment (allocated from cmd_raw)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    # 4. Actual moments (from actual_rpm)
    ax = axes[3]
    ax.plot(d['rpm_t'], d['rpm_Mx'], 'tab:red',   lw=0.8, label='actual Mx')
    ax.plot(d['rpm_t'], d['rpm_My'], 'tab:blue',  lw=0.8, label='actual My')
    ax.plot(d['rpm_t'], d['rpm_Mz'], 'tab:green', lw=0.8, label='actual Mz')
    ax.set_ylabel('Moment (Nm)')
    ax.set_title('Actual moment (allocated from actual_rpm)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    # 5. Thrust
    ax = axes[4]
    ax.plot(d['cmd_t'], d['cmd_F'], 'tab:purple', lw=0.8, label='cmd F')
    ax.plot(d['rpm_t'], d['rpm_F'], 'tab:orange', lw=0.8, label='actual F')
    ax.set_ylabel('Total thrust (N)')
    ax.set_title('Thrust')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    # 6. Position (mocap) + height z
    ax = axes[5]
    ax.plot(d['mocap_t'], d['mocap_px'], 'tab:red',   lw=0.8, label='x')
    ax.plot(d['mocap_t'], d['mocap_py'], 'tab:blue',  lw=0.8, label='y')
    ax.plot(d['mocap_t'], d['mocap_pz'], 'tab:green', lw=0.8, label='z (off sub)')
    ax.set_ylabel('Position (m)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Mocap position')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = f'{out_base}_analysis.png'
    plt.savefig(out, dpi=110)
    plt.close()
    print(f'[saved] {out}')


def plot_compare(data_dict, out_path):
    """Compare roll-axis response across 3 runs."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=False)
    colors = {'roll_test_01': 'tab:red', 'roll_test_02': 'tab:blue', 'roll_test_03': 'tab:green'}

    # roll angle
    ax = axes[0]
    for name, d in data_dict.items():
        ax.plot(d['mocap_t'], d['mocap_roll'], color=colors[name], lw=1.0, label=f'{name} (mocap)')
    ax.set_ylabel('Roll (deg)')
    ax.set_title('Roll angle — mocap')
    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # wx
    ax = axes[1]
    for name, d in data_dict.items():
        ax.plot(d['odom_t'], d['ekf_wx'], color=colors[name], lw=0.8, label=name)
    ax.set_ylabel('wx (rad/s)')
    ax.set_title('Body roll rate')
    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # cmd Mx
    ax = axes[2]
    for name, d in data_dict.items():
        ax.plot(d['cmd_t'], d['cmd_Mx'], color=colors[name], lw=0.8, label=name)
    ax.set_ylabel('cmd Mx (Nm)')
    ax.set_title('Commanded roll moment')
    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # actual Mx
    ax = axes[3]
    for name, d in data_dict.items():
        ax.plot(d['rpm_t'], d['rpm_Mx'], color=colors[name], lw=0.8, label=name)
    ax.set_ylabel('actual Mx (Nm)')
    ax.set_title('Actual roll moment (from actual_rpm)')
    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Phase plot: roll vs wx
    ax = axes[4]
    for name, d in data_dict.items():
        # interpolate wx to mocap time
        wx_i = np.interp(d['mocap_t'], d['odom_t'], d['ekf_wx'])
        ax.plot(d['mocap_roll'], wx_i, color=colors[name], lw=0.6, alpha=0.8, label=name)
    ax.set_xlabel('roll (deg)')
    ax.set_ylabel('wx (rad/s)')
    ax.set_title('Phase plot: roll vs wx  (convergence trajectory)')
    ax.axhline(0, color='k', lw=0.5, alpha=0.5); ax.axvline(0, color='k', lw=0.5, alpha=0.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f'[saved] {out_path}')


def print_stats_table(data_dict):
    rows = []
    header = f'{"bag":<14} | {"roll_rms":>8} | {"roll_peak":>9} | {"pitch_rms":>9} | {"pitch_peak":>10} | {"wx_rms":>7} | {"wx_peak":>8} | {"cmdMx_rms":>10} | {"cmdMx_peak":>10}'
    print(header)
    print('-' * len(header))
    for name, d in data_dict.items():
        s_r  = stats(d['mocap_roll'])
        s_p  = stats(d['mocap_pitch'])
        s_wx = stats(d['ekf_wx'])
        s_cm = stats(d['cmd_Mx'])
        print(f'{name:<14} | {s_r["rms"]:8.3f} | {s_r["peak"]:9.3f} | {s_p["rms"]:9.3f} | {s_p["peak"]:10.3f} | {s_wx["rms"]:7.3f} | {s_wx["peak"]:8.3f} | {s_cm["rms"]:10.4f} | {s_cm["peak"]:10.4f}')


def main():
    base_dir = '/home/user/drone_control_pkgs/bag_folder'
    bags = ['roll_test_01', 'roll_test_02', 'roll_test_03']
    data = {}
    for name in bags:
        db = f'{base_dir}/{name}/{name}_0.db3'
        print(f'loading {name} ...')
        data[name] = load_bag(db)
        plot_one(name, data[name], f'{base_dir}/{name}')

    plot_compare(data, f'{base_dir}/roll_tests_compare.png')

    print()
    print('=== Stats (roll/pitch in deg, wx in rad/s, Mx in Nm) ===')
    print_stats_table(data)


if __name__ == '__main__':
    main()
