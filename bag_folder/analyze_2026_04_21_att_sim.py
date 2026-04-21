#!/usr/bin/env python3
"""Analyze 2026_04_21_att_sim bag: NMPC moments, actual-RPM moments, RPY."""

import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BAG_DIR = Path(__file__).parent / '2026_04_21_att_sim'
DB_PATH = BAG_DIR / '2026_04_21_att_sim_0.db3'

# Drone parameters
L = 0.265
C_T = 1.386e-7
K_M = 0.01569
cos60 = np.cos(np.pi / 3)
sin60 = np.sin(np.pi / 3)

K_FORWARD = np.array([
    [1, 1, 1, 1, 1, 1],
    [L*cos60, L, L*cos60, -L*cos60, -L, -L*cos60],
    [-L*sin60, 0, L*sin60, L*sin60, 0, -L*sin60],
    [-K_M, K_M, -K_M, K_M, -K_M, K_M],
])


def read_bag(db_path):
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, type FROM topics")
    topics = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    topic_name_to_id = {v[0]: k for k, v in topics.items()}

    cursor.execute("SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp")
    rows = cursor.fetchall()
    conn.close()
    return topics, topic_name_to_id, rows


CDR_START = 4  # CDR encapsulation header size

def cdr_align(offset, alignment):
    """Align offset for CDR field (relative to CDR data start)."""
    pos = offset - CDR_START
    remainder = pos % alignment
    if remainder != 0:
        offset += alignment - remainder
    return offset


def parse_wrench_stamped(data):
    """Parse geometry_msgs/WrenchStamped from CDR bytes."""
    offset = CDR_START
    sec = struct.unpack_from('<i', data, offset)[0]; offset += 4
    nsec = struct.unpack_from('<I', data, offset)[0]; offset += 4
    str_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += str_len
    offset = cdr_align(offset, 8)
    fx, fy, fz = struct.unpack_from('<3d', data, offset); offset += 24
    tx, ty, tz = struct.unpack_from('<3d', data, offset)
    t = sec + nsec * 1e-9
    return t, fx, fy, fz, tx, ty, tz


def parse_odometry(data):
    """Parse nav_msgs/Odometry — extract quaternion and angular velocity."""
    offset = CDR_START
    sec = struct.unpack_from('<i', data, offset)[0]; offset += 4
    nsec = struct.unpack_from('<I', data, offset)[0]; offset += 4
    str_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += str_len
    offset = cdr_align(offset, 4)
    str_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += str_len
    offset = cdr_align(offset, 8)
    px, py, pz = struct.unpack_from('<3d', data, offset); offset += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', data, offset); offset += 32
    offset += 36 * 8
    vx, vy, vz = struct.unpack_from('<3d', data, offset); offset += 24
    wx, wy, wz = struct.unpack_from('<3d', data, offset)
    t = sec + nsec * 1e-9
    return t, qw, qx, qy, qz, wx, wy, wz


def parse_hexa_actual_rpm(data):
    """Parse ros2_libcanard_msgs/HexaActualRpm — Header + int32[6] rpm."""
    offset = CDR_START
    sec = struct.unpack_from('<i', data, offset)[0]; offset += 4
    nsec = struct.unpack_from('<I', data, offset)[0]; offset += 4
    str_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += str_len
    offset = cdr_align(offset, 4)
    rpms = struct.unpack_from('<6i', data, offset)
    t = sec + nsec * 1e-9
    return t, rpms


def quat_to_rpy(qw, qx, qy, qz):
    """Quaternion [w,x,y,z] to roll, pitch, yaw [rad] (ZYX convention)."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def main():
    topics, tid_map, rows = read_bag(DB_PATH)

    nmpc_id = tid_map.get('/nmpc/control')
    odom_id = tid_map.get('/mavros/local_position/odom')
    rpm_id = tid_map.get('/uav/actual_rpm')
    hgdo_id = tid_map.get('/hgdo/wrench')

    nmpc_t, nmpc_mx, nmpc_my, nmpc_mz, nmpc_fz = [], [], [], [], []
    odom_t, odom_roll, odom_pitch, odom_yaw = [], [], [], []
    odom_wx, odom_wy, odom_wz_list = [], [], []
    rpm_t, rpm_mx, rpm_my, rpm_mz, rpm_fz = [], [], [], [], []
    hgdo_t, hgdo_mx, hgdo_my, hgdo_mz = [], [], [], []
    hgdo_fx, hgdo_fy, hgdo_fz_list = [], [], []

    t0 = None
    for topic_id, timestamp, data in rows:
        if t0 is None:
            t0 = timestamp

        if topic_id == nmpc_id:
            t, fx, fy, fz, tx, ty, tz = parse_wrench_stamped(data)
            nmpc_t.append(t)
            nmpc_fz.append(fz)
            nmpc_mx.append(tx)
            nmpc_my.append(ty)
            nmpc_mz.append(tz)

        elif topic_id == odom_id:
            t, qw, qx, qy, qz, wx, wy, wz = parse_odometry(data)
            roll, pitch, yaw = quat_to_rpy(qw, qx, qy, qz)
            odom_t.append(t)
            odom_roll.append(roll)
            odom_pitch.append(pitch)
            odom_yaw.append(yaw)
            odom_wx.append(wx)
            odom_wy.append(wy)
            odom_wz_list.append(wz)

        elif topic_id == rpm_id:
            t, rpms = parse_hexa_actual_rpm(data)
            rpms_arr = np.array(rpms[:6], dtype=np.float64)
            thrusts = C_T * rpms_arr ** 2
            u = K_FORWARD @ thrusts
            rpm_t.append(t)
            rpm_fz.append(u[0])
            rpm_mx.append(u[1])
            rpm_my.append(u[2])
            rpm_mz.append(u[3])

        elif topic_id == hgdo_id:
            t, fx, fy, fz, tx, ty, tz = parse_wrench_stamped(data)
            hgdo_t.append(t)
            hgdo_fx.append(fx)
            hgdo_fy.append(fy)
            hgdo_fz_list.append(fz)
            hgdo_mx.append(tx)
            hgdo_my.append(ty)
            hgdo_mz.append(tz)

    # Convert to arrays and shift to relative time
    nmpc_t = np.array(nmpc_t); nmpc_mx = np.array(nmpc_mx)
    nmpc_my = np.array(nmpc_my); nmpc_mz = np.array(nmpc_mz)
    nmpc_fz = np.array(nmpc_fz)

    odom_t = np.array(odom_t); odom_roll = np.array(odom_roll)
    odom_pitch = np.array(odom_pitch); odom_yaw = np.array(odom_yaw)
    odom_wx = np.array(odom_wx); odom_wy = np.array(odom_wy)
    odom_wz = np.array(odom_wz_list)

    rpm_t = np.array(rpm_t); rpm_mx = np.array(rpm_mx)
    rpm_my = np.array(rpm_my); rpm_mz = np.array(rpm_mz)
    rpm_fz = np.array(rpm_fz)

    hgdo_t = np.array(hgdo_t); hgdo_mx = np.array(hgdo_mx)
    hgdo_my = np.array(hgdo_my); hgdo_mz = np.array(hgdo_mz)
    hgdo_fx = np.array(hgdo_fx); hgdo_fy = np.array(hgdo_fy)
    hgdo_fz_arr = np.array(hgdo_fz_list)

    # Common t0 for relative time
    all_t0 = min(nmpc_t[0] if len(nmpc_t) else 1e18,
                 odom_t[0] if len(odom_t) else 1e18,
                 rpm_t[0] if len(rpm_t) else 1e18)
    nmpc_t -= all_t0
    odom_t -= all_t0
    rpm_t -= all_t0
    hgdo_t -= all_t0

    # Degrees for RPY
    r2d = 180.0 / np.pi

    # ── Figure 1: RPY + Angular velocity ──
    fig1, ax1 = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    fig1.suptitle('2026_04_21_att_sim — Attitude & Angular Velocity', fontsize=14)

    for i, (label, data, color) in enumerate([
        ('Roll', odom_roll, 'b'), ('Pitch', odom_pitch, 'r'), ('Yaw', odom_yaw, 'g')
    ]):
        ax1[0, i].plot(odom_t, data * r2d, color, linewidth=0.8)
        ax1[0, i].set_ylabel(f'{label} [deg]')
        ax1[0, i].grid(True, alpha=0.3)

    for i, (label, data, color) in enumerate([
        ('wx', odom_wx, 'b'), ('wy', odom_wy, 'r'), ('wz', odom_wz, 'g')
    ]):
        ax1[1, i].plot(odom_t, data, color, linewidth=0.8)
        ax1[1, i].set_ylabel(f'{label} [rad/s]')
        ax1[1, i].set_xlabel('Time [s]')
        ax1[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / '2026_04_21_att_sim_rpy.png', dpi=150)
    print('Saved: 2026_04_21_att_sim_rpy.png')

    # ── Figure 2: NMPC cmd vs Actual(RPM) vs HGDO — overlay per axis ──
    fig2, ax2 = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig2.suptitle('Moment Comparison: NMPC cmd / Actual(RPM) / HGDO', fontsize=14)

    axis_labels = ['Mx (Roll)', 'My (Pitch)', 'Mz (Yaw)']
    nmpc_data = [nmpc_mx, nmpc_my, nmpc_mz]
    rpm_data_list = [rpm_mx, rpm_my, rpm_mz]
    hgdo_data = [hgdo_mx, hgdo_my, hgdo_mz]

    for i in range(3):
        ax2[i].plot(nmpc_t, nmpc_data[i], 'k', linewidth=0.9, label='NMPC cmd (published)')
        ax2[i].plot(rpm_t, rpm_data_list[i], 'C0', linewidth=0.8, alpha=0.7,
                    label='Actual (RPM)')
        ax2[i].plot(hgdo_t, hgdo_data[i], 'C3', linewidth=0.8, alpha=0.7,
                    label='HGDO torque')
        ax2[i].set_ylabel(f'{axis_labels[i]} [Nm]')
        ax2[i].legend(loc='upper right', fontsize=9)
        ax2[i].grid(True, alpha=0.3)

    ax2[2].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / '2026_04_21_att_sim_moments.png', dpi=150)
    print('Saved: 2026_04_21_att_sim_moments.png')

    # ── Figure 3: Effective moment (NMPC - HGDO) vs Actual ──
    fig3, ax3 = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig3.suptitle('Effective Moment (NMPC - HGDO) vs Actual(RPM)', fontsize=14)

    nmpc_interp_mx = np.interp(hgdo_t, nmpc_t, nmpc_mx)
    nmpc_interp_my = np.interp(hgdo_t, nmpc_t, nmpc_my)
    nmpc_interp_mz = np.interp(hgdo_t, nmpc_t, nmpc_mz)
    eff_mx = nmpc_interp_mx - hgdo_mx
    eff_my = nmpc_interp_my - hgdo_my
    eff_mz = nmpc_interp_mz - hgdo_mz
    eff_data = [eff_mx, eff_my, eff_mz]

    for i in range(3):
        ax3[i].plot(hgdo_t, eff_data[i], 'k', linewidth=0.9,
                    label='Effective (NMPC - HGDO)')
        ax3[i].plot(rpm_t, rpm_data_list[i], 'C0', linewidth=0.8, alpha=0.7,
                    label='Actual (RPM)')
        ax3[i].set_ylabel(f'{axis_labels[i]} [Nm]')
        ax3[i].legend(loc='upper right', fontsize=9)
        ax3[i].grid(True, alpha=0.3)

    ax3[2].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / '2026_04_21_att_sim_effective.png', dpi=150)
    print('Saved: 2026_04_21_att_sim_effective.png')

    # Print stats
    print(f'\nData points: odom={len(odom_t)}, nmpc={len(nmpc_t)}, '
          f'rpm={len(rpm_t)}, hgdo={len(hgdo_t)}')
    print(f'Duration: {max(odom_t[-1], nmpc_t[-1], rpm_t[-1]):.1f} s')
    print(f'\nRPY range [deg]:')
    print(f'  Roll:  [{odom_roll.min()*r2d:.1f}, {odom_roll.max()*r2d:.1f}]')
    print(f'  Pitch: [{odom_pitch.min()*r2d:.1f}, {odom_pitch.max()*r2d:.1f}]')
    print(f'  Yaw:   [{odom_yaw.min()*r2d:.1f}, {odom_yaw.max()*r2d:.1f}]')
    print(f'\nNMPC moment range [Nm]:')
    print(f'  Mx: [{nmpc_mx.min():.4f}, {nmpc_mx.max():.4f}]')
    print(f'  My: [{nmpc_my.min():.4f}, {nmpc_my.max():.4f}]')
    print(f'  Mz: [{nmpc_mz.min():.4f}, {nmpc_mz.max():.4f}]')
    print(f'\nActual (RPM) moment range [Nm]:')
    print(f'  Mx: [{rpm_mx.min():.4f}, {rpm_mx.max():.4f}]')
    print(f'  My: [{rpm_my.min():.4f}, {rpm_my.max():.4f}]')
    print(f'  Mz: [{rpm_mz.min():.4f}, {rpm_mz.max():.4f}]')
    print(f'\nHGDO torque range [Nm]:')
    print(f'  Mx: [{hgdo_mx.min():.4f}, {hgdo_mx.max():.4f}]')
    print(f'  My: [{hgdo_my.min():.4f}, {hgdo_my.max():.4f}]')
    print(f'  Mz: [{hgdo_mz.min():.4f}, {hgdo_mz.max():.4f}]')


if __name__ == '__main__':
    main()
