#!/usr/bin/env python3
"""Analyze bag 05 altitude overshoot, compare with bags 03 and 04."""

import sqlite3
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ── Parsers (from existing plot scripts) ──

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


def parse_wrench_stamped(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    fx, fy, fz = struct.unpack_from('<3d', data, 28)
    tx, ty, tz = struct.unpack_from('<3d', data, 52)
    t = sec + nsec * 1e-9
    return t, fx, fy, fz, tx, ty, tz


def parse_ref(data):
    """Parse drone_msgs/msg/Ref CDR:
    4-byte CDR encapsulation + Header (sec,nsec,frame_id) + float64[3] p + float64[3] v + float64 psi + float64 psi_dot.
    Alignment is relative to data start (offset 4 past encapsulation header).
    """
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    flen = struct.unpack_from('<I', data, 12)[0]
    # After frame_id string: 16 + flen
    # Align relative to data start (offset 4): (16+flen-4) aligned to 8, then +4
    rel = 16 + flen - 4  # relative to data start
    rel = (rel + 7) & ~7
    off = rel + 4  # back to absolute
    p = struct.unpack_from('<3d', data, off)
    off += 24
    v = struct.unpack_from('<3d', data, off)
    off += 24
    psi = struct.unpack_from('<d', data, off)[0]
    off += 8
    if off + 8 <= len(data):
        psi_dot = struct.unpack_from('<d', data, off)[0]
    else:
        psi_dot = 0.0
    t = sec + nsec * 1e-9
    return t, p, v, psi, psi_dot


def parse_rc_in(data):
    """Parse mavros_msgs/RCIn: Header + uint16 rssi + uint16[] channels.
    Channels start at offset 24 (after 4-byte CDR encap + header + rssi + array length).
    """
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    arr_len = struct.unpack_from('<I', data, 20)[0]
    channels = list(struct.unpack_from(f'<{min(arr_len, 16)}H', data, 24))
    t = sec + nsec * 1e-9
    return t, channels


def load_bag(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name FROM topics')
    topics = {name: tid for tid, name in c.fetchall()}

    # Odom
    tid = topics['/mavros/local_position/odom']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    odom_t, pz_list, vz_list = [], [], []
    quat_list = []
    vx_list, vy_list = [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = parse_odom(data)
        q = np.array([qx, qy, qz, qw])
        norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm < 1e-10:
            continue
        R = Rotation.from_quat(q / norm)
        v_world = R.as_matrix() @ np.array([vx, vy, vz])
        odom_t.append(t)
        pz_list.append(pz)
        vz_list.append(v_world[2])
    odom_t = np.array(odom_t)
    pz_arr = np.array(pz_list)
    vz_arr = np.array(vz_list)

    # Mocap
    tid = topics['/S550/pose']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    mocap_t, mocap_pz = [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw = parse_pose_stamped(data)
        mocap_t.append(t)
        mocap_pz.append(pz)
    mocap_t = np.array(mocap_t)
    mocap_pz = np.array(mocap_pz)

    # Reference (use bag timestamp since header stamp may be zero)
    ref_t, ref_pz, ref_vz = [], [], []
    if '/nmpc/ref' in topics:
        tid = topics['/nmpc/ref']
        c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
        for bag_ts, data in c.fetchall():
            try:
                t_hdr, p, v, psi, psi_dot = parse_ref(data)
                # Use header time if valid, else bag timestamp (in nanoseconds)
                if t_hdr > 1e9:
                    t = t_hdr
                else:
                    t = bag_ts * 1e-9
                ref_t.append(t)
                ref_pz.append(p[2])
                ref_vz.append(v[2])
            except Exception as e:
                print(f"  Warning: failed to parse ref message: {e}")
    ref_t = np.array(ref_t) if ref_t else np.array([])
    ref_pz = np.array(ref_pz) if ref_pz else np.array([])
    ref_vz = np.array(ref_vz) if ref_vz else np.array([])

    # NMPC control (thrust)
    mpc_t, mpc_fz = [], []
    if '/nmpc/control' in topics:
        tid = topics['/nmpc/control']
        c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
        for data, in c.fetchall():
            t, fx, fy, fz, tx, ty, tz = parse_wrench_stamped(data)
            mpc_t.append(t)
            mpc_fz.append(fz)
    mpc_t = np.array(mpc_t) if mpc_t else np.array([])
    mpc_fz = np.array(mpc_fz) if mpc_fz else np.array([])

    # HGDO disturbance (fz component)
    hgdo_t, hgdo_fz = [], []
    if '/hgdo/wrench' in topics:
        tid = topics['/hgdo/wrench']
        c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
        for data, in c.fetchall():
            t, fx, fy, fz, tx, ty, tz = parse_wrench_stamped(data)
            hgdo_t.append(t)
            hgdo_fz.append(fz)
    hgdo_t = np.array(hgdo_t) if hgdo_t else np.array([])
    hgdo_fz = np.array(hgdo_fz) if hgdo_fz else np.array([])

    # Kill time: detect when ch8 (NMPC enable, index 8) drops from 2000 to 1000
    kill_t = None
    nmpc_enable_t = None
    if '/mavros/rc/in' in topics:
        tid = topics['/mavros/rc/in']
        c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
        prev_ch8 = None
        for data, in c.fetchall():
            t, channels = parse_rc_in(data)
            if len(channels) > 8:
                ch8 = channels[8]
                if prev_ch8 is not None:
                    if prev_ch8 < 1500 and ch8 > 1500 and nmpc_enable_t is None:
                        nmpc_enable_t = t
                    if prev_ch8 > 1500 and ch8 < 1500 and nmpc_enable_t is not None:
                        kill_t = t
                        break
                prev_ch8 = ch8

    conn.close()

    # Align times
    t0 = odom_t[0]
    odom_t -= t0
    mocap_t -= t0
    if len(ref_t) > 0:
        ref_t -= t0
    if len(mpc_t) > 0:
        mpc_t -= t0
    if len(hgdo_t) > 0:
        hgdo_t -= t0
    if kill_t is not None:
        kill_t -= t0

    # Subtract initial pz
    pz0 = pz_arr[0]
    pz_arr = pz_arr - pz0
    mocap_pz0 = mocap_pz[0]
    mocap_pz = mocap_pz - mocap_pz0

    # Adjust ref_pz: the ref p[2] is likely absolute, subtract same offset
    if len(ref_pz) > 0:
        ref_pz = ref_pz - pz0

    return dict(
        odom_t=odom_t, pz=pz_arr, vz=vz_arr, pz0=pz0,
        mocap_t=mocap_t, mocap_pz=mocap_pz, mocap_pz0=mocap_pz0,
        ref_t=ref_t, ref_pz=ref_pz, ref_vz=ref_vz,
        mpc_t=mpc_t, mpc_fz=mpc_fz,
        hgdo_t=hgdo_t, hgdo_fz=hgdo_fz,
        kill_t=kill_t,
    )


def print_stats(name, d, target=0.2):
    print(f"\n{'='*60}")
    print(f"  Bag: {name}")
    print(f"{'='*60}")
    print(f"  Duration: {d['odom_t'][-1]:.1f} s")
    print(f"  EKF2 initial z offset (subtracted): {d['pz0']:.4f} m")
    print(f"  Mocap initial z offset (subtracted): {d['mocap_pz0']:.4f} m")

    max_z_ekf = np.max(d['pz'])
    max_z_idx = np.argmax(d['pz'])
    max_z_time = d['odom_t'][max_z_idx]
    print(f"  Max altitude (EKF2): {max_z_ekf:.4f} m at t={max_z_time:.2f} s")

    max_z_mocap = np.max(d['mocap_pz'])
    max_z_mocap_idx = np.argmax(d['mocap_pz'])
    max_z_mocap_time = d['mocap_t'][max_z_mocap_idx]
    print(f"  Max altitude (Mocap): {max_z_mocap:.4f} m at t={max_z_mocap_time:.2f} s")

    overshoot_ekf = (max_z_ekf - target) / target * 100
    overshoot_mocap = (max_z_mocap - target) / target * 100
    print(f"  Overshoot vs {target}m target (EKF2): {overshoot_ekf:.1f}%")
    print(f"  Overshoot vs {target}m target (Mocap): {overshoot_mocap:.1f}%")

    max_vz = np.max(d['vz'])
    max_vz_time = d['odom_t'][np.argmax(d['vz'])]
    print(f"  Max upward velocity (EKF2): {max_vz:.4f} m/s at t={max_vz_time:.2f} s")

    if d['kill_t'] is not None:
        print(f"  Kill switch activated at t={d['kill_t']:.2f} s")
        # altitude at kill
        kill_z = np.interp(d['kill_t'], d['odom_t'], d['pz'])
        print(f"  Altitude at kill (EKF2): {kill_z:.4f} m")
    else:
        print(f"  Kill switch: NOT detected")

    if len(d['ref_pz']) > 0:
        print(f"  Reference pz values: {d['ref_pz']}")
        print(f"  Reference pz times:  {d['ref_t']}")
        print(f"  Reference vz values: {d['ref_vz']}")

    if len(d['mpc_fz']) > 0:
        m = 3.146
        W = m * 9.81
        print(f"  MPC thrust u_mpc range: [{np.min(d['mpc_fz']):.2f}, {np.max(d['mpc_fz']):.2f}] N")
        above_hover = np.sum(d['mpc_fz'] > W) / len(d['mpc_fz']) * 100
        print(f"  MPC thrust > hover weight ({W:.1f}N): {above_hover:.1f}% of the time")

    if len(d['hgdo_fz']) > 0:
        print(f"  HGDO fz range: [{np.min(d['hgdo_fz']):.2f}, {np.max(d['hgdo_fz']):.2f}] N")
        print(f"  HGDO fz mean: {np.mean(d['hgdo_fz']):.2f} N")

    if len(d['mpc_fz']) > 0 and len(d['hgdo_fz']) > 0:
        hgdo_interp = np.interp(d['mpc_t'], d['hgdo_t'], d['hgdo_fz'])
        f_eff = d['mpc_fz'] - hgdo_interp
        print(f"  Effective thrust (u_mpc - hgdo) range: [{np.min(f_eff):.2f}, {np.max(f_eff):.2f}] N")
        print(f"  Effective thrust max (= motor command peak): {np.max(f_eff):.2f} N")


# ── Load all bags ──
bags = {}
bag_configs = [
    ('03', '/home/user/drone_control_pkgs/bag_folder/2026_03_31_03/2026_03_31_03_0.db3'),
    ('04', '/home/user/drone_control_pkgs/bag_folder/2026_03_31_04/2026_03_31_04_0.db3'),
    ('05', '/home/user/drone_control_pkgs/bag_folder/2026_03_31_05/2026_03_31_05_0.db3'),
]

for label, path in bag_configs:
    print(f"Loading bag {label}...")
    try:
        bags[label] = load_bag(path)
        print_stats(label, bags[label])
    except Exception as e:
        print(f"  ERROR loading bag {label}: {e}")
        import traceback; traceback.print_exc()

# ── Main analysis plot for bag 05 (zoomed to t=12-22s around the event) ──
d = bags['05']
m = 3.146  # from config
W = m * 9.81
t_zoom = (12, 22)

fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

# 1) Altitude: EKF2 + Mocap + Reference
ax = axes[0]
ax.plot(d['odom_t'], d['pz'], 'tab:blue', lw=1.2, label='EKF2 z (relative)')
ax.plot(d['mocap_t'], d['mocap_pz'], 'tab:cyan', lw=1.0, alpha=0.7, label='Mocap z (relative)')
if len(d['ref_t']) > 0:
    ax.step(d['ref_t'], d['ref_pz'], 'tab:red', lw=1.5, where='post', label=f'Reference z (0.2m abs = {d["ref_pz"][0]:.3f}m rel)')
ax.axhline(0.2, color='green', ls=':', lw=1.0, alpha=0.7, label='Intended target 0.2m AGL')
ax.axhline(0.0, color='k', ls='-', lw=0.5, alpha=0.3, label='Ground level')
if d['kill_t'] is not None:
    ax.axvline(d['kill_t'], color='red', ls='--', lw=1.2, alpha=0.8, label=f"Kill @ {d['kill_t']:.1f}s")
ax.set_ylabel('Altitude (m)')
ax.set_title('Bag 05 - Altitude vs Time (offset subtracted: ground=0)')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(t_zoom)
ax.set_ylim(-0.2, 0.7)

# 2) Vertical velocity
ax = axes[1]
ax.plot(d['odom_t'], d['vz'], 'tab:green', lw=0.8, label='vz (world frame)')
if len(d['ref_vz']) > 0:
    ax.step(d['ref_t'], d['ref_vz'], 'tab:red', lw=1.5, where='post', label='ref vz')
ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
if d['kill_t'] is not None:
    ax.axvline(d['kill_t'], color='red', ls='--', lw=1.2, alpha=0.8)
ax.set_ylabel('Vertical velocity (m/s)')
ax.set_title('Bag 05 - Vertical velocity')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# 3) MPC thrust (u_mpc) - this is the NMPC solver output, NOT the actual motor command
ax = axes[2]
if len(d['mpc_t']) > 0:
    ax.plot(d['mpc_t'], d['mpc_fz'], 'tab:purple', lw=0.8, label='u_mpc[0] (NMPC solver output)')
    ax.axhline(W, color='orange', ls=':', lw=1.0, label=f'Hover weight ({W:.1f}N)')
if d['kill_t'] is not None:
    ax.axvline(d['kill_t'], color='red', ls='--', lw=1.2, alpha=0.8)
ax.set_ylabel('Thrust (N)')
ax.set_title('Bag 05 - NMPC solver output (u_mpc)')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# 4) HGDO disturbance fz
ax = axes[3]
if len(d['hgdo_t']) > 0:
    ax.plot(d['hgdo_t'], d['hgdo_fz'], 'tab:brown', lw=0.8, label='HGDO fz (disturbance estimate)')
    ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
if d['kill_t'] is not None:
    ax.axvline(d['kill_t'], color='red', ls='--', lw=1.2, alpha=0.8)
ax.set_ylabel('Disturbance fz (N)')
ax.set_title('Bag 05 - HGDO estimated vertical disturbance')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# 5) Effective thrust = u_mpc - hgdo_fz (what actually goes to motors in flight)
ax = axes[4]
if len(d['mpc_t']) > 0 and len(d['hgdo_t']) > 0:
    # Interpolate HGDO to MPC timestamps
    hgdo_interp = np.interp(d['mpc_t'], d['hgdo_t'], d['hgdo_fz'])
    f_effective = d['mpc_fz'] - hgdo_interp
    ax.plot(d['mpc_t'], f_effective, 'tab:red', lw=1.0, label='f_comp = u_mpc - hgdo_fz (motor command)')
    ax.plot(d['mpc_t'], d['mpc_fz'], 'tab:purple', lw=0.6, alpha=0.5, label='u_mpc (NMPC only)')
    ax.axhline(W, color='orange', ls=':', lw=1.0, label=f'Hover weight ({W:.1f}N)')
    ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
    # Mark where airborne detection would fire
    ax.annotate('DOB comp kicks in\n(airborne detected)',
                xy=(16.5, 37), fontsize=8, color='red',
                ha='center')
if d['kill_t'] is not None:
    ax.axvline(d['kill_t'], color='red', ls='--', lw=1.2, alpha=0.8)
ax.set_ylabel('Thrust (N)')
ax.set_xlabel('Time (s)')
ax.set_title('Bag 05 - Effective motor thrust (with DOB compensation)')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/drone_control_pkgs/bag_folder/2026_03_31_05/analysis.png', dpi=150)
plt.close()
print("\nSaved: /home/user/drone_control_pkgs/bag_folder/2026_03_31_05/analysis.png")


# ── Comparison plot: bags 03, 04, 05 altitude ──
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

for idx, (label, color) in enumerate([('03', 'tab:green'), ('04', 'tab:blue'), ('05', 'tab:red')]):
    if label not in bags:
        continue
    d = bags[label]
    ax = axes[idx]
    ax.plot(d['odom_t'], d['pz'], color=color, lw=1.0, label=f'EKF2 z (bag {label})')
    ax.plot(d['mocap_t'], d['mocap_pz'], color='tab:cyan', lw=0.8, alpha=0.7, label='Mocap z')
    if len(d['ref_t']) > 0:
        ax.step(d['ref_t'], d['ref_pz'], 'tab:orange', lw=1.5, where='post', label='Reference z')
    ax.axhline(0.2, color='green', ls=':', lw=1.0, alpha=0.5, label='Target 0.2m')
    if d['kill_t'] is not None:
        ax.axvline(d['kill_t'], color='red', ls='--', lw=1.2, alpha=0.8, label=f"Kill @ {d['kill_t']:.1f}s")
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f'Bag {label} - Altitude (duration={d["odom_t"][-1]:.1f}s, max_z={np.max(d["pz"]):.3f}m)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

axes[2].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('/home/user/drone_control_pkgs/bag_folder/2026_03_31_05/comparison_03_04_05.png', dpi=150)
plt.close()
print("Saved: /home/user/drone_control_pkgs/bag_folder/2026_03_31_05/comparison_03_04_05.png")

# ── Comparison: MPC thrust across bags ──
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
m = 3.144
W = m * 9.81

for idx, (label, color) in enumerate([('03', 'tab:green'), ('04', 'tab:blue'), ('05', 'tab:red')]):
    if label not in bags:
        continue
    d = bags[label]
    ax = axes[idx]
    if len(d['mpc_t']) > 0:
        ax.plot(d['mpc_t'], d['mpc_fz'], color=color, lw=0.8, label=f'MPC fz (bag {label})')
        ax.axhline(W, color='orange', ls=':', lw=1.0, label=f'Hover ({W:.1f}N)')
    if d['kill_t'] is not None:
        ax.axvline(d['kill_t'], color='red', ls='--', lw=1.2, alpha=0.8)
    ax.set_ylabel('Thrust (N)')
    ax.set_title(f'Bag {label} - MPC thrust')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

axes[2].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('/home/user/drone_control_pkgs/bag_folder/2026_03_31_05/comparison_thrust.png', dpi=150)
plt.close()
print("Saved: /home/user/drone_control_pkgs/bag_folder/2026_03_31_05/comparison_thrust.png")
