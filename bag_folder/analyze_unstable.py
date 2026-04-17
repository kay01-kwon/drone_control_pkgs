#!/usr/bin/env python3
"""Analyze sim_unstable bag with NMPC predicted trajectory."""

import sqlite3, struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def read_bag(db_path, topic_name, parse_fn):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT id FROM topics WHERE name=?", (topic_name,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return []
    tid = row[0]
    cur.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))
    results = []
    for ts, blob in cur.fetchall():
        parsed = parse_fn(blob)
        if parsed is not None:
            results.append((ts * 1e-9, parsed))
    conn.close()
    return results

def parse_odom(blob):
    try:
        off = 4
        off += 4; off += 4
        fid_len = struct.unpack_from('<I', blob, off)[0]; off += 4
        off += fid_len; off += (4 - off % 4) % 4
        cfl = struct.unpack_from('<I', blob, off)[0]; off += 4
        off += cfl
        data_off = off - 4
        data_off += (8 - data_off % 8) % 8
        off = data_off + 4
        px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
        qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
        off += 6*8
        vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
        wx, wy, wz = struct.unpack_from('<3d', blob, off)
        return np.array([px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz])
    except:
        return None

def parse_wrench(blob):
    try:
        fid_len = struct.unpack_from('<I', blob, 12)[0]
        data_off = 12 + fid_len
        data_off += (8 - data_off % 8) % 8
        off = 4 + data_off
        vals = struct.unpack_from('<6d', blob, off)
        return np.array(vals)
    except:
        return None

def parse_ref(blob):
    for test_off in [20, 24, 16, 28]:
        try:
            vals = struct.unpack_from('<8d', blob, test_off)
            p = np.array(vals[0:3])
            if abs(p[2]) < 10 and all(abs(x) < 100 for x in p):
                return np.array(vals)
        except:
            continue
    return None

def parse_rc(blob):
    try:
        nch = struct.unpack_from('<I', blob, 20)[0]
        channels = struct.unpack_from(f'<{nch}H', blob, 24)
        return np.array(channels, dtype=np.float64)
    except:
        return None

def parse_rpm(blob):
    try:
        rpms = struct.unpack_from('<6i', blob, 20)
        return np.array(rpms, dtype=np.float64)
    except:
        return None

def parse_path(blob):
    """Parse nav_msgs/Path CDR: extract predicted poses (position + quaternion)."""
    try:
        off = 4  # CDR encapsulation header
        # Header: stamp (sec + nanosec)
        off += 4; off += 4
        # frame_id string
        fid_len = struct.unpack_from('<I', blob, off)[0]; off += 4
        off += fid_len
        off += (4 - off % 4) % 4
        # poses sequence length
        n_poses = struct.unpack_from('<I', blob, off)[0]; off += 4

        poses = []
        for _ in range(n_poses):
            # PoseStamped header: stamp (sec+nanosec)
            off += 4; off += 4
            # frame_id string
            ps_fid_len = struct.unpack_from('<I', blob, off)[0]; off += 4
            off += ps_fid_len
            # Align to 8 bytes for doubles (relative to CDR data start at byte 4)
            data_off = off - 4
            data_off += (8 - data_off % 8) % 8
            off = data_off + 4
            # Pose: position (3 doubles) + orientation (4 doubles)
            px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
            qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
            poses.append([px, py, pz, qw, qx, qy, qz])
        return poses
    except:
        return None

def rc_to_mode(ch):
    if ch.shape[0] > 8 and ch[8] < 1200: return 'KILL'
    if ch.shape[0] > 7 and ch[7] < 1200: return 'DISARMED'
    if ch.shape[0] > 7 and ch[7] > 1700:
        if ch.shape[0] > 5 and ch[5] > 1700: return 'AUTO'
        elif ch.shape[0] > 5 and ch[5] > 1200: return 'MANUAL_STAB'
        else: return 'ARMED'
    return 'UNK'

def quat_to_euler(qw, qx, qy, qz):
    sinr = 2*(qw*qx + qy*qz); cosr = 1 - 2*(qx*qx + qy*qy)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2*(qw*qy - qz*qx), -1, 1)
    pitch = np.arcsin(sinp)
    siny = 2*(qw*qz + qx*qy); cosy = 1 - 2*(qy*qy + qz*qz)
    yaw = np.arctan2(siny, cosy)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def analyze_bag(db_path, bag_name, output_png):
    print(f"\n{'='*60}")
    print(f"Analyzing: {bag_name}")
    print(f"{'='*60}")

    od_raw = read_bag(db_path, '/mavros/local_position/odom', parse_odom)
    od_sim_raw = read_bag(db_path, '/mavros/local_position/odom_sim', parse_odom)
    wr_raw = read_bag(db_path, '/hgdo/wrench', parse_wrench)
    nm_raw = read_bag(db_path, '/nmpc/control', parse_wrench)
    ref_raw = read_bag(db_path, '/nmpc/ref', parse_ref)
    rc_raw = read_bag(db_path, '/mavros/rc/in', parse_rc)
    rpm_raw = read_bag(db_path, '/uav/actual_rpm', parse_rpm)
    path_raw = read_bag(db_path, '/nmpc/state', parse_path)

    if not od_raw:
        print("No odom data!"); return

    t0 = od_raw[0][0]
    od_t = np.array([d[0]-t0 for d in od_raw])
    od_v = np.array([d[1] for d in od_raw])
    p_off = od_v[0, 0:3].copy()
    print(f"Initial pos (raw): ({p_off[0]:.3f}, {p_off[1]:.3f}, {p_off[2]:.3f}) m")
    od_v[:, 0:3] -= p_off

    rolls, pitches, yaws = [], [], []
    for s in od_v:
        r, p, y = quat_to_euler(s[6], s[7], s[8], s[9])
        rolls.append(r); pitches.append(p); yaws.append(y)
    rolls = np.array(rolls); pitches = np.array(pitches); yaws = np.array(yaws)

    nm_t = np.array([d[0]-t0 for d in nm_raw]) if nm_raw else np.array([])
    nm_v = np.array([d[1] for d in nm_raw]) if nm_raw else np.zeros((0,6))
    wr_t = np.array([d[0]-t0 for d in wr_raw]) if wr_raw else np.array([])
    wr_v = np.array([d[1] for d in wr_raw]) if wr_raw else np.zeros((0,6))
    ref_t = np.array([d[0]-t0 for d in ref_raw]) if ref_raw else np.array([])
    ref_v = np.array([d[1] for d in ref_raw]) if ref_raw else np.zeros((0,8))
    rc_t = np.array([d[0]-t0 for d in rc_raw]) if rc_raw else np.array([])
    rc_v = np.array([d[1] for d in rc_raw]) if rc_raw else np.zeros((0,16))
    rpm_t = np.array([d[0]-t0 for d in rpm_raw]) if rpm_raw else np.array([])
    rpm_v = np.array([d[1] for d in rpm_raw]) if rpm_raw else np.zeros((0,6))

    # NMPC predicted trajectory: extract first and last pose of each Path
    path_t = np.array([d[0]-t0 for d in path_raw]) if path_raw else np.array([])
    pred_first = []  # current state as seen by NMPC
    pred_last = []   # end-of-horizon prediction
    pred_rolls_first, pred_pitches_first = [], []
    pred_rolls_last, pred_pitches_last = [], []
    for _, poses in path_raw:
        if poses and len(poses) > 0:
            p0 = poses[0]
            pred_first.append(p0[:3])
            r0, p0_att, _ = quat_to_euler(p0[3], p0[4], p0[5], p0[6])
            pred_rolls_first.append(r0)
            pred_pitches_first.append(p0_att)
            pN = poses[-1]
            pred_last.append(pN[:3])
            rN, pN_att, _ = quat_to_euler(pN[3], pN[4], pN[5], pN[6])
            pred_rolls_last.append(rN)
            pred_pitches_last.append(pN_att)
    pred_first = np.array(pred_first) if pred_first else np.zeros((0,3))
    pred_last = np.array(pred_last) if pred_last else np.zeros((0,3))
    pred_rolls_first = np.array(pred_rolls_first)
    pred_pitches_first = np.array(pred_pitches_first)
    pred_rolls_last = np.array(pred_rolls_last)
    pred_pitches_last = np.array(pred_pitches_last)

    # Subtract initial offset from predictions too
    if len(pred_first) > 0:
        pred_first -= p_off
        pred_last -= p_off

    C_T = 1.386e-7
    W = 3.146 * 9.81
    total_thrust = np.array([C_T * np.sum(r**2) for r in rpm_v]) if len(rpm_v) > 0 else np.array([])

    modes = [rc_to_mode(rc_v[i]) for i in range(len(rc_t))] if len(rc_t) > 0 else []

    print(f"\nFlight mode transitions:")
    if modes:
        prev = modes[0]
        print(f"  t={rc_t[0]:.1f}s: {prev}")
        for i in range(1, len(modes)):
            if modes[i] != prev:
                print(f"  t={rc_t[i]:.1f}s: {prev} -> {modes[i]}")
                prev = modes[i]

    print(f"\nRef msgs ({len(ref_t)}):")
    for i in range(min(5, len(ref_t))):
        r = ref_v[i]; print(f"  t={ref_t[i]:.1f}s: p=({r[0]:.3f},{r[1]:.3f},{r[2]:.3f})")

    print(f"\nAlt: [{od_v[:,2].min():.4f}, {od_v[:,2].max():.4f}] m, Duration: {od_t[-1]:.1f}s")
    print(f"Roll: [{rolls.min():.1f}, {rolls.max():.1f}] deg")
    print(f"Pitch: [{pitches.min():.1f}, {pitches.max():.1f}] deg")

    if len(nm_t) > 0:
        print(f"NMPC Fz: [{nm_v[:,2].min():.1f}, {nm_v[:,2].max():.1f}] N (W={W:.1f}N)")
        print(f"NMPC Mx: [{nm_v[:,3].min():.2f}, {nm_v[:,3].max():.2f}] Nm")
        print(f"NMPC My: [{nm_v[:,4].min():.2f}, {nm_v[:,4].max():.2f}] Nm")

    if len(path_t) > 0:
        print(f"\nNMPC predicted trajectory: {len(path_t)} msgs, {len(path_raw[0][1])} poses each")
        if len(pred_rolls_last) > 0:
            print(f"Pred horizon-end roll: [{pred_rolls_last.min():.1f}, {pred_rolls_last.max():.1f}] deg")
            print(f"Pred horizon-end pitch: [{pred_pitches_last.min():.1f}, {pred_pitches_last.max():.1f}] deg")

    # --- PLOT ---
    fig, axes = plt.subplots(5, 2, figsize=(18, 22))
    fig.suptitle(f'{bag_name} (Q_p=4, Q_att=20, soft Z=100, R=1)', fontsize=14)

    colors_mode = {'AUTO':'green', 'ARMED':'orange', 'MANUAL_STAB':'red',
                   'DISARMED':'gray', 'KILL':'black', 'UNK':'purple'}
    def shade_modes(ax):
        if not modes: return
        prev_t = rc_t[0]; prev_m = modes[0]
        for i in range(1, len(modes)):
            if modes[i] != prev_m or i == len(modes)-1:
                c = colors_mode.get(prev_m, 'white')
                ax.axvspan(prev_t, rc_t[i], alpha=0.08, color=c)
                prev_t = rc_t[i]; prev_m = modes[i]

    # 1. Position + NMPC predicted
    ax = axes[0,0]
    for i, (l,c) in enumerate(zip('xyz', 'rgb')):
        ax.plot(od_t, od_v[:,i], c, label=f'{l}_odom', alpha=0.8, lw=0.8)
        if len(path_t) > 0 and len(pred_last) > 0:
            ax.plot(path_t, pred_last[:,i], c, ls=':', alpha=0.4, lw=0.6, label=f'{l}_pred_end')
    if len(ref_t) > 0:
        for i, c in enumerate('rgb'):
            ax.step(ref_t, ref_v[:,i], c, ls='--', alpha=0.4)
    shade_modes(ax); ax.set_ylabel('Pos [m]'); ax.legend(fontsize=6); ax.set_title('Position (odom vs predicted end)'); ax.grid(True, alpha=0.3)

    # 2. Attitude + NMPC predicted attitude
    ax = axes[0,1]
    ax.plot(od_t, rolls, 'r', label='roll', alpha=0.8, lw=0.8)
    ax.plot(od_t, pitches, 'g', label='pitch', alpha=0.8, lw=0.8)
    if len(path_t) > 0 and len(pred_rolls_last) > 0:
        ax.plot(path_t, pred_rolls_last, 'r', ls=':', alpha=0.4, lw=0.6, label='roll_pred_end')
        ax.plot(path_t, pred_pitches_last, 'g', ls=':', alpha=0.4, lw=0.6, label='pitch_pred_end')
    ax.axhline(20, color='gray', ls='--', alpha=0.3); ax.axhline(-20, color='gray', ls='--', alpha=0.3)
    shade_modes(ax); ax.set_ylabel('Angle [deg]'); ax.legend(fontsize=6); ax.set_title('Attitude (odom vs predicted end)'); ax.grid(True, alpha=0.3)

    # 3. Velocity
    ax = axes[1,0]
    for i, (l,c) in enumerate(zip(['vx','vy','vz'], 'rgb')):
        ax.plot(od_t, od_v[:,3+i], c, label=l, alpha=0.8, lw=0.8)
    shade_modes(ax); ax.set_ylabel('Vel [m/s]'); ax.legend(fontsize=7); ax.set_title('Velocity'); ax.grid(True, alpha=0.3)

    # 4. Angular rates
    ax = axes[1,1]
    for i, (l,c) in enumerate(zip(['wx','wy','wz'], 'rgb')):
        ax.plot(od_t, od_v[:,10+i], c, label=l, alpha=0.8, lw=0.8)
    shade_modes(ax); ax.set_ylabel('Rate [rad/s]'); ax.legend(fontsize=7); ax.set_title('Angular Rates'); ax.grid(True, alpha=0.3)

    # 5. NMPC Fz
    ax = axes[2,0]
    if len(nm_t) > 0:
        ax.plot(nm_t, nm_v[:,2], 'b', label='Fz', alpha=0.8, lw=0.8)
        ax.axhline(W, color='r', ls='--', alpha=0.5, label=f'W={W:.1f}N')
    shade_modes(ax); ax.set_ylabel('Force [N]'); ax.legend(fontsize=7); ax.set_title('NMPC Fz'); ax.grid(True, alpha=0.3)

    # 6. NMPC Moments
    ax = axes[2,1]
    if len(nm_t) > 0:
        for i, (l,c) in enumerate(zip(['Mx','My','Mz'], 'rgb')):
            ax.plot(nm_t, nm_v[:,3+i], c, label=l, alpha=0.8, lw=0.8)
    shade_modes(ax); ax.set_ylabel('Moment [Nm]'); ax.legend(fontsize=7); ax.set_title('NMPC Moments'); ax.grid(True, alpha=0.3)

    # 7. HGDO forces
    ax = axes[3,0]
    if len(wr_t) > 0:
        for i, (l,c) in enumerate(zip(['fx','fy','fz'], 'rgb')):
            ax.plot(wr_t, wr_v[:,i], c, label=l, alpha=0.8, lw=0.8)
    shade_modes(ax); ax.set_ylabel('Force [N]'); ax.legend(fontsize=7); ax.set_title('HGDO Force (body)'); ax.grid(True, alpha=0.3)

    # 8. HGDO torques
    ax = axes[3,1]
    if len(wr_t) > 0:
        for i, (l,c) in enumerate(zip(['tx','ty','tz'], 'rgb')):
            ax.plot(wr_t, wr_v[:,3+i], c, label=l, alpha=0.8, lw=0.8)
    shade_modes(ax); ax.set_ylabel('Torque [Nm]'); ax.legend(fontsize=7); ax.set_title('HGDO Torque (body)'); ax.grid(True, alpha=0.3)

    # 9. Thrust vs Weight
    ax = axes[4,0]
    if len(rpm_t) > 0 and len(total_thrust) > 0:
        ax.plot(rpm_t, total_thrust, 'b', label='Thrust', alpha=0.8, lw=0.8)
        ax.axhline(W, color='r', ls='--', alpha=0.5, label=f'W={W:.1f}N')
    shade_modes(ax); ax.set_ylabel('Thrust [N]'); ax.legend(fontsize=7); ax.set_title('Thrust vs Weight'); ax.grid(True, alpha=0.3); ax.set_xlabel('Time [s]')

    # 10. Motor RPMs
    ax = axes[4,1]
    if len(rpm_t) > 0:
        for i in range(min(6, rpm_v.shape[1])):
            ax.plot(rpm_t, rpm_v[:,i], alpha=0.5, lw=0.6, label=f'M{i+1}')
    shade_modes(ax); ax.set_ylabel('RPM'); ax.legend(fontsize=6); ax.set_title('Motor RPMs'); ax.grid(True, alpha=0.3); ax.set_xlabel('Time [s]')

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"\nSaved: {output_png}")
    plt.close()

if __name__ == '__main__':
    base = Path('/home/user/drone_control_pkgs/bag_folder')
    db = base / '2026_04_17_sim_unstable' / '2026_04_17_sim_unstable_0.db3'
    if db.exists():
        analyze_bag(db, 'sim_unstable', str(base / 'sim_unstable_analysis.png'))
    else:
        print(f"DB not found: {db}")
