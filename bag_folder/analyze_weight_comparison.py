#!/usr/bin/env python3
"""Compare 3 NMPC attitude experiments with different angular velocity weights.
   exp01: w_xy=0.3, exp02: w_xy=0.6, exp03: w_xy=0.8
   Des RPY = (0,0,0), des_thrust = 32N, HGDO compensation.
"""

import sqlite3, struct, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)

EXPERIMENTS = {
    '01': {'db': '2026_04_24_att_exp_01/2026_04_24_att_exp_01_0.db3',
           'label': 'Exp01 (w_xy=0.3)', 'color': 'C0',
           'Q': [20,20,2, 0.3,0.3,0.01], 'R': 1.0},
    '02': {'db': '2026_04_24_att_exp_02/2026_04_24_att_exp_02_0.db3',
           'label': 'Exp02 (w_xy=0.6)', 'color': 'C1',
           'Q': [20,20,2, 0.6,0.6,0.01], 'R': 1.0},
    '03': {'db': '2026_04_24_att_exp_03/2026_04_24_att_exp_03_0.db3',
           'label': 'Exp03 (w_xy=0.8)', 'color': 'C2',
           'Q': [20,20,2, 0.8,0.8,0.01], 'R': 1.0},
}


def parse_odom(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    if off % 4: off += 4 - off % 4
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    if off % 4: off += 4 - off % 4
    off += 4  # XCDR2 padding
    px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
    off += 36 * 8
    vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
    wx, wy, wz = struct.unpack_from('<3d', blob, off)
    return np.array([px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz])


def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    if off % 4: off += 4 - off % 4
    return np.array(struct.unpack_from('<6d', blob, off))


def quat_to_rpy(q):
    qw, qx, qy, qz = q
    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    sinp = np.clip(2*(qw*qy - qz*qx), -1, 1)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    return np.array([roll, pitch, yaw])


def quat_to_rotm(q):
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),    1-2*(qx**2+qy**2)],
    ])


def fetch(conn, topic_name, parser):
    c = conn.cursor()
    c.execute('SELECT id FROM topics WHERE name=?', (topic_name,))
    row = c.fetchone()
    if row is None:
        return np.array([]), np.array([])
    topic_id = row[0]
    c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (topic_id,))
    ts_list, data_list = [], []
    for ts, data in c.fetchall():
        try:
            parsed = parser(bytes(data))
            ts_list.append(ts)
            data_list.append(parsed)
        except Exception:
            pass
    return np.array(ts_list, dtype=np.float64), np.array(data_list)


# ── Load all experiments ──
all_data = {}
for exp_id, cfg in EXPERIMENTS.items():
    db_path = os.path.join(_HERE, cfg['db'])
    conn = sqlite3.connect(db_path)
    
    odom_ts, odom = fetch(conn, '/mavros/local_position/odom', parse_odom)
    ctrl_ts, ctrl = fetch(conn, '/nmpc/control', parse_wrench)
    hgdo_ts, hgdo = fetch(conn, '/hgdo/wrench', parse_wrench)
    conn.close()
    
    t0 = odom_ts[0]
    odom_t = (odom_ts - t0) * 1e-9
    ctrl_t = (ctrl_ts - t0) * 1e-9
    hgdo_t = (hgdo_ts - t0) * 1e-9
    
    rpy = np.array([quat_to_rpy(odom[i, 6:10]) for i in range(len(odom))])
    rpy_deg = np.degrees(rpy)
    
    # Find ctrl start (AUTO mode)
    ctrl_start = ctrl_t[0] if len(ctrl_t) > 0 else 0
    
    # Steady-state mask: after initial transient (5s after ctrl start)
    ss_start = ctrl_start + 5.0
    mask_ss = odom_t >= ss_start
    mask_ctrl_ss = ctrl_t >= ss_start
    mask_hgdo_ss = hgdo_t >= ss_start
    
    all_data[exp_id] = {
        'odom_t': odom_t, 'odom': odom, 'rpy_deg': rpy_deg,
        'ctrl_t': ctrl_t, 'ctrl': ctrl,
        'hgdo_t': hgdo_t, 'hgdo': hgdo,
        'ctrl_start': ctrl_start, 'ss_start': ss_start,
        'mask_ss': mask_ss, 'mask_ctrl_ss': mask_ctrl_ss,
        'mask_hgdo_ss': mask_hgdo_ss,
        'cfg': cfg,
    }
    
    print(f"Exp{exp_id} ({cfg['label']}): {odom_t[-1]:.1f}s, ctrl from {ctrl_start:.1f}s")
    if mask_ss.sum() > 0:
        for i, lbl in enumerate(['Roll','Pitch','Yaw']):
            vals = rpy_deg[mask_ss, i]
            print(f"  {lbl} SS: mean={vals.mean():+.2f}, std={vals.std():.3f}, range=[{vals.min():.1f},{vals.max():.1f}] deg")
        for i, lbl in enumerate(['wx','wy','wz']):
            vals = odom[mask_ss, 10+i]
            print(f"  {lbl} SS: mean={vals.mean():+.4f}, std={vals.std():.4f} rad/s")


# ═══════════════════════════════════════════════════════════
# PLOT 1: RPY comparison (3 columns, one per experiment)
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 3, figsize=(22, 14), sharex='col')

for col, exp_id in enumerate(['01', '02', '03']):
    d = all_data[exp_id]
    ot = d['odom_t'] - d['ctrl_start']  # align to ctrl start
    ct = d['ctrl_t'] - d['ctrl_start']
    
    # Roll, Pitch
    ax = axes[0, col]
    ax.plot(ot, d['rpy_deg'][:, 0], 'r', alpha=0.8, linewidth=0.7, label='Roll')
    ax.plot(ot, d['rpy_deg'][:, 1], 'g', alpha=0.8, linewidth=0.7, label='Pitch')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_ylabel('Angle [deg]')
    ax.set_title(d['cfg']['label'])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-25, 25])
    
    # Angular velocity
    ax = axes[1, col]
    ax.plot(ot, d['odom'][:, 10], 'r', alpha=0.7, linewidth=0.5, label='wx')
    ax.plot(ot, d['odom'][:, 11], 'g', alpha=0.7, linewidth=0.5, label='wy')
    ax.plot(ot, d['odom'][:, 12], 'b', alpha=0.7, linewidth=0.5, label='wz')
    ax.set_ylabel('Angular vel [rad/s]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.5, 1.5])
    
    # NMPC Moments
    ax = axes[2, col]
    ax.plot(ct, d['ctrl'][:, 3], 'r', alpha=0.7, linewidth=0.5, label='Mx')
    ax.plot(ct, d['ctrl'][:, 4], 'g', alpha=0.7, linewidth=0.5, label='My')
    ax.plot(ct, d['ctrl'][:, 5], 'b', alpha=0.7, linewidth=0.5, label='Mz')
    ax.set_ylabel('Moment [Nm]')
    ax.set_xlabel('Time [s] (from ctrl start)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 0.5])

plt.suptitle('NMPC Attitude Experiments — Weight Comparison\n'
             'Des RPY=(0,0,0), Des Thrust=32N, HGDO compensation',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('weight_comparison_rpy.png', dpi=150)
print('\nSaved weight_comparison_rpy.png')


# ═══════════════════════════════════════════════════════════
# PLOT 2: Overlay comparison (same axes, all 3 experiments)
# ═══════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(4, 2, figsize=(20, 18))

# Roll overlay
ax = axes2[0, 0]
for exp_id in ['01', '02', '03']:
    d = all_data[exp_id]
    ot = d['odom_t'] - d['ctrl_start']
    ax.plot(ot, d['rpy_deg'][:, 0], color=d['cfg']['color'], alpha=0.7,
            linewidth=0.7, label=d['cfg']['label'])
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_ylabel('Roll [deg]')
ax.set_title('Roll Comparison')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Pitch overlay
ax = axes2[0, 1]
for exp_id in ['01', '02', '03']:
    d = all_data[exp_id]
    ot = d['odom_t'] - d['ctrl_start']
    ax.plot(ot, d['rpy_deg'][:, 1], color=d['cfg']['color'], alpha=0.7,
            linewidth=0.7, label=d['cfg']['label'])
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_ylabel('Pitch [deg]')
ax.set_title('Pitch Comparison')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# wx overlay
ax = axes2[1, 0]
for exp_id in ['01', '02', '03']:
    d = all_data[exp_id]
    ot = d['odom_t'] - d['ctrl_start']
    ax.plot(ot, d['odom'][:, 10], color=d['cfg']['color'], alpha=0.6,
            linewidth=0.5, label=d['cfg']['label'])
ax.set_ylabel('wx [rad/s]')
ax.set_title('Roll Rate (wx)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# wy overlay
ax = axes2[1, 1]
for exp_id in ['01', '02', '03']:
    d = all_data[exp_id]
    ot = d['odom_t'] - d['ctrl_start']
    ax.plot(ot, d['odom'][:, 11], color=d['cfg']['color'], alpha=0.6,
            linewidth=0.5, label=d['cfg']['label'])
ax.set_ylabel('wy [rad/s]')
ax.set_title('Pitch Rate (wy)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Mx overlay
ax = axes2[2, 0]
for exp_id in ['01', '02', '03']:
    d = all_data[exp_id]
    ct = d['ctrl_t'] - d['ctrl_start']
    ax.plot(ct, d['ctrl'][:, 3], color=d['cfg']['color'], alpha=0.6,
            linewidth=0.5, label=d['cfg']['label'])
ax.set_ylabel('Mx [Nm]')
ax.set_title('NMPC Moment Mx (Roll)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# My overlay
ax = axes2[2, 1]
for exp_id in ['01', '02', '03']:
    d = all_data[exp_id]
    ct = d['ctrl_t'] - d['ctrl_start']
    ax.plot(ct, d['ctrl'][:, 4], color=d['cfg']['color'], alpha=0.6,
            linewidth=0.5, label=d['cfg']['label'])
ax.set_ylabel('My [Nm]')
ax.set_title('NMPC Moment My (Pitch)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# HGDO tx overlay
ax = axes2[3, 0]
for exp_id in ['01', '02', '03']:
    d = all_data[exp_id]
    ht = d['hgdo_t'] - d['ctrl_start']
    ax.plot(ht, d['hgdo'][:, 3], color=d['cfg']['color'], alpha=0.6,
            linewidth=0.5, label=d['cfg']['label'])
ax.set_ylabel('tau_x [Nm]')
ax.set_xlabel('Time [s]')
ax.set_title('HGDO Torque tx (Roll)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# HGDO ty overlay
ax = axes2[3, 1]
for exp_id in ['01', '02', '03']:
    d = all_data[exp_id]
    ht = d['hgdo_t'] - d['ctrl_start']
    ax.plot(ht, d['hgdo'][:, 4], color=d['cfg']['color'], alpha=0.6,
            linewidth=0.5, label=d['cfg']['label'])
ax.set_ylabel('tau_y [Nm]')
ax.set_xlabel('Time [s]')
ax.set_title('HGDO Torque ty (Pitch)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('Weight Comparison Overlay — w_xy: 0.3 vs 0.6 vs 0.8\n'
             'Des RPY=(0,0,0), Des Thrust=32N',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('weight_comparison_overlay.png', dpi=150)
print('Saved weight_comparison_overlay.png')


# ═══════════════════════════════════════════════════════════
# PLOT 3: Statistics bar chart
# ═══════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))

exp_labels = []
roll_std, pitch_std, yaw_std = [], [], []
wx_std, wy_std, wz_std = [], [], []
mx_std, my_std = [], []

for exp_id in ['01', '02', '03']:
    d = all_data[exp_id]
    mask = d['mask_ss']
    mask_c = d['mask_ctrl_ss']
    exp_labels.append(d['cfg']['label'])
    
    roll_std.append(d['rpy_deg'][mask, 0].std())
    pitch_std.append(d['rpy_deg'][mask, 1].std())
    yaw_std.append(d['rpy_deg'][mask, 2].std())
    
    wx_std.append(d['odom'][mask, 10].std())
    wy_std.append(d['odom'][mask, 11].std())
    wz_std.append(d['odom'][mask, 12].std())
    
    mx_std.append(d['ctrl'][mask_c, 3].std())
    my_std.append(d['ctrl'][mask_c, 4].std())

x = np.arange(3)
w = 0.25

ax = axes3[0]
ax.bar(x - w, roll_std, w, label='Roll σ', color='r', alpha=0.7)
ax.bar(x, pitch_std, w, label='Pitch σ', color='g', alpha=0.7)
ax.bar(x + w, yaw_std, w, label='Yaw σ', color='b', alpha=0.7)
ax.set_ylabel('Std [deg]')
ax.set_title('RPY Oscillation (Steady-State σ)')
ax.set_xticks(x)
ax.set_xticklabels([f'w_xy={v}' for v in ['0.3','0.6','0.8']])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes3[1]
ax.bar(x - w/2, wx_std, w, label='wx σ', color='r', alpha=0.7)
ax.bar(x + w/2, wy_std, w, label='wy σ', color='g', alpha=0.7)
ax.set_ylabel('Std [rad/s]')
ax.set_title('Angular Velocity Oscillation (Steady-State σ)')
ax.set_xticks(x)
ax.set_xticklabels([f'w_xy={v}' for v in ['0.3','0.6','0.8']])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes3[2]
ax.bar(x - w/2, mx_std, w, label='Mx σ', color='r', alpha=0.7)
ax.bar(x + w/2, my_std, w, label='My σ', color='g', alpha=0.7)
ax.set_ylabel('Std [Nm]')
ax.set_title('NMPC Moment Oscillation (Steady-State σ)')
ax.set_xticks(x)
ax.set_xticklabels([f'w_xy={v}' for v in ['0.3','0.6','0.8']])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Steady-State Oscillation Comparison (5s after ctrl start)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('weight_comparison_stats.png', dpi=150)
print('Saved weight_comparison_stats.png')

# Print summary table
print('\n' + '='*80)
print('STEADY-STATE STATISTICS (5s after ctrl start)')
print('='*80)
print(f'{"Metric":<25} {"w_xy=0.3":>12} {"w_xy=0.6":>12} {"w_xy=0.8":>12}  {"Best":>8}')
print('-'*80)
metrics = [
    ('Roll σ [deg]', roll_std),
    ('Pitch σ [deg]', pitch_std),
    ('Yaw σ [deg]', yaw_std),
    ('wx σ [rad/s]', wx_std),
    ('wy σ [rad/s]', wy_std),
    ('wz σ [rad/s]', wz_std),
    ('Mx σ [Nm]', mx_std),
    ('My σ [Nm]', my_std),
]
for name, vals in metrics:
    best_idx = np.argmin(vals)
    best_label = ['0.3','0.6','0.8'][best_idx]
    print(f'{name:<25} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f}  {best_label:>8}')

# Mean values
print('\n' + '-'*80)
print(f'{"Mean offset":<25} {"w_xy=0.3":>12} {"w_xy=0.6":>12} {"w_xy=0.8":>12}')
print('-'*80)
for exp_id, label in zip(['01','02','03'], ['0.3','0.6','0.8']):
    d = all_data[exp_id]
    mask = d['mask_ss']
    r_mean = d['rpy_deg'][mask, 0].mean()
    p_mean = d['rpy_deg'][mask, 1].mean()
    y_mean = d['rpy_deg'][mask, 2].mean()
    if exp_id == '01':
        print(f'{"Roll mean [deg]":<25} {r_mean:>+12.2f}', end='')
        r_means = [r_mean]
    elif exp_id == '02':
        print(f' {r_mean:>+12.2f}', end='')
        r_means.append(r_mean)
    else:
        print(f' {r_mean:>+12.2f}')
        r_means.append(r_mean)

for exp_id in ['01','02','03']:
    d = all_data[exp_id]
    mask = d['mask_ss']
    p_mean = d['rpy_deg'][mask, 1].mean()
    if exp_id == '01':
        print(f'{"Pitch mean [deg]":<25} {p_mean:>+12.2f}', end='')
    elif exp_id == '02':
        print(f' {p_mean:>+12.2f}', end='')
    else:
        print(f' {p_mean:>+12.2f}')

for exp_id in ['01','02','03']:
    d = all_data[exp_id]
    mask = d['mask_ss']
    y_mean = d['rpy_deg'][mask, 2].mean()
    if exp_id == '01':
        print(f'{"Yaw mean [deg]":<25} {y_mean:>+12.2f}', end='')
    elif exp_id == '02':
        print(f' {y_mean:>+12.2f}', end='')
    else:
        print(f' {y_mean:>+12.2f}')

