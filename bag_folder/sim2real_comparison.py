#!/usr/bin/env python3
"""Sim2Real comparison: 2026_04_01_sim vs 2026_03_31_05 (real flight)."""

import sqlite3
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ── Constants ──
C_T = 1.386e-07
k_m = 0.01569
l = 0.265
MaxBit = 8191
MaxRpm = 9800
m = 3.146
W = m * 9.81

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


def parse_odom(data, pos_off, q_off, v_off, w_off):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, pos_off)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, q_off)
    vx, vy, vz = struct.unpack_from('<3d', data, v_off)
    wx, wy, wz = struct.unpack_from('<3d', data, w_off)
    t = sec + nsec * 1e-9
    return t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz


def parse_wrench_stamped(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    fx, fy, fz = struct.unpack_from('<3d', data, 28)
    tx, ty, tz = struct.unpack_from('<3d', data, 52)
    return sec + nsec * 1e-9, fx, fy, fz, tx, ty, tz


def parse_actual_rpm(data):
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    off = (off + 3) & ~3
    rpms = np.array(struct.unpack_from('<6i', data, off), dtype=np.float64)
    return sec + nsec * 1e-9, rpms


def parse_cmd_raw(data):
    off = 4
    sec = struct.unpack_from('<I', data, off)[0]; off += 4
    nsec = struct.unpack_from('<I', data, off)[0]; off += 4
    flen = struct.unpack_from('<I', data, off)[0]; off += 4
    off += flen
    if off % 2 != 0: off += 1
    cmds = np.array(struct.unpack_from('<6h', data, off), dtype=np.float64)
    return sec + nsec * 1e-9, cmds


def load_common(db_path, odom_offsets):
    """Load odom, hgdo, nmpc/control, actual_rpm, cmd_raw from a bag."""
    pos_off, q_off, v_off, w_off = odom_offsets
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name FROM topics')
    topics = {name: tid for tid, name in c.fetchall()}

    # Odom
    tid = topics['/mavros/local_position/odom']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    odom_t, odom_px, odom_py, odom_pz = [], [], [], []
    odom_vx, odom_vy, odom_vz = [], [], []
    odom_wx, odom_wy, odom_wz = [], [], []
    odom_roll, odom_pitch, odom_yaw = [], [], []
    for data, in c.fetchall():
        t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = parse_odom(
            data, pos_off, q_off, v_off, w_off)
        q = np.array([qx, qy, qz, qw]); norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm < 1e-10: continue
        R_mat = Rotation.from_quat(q / norm)
        roll, pitch, yaw = R_mat.as_euler('xyz', degrees=True)
        v_world = R_mat.as_matrix() @ np.array([vx, vy, vz])
        odom_t.append(t); odom_px.append(px); odom_py.append(py); odom_pz.append(pz)
        odom_vx.append(v_world[0]); odom_vy.append(v_world[1]); odom_vz.append(v_world[2])
        odom_wx.append(wx); odom_wy.append(wy); odom_wz.append(wz)
        odom_roll.append(roll); odom_pitch.append(pitch); odom_yaw.append(yaw)

    # HGDO
    tid = topics['/hgdo/wrench']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    hgdo_t, hgdo_fx, hgdo_fy, hgdo_fz = [], [], [], []
    hgdo_tx, hgdo_ty, hgdo_tz = [], [], []
    for data, in c.fetchall():
        t, fx, fy, fz, tx, ty, tz = parse_wrench_stamped(data)
        hgdo_t.append(t); hgdo_fx.append(fx); hgdo_fy.append(fy); hgdo_fz.append(fz)
        hgdo_tx.append(tx); hgdo_ty.append(ty); hgdo_tz.append(tz)

    # NMPC control
    tid = topics['/nmpc/control']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    mpc_t, mpc_F, mpc_Mx, mpc_My, mpc_Mz = [], [], [], [], []
    for data, in c.fetchall():
        t, fx, fy, fz, tx, ty, tz = parse_wrench_stamped(data)
        mpc_t.append(t); mpc_F.append(fz)
        mpc_Mx.append(tx); mpc_My.append(ty); mpc_Mz.append(tz)

    # Actual RPM
    tid = topics['/uav/actual_rpm']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    rpm_t, rpm_raw = [], []
    for data, in c.fetchall():
        t, rpms = parse_actual_rpm(data)
        rpm_t.append(t); rpm_raw.append(rpms.copy())

    # Cmd raw
    tid = topics['/uav/cmd_raw']
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,))
    cmd_t, cmd_F, cmd_rpm_raw = [], [], []
    for data, in c.fetchall():
        t, cmds = parse_cmd_raw(data)
        rpms = cmds * MaxRpm / MaxBit
        cmd_rpm_raw.append(rpms.copy())
        thrusts = C_T * rpms ** 2
        u = K_forward @ thrusts
        cmd_t.append(t); cmd_F.append(u[0])

    conn.close()

    # Convert
    odom_t = np.array(odom_t)
    t0 = odom_t[0]
    odom_t -= t0
    hgdo_t = np.array(hgdo_t) - t0
    mpc_t = np.array(mpc_t) - t0
    rpm_t = np.array(rpm_t) - t0
    cmd_t = np.array(cmd_t) - t0

    odom_pz = np.array(odom_pz); odom_pz -= odom_pz[0]

    return dict(
        odom_t=odom_t,
        odom_px=np.array(odom_px) - np.array(odom_px)[0],
        odom_py=np.array(odom_py) - np.array(odom_py)[0],
        odom_pz=odom_pz,
        odom_vx=np.array(odom_vx), odom_vy=np.array(odom_vy), odom_vz=np.array(odom_vz),
        odom_wx=np.array(odom_wx), odom_wy=np.array(odom_wy), odom_wz=np.array(odom_wz),
        odom_roll=np.array(odom_roll), odom_pitch=np.array(odom_pitch), odom_yaw=np.array(odom_yaw),
        hgdo_t=hgdo_t,
        hgdo_fx=np.array(hgdo_fx), hgdo_fy=np.array(hgdo_fy), hgdo_fz=np.array(hgdo_fz),
        hgdo_tx=np.array(hgdo_tx), hgdo_ty=np.array(hgdo_ty), hgdo_tz=np.array(hgdo_tz),
        mpc_t=mpc_t, mpc_F=np.array(mpc_F),
        mpc_Mx=np.array(mpc_Mx), mpc_My=np.array(mpc_My), mpc_Mz=np.array(mpc_Mz),
        rpm_t=rpm_t, rpm_raw=np.array(rpm_raw),
        cmd_t=cmd_t, cmd_F=np.array(cmd_F), cmd_rpm_raw=np.array(cmd_rpm_raw),
    )


def find_flight_window(d, thrust_threshold=None):
    """Find approximate flight window based on cmd thrust > hover weight * 0.7."""
    if thrust_threshold is None:
        thrust_threshold = W * 0.7
    flying = d['cmd_F'] > thrust_threshold
    if not np.any(flying):
        return 0, d['cmd_t'][-1]
    first = np.argmax(flying)
    last = len(flying) - 1 - np.argmax(flying[::-1])
    return d['cmd_t'][first], d['cmd_t'][last]


def compute_noise_stats(signal, t, t_start, t_end):
    """Compute std and mean of signal within a time window."""
    mask = (t >= t_start) & (t <= t_end)
    if np.sum(mask) < 10:
        return 0, 0
    seg = signal[mask]
    return np.mean(seg), np.std(seg)


def compute_psd(signal, dt_mean):
    """Compute power spectral density."""
    n = len(signal)
    if n < 64:
        return np.array([]), np.array([])
    # Detrend
    signal = signal - np.mean(signal)
    # Hanning window
    window = np.hanning(n)
    signal = signal * window
    fft_vals = np.fft.rfft(signal)
    psd = np.abs(fft_vals) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=dt_mean)
    return freqs, psd


# ── Load bags ──
print("Loading SIM bag (2026_04_01_sim)...")
sim = load_common(
    'bag_folder/2026_04_01_sim/2026_04_01_sim_0.db3',
    odom_offsets=(36, 60, 380, 404)
)

print("Loading REAL bag (2026_03_31_05)...")
real = load_common(
    'bag_folder/2026_03_31_05/2026_03_31_05_0.db3',
    odom_offsets=(44, 68, 388, 412)
)

# Find flight windows
sim_start, sim_end = find_flight_window(sim)
real_start, real_end = find_flight_window(real)
print(f"\nSIM flight window: {sim_start:.1f} - {sim_end:.1f} s")
print(f"REAL flight window: {real_start:.1f} - {real_end:.1f} s")

# ── Stats comparison ──
print(f"\n{'='*70}")
print(f"{'Metric':<40} {'SIM':>12} {'REAL':>12}")
print(f"{'='*70}")

for label, key, t_key in [
    ('HGDO fz mean (flight)', 'hgdo_fz', 'hgdo_t'),
    ('HGDO fz std (flight)', 'hgdo_fz', 'hgdo_t'),
    ('HGDO fx mean (flight)', 'hgdo_fx', 'hgdo_t'),
    ('HGDO fy mean (flight)', 'hgdo_fy', 'hgdo_t'),
    ('odom vz std (flight)', 'odom_vz', 'odom_t'),
    ('odom wx std (flight)', 'odom_wx', 'odom_t'),
    ('odom wy std (flight)', 'odom_wy', 'odom_t'),
    ('odom wz std (flight)', 'odom_wz', 'odom_t'),
]:
    sim_mean, sim_std = compute_noise_stats(sim[key], sim[t_key], sim_start, sim_end)
    real_mean, real_std = compute_noise_stats(real[key], real[t_key], real_start, real_end)
    if 'std' in label:
        print(f"  {label:<40} {sim_std:>12.4f} {real_std:>12.4f}")
    else:
        print(f"  {label:<40} {sim_mean:>12.4f} {real_mean:>12.4f}")

# HGDO full range
print(f"\n  {'HGDO fz range (full)':<40} [{sim['hgdo_fz'].min():.1f}, {sim['hgdo_fz'].max():.1f}] [{real['hgdo_fz'].min():.1f}, {real['hgdo_fz'].max():.1f}]")
print(f"  {'HGDO fz range (flight)':<40}", end='')
sim_mask = (sim['hgdo_t'] >= sim_start) & (sim['hgdo_t'] <= sim_end)
real_mask = (real['hgdo_t'] >= real_start) & (real['hgdo_t'] <= real_end)
print(f" [{sim['hgdo_fz'][sim_mask].min():.1f}, {sim['hgdo_fz'][sim_mask].max():.1f}]", end='')
print(f" [{real['hgdo_fz'][real_mask].min():.1f}, {real['hgdo_fz'][real_mask].max():.1f}]")

print(f"{'='*70}")

# ═══════════════════════════════════════════════════════════
# PLOT 1: HGDO disturbance comparison (sim vs real)
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 2, figsize=(18, 16))
fig.suptitle('Sim2Real: HGDO Disturbance Comparison\n'
             'LEFT=SIM (2026_04_01_sim)  RIGHT=REAL (2026_03_31_05)', fontsize=14, fontweight='bold')

for col, (d, name, fs, fe) in enumerate([
    (sim, 'SIM', sim_start, sim_end),
    (real, 'REAL', real_start, real_end),
]):
    # Row 0: HGDO force xyz
    ax = axes[0, col]
    ax.plot(d['hgdo_t'], d['hgdo_fx'], 'tab:red', lw=0.6, alpha=0.8, label='fx')
    ax.plot(d['hgdo_t'], d['hgdo_fy'], 'tab:blue', lw=0.6, alpha=0.8, label='fy')
    ax.plot(d['hgdo_t'], d['hgdo_fz'], 'tab:green', lw=0.8, label='fz')
    ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
    ax.axvspan(fs, fe, alpha=0.1, color='yellow')
    ax.set_ylabel('Force (N)')
    ax.set_title(f'{name} - HGDO disturbance force')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Row 1: HGDO fz zoomed to flight
    ax = axes[1, col]
    mask = (d['hgdo_t'] >= fs) & (d['hgdo_t'] <= fe)
    ax.plot(d['hgdo_t'][mask], d['hgdo_fz'][mask], 'tab:green', lw=0.8, label='fz')
    ax.plot(d['hgdo_t'][mask], d['hgdo_fx'][mask], 'tab:red', lw=0.6, alpha=0.7, label='fx')
    ax.plot(d['hgdo_t'][mask], d['hgdo_fy'][mask], 'tab:blue', lw=0.6, alpha=0.7, label='fy')
    ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
    fz_flight = d['hgdo_fz'][mask]
    ax.set_ylabel('Force (N)')
    ax.set_title(f'{name} - HGDO force (flight only) | fz: [{fz_flight.min():.1f}, {fz_flight.max():.1f}] mean={fz_flight.mean():.2f}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Row 2: HGDO torque zoomed to flight
    ax = axes[2, col]
    ax.plot(d['hgdo_t'][mask], d['hgdo_tx'][mask], 'tab:red', lw=0.8, label='tx (roll)')
    ax.plot(d['hgdo_t'][mask], d['hgdo_ty'][mask], 'tab:blue', lw=0.8, label='ty (pitch)')
    ax.plot(d['hgdo_t'][mask], d['hgdo_tz'][mask], 'tab:green', lw=0.8, label='tz (yaw)')
    ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
    ax.set_ylabel('Torque (Nm)')
    ax.set_title(f'{name} - HGDO torque (flight only)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Row 3: MPC thrust + cmd thrust
    ax = axes[3, col]
    ax.plot(d['mpc_t'], d['mpc_F'], 'tab:purple', lw=0.8, label='MPC F')
    ax.plot(d['cmd_t'], d['cmd_F'], 'tab:cyan', lw=0.6, alpha=0.7, label='cmd F')
    ax.axhline(W, color='orange', ls=':', lw=1.0, label=f'Hover ({W:.1f}N)')
    ax.axvspan(fs, fe, alpha=0.1, color='yellow')
    ax.set_ylabel('Thrust (N)'); ax.set_xlabel('Time (s)')
    ax.set_title(f'{name} - Thrust commands')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = 'bag_folder/sim2real_hgdo_comparison.png'
plt.savefig(out, dpi=150); plt.close()
print(f'\nSaved: {out}')

# ═══════════════════════════════════════════════════════════
# PLOT 2: Velocity & angular velocity noise comparison
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 2, figsize=(18, 16))
fig.suptitle('Sim2Real: Odom Noise Comparison (flight window)\n'
             'LEFT=SIM  RIGHT=REAL', fontsize=14, fontweight='bold')

for col, (d, name, fs, fe) in enumerate([
    (sim, 'SIM', sim_start, sim_end),
    (real, 'REAL', real_start, real_end),
]):
    mask = (d['odom_t'] >= fs) & (d['odom_t'] <= fe)

    # Row 0: Linear velocity
    ax = axes[0, col]
    ax.plot(d['odom_t'][mask], d['odom_vx'][mask], 'tab:red', lw=0.6, label='vx')
    ax.plot(d['odom_t'][mask], d['odom_vy'][mask], 'tab:blue', lw=0.6, label='vy')
    ax.plot(d['odom_t'][mask], d['odom_vz'][mask], 'tab:green', lw=0.8, label='vz')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f'{name} - Linear velocity (world)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Row 1: Angular velocity
    ax = axes[1, col]
    ax.plot(d['odom_t'][mask], d['odom_wx'][mask], 'tab:red', lw=0.6, label='wx')
    ax.plot(d['odom_t'][mask], d['odom_wy'][mask], 'tab:blue', lw=0.6, label='wy')
    ax.plot(d['odom_t'][mask], d['odom_wz'][mask], 'tab:green', lw=0.6, label='wz')
    ax.set_ylabel('Angular vel (rad/s)')
    ax.set_title(f'{name} - Angular velocity (body)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Row 2: Position
    ax = axes[2, col]
    ax.plot(d['odom_t'][mask], d['odom_px'][mask], 'tab:red', lw=0.8, label='x')
    ax.plot(d['odom_t'][mask], d['odom_py'][mask], 'tab:blue', lw=0.8, label='y')
    ax.plot(d['odom_t'][mask], d['odom_pz'][mask], 'tab:green', lw=0.8, label='z')
    ax.set_ylabel('Position (m)')
    ax.set_title(f'{name} - Position')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Row 3: RPY
    ax = axes[3, col]
    ax.plot(d['odom_t'][mask], d['odom_roll'][mask], 'tab:red', lw=0.6, label='roll')
    ax.plot(d['odom_t'][mask], d['odom_pitch'][mask], 'tab:blue', lw=0.6, label='pitch')
    ax.plot(d['odom_t'][mask], d['odom_yaw'][mask], 'tab:green', lw=0.6, label='yaw')
    ax.set_ylabel('Angle (deg)'); ax.set_xlabel('Time (s)')
    ax.set_title(f'{name} - RPY')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = 'bag_folder/sim2real_odom_noise.png'
plt.savefig(out, dpi=150); plt.close()
print(f'Saved: {out}')

# ═══════════════════════════════════════════════════════════
# PLOT 3: PSD comparison (velocity noise frequency content)
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(18, 12))
fig.suptitle('Sim2Real: Velocity Noise PSD (flight window)\n'
             'LEFT=Linear velocity  RIGHT=Angular velocity', fontsize=14, fontweight='bold')

for row, (v_key, w_key, v_label, w_label) in enumerate([
    ('odom_vx', 'odom_wx', 'vx', 'wx'),
    ('odom_vy', 'odom_wy', 'vy', 'wy'),
    ('odom_vz', 'odom_wz', 'vz', 'wz'),
]):
    # Linear velocity PSD
    ax = axes[row, 0]
    for d, name, fs, fe, color in [
        (sim, 'SIM', sim_start, sim_end, 'tab:blue'),
        (real, 'REAL', real_start, real_end, 'tab:red'),
    ]:
        mask = (d['odom_t'] >= fs) & (d['odom_t'] <= fe)
        seg = d[v_key][mask]
        dt = np.mean(np.diff(d['odom_t'][mask])) if np.sum(mask) > 10 else 0.01
        freqs, psd = compute_psd(seg, dt)
        if len(freqs) > 1:
            ax.semilogy(freqs[1:], psd[1:], color=color, lw=0.8, label=f'{name} (dt={dt*1000:.1f}ms)')
    ax.set_ylabel('PSD')
    ax.set_title(f'{v_label} PSD')
    ax.set_xlim(0, 50)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Angular velocity PSD
    ax = axes[row, 1]
    for d, name, fs, fe, color in [
        (sim, 'SIM', sim_start, sim_end, 'tab:blue'),
        (real, 'REAL', real_start, real_end, 'tab:red'),
    ]:
        mask = (d['odom_t'] >= fs) & (d['odom_t'] <= fe)
        seg = d[w_key][mask]
        dt = np.mean(np.diff(d['odom_t'][mask])) if np.sum(mask) > 10 else 0.01
        freqs, psd = compute_psd(seg, dt)
        if len(freqs) > 1:
            ax.semilogy(freqs[1:], psd[1:], color=color, lw=0.8, label=f'{name} (dt={dt*1000:.1f}ms)')
    ax.set_ylabel('PSD')
    ax.set_title(f'{w_label} PSD')
    ax.set_xlim(0, 50)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

axes[2, 0].set_xlabel('Frequency (Hz)')
axes[2, 1].set_xlabel('Frequency (Hz)')
plt.tight_layout()
out = 'bag_folder/sim2real_psd.png'
plt.savefig(out, dpi=150); plt.close()
print(f'Saved: {out}')

# ═══════════════════════════════════════════════════════════
# PLOT 4: RPM comparison (6x2: LEFT=REAL, RIGHT=SIM, per motor)
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(6, 2, figsize=(18, 24))
fig.suptitle('Sim2Real: Cmd vs Actual RPM per Motor\n'
             'LEFT=REAL (2026_03_31_05)  RIGHT=SIM (2026_04_01_sim)', fontsize=14, fontweight='bold')

rpm_duration = 3.0  # seconds from flight start
print(f"\n{'='*50}")
print(f"{'Cmd vs Actual RPM RMSE (per motor, first 3s)':^50}")
print(f"{'='*50}")
print(f"  {'Motor':<10} {'REAL':>12} {'SIM':>12}")
print(f"  {'-'*34}")

for col, (d, name, fs, fe) in enumerate([
    (real, 'REAL', real_start, real_start + rpm_duration),
    (sim, 'SIM', sim_start, sim_start + rpm_duration),
]):
    rpm_mask = (d['rpm_t'] >= fs) & (d['rpm_t'] <= fe)
    cmd_mask = (d['cmd_t'] >= fs) & (d['cmd_t'] <= fe)
    # Interpolate cmd RPM onto actual RPM time grid for RMSE
    rmse_values = []
    for i in range(6):
        cmd_interp = np.interp(d['rpm_t'][rpm_mask], d['cmd_t'][cmd_mask],
                               d['cmd_rpm_raw'][cmd_mask, i])
        rmse = np.sqrt(np.mean((cmd_interp - d['rpm_raw'][rpm_mask, i]) ** 2))
        rmse_values.append(rmse)
    if col == 0:
        real_rmse = rmse_values
    else:
        sim_rmse = rmse_values

    for i in range(6):
        ax = axes[i, col]
        ax.plot(d['cmd_t'][cmd_mask], d['cmd_rpm_raw'][cmd_mask, i],
                'tab:red', lw=0.6, alpha=0.8, label='Cmd RPM')
        ax.plot(d['rpm_t'][rpm_mask], d['rpm_raw'][rpm_mask, i],
                'tab:blue', lw=0.6, alpha=0.8, label='Actual RPM')
        ax.set_xlim(fs, fe)
        ax.set_ylabel('RPM')
        ax.set_title(f'{name} - Motor {i+1} (RMSE={rmse_values[i]:.1f})')
        ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)
        if i == 5:
            ax.set_xlabel('Time (s)')

for i in range(6):
    print(f"  M{i+1:<9} {real_rmse[i]:>12.1f} {sim_rmse[i]:>12.1f}")
print(f"{'='*50}")

plt.tight_layout()
out = 'bag_folder/sim2real_rpm.png'
plt.savefig(out, dpi=150); plt.close()
print(f'\nSaved: {out}')

print("\n=== Sim2Real analysis complete ===")
