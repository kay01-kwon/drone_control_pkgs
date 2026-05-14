#!/usr/bin/env python3
"""Inner-loop attitude diagnostic.

For each bag we compare:
  1. Motor-commanded body torque   = (u_mpc - tau_dist)              [from /nmpc/control - /hgdo/wrench]
  2. Implied actual body torque    = J·dw/dt + w × Jw                [from odom angular velocity]
  3. Desired vs actual roll/pitch with cross-correlation lag         [phase delay]

If (1) and (2) match, motors/allocation are fine and the issue is upstream
(NMPC OCP, gains, model inertia).  If (1) ≫ (2), motors aren't delivering the
torque the controller demands (motor lag, allocation gain, mass/inertia model
error, or saturation).  Large attitude lag also points to slow inner dynamics.

Usage: python3 inner_loop_diag.py [<bag_subdir> [<date_dir>]]
"""

import os, sys, sqlite3, struct, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Drone inertia / mass (from drone_control/config/nmpc_att_with_hgdo.yaml) ──
J_DIAG = np.array([0.06, 0.06, 0.08])
M      = 3.188

_HERE = os.path.dirname(os.path.abspath(__file__))
BAG_SUBDIR = sys.argv[1] if len(sys.argv) > 1 else '02_ct_1p255'
DATE_DIR   = sys.argv[2] if len(sys.argv) > 2 else '2026_05_05_free_flight'
BAG_DIR = os.path.join(_HERE, DATE_DIR, BAG_SUBDIR)
db_path = glob.glob(os.path.join(BAG_DIR, '*.db3'))
if not db_path:
    raise SystemExit(f'No .db3 in {BAG_DIR}')
DB = db_path[0]
_parts = BAG_SUBDIR.replace('\\', '/').split('/')
OUT_DIR = os.path.join(_HERE, DATE_DIR, *_parts[:-1]) if len(_parts) > 1 else os.path.join(_HERE, DATE_DIR)
TAG = _parts[-1]
print(f'Analyzing: {DB}')


def _align(off, n):
    rel = off - 4
    return off + ((-rel) % n)

def parse_odom(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    slen2 = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen2
    off = _align(off, 8)
    px, py, pz = struct.unpack_from('<3d', blob, off); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', blob, off); off += 32
    off += 36 * 8
    vx, vy, vz = struct.unpack_from('<3d', blob, off); off += 24
    wx, wy, wz = struct.unpack_from('<3d', blob, off); off += 24
    return np.array([px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz])

def parse_wrench(blob):
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 8)
    return np.array(struct.unpack_from('<6d', blob, off))

MAX_BIT, MAX_RPM = 8191, 9800

def parse_cmd_raw(blob):
    """HexaCmdRaw: header + int16[6] cmd_raw → convert back to RPM."""
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 2)
    raw = np.array(struct.unpack_from('<6h', blob, off), dtype=np.float64)
    return raw * MAX_RPM / MAX_BIT

def parse_actual_rpm(blob):
    """HexaActualRpm: header + uint32[6] rpm (trailing fields zero, ignored)."""
    off = 4 + 8
    slen = struct.unpack_from('<I', blob, off)[0]; off += 4 + slen
    off = _align(off, 4)
    return np.array(struct.unpack_from('<6I', blob, off), dtype=np.float64)

def quat_to_rpy(q):
    qw, qx, qy, qz = q
    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    sinp = np.clip(2*(qw*qy - qz*qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    return np.array([roll, pitch, yaw])


conn = sqlite3.connect(DB)
c = conn.cursor()
tid = {n: i for i, n in c.execute('SELECT id, name FROM topics').fetchall()}
def fetch(t, p):
    rows = c.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
                     (tid[t],)).fetchall()
    ts = np.array([r[0] for r in rows], dtype=np.float64)
    dat = np.array([p(bytes(r[1])) for r in rows])
    return ts, dat
odom_ts, odom = fetch('/mavros/local_position/odom', parse_odom)
hgdo_ts, hgdo = fetch('/hgdo/wrench', parse_wrench)
ctrl_ts, ctrl = fetch('/nmpc/control', parse_wrench)
cmd_ts,  cmd_rpm  = fetch('/uav/cmd_raw',    parse_cmd_raw)
act_ts,  act_rpm  = fetch('/uav/actual_rpm', parse_actual_rpm)
conn.close()

t0 = min(odom_ts[0], hgdo_ts[0], ctrl_ts[0], cmd_ts[0], act_ts[0])
odom_t = (odom_ts - t0) * 1e-9
hgdo_t = (hgdo_ts - t0) * 1e-9
ctrl_t = (ctrl_ts - t0) * 1e-9
cmd_t  = (cmd_ts  - t0) * 1e-9
act_t  = (act_ts  - t0) * 1e-9

# ── Implied torque from omega (body frame) ──
# τ = J·dω/dt + ω × (J·ω)
w = odom[:, 10:13]                                    # body angular velocity (rad/s)
# Median filter then central difference for derivative — IMU noise filtering.
def smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, mode='same')
w_s = np.column_stack([smooth(w[:, k], 5) for k in range(3)])
dt = np.gradient(odom_t)
dw = np.column_stack([np.gradient(w_s[:, k], odom_t) for k in range(3)])
Jw = w_s * J_DIAG[None, :]
gyro_term = np.cross(w_s, Jw)
tau_implied = dw * J_DIAG[None, :] + gyro_term       # body-frame torque [N·m]

# ── Motor commanded torque (controller subtracts HGDO from u_mpc) ──
hgdo_tau_at_ctrl = np.column_stack([
    np.interp(ctrl_t, hgdo_t, hgdo[:, 3 + k]) for k in range(3)])
tau_motor_cmd = ctrl[:, 3:6] - hgdo_tau_at_ctrl       # what the allocator was told to make

# Interp implied to ctrl_t for direct comparison
tau_implied_at_ctrl = np.column_stack([
    np.interp(ctrl_t, odom_t, tau_implied[:, k]) for k in range(3)])

# ── Desired vs actual roll/pitch, with cross-correlation delay ──
rpy = np.array([quat_to_rpy(odom[i, 6:10]) for i in range(len(odom))])

def force_to_rp(fx, fy, f_col, psi):
    fz_sq = f_col**2 - fx**2 - fy**2
    fz = np.sqrt(max(fz_sq, 0.0))
    n = np.sqrt(fx**2 + fy**2 + fz**2)
    if n < 1e-6: return 0.0, 0.0
    zb = np.array([fx, fy, fz]) / n
    xc = np.array([np.cos(psi), np.sin(psi), 0.0])
    yb = np.cross(zb, xc)
    yn = np.linalg.norm(yb)
    yb = (np.array([-np.sin(psi), np.cos(psi), 0.0]) if yn < 1e-6 else yb / yn)
    xb = np.cross(yb, zb)
    R = np.column_stack((xb, yb, zb))
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    return roll, pitch

psi_at_ctrl = np.interp(ctrl_t, odom_t, np.unwrap(rpy[:, 2]))
des_rp = np.zeros((len(ctrl_t), 2))
for i in range(len(ctrl_t)):
    des_rp[i] = force_to_rp(ctrl[i, 0], ctrl[i, 1], ctrl[i, 2], psi_at_ctrl[i])

# Uniform-resample desired and actual to find lag
T_LO = max(ctrl_t[0], odom_t[0]) + 2.0
T_HI = min(ctrl_t[-1], odom_t[-1]) - 2.0
dt_u = 0.01
t_u = np.arange(T_LO, T_HI, dt_u)
roll_des_u  = np.interp(t_u, ctrl_t, des_rp[:, 0])
pitch_des_u = np.interp(t_u, ctrl_t, des_rp[:, 1])
roll_act_u  = np.interp(t_u, odom_t, rpy[:, 0])
pitch_act_u = np.interp(t_u, odom_t, rpy[:, 1])

def xcorr_lag(a, b, dt, max_lag_s=0.5):
    a = a - a.mean(); b = b - b.mean()
    n = len(a); max_k = int(max_lag_s / dt)
    lags = np.arange(-max_k, max_k + 1)
    c = np.zeros_like(lags, dtype=float)
    sa = a.std(); sb = b.std()
    if sa < 1e-9 or sb < 1e-9: return lags*dt, c, 0.0
    for j, k in enumerate(lags):
        if k >= 0:
            c[j] = np.mean(a[k:] * b[:n - k]) / (sa * sb)
        else:
            c[j] = np.mean(a[:n + k] * b[-k:]) / (sa * sb)
    j_max = np.argmax(c)
    return lags * dt, c, lags[j_max] * dt

lag_t_r, ccr, lag_r = xcorr_lag(roll_act_u,  roll_des_u,  dt_u)
lag_t_p, ccp, lag_p = xcorr_lag(pitch_act_u, pitch_des_u, dt_u)

# ── Stats ──
def stats_lin(name, x, unit='N·m'):
    print(f'  {name:22s}  std={np.std(x):7.4f} {unit}  '
          f'min={np.min(x):+8.4f}  max={np.max(x):+8.4f}')

print('\n== Inner-loop diagnostic ==')
print(f'-- Body torque comparison (J={J_DIAG.tolist()} kg·m²) --')
for k, ax in enumerate(['Mx', 'My', 'Mz']):
    stats_lin(f'{ax}_motor_cmd',  tau_motor_cmd[:, k])
    stats_lin(f'{ax}_implied',    tau_implied_at_ctrl[:, k])
    # ratio of implied/cmd std and correlation
    sc = np.std(tau_motor_cmd[:, k]); si = np.std(tau_implied_at_ctrl[:, k])
    a = tau_motor_cmd[:, k] - tau_motor_cmd[:, k].mean()
    b = tau_implied_at_ctrl[:, k] - tau_implied_at_ctrl[:, k].mean()
    corr = np.mean(a * b) / max(np.std(a) * np.std(b), 1e-12)
    print(f'    std(implied)/std(cmd) = {si/max(sc,1e-9):.3f},  corr={corr:+.3f}')

print(f'\n-- Attitude tracking lag (peak xcorr lag) --')
print(f'  roll  delay = {lag_r*1000:+.0f} ms   max xcorr = {np.max(ccr):.3f}')
print(f'  pitch delay = {lag_p*1000:+.0f} ms   max xcorr = {np.max(ccp):.3f}')

# ── Plots ──
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for k, ax_name in enumerate(['Mx', 'My', 'Mz']):
    axes[k].plot(ctrl_t, tau_motor_cmd[:, k], 'b', label=f'{ax_name} commanded (u_mpc − τ_hgdo)')
    axes[k].plot(odom_t, tau_implied[:, k],   'r', alpha=0.75, label=f'{ax_name} implied (J·α + ω×Jω)')
    axes[k].set_ylabel(f'{ax_name} [N·m]'); axes[k].grid(alpha=0.3); axes[k].legend(loc='upper right', fontsize=9)
axes[0].set_title(f'{TAG}  —  Commanded torque vs Implied torque from ω  (J={J_DIAG.tolist()})')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_inner_torque_check.png'), dpi=120)
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(lag_t_r * 1000, ccr, 'r'); axes[0].axvline(lag_r * 1000, color='k', ls='--', alpha=0.5)
axes[0].set_xlabel('Lag [ms]  (positive ⇒ actual lags desired)'); axes[0].set_ylabel('xcorr')
axes[0].set_title(f'Roll attitude tracking lag = {lag_r*1000:+.0f} ms')
axes[0].grid(alpha=0.3)
axes[1].plot(lag_t_p * 1000, ccp, 'g'); axes[1].axvline(lag_p * 1000, color='k', ls='--', alpha=0.5)
axes[1].set_xlabel('Lag [ms]'); axes[1].set_ylabel('xcorr')
axes[1].set_title(f'Pitch attitude tracking lag = {lag_p*1000:+.0f} ms')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_inner_attitude_lag.png'), dpi=120)
plt.close()

# ── Visual verification: shift actual back by measured lag ──
# If the measured lag is real, shifting actual by −lag should overlap desired.
# Pick a 10-second airborne window with reasonable signal variation.
def pick_window(des, t, win_s=10.0):
    n = len(t)
    seg = int(win_s / dt_u)
    if seg >= n: return 0, n
    # find segment with highest std (most signal)
    best_i, best_v = 0, -1
    for i in range(0, n - seg, seg // 4):
        v = des[i:i + seg].std()
        if v > best_v: best_v, best_i = v, i
    return best_i, best_i + seg

i0, i1 = pick_window(roll_des_u, t_u)

def shift_signal(x, lag_samples):
    out = np.full_like(x, np.nan)
    if lag_samples >= 0:
        out[:len(x) - lag_samples] = x[lag_samples:]
    else:
        out[-lag_samples:] = x[:len(x) + lag_samples]
    return out

lag_r_samp = int(round(lag_r / dt_u))
lag_p_samp = int(round(lag_p / dt_u))
roll_act_shifted  = shift_signal(roll_act_u,  lag_r_samp)
pitch_act_shifted = shift_signal(pitch_act_u, lag_p_samp)

fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex='col')
# Top row: Roll
axes[0, 0].plot(t_u[i0:i1], np.degrees(roll_des_u[i0:i1]),  'k', lw=1.4, label='desired')
axes[0, 0].plot(t_u[i0:i1], np.degrees(roll_act_u[i0:i1]),  'r', lw=1.0, alpha=0.85,
                label='actual (raw)')
axes[0, 0].set_ylabel('Roll [deg]'); axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend(loc='upper right', fontsize=9)
axes[0, 0].set_title(f'Roll — original (no shift)')

axes[0, 1].plot(t_u[i0:i1], np.degrees(roll_des_u[i0:i1]),       'k', lw=1.4, label='desired')
axes[0, 1].plot(t_u[i0:i1], np.degrees(roll_act_shifted[i0:i1]), 'b', lw=1.0, alpha=0.85,
                label=f'actual shifted by −{lag_r*1000:.0f} ms')
axes[0, 1].set_ylabel('Roll [deg]'); axes[0, 1].grid(alpha=0.3)
axes[0, 1].legend(loc='upper right', fontsize=9)
axes[0, 1].set_title(f'Roll — shifted (if lag is real, this should overlap)')

# Bottom row: Pitch
axes[1, 0].plot(t_u[i0:i1], np.degrees(pitch_des_u[i0:i1]),  'k', lw=1.4, label='desired')
axes[1, 0].plot(t_u[i0:i1], np.degrees(pitch_act_u[i0:i1]),  'g', lw=1.0, alpha=0.85,
                label='actual (raw)')
axes[1, 0].set_ylabel('Pitch [deg]'); axes[1, 0].set_xlabel('Time [s]'); axes[1, 0].grid(alpha=0.3)
axes[1, 0].legend(loc='upper right', fontsize=9)
axes[1, 0].set_title('Pitch — original (no shift)')

axes[1, 1].plot(t_u[i0:i1], np.degrees(pitch_des_u[i0:i1]),       'k', lw=1.4, label='desired')
axes[1, 1].plot(t_u[i0:i1], np.degrees(pitch_act_shifted[i0:i1]), 'b', lw=1.0, alpha=0.85,
                label=f'actual shifted by −{lag_p*1000:.0f} ms')
axes[1, 1].set_ylabel('Pitch [deg]'); axes[1, 1].set_xlabel('Time [s]'); axes[1, 1].grid(alpha=0.3)
axes[1, 1].legend(loc='upper right', fontsize=9)
axes[1, 1].set_title('Pitch — shifted')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_lag_verify.png'), dpi=120)
plt.close()

# ── Motor RPM tracking (cmd vs actual per motor) ──
# Resample both to a common 200 Hz grid in the airborne segment for xcorr.
RPM_HOVER = np.median(act_rpm[(act_rpm > 0).all(axis=1)], axis=0) if (act_rpm > 0).any() else np.zeros(6)
T_LO2 = max(cmd_t[0], act_t[0]) + 1.0
T_HI2 = min(cmd_t[-1], act_t[-1]) - 1.0
dt_m = 0.005
t_m = np.arange(T_LO2, T_HI2, dt_m)
cmd_u = np.column_stack([np.interp(t_m, cmd_t, cmd_rpm[:, k]) for k in range(6)])
act_u = np.column_stack([np.interp(t_m, act_t, act_rpm[:, k]) for k in range(6)])

# Mask airborne (use any motor cmd > 1.5x rotor_min ≈ 3000)
m_air = cmd_u.mean(axis=1) > 3000.0

print(f'\n-- Motor RPM tracking (cmd vs actual) — {m_air.sum()*dt_m:.1f}s airborne sample --')
motor_lags = np.zeros(6)
for k in range(6):
    a = act_u[m_air, k] - act_u[m_air, k].mean()
    c = cmd_u[m_air, k] - cmd_u[m_air, k].mean()
    if a.std() < 1e-6 or c.std() < 1e-6:
        continue
    max_k = int(0.2 / dt_m)
    lags = np.arange(-max_k, max_k + 1)
    corr_curve = np.zeros_like(lags, dtype=float)
    for j, kk in enumerate(lags):
        if kk >= 0: corr_curve[j] = np.mean(a[kk:] * c[:len(a) - kk])
        else:       corr_curve[j] = np.mean(a[:len(a) + kk] * c[-kk:])
    corr_curve /= (a.std() * c.std())
    j_pk = np.argmax(corr_curve)
    motor_lags[k] = lags[j_pk] * dt_m
    err = act_u[m_air, k] - cmd_u[m_air, k]
    print(f'  motor {k}: lag={motor_lags[k]*1000:+5.0f} ms  '
          f'peak xcorr={corr_curve[j_pk]:.3f}  '
          f'std(act−cmd)={err.std():6.1f} RPM  '
          f'mean(cmd)={cmd_u[m_air, k].mean():.0f} RPM')

fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)
for k in range(6):
    axes[k].plot(cmd_t, cmd_rpm[:, k], 'b', lw=0.9, label=f'motor {k} cmd')
    axes[k].plot(act_t, act_rpm[:, k], 'r', lw=0.7, alpha=0.85, label=f'motor {k} actual')
    axes[k].set_ylabel('RPM'); axes[k].grid(alpha=0.3)
    axes[k].legend(loc='upper right', fontsize=8)
axes[0].set_title(f'{TAG}  —  Motor RPM: cmd vs actual')
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{TAG}_motor_rpm_track.png'), dpi=120)
plt.close()

print('\nSaved:')
print(f'  - {TAG}_inner_torque_check.png')
print(f'  - {TAG}_inner_attitude_lag.png')
print(f'  - {TAG}_lag_verify.png')
print(f'  - {TAG}_motor_rpm_track.png')
