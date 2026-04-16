#!/usr/bin/env python3
"""Analyze 2026_04_17_sim bag: position-tracking instability with Q_p=4 tuning.

Topics:
  /mavros/local_position/odom    - actual drone state (pos world, vel ?, q, w body)
  /nmpc/ref                      - reference (p, v, psi, psi_dot)
  /nmpc/control                  - NMPC output wrench (Fx,Fy,Fz,Mx,My,Mz) body
  /hgdo/wrench                   - DOB disturbance estimate (body)
  /uav/actual_rpm                - actual rotor RPMs
  /uav/cmd_raw                   - commanded RPMs
"""

import sqlite3, struct, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

BAG = '/home/user/drone_control_pkgs/bag_folder/2026_04_17_sim/2026_04_17_sim_0.db3'
OUT = '/home/user/drone_control_pkgs/bag_folder/2026_04_17_sim_analysis.png'


def parse_odom(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, 44)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 68)
    vx, vy, vz = struct.unpack_from('<3d', data, 388)
    wx, wy, wz = struct.unpack_from('<3d', data, 412)
    return sec + nsec*1e-9, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz


def parse_wrench(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    fx, fy, fz = struct.unpack_from('<3d', data, 28)
    tx, ty, tz = struct.unpack_from('<3d', data, 52)
    return sec + nsec*1e-9, fx, fy, fz, tx, ty, tz


def parse_ref(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    # After header (approx. frame_id 4-byte length + padding ...); use offset from actual layout
    # For drone_msgs/Ref: header + float64[3] p + float64[3] v + float64 psi + float64 psi_dot
    # Header layout: stamp(8) + frame_id (string: uint32 len + bytes aligned)
    # frame_id offset at byte 12; need alignment to 8 for subsequent fields
    # Common layout gives p at offset 24 if frame_id is empty with 4-byte len 0
    off = 20
    px, py, pz = struct.unpack_from('<3d', data, off)
    vx, vy, vz = struct.unpack_from('<3d', data, off + 24)
    psi = struct.unpack_from('<d', data, off + 48)[0]
    psi_dot = 0.0
    return sec + nsec*1e-9, px, py, pz, vx, vy, vz, psi, psi_dot


def load(path):
    conn = sqlite3.connect(path); c = conn.cursor()
    c.execute('SELECT id,name FROM topics')
    topics = {n:i for i,n in c.fetchall()}

    od = {'t':[], 'p':[], 'v':[], 'rpy':[], 'q':[], 'w':[]}
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/mavros/local_position/odom'],))
    for (data,) in c.fetchall():
        t,px,py,pz,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz = parse_odom(data)
        n = (qx*qx+qy*qy+qz*qz+qw*qw)**0.5
        if n < 1e-9: continue
        R = Rotation.from_quat([qx/n,qy/n,qz/n,qw/n])
        rpy = R.as_euler('xyz', degrees=True)
        od['t'].append(t); od['p'].append([px,py,pz])
        od['v'].append([vx,vy,vz])  # raw twist (may be body or world)
        od['rpy'].append(rpy); od['q'].append([qw, qx, qy, qz])
        od['w'].append([wx, wy, wz])
    for k in od: od[k] = np.array(od[k])

    mpc = {'t':[], 'f':[], 'tq':[]}
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/nmpc/control'],))
    for (data,) in c.fetchall():
        t,fx,fy,fz,tx,ty,tz = parse_wrench(data)
        mpc['t'].append(t); mpc['f'].append([fx,fy,fz]); mpc['tq'].append([tx,ty,tz])
    for k in mpc: mpc[k] = np.array(mpc[k])

    hg = {'t':[], 'f':[], 'tq':[]}
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
              (topics['/hgdo/wrench'],))
    for (data,) in c.fetchall():
        t,fx,fy,fz,tx,ty,tz = parse_wrench(data)
        hg['t'].append(t); hg['f'].append([fx,fy,fz]); hg['tq'].append([tx,ty,tz])
    for k in hg: hg[k] = np.array(hg[k])

    ref = {'t':[], 'p':[], 'v':[], 'psi':[]}
    if '/nmpc/ref' in topics:
        c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
                  (topics['/nmpc/ref'],))
        for (data,) in c.fetchall():
            try:
                t, px, py, pz, vx, vy, vz, psi, pdt = parse_ref(data)
                ref['t'].append(t); ref['p'].append([px,py,pz])
                ref['v'].append([vx,vy,vz]); ref['psi'].append(psi)
            except Exception:
                pass
        for k in ref: ref[k] = np.array(ref[k])

    conn.close()
    t0 = od['t'][0]
    od['t'] -= t0
    if len(mpc['t']): mpc['t'] -= t0
    if len(hg['t']):  hg['t']  -= t0
    if len(ref['t']): ref['t'] -= t0

    # Subtract initial position (to match node behaviour)
    od['p'][:,0] -= od['p'][0,0]
    od['p'][:,1] -= od['p'][0,1]
    od['p'][:,2] -= od['p'][0,2]

    return od, mpc, hg, ref


def print_stats(od, mpc, hg, ref):
    print(f"Duration: {od['t'][-1]:.2f} s")
    print(f"Samples:  odom {len(od['t'])},  mpc {len(mpc['t'])},"
          f"  hgdo {len(hg['t'])},  ref {len(ref['t'])}")
    print()
    print(f"Position (world, relative to start):")
    for i, ax in enumerate('xyz'):
        print(f"  p{ax}: range [{od['p'][:,i].min():+.3f},{od['p'][:,i].max():+.3f}] m,"
              f"  std {od['p'][:,i].std():.3f} m")
    print(f"Velocity (raw twist):")
    for i, ax in enumerate('xyz'):
        print(f"  v{ax}: range [{od['v'][:,i].min():+.3f},{od['v'][:,i].max():+.3f}] m/s,"
              f"  std {od['v'][:,i].std():.3f}")
    print(f"Attitude (deg):")
    for i, ax in enumerate(['roll','pitch','yaw']):
        print(f"  {ax:5s}: range [{od['rpy'][:,i].min():+.2f},{od['rpy'][:,i].max():+.2f}],"
              f"  std {od['rpy'][:,i].std():.3f}")
    print(f"Angular rates (body, rad/s):")
    for i, ax in enumerate(['wx','wy','wz']):
        print(f"  {ax}: std {od['w'][:,i].std():.4f}, peak {np.abs(od['w'][:,i]).max():.3f}")
    if len(mpc['t']):
        print(f"NMPC control:")
        print(f"  Fz: mean {mpc['f'][:,2].mean():.2f}, range [{mpc['f'][:,2].min():.2f},{mpc['f'][:,2].max():.2f}] N")
        for i, ax in enumerate(['Mx','My','Mz']):
            print(f"  {ax}: std {mpc['tq'][:,i].std():.4f}, peak {np.abs(mpc['tq'][:,i]).max():.3f} Nm")
    if len(hg['t']):
        print(f"HGDO disturbance (body):")
        for i, ax in enumerate(['fx','fy','fz']):
            print(f"  {ax}: mean {hg['f'][:,i].mean():+.3f}, std {hg['f'][:,i].std():.3f}, peak {np.abs(hg['f'][:,i]).max():.3f} N")
        for i, ax in enumerate(['tx','ty','tz']):
            print(f"  {ax}: mean {hg['tq'][:,i].mean():+.4f}, std {hg['tq'][:,i].std():.4f}, peak {np.abs(hg['tq'][:,i]).max():.3f} Nm")


def analyze_oscillation(t, sig, label):
    """FFT-based dominant frequency of a signal."""
    if len(t) < 20: return None
    dt = np.mean(np.diff(t))
    fs = 1.0/dt
    s = sig - sig.mean()
    N = len(s)
    F = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(N, d=dt)
    amps = np.abs(F)
    # skip DC
    if len(amps) < 3: return None
    idx = np.argmax(amps[1:]) + 1
    f_peak = freqs[idx]
    return f_peak, amps[idx]/(N/2)


def find_divergence_start(od, thresh_deg=30.0):
    """Find the time when roll or pitch first exceeds threshold."""
    for i in range(len(od['t'])):
        if abs(od['rpy'][i,0]) > thresh_deg or abs(od['rpy'][i,1]) > thresh_deg:
            return od['t'][i]
    return None


def main():
    print(f"Loading {BAG}...")
    od, mpc, hg, ref = load(BAG)
    print_stats(od, mpc, hg, ref)

    t_div = find_divergence_start(od, thresh_deg=30.0)
    print(f"\nDivergence start (|att|>30°): t = {t_div if t_div else 'N/A'} s")

    # Pre-divergence window for stable-phase stats
    if t_div is not None:
        pre_mask = (od['t'] > 3.0) & (od['t'] < t_div - 0.5)
    else:
        pre_mask = od['t'] > 3.0

    if pre_mask.sum() > 20:
        print(f"\nPre-divergence (3.0s → {t_div-0.5 if t_div else 'end'}s) stats:")
        print(f"  pos error std: x={od['p'][pre_mask,0].std():.3f}, y={od['p'][pre_mask,1].std():.3f}, z={od['p'][pre_mask,2].std():.3f} m")
        print(f"  roll  std={od['rpy'][pre_mask,0].std():.2f} deg, peak={np.abs(od['rpy'][pre_mask,0]).max():.2f}")
        print(f"  pitch std={od['rpy'][pre_mask,1].std():.2f} deg, peak={np.abs(od['rpy'][pre_mask,1]).max():.2f}")

    # Oscillation analysis in growing-oscillation window (immediately before/during divergence)
    if t_div is not None:
        osc_mask = (od['t'] > max(t_div - 2.0, 3.0)) & (od['t'] < t_div + 1.0)
        if osc_mask.sum() > 20:
            print(f"\nOscillation window ({max(t_div-2,3):.1f}→{t_div+1:.1f}s, dominant freq):")
            for i, ax in enumerate(['roll','pitch']):
                res = analyze_oscillation(od['t'][osc_mask], od['rpy'][osc_mask,i], ax)
                if res:
                    print(f"  {ax}: {res[0]:.2f} Hz  (amp {res[1]:.3f} deg)")
            for i, ax in enumerate(['wx','wy','wz']):
                res = analyze_oscillation(od['t'][osc_mask], od['w'][osc_mask,i], ax)
                if res:
                    print(f"  {ax}: {res[0]:.2f} Hz  (amp {res[1]:.3f} rad/s)")

    # Restrict window for clarity (stable + divergence onset)
    if t_div is not None:
        t_plot_end = min(t_div + 2.5, od['t'][-1])
    else:
        t_plot_end = od['t'][-1]
    od_m = od['t'] <= t_plot_end
    mpc_m = mpc['t'] <= t_plot_end if len(mpc['t']) else None
    hg_m = hg['t'] <= t_plot_end if len(hg['t']) else None

    # Plot
    fig, axs = plt.subplots(5, 2, figsize=(15, 18))

    # 1 Position with reference
    ax = axs[0,0]
    for i, c, lbl in [(0,'r','x'),(1,'g','y'),(2,'b','z')]:
        ax.plot(od['t'][od_m], od['p'][od_m,i], c+'-', label=f'p{lbl}', lw=1)
    if len(ref['t']):
        ref_m = ref['t'] <= t_plot_end
        for i, c, lbl in [(0,'r','x'),(1,'g','y'),(2,'b','z')]:
            ax.plot(ref['t'][ref_m], ref['p'][ref_m,i], c+'--', alpha=0.6, label=f'ref{lbl}')
    if t_div is not None:
        ax.axvline(t_div, color='k', ls=':', alpha=0.5, label='divergence')
    ax.set_xlabel('t [s]'); ax.set_ylabel('pos [m]'); ax.grid(True); ax.legend(loc='best', fontsize=8)
    ax.set_title('Position (world, relative to start) with ref')

    # 2 Velocity
    ax = axs[0,1]
    for i, c, lbl in [(0,'r','x'),(1,'g','y'),(2,'b','z')]:
        ax.plot(od['t'][od_m], od['v'][od_m,i], c+'-', label=f'v{lbl}', lw=1)
    if t_div: ax.axvline(t_div, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('t [s]'); ax.set_ylabel('v [m/s]'); ax.grid(True); ax.legend(loc='best', fontsize=8)
    ax.set_title('Velocity (raw twist from odom)')

    # 3 Attitude (roll, pitch, yaw)
    ax = axs[1,0]
    for i, c, lbl in [(0,'r','roll'),(1,'g','pitch'),(2,'b','yaw')]:
        ax.plot(od['t'][od_m], od['rpy'][od_m,i], c+'-', label=lbl, lw=1)
    if t_div: ax.axvline(t_div, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('t [s]'); ax.set_ylabel('deg'); ax.grid(True); ax.legend(loc='best', fontsize=8)
    ax.set_title('Attitude (roll, pitch, yaw)')

    # 4 Angular rates
    ax = axs[1,1]
    for i, c, lbl in [(0,'r','wx'),(1,'g','wy'),(2,'b','wz')]:
        ax.plot(od['t'][od_m], od['w'][od_m,i], c+'-', label=lbl, lw=1)
    if t_div: ax.axvline(t_div, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('t [s]'); ax.set_ylabel('rad/s'); ax.grid(True); ax.legend(loc='best', fontsize=8)
    ax.set_title('Angular rates (body)')

    # 5 NMPC Force
    ax = axs[2,0]
    if len(mpc['t']) and mpc_m is not None:
        for i, c, lbl in [(0,'r','Fx'),(1,'g','Fy'),(2,'b','Fz')]:
            ax.plot(mpc['t'][mpc_m], mpc['f'][mpc_m,i], c+'-', label=lbl, lw=1)
    if t_div: ax.axvline(t_div, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('t [s]'); ax.set_ylabel('N'); ax.grid(True); ax.legend(loc='best', fontsize=8)
    ax.set_title('NMPC force (body, Fx/Fy unused = 0)')

    # 6 NMPC Moment
    ax = axs[2,1]
    if len(mpc['t']) and mpc_m is not None:
        for i, c, lbl in [(0,'r','Mx'),(1,'g','My'),(2,'b','Mz')]:
            ax.plot(mpc['t'][mpc_m], mpc['tq'][mpc_m,i], c+'-', label=lbl, lw=1)
    if t_div: ax.axvline(t_div, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('t [s]'); ax.set_ylabel('Nm'); ax.grid(True); ax.legend(loc='best', fontsize=8)
    ax.set_title('NMPC moment (body)')

    # 7 HGDO force
    ax = axs[3,0]
    if len(hg['t']) and hg_m is not None:
        for i, c, lbl in [(0,'r','fx'),(1,'g','fy'),(2,'b','fz')]:
            ax.plot(hg['t'][hg_m], hg['f'][hg_m,i], c+'-', label=lbl, lw=1)
    if t_div: ax.axvline(t_div, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('t [s]'); ax.set_ylabel('N'); ax.grid(True); ax.legend(loc='best', fontsize=8)
    ax.set_title('HGDO disturbance force (body)')

    # 8 HGDO torque
    ax = axs[3,1]
    if len(hg['t']) and hg_m is not None:
        for i, c, lbl in [(0,'r','tx'),(1,'g','ty'),(2,'b','tz')]:
            ax.plot(hg['t'][hg_m], hg['tq'][hg_m,i], c+'-', label=lbl, lw=1)
    if t_div: ax.axvline(t_div, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('t [s]'); ax.set_ylabel('Nm'); ax.grid(True); ax.legend(loc='best', fontsize=8)
    ax.set_title('HGDO disturbance torque (body)')

    # 9 XY trajectory
    ax = axs[4,0]
    ax.plot(od['p'][od_m,0], od['p'][od_m,1], 'b-', lw=1, alpha=0.7)
    ax.plot(od['p'][0,0], od['p'][0,1], 'go', label='start')
    if t_div is not None:
        div_idx = np.argmin(np.abs(od['t'] - t_div))
        ax.plot(od['p'][div_idx,0], od['p'][div_idx,1], 'r^', label=f'divergence @ {t_div:.2f}s')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.grid(True); ax.legend()
    ax.set_aspect('equal'); ax.set_title('XY trajectory (up to divergence)')

    # 10 Pitch vs wy (cascade oscillation diagnostic)
    ax = axs[4,1]
    ax.plot(od['t'][od_m], od['rpy'][od_m,1], 'g-', label='pitch [deg]', lw=1)
    ax2 = ax.twinx()
    ax2.plot(od['t'][od_m], od['w'][od_m,1], 'm-', alpha=0.6, label='wy [rad/s]', lw=1)
    if t_div: ax.axvline(t_div, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('t [s]'); ax.set_ylabel('pitch [deg]', color='g')
    ax2.set_ylabel('wy [rad/s]', color='m'); ax.grid(True)
    ax.set_title('Pitch & wy (cascade oscillation indicator)')

    plt.tight_layout()
    plt.savefig(OUT, dpi=100)
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
