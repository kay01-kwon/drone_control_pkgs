#!/usr/bin/env python3
"""Compare 04_01_sim vs 04_07_sim bags."""
import sqlite3, struct, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def parse_odom(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    px, py, pz = struct.unpack_from('<3d', data, 44)
    qx, qy, qz, qw = struct.unpack_from('<4d', data, 68)
    vx, vy, vz = struct.unpack_from('<3d', data, 388)
    return sec + nsec*1e-9, px, py, pz, qx, qy, qz, qw, vx, vy, vz


def parse_wrench(data):
    sec = struct.unpack_from('<I', data, 4)[0]
    nsec = struct.unpack_from('<I', data, 8)[0]
    fx, fy, fz = struct.unpack_from('<3d', data, 28)
    tx, ty, tz = struct.unpack_from('<3d', data, 52)
    return sec + nsec*1e-9, fx, fy, fz, tx, ty, tz


def load(path):
    conn = sqlite3.connect(path); c = conn.cursor()
    c.execute('SELECT id,name FROM topics')
    topics = {n:i for i,n in c.fetchall()}
    od = {'t':[], 'p':[], 'v':[], 'rpy':[]}
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/mavros/local_position/odom'],))
    for (data,) in c.fetchall():
        t,px,py,pz,qx,qy,qz,qw,vx,vy,vz = parse_odom(data)
        n = (qx*qx+qy*qy+qz*qz+qw*qw)**0.5
        if n < 1e-9: continue
        R = Rotation.from_quat([qx/n,qy/n,qz/n,qw/n])
        rpy = R.as_euler('xyz', degrees=True)
        vw = R.as_matrix() @ np.array([vx,vy,vz])
        od['t'].append(t); od['p'].append([px,py,pz]); od['v'].append(vw); od['rpy'].append(rpy)
    for k in od: od[k] = np.array(od[k])
    mpc = {'t':[], 'fz':[]}
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/nmpc/control'],))
    for (data,) in c.fetchall():
        t,fx,fy,fz,tx,ty,tz = parse_wrench(data)
        mpc['t'].append(t); mpc['fz'].append(fz)
    for k in mpc: mpc[k] = np.array(mpc[k])
    hg = {'t':[], 'f':[], 'tq':[]}
    c.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topics['/hgdo/wrench'],))
    for (data,) in c.fetchall():
        t,fx,fy,fz,tx,ty,tz = parse_wrench(data)
        hg['t'].append(t); hg['f'].append([fx,fy,fz]); hg['tq'].append([tx,ty,tz])
    for k in hg: hg[k] = np.array(hg[k])
    conn.close()
    t0 = od['t'][0]
    od['t'] -= t0; mpc['t'] -= t0; hg['t'] -= t0
    od['p'][:,2] -= od['p'][0,2]
    return od, mpc, hg


def stats(name, od, mpc, hg):
    print(f"\n=== {name} ===")
    print(f"  duration: {od['t'][-1]:.2f} s")
    print(f"  pos x range: [{od['p'][:,0].min():+.3f}, {od['p'][:,0].max():+.3f}] m  (peak |x|={np.abs(od['p'][:,0]).max():.3f})")
    print(f"  pos y range: [{od['p'][:,1].min():+.3f}, {od['p'][:,1].max():+.3f}] m  (peak |y|={np.abs(od['p'][:,1]).max():.3f})")
    print(f"  pos z range: [{od['p'][:,2].min():+.3f}, {od['p'][:,2].max():+.3f}] m")
    print(f"  vx peak: {np.abs(od['v'][:,0]).max():.3f} m/s")
    print(f"  vy peak: {np.abs(od['v'][:,1]).max():.3f} m/s")
    print(f"  vz peak: {np.abs(od['v'][:,2]).max():.3f} m/s")
    print(f"  roll  peak: {np.abs(od['rpy'][:,0]).max():.2f} deg")
    print(f"  pitch peak: {np.abs(od['rpy'][:,1]).max():.2f} deg")
    print(f"  yaw   range: [{od['rpy'][:,2].min():+.2f}, {od['rpy'][:,2].max():+.2f}] deg")
    if len(mpc['fz']):
        print(f"  NMPC Fz range: [{mpc['fz'].min():.2f}, {mpc['fz'].max():.2f}] N")
    if len(hg['f']):
        print(f"  HGDO |f| max: {np.linalg.norm(hg['f'], axis=1).max():.3f} N")
        print(f"  HGDO |tq| max: {np.linalg.norm(hg['tq'], axis=1).max():.4f} Nm")
    # settling: time after t=8s at which |pos|<0.02m sustained
    mask = od['t'] > 8.0
    if mask.any():
        rss = np.linalg.norm(od['p'][mask,:2], axis=1)
        print(f"  RMS xy after 8s: {np.sqrt((rss**2).mean()):.4f} m")
        print(f"  RMS vz  after 8s: {np.sqrt((od['v'][mask,2]**2).mean()):.4f} m/s")


bags = {}
for tag, p in [('04_01','bag_folder/2026_04_01_sim/2026_04_01_sim_0.db3'),
               ('04_07','bag_folder/2026_04_07_sim/2026_04_07_sim_0.db3')]:
    bags[tag] = load(p)
    stats(tag, *bags[tag])

# overlay plot
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
labels = {'04_01':('tab:red','04_01_sim'), '04_07':('tab:blue','04_07_sim')}
for tag,(od,mpc,hg) in bags.items():
    c,lab = labels[tag]
    axes[0,0].plot(od['t'], od['p'][:,0], c, lw=1, label=lab)
    axes[0,1].plot(od['t'], od['p'][:,1], c, lw=1, label=lab)
    axes[0,2].plot(od['t'], od['p'][:,2], c, lw=1, label=lab)
    axes[1,0].plot(od['t'], od['v'][:,0], c, lw=1)
    axes[1,1].plot(od['t'], od['v'][:,1], c, lw=1)
    axes[1,2].plot(od['t'], od['v'][:,2], c, lw=1)
    axes[2,0].plot(od['t'], od['rpy'][:,0], c, lw=1)
    axes[2,1].plot(od['t'], od['rpy'][:,1], c, lw=1)
    axes[2,2].plot(od['t'], od['rpy'][:,2], c, lw=1)
    if len(mpc['fz']):
        axes[3,0].plot(mpc['t'], mpc['fz'], c, lw=1)
    if len(hg['f']):
        axes[3,1].plot(hg['t'], np.linalg.norm(hg['f'], axis=1), c, lw=1)
        axes[3,2].plot(hg['t'], np.linalg.norm(hg['tq'], axis=1), c, lw=1)
titles = [['x [m]','y [m]','z [m]'],
          ['vx [m/s]','vy [m/s]','vz [m/s]'],
          ['roll [deg]','pitch [deg]','yaw [deg]'],
          ['NMPC Fz [N]','|HGDO F| [N]','|HGDO Tau| [Nm]']]
for i in range(4):
    for j in range(3):
        axes[i,j].set_title(titles[i][j]); axes[i,j].grid(alpha=0.3)
        axes[i,j].set_xlabel('t [s]')
axes[0,0].legend(fontsize=8)
plt.tight_layout()
plt.savefig('bag_folder/compare_04_01_vs_04_07.png', dpi=130)
print("\nSaved: bag_folder/compare_04_01_vs_04_07.png")
