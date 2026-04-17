#!/usr/bin/env python3
"""Check NMPC moments near target altitude in sim_new bag."""

import sqlite3, struct
import numpy as np
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
        off = 4; off += 4; off += 4
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

base = Path('/home/user/drone_control_pkgs/bag_folder')
db = base / '2026_04_17_sim_new' / '2026_04_17_sim_new_0.db3'

od_raw = read_bag(db, '/mavros/local_position/odom', parse_odom)
nm_raw = read_bag(db, '/nmpc/control', parse_wrench)
wr_raw = read_bag(db, '/hgdo/wrench', parse_wrench)
ref_raw = read_bag(db, '/nmpc/ref', parse_ref)

t0 = od_raw[0][0]
od_t = np.array([d[0]-t0 for d in od_raw])
od_v = np.array([d[1] for d in od_raw])
p_off = od_v[0, 0:3].copy()
od_v[:, 0:3] -= p_off

nm_t = np.array([d[0]-t0 for d in nm_raw])
nm_v = np.array([d[1] for d in nm_raw])
wr_t = np.array([d[0]-t0 for d in wr_raw])
wr_v = np.array([d[1] for d in wr_raw])
ref_t = np.array([d[0]-t0 for d in ref_raw])
ref_v = np.array([d[1] for d in ref_raw])

print("Ref messages:")
for i in range(len(ref_t)):
    r = ref_v[i]
    print(f"  t={ref_t[i]:.1f}s: p=({r[0]:.3f},{r[1]:.3f},{r[2]:.3f})")

# Find when drone reaches target altitude (ref_z)
ref_z = ref_v[0, 2] if len(ref_v) > 0 else 0.5
print(f"\nTarget altitude: {ref_z:.3f} m")

# Find time range where altitude is within 5cm of target
alt = od_v[:, 2]
near_target = np.abs(alt - ref_z) < 0.05
if np.any(near_target):
    idx_near = np.where(near_target)[0]
    t_start = od_t[idx_near[0]]
    t_end = od_t[idx_near[-1]]
    print(f"Near target altitude ({ref_z:.3f}m ± 0.05m): t={t_start:.1f}s to t={t_end:.1f}s")

    # NMPC moments in this time range
    nm_mask = (nm_t >= t_start) & (nm_t <= t_end)
    if np.any(nm_mask):
        mx = nm_v[nm_mask, 3]
        my = nm_v[nm_mask, 4]
        mz = nm_v[nm_mask, 5]
        fz = nm_v[nm_mask, 2]
        print(f"\nNMPC during hover near target:")
        print(f"  Fz:  mean={fz.mean():.2f}, std={fz.std():.2f}, range=[{fz.min():.2f}, {fz.max():.2f}] N")
        print(f"  Mx:  mean={mx.mean():.4f}, std={mx.std():.4f}, range=[{mx.min():.4f}, {mx.max():.4f}] Nm")
        print(f"  My:  mean={my.mean():.4f}, std={my.std():.4f}, range=[{my.min():.4f}, {my.max():.4f}] Nm")
        print(f"  Mz:  mean={mz.mean():.4f}, std={mz.std():.4f}, range=[{mz.min():.4f}, {mz.max():.4f}] Nm")

    # HGDO wrench in this time range
    wr_mask = (wr_t >= t_start) & (wr_t <= t_end)
    if np.any(wr_mask):
        print(f"\nHGDO during hover near target:")
        for i, lbl in enumerate(['fx','fy','fz','tx','ty','tz']):
            v = wr_v[wr_mask, i]
            print(f"  {lbl}: mean={v.mean():.4f}, std={v.std():.4f}, range=[{v.min():.4f}, {v.max():.4f}]")

    # Position during this time
    px = od_v[near_target, 0]
    py = od_v[near_target, 1]
    pz = od_v[near_target, 2]
    print(f"\nPosition during hover near target:")
    print(f"  px: mean={px.mean():.4f}, std={px.std():.4f}")
    print(f"  py: mean={py.mean():.4f}, std={py.std():.4f}")
    print(f"  pz: mean={pz.mean():.4f}, std={pz.std():.4f}")

    # Also check steady-state (last 10s of near-target)
    ss_start = max(t_start, t_end - 10)
    nm_ss = (nm_t >= ss_start) & (nm_t <= t_end)
    if np.any(nm_ss):
        print(f"\nSteady-state (last 10s, t={ss_start:.1f}-{t_end:.1f}s):")
        print(f"  Mx: mean={nm_v[nm_ss,3].mean():.4f} Nm")
        print(f"  My: mean={nm_v[nm_ss,4].mean():.4f} Nm")
        print(f"  Mz: mean={nm_v[nm_ss,5].mean():.4f} Nm")
        print(f"  Fz: mean={nm_v[nm_ss,2].mean():.2f} N (W={3.146*9.81:.1f}N)")
