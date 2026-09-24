#!/usr/bin/env python3
"""Thrust ratio T/W_nom versus time, with W_nom fixed at the NOMINAL mass.

T(t)      = C_T * sum(rpm_i^2)      from /uav/actual_rpm
W_nom     = m_nom * g               m_nom = 3.0 kg (yaml nominal), NOT the
                                    measured mass, so the same reference is
                                    used for every dataset
Overlays the unity line (the bare  T >= W  detection threshold) and the
mocap-derived airborne interval, and prints the hover statistics that
determine how often the ratio crosses unity.

Usage: python3 _thrust_ratio.py [m_nom]        (default 3.0)
"""
import os, sqlite3, struct, glob, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
M_NOM = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
G     = 9.81
C_T   = 1.3175e-07
W_NOM = M_NOM * G

BAGS = [('01/hgdo/ff_pivot_based', 'no payload, HGDO, ff pivot-based'),
        ('01/l1/wo_ff',            'no payload, L1, w/o ff'),
        ('02/hgdo/ff_pivot_based', 'payload,    HGDO, ff pivot-based'),
        ('02/l1/wo_ff',            'payload,    L1, w/o ff')]


def _al(o, n):
    return o + (-(o - 4)) % n


def p_rpm(b):
    o = 4 + 8; s = struct.unpack_from('<I', b, o)[0]; o += 4 + s; o = _al(o, 4)
    return np.array(struct.unpack_from('<6I', b, o), float)


def p_oz(b):
    o = 4 + 8; s = struct.unpack_from('<I', b, o)[0]; o += 4 + s; o = _al(o, 4)
    s2 = struct.unpack_from('<I', b, o)[0]; o += 4 + s2; o = _al(o, 8)
    return struct.unpack_from('<d', b, o + 16)[0]


def load(bag):
    db = glob.glob(os.path.join(_HERE, bag, '*.db3'))[0]
    con = sqlite3.connect(db); cur = con.cursor()
    cur.execute("SELECT id,name FROM topics"); tid = {n: i for i, n in cur.fetchall()}
    cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={tid['/mavros/local_position/odom']}")
    t0 = cur.fetchone()[0]

    def f(topic, parser):
        cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={tid[topic]} ORDER BY timestamp")
        r = cur.fetchall()
        return np.array([(t - t0) * 1e-9 for t, _ in r]), np.array([parser(b) for _, b in r])

    rt, rpm = f('/uav/actual_rpm', p_rpm)
    ot, oz  = f('/mavros/local_position/odom', p_oz)
    con.close()
    T  = C_T * (rpm ** 2).sum(1)
    z0 = oz[ot < 5].mean()
    zr = np.interp(rt, ot, oz - z0)
    return rt, T / W_NOM, zr


fig, axes = plt.subplots(len(BAGS), 1, figsize=(12, 11), sharex=False)
print(f"W_nom = {M_NOM} kg * g = {W_NOM:.3f} N   (fixed reference for all datasets)\n")
print(f"{'bag':26s} {'hover T/Wnom':>16s} {'sigma[%Wnom]':>13s} {'(mu-1)/sigma':>13s} {'crossings':>10s}")

for ax, (bag, label) in zip(axes, BAGS):
    t, R, zr = load(bag)
    ab = zr > 0.05
    i0 = int(np.argmax(ab)); i1 = len(ab) - 1 - int(np.argmax(ab[::-1]))
    t_lo, t_hi = t[i0], t[i1]
    win = (t > t_lo + 1) & (t < t_hi - 1)
    mu, sd = R[win].mean(), R[win].std()
    ncr = int(np.abs(np.diff((R[win] > 1.0).astype(int))).sum())
    print(f"{bag:26s} {mu:8.4f} +- {sd:.4f} {100*sd:12.2f} {(mu-1)/sd:+13.2f} {ncr:10d}")

    ax.axvspan(t_lo, t_hi, color='g', alpha=0.06, label='airborne (mocap $z>5$ cm)')
    ax.plot(t, R, lw=0.8, color='tab:blue')
    ax.axhline(1.0, color='r', ls='--', lw=1.2, label=r'$T/W_{\rm nom}=1$  (bare threshold)')
    ax.fill_between([t_lo + 1, t_hi - 1], mu - sd, mu + sd,
                    color='tab:orange', alpha=0.25, label=r'hover mean $\pm\,1\sigma$')
    ax.axhline(mu, color='tab:orange', lw=1.0)
    ax.set_ylabel(r'$T/W_{\rm nom}$')
    ax.set_ylim(0, 1.25)
    ax.grid(alpha=0.3)
    ax.set_title(f'{bag}   —   {label}   |   hover ${mu:.3f}\\pm{sd:.3f}$, '
                 f'$({mu:.3f}-1)/\\sigma={((mu-1)/sd):+.2f}$, {ncr} crossings',
                 fontsize=9)
    if ax is axes[0]:
        ax.legend(fontsize=8, loc='lower right', ncol=3)

axes[-1].set_xlabel('time [s]')
fig.suptitle(rf'Thrust ratio $T/W_{{\rm nom}}$ ($W_{{\rm nom}}={M_NOM}$ kgf $= {W_NOM:.1f}$ N)', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.985])
out = os.path.join(_HERE, 'thrust_ratio.png')
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")
