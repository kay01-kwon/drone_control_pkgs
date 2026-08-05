#!/usr/bin/env python3
"""Admissible moment set of the hexarotor at fixed total thrust, for the
implemented allocator (pseudo-inverse + per-rotor clamp), compared with an
optimal (LP / l1-dual) allocator.

Constraint:  dT = G M,   -dn <= dT_i <= up,   G = pinv(K)[:, 1:4]
             up = T_max - Tbar,  dn = Tbar - T_min,  Tbar = T_tot/6

Panels
  (a) roll-pitch slice at Mz = 0, 0.1, 0.2, 0.3 N.m  -> shrinking hexagon
  (b) roll-yaw slice at My = 0, with the budget line |Mx|/Mx_max+|Mz|/Mz_max=1
  (c) 3-D admissible set (pinv) : thin slab along yaw
  (d) single-axis capacity: pinv vs optimal allocator

Usage: python3 _alloc_moment_set.py [T_tot_kgf]   (default 3.066)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

_HERE = os.path.dirname(os.path.abspath(__file__))
M_KGF = float(sys.argv[1]) if len(sys.argv) > 1 else 3.066

C_T = 1.3175e-07; K_M = 0.01569; L = 0.265
RPM_MIN, RPM_MAX = 2000.0, 7900.0
G_ACC = 9.81

c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)
ly = L * np.array([c, 1, c, -c, -1, -c])
lx = L * np.array([s, 0, -s, -s, 0, s])
sg = np.array([-1, 1, -1, 1, -1, 1])
K = np.vstack([np.ones(6), ly, -lx, K_M * sg])
G = np.linalg.pinv(K)[:, 1:4]                  # rotor thrust per [Mx,My,Mz]

T_tot = M_KGF * G_ACC
Tbar = T_tot / 6
T_min = C_T * RPM_MIN**2
T_max = C_T * RPM_MAX**2
up = T_max - Tbar
dn = Tbar - T_min


def feasible(M):
    dT = G @ np.asarray(M, float)
    return np.all(dT <= up + 1e-12) and np.all(-dT <= dn + 1e-12)


def radial(dirv, hi=20.0):
    """largest a>=0 with a*dirv feasible (bisection)."""
    dirv = np.asarray(dirv, float)
    if not feasible(dirv * 1e-9):
        return 0.0
    lo_, hi_ = 0.0, hi
    if feasible(dirv * hi_):
        return hi_
    for _ in range(60):
        mid = 0.5 * (lo_ + hi_)
        if feasible(dirv * mid):
            lo_ = mid
        else:
            hi_ = mid
    return lo_


# single-axis capacities
def axis_cap(j):
    e = np.zeros(3); e[j] = 1.0
    return radial(e), radial(-e)


caps = [axis_cap(j) for j in range(3)]
Mx_max, My_max, Mz_max = caps[0][0], caps[1][0], caps[2][0]

# optimal (l1-dual) capacities: Mj_max = dTmax * 1/||row||... -> ||a||_1 * dTmax
dTsym = min(up, dn)
opt_Mx = np.sum(np.abs(ly)) * dTsym
opt_My = np.sum(np.abs(lx)) * dTsym
opt_Mz = 6 * K_M * dTsym

print(f"T_tot = {M_KGF}*g = {T_tot:.3f} N,  Tbar = {Tbar:.4f} N")
print(f"rotor bounds: [{T_min:.4f}, {T_max:.4f}] N -> up={up:.4f}, dn={dn:.4f}")
print(f"\nimplemented (pinv+clamp) clamp-onset capacity:")
print(f"  Mx = +-{Mx_max:.4f}   My = +-{My_max:.4f}   Mz = +-{Mz_max:.4f}  N.m")
print(f"  ||pinv col||_inf : Mx {np.abs(G[:,0]).max():.4f}  My {np.abs(G[:,1]).max():.4f}  Mz {np.abs(G[:,2]).max():.4f} N/(N.m)")
print(f"optimal (l1-dual) capacity:")
print(f"  Mx = +-{opt_Mx:.4f}   My = +-{opt_My:.4f}   Mz = +-{opt_Mz:.4f}  N.m")
print(f"  loss: Mx {100*(1-Mx_max/opt_Mx):.1f}%   My {100*(1-My_max/opt_My):.1f}%   Mz {100*(1-Mz_max/opt_Mz):.1f}%")
print(f"\nyaw/roll cost ratio (pinv) = {np.abs(G[:,2]).max()/np.abs(G[:,0]).max():.2f}")

# ---- boundary curves ----
def slice_xy(Mz, n=721):
    th = np.linspace(0, 2 * np.pi, n)
    r = np.array([radial([np.cos(t), np.sin(t), 0]) if Mz == 0 else 0.0 for t in th])
    if Mz != 0:
        r = np.array([_radial_off(t, Mz) for t in th])
    return r * np.cos(th), r * np.sin(th)


def _radial_off(t, Mz):
    d = np.array([np.cos(t), np.sin(t), 0.0])
    if not feasible([0, 0, Mz]):
        return 0.0
    lo_, hi_ = 0.0, 6.0
    if feasible(np.array([0, 0, Mz]) + d * hi_):
        return hi_
    for _ in range(50):
        mid = 0.5 * (lo_ + hi_)
        if feasible(np.array([0, 0, Mz]) + d * mid):
            lo_ = mid
        else:
            hi_ = mid
    return lo_


fig = plt.figure(figsize=(14, 10))

# (a) roll-pitch slices
ax = fig.add_subplot(2, 2, 1)
for Mz, col in zip([0.0, 0.10, 0.20, 0.28], ['k', 'tab:blue', 'tab:orange', 'tab:red']):
    th = np.linspace(0, 2 * np.pi, 721)
    r = np.array([_radial_off(t, Mz) for t in th])
    ax.plot(r * np.cos(th), r * np.sin(th), color=col, lw=1.6,
            label=f'$M_z$={Mz:.2f} N·m')
ax.set_xlabel('$M_x$ [N·m]'); ax.set_ylabel('$M_y$ [N·m]')
ax.set_title('(a) roll–pitch set shrinks as yaw grows')
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_aspect('equal')

# (b) roll-yaw slice + budget line
ax = fig.add_subplot(2, 2, 2)
th = np.linspace(0, 2 * np.pi, 1441)
pts = []
for t in th:
    d = np.array([np.cos(t), 0.0, np.sin(t) * Mz_max / Mx_max])
    a = radial(d)
    pts.append([a * d[0], a * d[2]])
pts = np.array(pts)
ax.plot(pts[:, 0], pts[:, 1], 'b', lw=1.8, label='admissible boundary (pinv+clamp)')
xx = np.linspace(-Mx_max, Mx_max, 200)
ax.plot(xx, Mz_max * (1 - np.abs(xx) / Mx_max), 'r--', lw=1.2,
        label=r'$|M_x|/M_x^{max}+|M_z|/M_z^{max}=1$')
ax.plot(xx, -Mz_max * (1 - np.abs(xx) / Mx_max), 'r--', lw=1.2)
for mz, lab in [(0.15, '50%'), (0.30, '0%')]:
    if mz <= Mz_max:
        ax.plot([0, Mx_max * (1 - mz / Mz_max)], [mz, mz], ':', color='gray', lw=1)
        ax.annotate(f'$M_z$={mz:.2f}: roll {lab}', (0.05, mz), fontsize=7, color='gray')
ax.set_xlabel('$M_x$ [N·m]'); ax.set_ylabel('$M_z$ [N·m]')
ax.set_title('(b) roll–yaw trade-off ($M_y=0$)')
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.grid(alpha=0.3); ax.legend(fontsize=7)

# (c) 3-D set
ax = fig.add_subplot(2, 2, 3, projection='3d')
dirs = []
for u_ in np.linspace(-1, 1, 26):
    for ph in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        st = np.sqrt(max(1 - u_**2, 0))
        dirs.append([st * np.cos(ph), st * np.sin(ph), u_ * Mz_max / Mx_max])
V = np.array([radial(d) * np.array(d) for d in dirs])
V = V[np.linalg.norm(V, axis=1) > 1e-9]
hull = ConvexHull(V)
polys = [V[simplex] for simplex in hull.simplices]
pc = Poly3DCollection(polys, alpha=0.35, facecolor='tab:blue', edgecolor='none')
ax.add_collection3d(pc)
ax.set_xlabel('$M_x$'); ax.set_ylabel('$M_y$'); ax.set_zlabel('$M_z$')
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(-0.35, 0.35)
ax.set_title('(c) admissible set: thin slab along yaw', fontsize=10)
ax.view_init(elev=18, azim=35)

# (d) capacity bars
ax = fig.add_subplot(2, 2, 4)
lbl = ['$M_x$ (roll)', '$M_y$ (pitch)', '$M_z$ (yaw)']
xpos = np.arange(3)
ax.bar(xpos - 0.2, [opt_Mx, opt_My, opt_Mz], 0.4, label='optimal ($\\ell_1$-dual)', color='tab:green')
ax.bar(xpos + 0.2, [Mx_max, My_max, Mz_max], 0.4, label='implemented (pinv+clamp)', color='tab:blue')
for i, (o, p) in enumerate(zip([opt_Mx, opt_My, opt_Mz], [Mx_max, My_max, Mz_max])):
    ax.annotate(f'{p:.2f}', (i + 0.2, p), ha='center', va='bottom', fontsize=8)
    ax.annotate(f'{o:.2f}', (i - 0.2, o), ha='center', va='bottom', fontsize=8)
    ax.annotate(f'−{100*(1-p/o):.0f}%', (i + 0.2, p / 2), ha='center', fontsize=8, color='white')
ax.set_xticks(xpos); ax.set_xticklabels(lbl)
ax.set_ylabel('clamp-onset capacity [N·m]'); ax.set_yscale('log')
ax.set_title('(d) capacity: pinv+clamp vs optimal'); ax.grid(alpha=0.3, axis='y')
ax.legend(fontsize=8)

fig.suptitle(f'Hexarotor admissible moment set at fixed total thrust '
             f'$T_{{tot}}$={M_KGF}$g$={T_tot:.1f} N '
             f'($\\bar T$={Tbar:.2f} N, rotor {RPM_MIN:.0f}–{RPM_MAX:.0f} RPM)', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(_HERE, f'alloc_moment_set_{M_KGF:.3f}kgf.png')
plt.savefig(out, dpi=130)
print(f"\nSaved: {out}")
