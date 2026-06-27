import sqlite3, struct, glob
import numpy as np
def _align(o,n): return o+(-(o-4))%n
def p_rpm(b):
    o=4+8;s=struct.unpack_from('<I',b,o)[0];o+=4+s;o=_align(o,4)
    return np.array(struct.unpack_from('<6I',b,o),float)
def p_odom(b):
    o=4+8;s=struct.unpack_from('<I',b,o)[0];o+=4+s;o=_align(o,4)
    s2=struct.unpack_from('<I',b,o)[0];o+=4+s2;o=_align(o,8)
    pz=struct.unpack_from('<d',b,o+16)[0]
    o+=3*8+4*8+36*8+3*8
    wx,wy,wz=struct.unpack_from('<3d',b,o)
    return pz,wx,wy,wz
CT=1.3175e-7
def analyze(db):
    con=sqlite3.connect(db);cur=con.cursor()
    cur.execute("SELECT id,name FROM topics");t={n:i for i,n in cur.fetchall()}
    cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={t['/mavros/local_position/odom']}");t0=cur.fetchone()[0]
    def f(tp,pr):
        cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={t[tp]} ORDER BY timestamp")
        r=cur.fetchall();return np.array([(x-t0)*1e-9 for x,_ in r]),np.array([pr(b) for _,b in r])
    at,rpm=f('/uav/actual_rpm',p_rpm); ot,od=f('/mavros/local_position/odom',p_odom)
    con.close()
    pz=od[:,0];z=pz-pz[ot<5].mean() if (ot<5).any() else pz;ab=z>0.05
    t_to=ot[np.argmax(ab)];t_land=ot[len(ab)-1-np.argmax(ab[::-1])]
    lo,hi=t_to+2,t_land-2
    # total thrust over time
    Tt=CT*(rpm**2).sum(axis=1)
    # bin by thrust: in 1s windows, compute thrust mean and omega(<2Hz proxy: std of wx,wy)
    win=1.0
    res=[]
    tcur=lo
    while tcur+win<hi:
        m_o=(ot>=tcur)&(ot<tcur+win); m_a=(at>=tcur)&(at<tcur+win)
        if m_o.sum()>10 and m_a.sum()>5:
            wstd=np.sqrt(od[m_o,1].std()**2+od[m_o,2].std()**2)  # combined wx,wy std
            res.append((Tt[m_a].mean(), wstd))
        tcur+=win
    res=np.array(res)
    if len(res)<3: return None
    # correlation thrust vs omega std
    c=np.corrcoef(res[:,0],res[:,1])[0,1]
    return res[:,0].mean(), res[:,0].min(), res[:,0].max(), res[:,1].mean(), c, len(res)
print(f"{'case':22s} {'T_mean':>7s} {'T_min':>7s} {'T_max':>7s} {'wstd_mean':>9s} {'corr(T,w)':>9s} {'n':>3s}")
for ds in ['01','02']:
    for dob in ['hgdo','l1']:
        for c in ['wo_ff','ff_pivot_free','ff_pivot_based']:
            g=glob.glob(f'{ds}/{dob}/{c}/*.db3')
            if not g: continue
            r=analyze(g[0])
            if r: print(f"{ds}/{dob}/{c:14s} {r[0]:7.1f} {r[1]:7.1f} {r[2]:7.1f} {r[3]:9.3f} {r[4]:+9.3f} {r[5]:3d}")
