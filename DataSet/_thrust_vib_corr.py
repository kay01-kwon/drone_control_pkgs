import sqlite3, struct, glob
import numpy as np
from scipy import signal
def _align(o,n): return o+(-(o-4))%n
def p_rpm(b):
    o=4+8;s=struct.unpack_from('<I',b,o)[0];o+=4+s;o=_align(o,4)
    return np.array(struct.unpack_from('<6I',b,o),float)
def p_imu(b):
    o=4+8;s=struct.unpack_from('<I',b,o)[0];o+=4+s;o=_align(o,8)
    o+=4*8+9*8
    wx,wy,wz=struct.unpack_from('<3d',b,o);o+=24+9*8
    ax,ay,az=struct.unpack_from('<3d',b,o)
    return wx,wy,wz,ax,ay,az
def p_oz(b):
    o=4+8;s=struct.unpack_from('<I',b,o)[0];o+=4+s;o=_align(o,4)
    s2=struct.unpack_from('<I',b,o)[0];o+=4+s2;o=_align(o,8)
    return struct.unpack_from('<d',b,o+16)[0]
CT=1.3175e-7
def analyze(db):
    con=sqlite3.connect(db);cur=con.cursor()
    cur.execute("SELECT id,name FROM topics");t={n:i for i,n in cur.fetchall()}
    cur.execute(f"SELECT MIN(timestamp) FROM messages WHERE topic_id={t['/mavros/local_position/odom']}");t0=cur.fetchone()[0]
    def f(tp,pr):
        cur.execute(f"SELECT timestamp,data FROM messages WHERE topic_id={t[tp]} ORDER BY timestamp")
        r=cur.fetchall();return np.array([(x-t0)*1e-9 for x,_ in r]),np.array([pr(b) for _,b in r])
    at,rpm=f('/uav/actual_rpm',p_rpm); it,imu=f('/mavros/imu/data_raw',p_imu)
    ot,oz=f('/mavros/local_position/odom',p_oz)
    con.close()
    z=oz-oz[ot<5].mean() if (ot<5).any() else oz;ab=z>0.05
    t_to=ot[np.argmax(ab)];t_land=ot[len(ab)-1-np.argmax(ab[::-1])]
    lo,hi=t_to+2,t_land-2
    Tt=CT*(rpm**2).sum(axis=1)
    fs=1/np.median(np.diff(it))
    # high-freq accel vibration RMS (>20Hz) in 1s windows vs thrust
    win=1.0;res=[]
    tc=lo
    while tc+win<hi:
        ma=(at>=tc)&(at<tc+win);mi=(it>=tc)&(it<tc+win)
        if ma.sum()>5 and mi.sum()>30:
            azw=imu[mi,5]-imu[mi,5].mean()
            # >20Hz band rms via highpass
            sos=signal.butter(2,20/(fs/2),'hp',output='sos')
            azhf=signal.sosfilt(sos,azw)
            res.append((Tt[ma].mean(),azhf.std()))
        tc+=win
    res=np.array(res)
    if len(res)<4:return None
    c=np.corrcoef(res[:,0],res[:,1])[0,1]
    return res[:,0].mean(),res[:,1].mean(),res[:,1].min(),res[:,1].max(),c,len(res)
print(f"{'case':22s} {'T_mean':>7s} {'vib_az':>7s} {'vmin':>6s} {'vmax':>6s} {'corr(T,vib)':>11s}")
for ds in ['01','02']:
    for dob in ['hgdo','l1']:
        for c in ['wo_ff','ff_pivot_free','ff_pivot_based']:
            g=glob.glob(f'{ds}/{dob}/{c}/*.db3')
            if not g:continue
            r=analyze(g[0])
            if r:print(f"{ds}/{dob}/{c:14s} {r[0]:7.1f} {r[1]:7.2f} {r[2]:6.2f} {r[3]:6.2f} {r[4]:+11.3f}")
