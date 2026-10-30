import numpy as np, sys, warnings
import active_subspaces as ac
d=sys.argv[1]
np.set_printoptions(linewidth=220,precision=3,suppress=True)
def rd(fn):
    with open(fn) as f: hdr=[h.strip('"') for h in f.readline().split()]
    return hdr, np.loadtxt(fn, skiprows=1, ndmin=2)
names,X=rd(f'{d}/stomata_realization_inputs.txt'); _,B=rd(f'{d}/param_ranges.txt')
sp=ac.Uniform(lower=B[:,0],upper=B[:,1],names=names)
months=["September","October","November","December","January","February","March","April","May","June","July","August"]
hours=["GMT_12","GMT_13","GMT_14","GMT_15","GMT_16","GMT_17","GMT_18","GMT_19","GMT_20","GMT_21","GMT_22","GMT_23","GMT_0","GMT_1"]
nW=0
for fn,cols in [('mo_avg_tran_veg',months),('mo_avg_rssha',months),('mo_avg_rssun',months),('JJA_hr_avg_tran_veg',hours),('JJA_hr_avg_rssha',hours),('JJA_hr_avg_rssun',hours)]:
    h,F=rd(f'{d}/Tonzi_Stomata_SC.{fn}.txt'); F=F[:,[h.index(c) for c in cols]]
    XX=sp.to_normalized(X)
    print('==',fn)
    for i,c in enumerate(cols):
        f=F[:,i]
        with warnings.catch_warnings(record=True) as Wn:
            warnings.simplefilter('always')
            s=ac.active_subspace(X=X,f=f,space=sp,estimator='ols',n=1)
        nW+=len(Wn)
        A=np.hstack([np.ones((len(f),1)),XX]); u=np.linalg.lstsq(A,f,rcond=None)[0][1:]
        wl=u/np.linalg.norm(u); wl=wl*np.sign(wl[0])
        w1=s.W1[:,0]
        # without row 0
        u2=np.linalg.lstsq(A[1:],f[1:],rcond=None)[0][1:]; 
        y=XX@wl; r=np.corrcoef(y,f)[0,1]**2
        top=[names[j] for j in np.argsort(-abs(wl))[:4]]
        print(f' {c:10s} |cos(v1,leg)|={abs(w1@wl):.10f} sgn(v1·leg)={np.sign(w1@wl):+.0f} sgn(v1·u)={np.sign(w1@u):+.0f} cos(drop row0)={abs(u2@u)/np.linalg.norm(u)/np.linalg.norm(u2):.4f} linR2={r:.2f} top={top} w[0]={wl[0]:+.3f}')
print('total warnings', nW)
