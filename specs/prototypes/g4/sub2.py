import numpy as np, sys
d=sys.argv[1]; np.set_printoptions(precision=3,suppress=True,linewidth=200)
X=np.loadtxt(f'{d}/inputs-ShortCorr.GW02.txt'); fs=np.loadtxt(f'{d}/RunOff02_Saturated_10k.txt'); fu=np.loadtxt(f'{d}/RunOff02_Unsaturated_10k.txt')
def ols(X,f):
    a=np.linalg.lstsq(np.hstack([np.ones((len(f),1)),X]),f,rcond=None)[0][1:]; return a/np.linalg.norm(a)
lo,hi=X.min(0),X.max(0); Xmm=2*(X-lo)/(hi-lo)-1
nz=fu>0
variants={'gauss_all':(X,None),'minmax_all':(Xmm,None)}
for name,f in [('sat',fs),('unsat',fu)]:
    print('==',name)
    for vn,(Z,_) in variants.items():
        w=ols(Z,f); y=Z@w
        print(f' {vn:12s} w[:8]={w[:8]} y-range=[{y.min():.2f},{y.max():.2f}]')
    if name=='unsat':
        for vn,Z in [('gauss_nz',X[nz]),('minmax_nz',Xmm[nz])]:
            w=ols(Z,fu[nz]); y=Z@w; print(f' {vn:12s} w[:8]={w[:8]} y-range=[{y.min():.2f},{y.max():.2f}]')
            wl=ols(Z,np.log10(fu[nz])); print(f' {vn+"_log":12s} w[:8]={wl[:8]}')
        print(' cos(gauss_all, gauss_nz)=', abs(ols(X,fu)@ols(X[nz],fu[nz])), ' cos(gauss_all,minmax_all)=',abs(ols(X,fu)@ols(Xmm,fu)))
print('per-column |min|,max range', np.abs(lo).min(), np.abs(lo).max(), hi.min(), hi.max())
