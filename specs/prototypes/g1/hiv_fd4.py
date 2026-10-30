import numpy as np, time, warnings, sys
warnings.simplefilter('ignore')
exec(open('hiv_fd2.py').read().split('s=0; q=X[s]')[0])
which=sys.argv[1]
if which=='legacy':
    t0=time.time(); p=X[:1]; hh=1e-7*half
    base=Tcells(p,np.linspace(1,3400,3400))[0,-1]; g=[]
    for j in range(27):
        pp=p.copy(); pp[0,j]+=hh[j]; g.append((Tcells(pp,np.linspace(1,3400,3400))[0,-1]-base)/hh[j]*half[j])
    g=np.array(g); ship=DF[0,-27:]; print('legacy-path py3 repro t=3400 s0: relerr', np.linalg.norm(g-ship)/np.linalg.norm(ship), '%.1fs'%(time.time()-t0), flush=True)
else:
    def rhs_f(t,x,q): return [0.0]*7 if x[0] < .1 else rhs(t,x,q)
    def T_f(q):
        s=solve_ivp(rhs_f,(0,times[-1]),[1000,0,0,30,0,5/.015,1],method='LSODA',t_eval=times,args=(q,),rtol=1e-9,atol=1e-9); return s.y[0]
    fro=np.where(F[:,-1]<0.1001)[0]; nf=np.where(F[:,-1]>1)[0]
    for s in list(nf[:4])+list(fro[:3]):
        t0=time.time(); q=X[s]; h=1e-4; G=np.empty((21,27))
        for j in range(27):
            e=np.zeros(27); e[j]=h*half[j]; G[:,j]=(T_f(q+e)-T_f(q-e))/(2*h)
        for i in (19,20):
            ship=DF[s,27*i:27*(i+1)]; c=abs(G[i]@ship)/(np.linalg.norm(G[i])*np.linalg.norm(ship)+1e-300)
            print(f's={s} t={times[i]} F={F[s,i]:.4g} |g|={np.linalg.norm(G[i]):.3g} |ship|={np.linalg.norm(ship):.3g} cos={c:.3f} ({time.time()-t0:.0f}s)', flush=True)
