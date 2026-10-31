import numpy as np, time, warnings
from scipy.integrate import solve_ivp
from hiv_model_copy import Tcells
d='/Users/paulconstantine/PycharmProjects/as-data-sets/epidemiology_systems_biology/HIV_Long_Term_Dynamics/'
X=np.loadtxt(d+'inputs.csv',delimiter=',',skiprows=1)[:,1:]; F=np.loadtxt(d+'outputs.csv',delimiter=',',skiprows=1)[:,1:]; DF=np.loadtxt(d+'gradients.csv',delimiter=',',skiprows=1)[:,1:]
nominal = np.array([10, .15, 5, .2, 55.6, 3.87e-3, 1e-6, 4.5e-4, 7.45e-4, 5.22e-4, 3e-6, 3.3e-4, 6e-9, .537, .285, 7.79e-6, 1e-6, 4e-5, .01, .28, .05, .005, .005, .015, 2.39, 3e-4, .97])
xl=.975*nominal; xu=1.025*nominal; half=(xu-xl)/2
times = np.array([5, 15, 24, 38, 40, 45, 50, 55, 65, 90, 140, 500, 750, 1000, 1600, 1800, 2000, 2200, 2400, 2800, 3400])
def rhs(t,x,q):
    s1,s2,s3,p1,C1,K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12,K13,d1,d2,d3,d4,d5,d6,d7,a1,psi=q
    T,TI,TL,M,MI,CTL,V=x
    inf=(K1*V+K2*MI)*T
    return [s1+p1*T*V/(C1+V)-d1*T-inf, psi*inf+a1*TL-d2*TI-K3*TI*CTL, (1-psi)*inf-a1*TL-d3*TL,
            s2+K4*M*V-K5*M*V-d4*M, K5*M*V-d5*MI-K6*MI*CTL, s3+(K7*TI+K8*MI)*CTL-d6*CTL,
            K9*TI+K10*MI-K11*T*V-(K12+K13)*M*V-d7*V]
def T_ivp(q, rtol=1e-10):
    y0=[1000,0,0,.15/.005,0,5/.015,1]
    s=solve_ivp(rhs,(0,times[-1]),y0,method='LSODA',t_eval=times,args=(q,),rtol=rtol,atol=1e-10)
    return s.y[0]
s=0; q=X[s]
t0=time.time(); base=T_ivp(q); print('ivp solve %.2fs'%(time.time()-t0), 'T(t) ivp vs shipped rel', np.max(abs(base-F[s])/F[s]))
print('ivp T(3400)=',base[-1],' shipped', F[s,-1], ' min T over traj?')
for h in [1e-3,1e-4]:
    G=np.empty((21,27))
    for j in range(27):
        e=np.zeros(27); e[j]=h*half[j]
        G[:,j]=(T_ivp(q+e)-T_ivp(q-e))/(2*h)
    for i in [2,11,14,17,20]:
        ship=DF[s,27*i:27*(i+1)]
        c=abs(G[i]@ship)/np.linalg.norm(G[i])/np.linalg.norm(ship)
        print(f'h={h:g} t={times[i]}: |g_ivp|={np.linalg.norm(G[i]):.4g} |shipped|={np.linalg.norm(ship):.4g} relerr={np.linalg.norm(G[i]-ship)/np.linalg.norm(G[i]):.2e} cos={c:.4f}')
