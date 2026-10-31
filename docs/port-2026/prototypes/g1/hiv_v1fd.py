import numpy as np, time, warnings
import active_subspaces as ac
exec(open('hiv_fd5.py').read().split('fro=np.where')[0])
space=ac.Uniform(lower=xl,upper=xu)
def T24(P): return np.array([solve_ivp(rhs,(0,24),[1000,0,0,30,0,5/.015,1],method='LSODA',t_eval=[24],args=(q,),rtol=1e-10,atol=1e-10).y[0,-1] for q in P])
rng=np.random.default_rng(0); P=space.sample(20,rng=rng)
t0=time.time(); gr=ac.finite_difference_gradients(T24,X=P,space=space,h=1e-6); print('v1 FD t=24, N=20: %.1fs'%(time.time()-t0))
sub=ac.active_subspace(grads=gr); print(sub.eigenvalues[:3], sub.n, np.round(sub.W[[5,13,24,26],0],3))
t0=time.time(); [T_ev(q) for q in P[:5]]; print('5 full 3400-day solves: %.2fs'%(time.time()-t0))
