# Item 4: activity scores (Constantine & Diaz 2017; Diaz et al. 2018 Fig. 9), Ebola 6^8 rule
import numpy as np
from scipy.special import roots_legendre
import active_subspaces as ac
print("names containing 'activ' or 'sobol' or 'sensitiv':", [n for n in dir(ac) if any(k in n.lower() for k in ("activ", "sobol", "sensitiv", "score"))])
names = ['b1', 'b2', 'b3', 'rho1', 'g1', 'g2', 'om', 'psi']
lbL = np.array([.1, .1, .05, .41, .0276, .081, .25, .0833]); ubL = np.array([.4, .4, .2, 1, .1702, .21, .5, .7])
lbS = np.array([.1, .1, .05, .41, .0275, .1236, .25, .0833]); ubS = np.array([.4, .4, .2, 1, .1569, .384, .5, .7])
def R0_grad(x):
    b1,b2,b3,r1,g1,g2,om,p = x.T; D=g1+p; R=(b1 + b2*r1*g1/om + b3*p/g2)/(g1 + p)
    return np.column_stack([1/D, r1*g1/om/D, p/g2/D, b2*g1/om/D, (b2*r1/om - R)/D, -b3*p/g2**2/D, -b2*r1*g1/om**2/D, (b3/g2 - R)/D])
t, w = roots_legendre(6); w = w / 2
T = np.stack(np.meshgrid(*([t] * 8), indexing='ij'), -1).reshape(-1, 8)
Wq = np.prod(np.stack(np.meshgrid(*([w] * 8), indexing='ij'), -1).reshape(-1, 8), 1)
for nm, lb, ub in (("Liberia", lbL, ubL), ("Sierra Leone", lbS, ubS)):
    sp = ac.Uniform(lb, ub, names=names)
    sub = ac.active_subspace(space=sp, grads=R0_grad(sp.from_normalized(T)), quad_weights=Wq)
    a = (sub.eigenvalues[:2] * sub.W[:, :2] ** 2).sum(1)
    print(f"{nm}: lambda1={sub.eigenvalues[0]:.4f}, activity scores alpha(2)/max: {dict(zip(names, np.round(a / a.max(), 3)))}")
