import numpy as np, time, sys
sys.path.insert(0, '.')
from fvsolver import PDESolver
R='/Users/paulconstantine/PycharmProjects/as-data-sets/uq_benchmarks/Elliptic_PDE_Random_Coefficients/'
s = PDESolver(R+'kl-long.npz'); rng = np.random.default_rng(1)
x = rng.standard_normal(100); a = s.coeff(x)
u = rng.standard_normal((99, 99)); L = s.matrix(a)
print('matrix vs legacy residual:', np.abs((L@u.ravel()+1).reshape(99,99) - s.residual(u, a)).max())
t=time.time(); q, g = s.qoi_and_grad(x); print('q', q, 'time', time.time()-t)
print('residual at solution', np.abs(s.residual(s.solve(x)[2], a)).max())
for j in [0, 5, 50, 99]:
    e = np.zeros(100); e[j] = 1e-6
    fd = (s.qoi_and_grad(x+e)[0]-s.qoi_and_grad(x-e)[0])/2e-6
    print('grad', j, g[j], fd, abs(g[j]-fd)/abs(fd))
d = s.observations(rng.standard_normal(100)); 
mf, gm = s.misfit_and_grad(x, d, 1e-4*np.sum(d**2))
e = rng.standard_normal(100)*1e-6
print('misfit dir-deriv', gm@e/1e-6, (s.misfit_and_grad(x+e,d,1e-4*np.sum(d**2))[0]-s.misfit_and_grad(x-e,d,1e-4*np.sum(d**2))[0])/2e-6)
