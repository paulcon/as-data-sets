import numpy as np, time, sys, warnings
sys.path.insert(0, '.')
from fvsolver import PDESolver
import active_subspaces as ac
s = PDESolver('/Users/paulconstantine/PycharmProjects/as-data-sets/uq_benchmarks/PDE_Inverse_Problem_MCMC/kl-short.npz')
rng = np.random.default_rng(1510)
xt = rng.standard_normal(100); obs = s.observations(xt); sig2 = 1e-4*obs@obs; d = obs + np.sqrt(sig2)*rng.standard_normal(7)
t = time.time(); [s.observations(rng.standard_normal(100)) for _ in range(50)]; tv = (time.time()-t)/50
print(f'value-only solve {tv*1e3:.1f} ms; sig2={sig2:.3e}')
X = rng.standard_normal((1000, 100)); t = time.time()
G = np.array([s.misfit_and_grad(x, d, sig2)[1] for x in X]); print('1000 misfit grads', round(time.time()-t, 1), 's')
sub = ac.active_subspace(space=ac.Gaussian.reference(100), grads=G, nboot=100, rng=0, n=2)
print('eigenvalues[:7]', sub.eigenvalues[:7]); print('ratios', sub.eigenvalues[:5]/sub.eigenvalues[1:6])
W1, W2 = sub.W1, sub.W2
mis = lambda x: .5*np.sum((s.observations(x)-d)**2)/sig2
def ghat(y, M=10):
    Z = rng.standard_normal((M, 98)); return np.mean([mis(W1@y + W2@z) for z in Z])
y = rng.standard_normal(2); g = ghat(y); acc = 0; N = 100; t = time.time()
for i in range(N):
    yp = y + 0.1*rng.standard_normal(2); gp = ghat(yp)
    if np.log(rng.uniform()) < (g - gp) + .5*(y@y - yp@yp): y, g = yp, gp; acc += 1
print(f'AV chain {N} steps (M=10, cached current) {time.time()-t:.1f}s, acceptance {acc/N:.2f}')
