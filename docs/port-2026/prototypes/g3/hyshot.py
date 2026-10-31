import sys, warnings, numpy as np, active_subspaces as ac
np.set_printoptions(precision=4, suppress=True, linewidth=150)
D = sys.argv[1]
names = ["AoA","Turb int","Turb len","Stag pres","Stag enth","Cowl trans","Ramp trans"]
xl = np.array([2.6, 0.1, 122.5, 1.6448e7, 3.0551e6, 0.03, 0.087])
xu = np.array([4.6, 1.9, 367.5, 1.9012e7, 3.4280e6, 0.07, 0.203])
space = ac.Uniform(xl, xu, names=names)
d35 = np.loadtxt(f"{D}/HyShotII35.txt", delimiter=",", skiprows=1)
d30 = np.loadtxt(f"{D}/HyShotII30.txt", delimiter=",")
X35 = d35[:, 1:8].copy(); X35[-2:] = space.from_normalized(d35[-2:, 1:8])
f35 = d35[:, 9]
X30 = space.from_normalized(d30[:, :7]); f30 = d30[:, 7]
print("shapes", X35.shape, X30.shape)
leg30 = np.array([0.6506,0.5565,-0.0002,0.3685,-0.3566,-0.0196,0.0607])
leg35 = np.array([0.6996,0.4823,-0.027,0.1997,-0.4738,-0.0602,0.0957])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    s30 = ac.active_subspace(X=X30[:-2], f=f30[:-2], space=space, estimator="ols", n=1, nboot=500, rng=0)
    s35 = ac.active_subspace(X=X35[:-2], f=f35[:-2], space=space, estimator="ols", n=1, nboot=500, rng=0)
    print("warnings:", sorted({str(x.category.__name__)+': '+str(x.message)[:120] for x in w}))
for s, leg, nm in [(s30, leg30, "0.30"), (s35, leg35, "0.35")]:
    w1 = s.W1[:, 0]
    print(nm, "v1 w:", w1, " legacy:", leg, " max|diff|", np.abs(w1-leg).max(), "lambda1", s.eigenvalues[0])
# extremes: are rows -2/-1 the argmin/argmax of w^T x?
for s, d, nm in [(s30, d30[:, :7], "0.30"), (s35, d35[:, 1:8], "0.35")]:
    w1 = s.W1[:, 0]
    print(nm, "sign(w) =", np.sign(w1), " stored min row =", np.sign(d[-2]), " max row =", np.sign(d[-1]))
# legacy-style bootstrap but with lstsq, on the SAME 50 points (legacy used all 52)
rng = np.random.default_rng(0)
Z = d30[:, :7]
def boot(Z, f, N):
    M = len(f); out = np.empty((N, 7))
    for k in range(N):
        i = rng.integers(0, M, M); A = np.hstack([np.ones((M,1)), Z[i]])
        u = np.linalg.lstsq(A, f[i], rcond=None)[0][1:]; out[k] = u/np.linalg.norm(u)
    return out
b50 = boot(Z[:-2], f30[:-2], 10000); b52 = boot(Z, f30, 10000)
print("boot range 50pts min", b50.min(0), "\n               max", b50.max(0))
print("boot range 52pts min", b52.min(0), "\n               max", b52.max(0))
# v1 bootstrap replicate fields
print([a for a in dir(s30) if not a.startswith('_')])
