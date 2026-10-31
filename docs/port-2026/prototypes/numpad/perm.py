import sys, time; sys.path.insert(0,'.')
import numpy as np, scipy.sparse.linalg as spla
from pde_model import PDESolver
S = PDESolver('short', datadir='/Users/paulconstantine/PycharmProjects/as-data-sets/Elliptic_PDE')
A = S.stiffness(S.coeff(np.random.default_rng(0).standard_normal(100))); b = np.ones(S.N)
for spec in ['COLAMD','MMD_AT_PLUS_A','MMD_ATA','NATURAL']:
    t=time.perf_counter()
    for _ in range(20): lu = spla.splu(A, permc_spec=spec); lu.solve(b)
    print(spec, f'{1e3*(time.perf_counter()-t)/20:.1f} ms  nnz(L+U)={lu.L.nnz+lu.U.nnz}')
t=time.perf_counter()
for _ in range(20): lu = spla.splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0., options=dict(SymmetricMode=True)); lu.solve(b)
print('MMD_AT_PLUS_A+symmetric', f'{1e3*(time.perf_counter()-t)/20:.1f} ms  nnz(L+U)={lu.L.nnz+lu.U.nnz}')
t=time.perf_counter()
for _ in range(20): spla.spsolve(A, b)
print('spsolve', f'{1e3*(time.perf_counter()-t)/20:.1f} ms')
t=time.perf_counter()
for _ in range(20): spla.cg(A, b, rtol=1e-12)
print('cg (no precond)', f'{1e3*(time.perf_counter()-t)/20:.1f} ms')
