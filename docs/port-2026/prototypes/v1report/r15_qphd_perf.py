# Performance: QPHD at m = 100, M = 10000 (Subsurface_Permeability)
import numpy as np, time, scipy.linalg, tracemalloc
import active_subspaces as ac
from active_subspaces.subspaces import _quadratic_design
from common import *
X, fs, fu = subsurface()
sp = ac.Gaussian.reference(100)
t = time.time(); q = ac.active_subspace(X=X, f=fs, space=sp, estimator="qphd"); tq = time.time() - t
print(f"v1 QPHD (gelsd) m=100 M=10000: {tq:.1f}s, |lambda|[:4]={np.abs(q.eigenvalues[:4]).round(1)}")
A = _quadratic_design(X); print("design", A.shape, f"{A.nbytes/1e9:.2f} GB")
for drv in ("gelsy",):
    t = time.time(); c = scipy.linalg.lstsq(A, fs, lapack_driver=drv, check_finite=False)[0]; print(f"scipy lstsq {drv}: {time.time()-t:.1f}s")
t = time.time(); Q, R = np.linalg.qr(A); c2 = scipy.linalg.solve_triangular(R, Q.T @ fs); print(f"numpy QR + solve: {time.time()-t:.1f}s")
m = 100; g = c2[1:m + 1]; H = np.diag(2 * c2[m + 1:2 * m + 1]); ju, ku = np.triu_indices(m, 1); H[ju, ku] = c2[2 * m + 1:]; H[ku, ju] = c2[2 * m + 1:]
lam = np.linalg.eigvalsh(np.outer(g, g) + H @ H)[::-1]
print("QR eigenvalues match gelsd:", np.allclose(np.abs(lam[:10]), np.abs(q.eigenvalues[:10]), rtol=1e-6))
