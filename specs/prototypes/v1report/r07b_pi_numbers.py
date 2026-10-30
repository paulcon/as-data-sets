# Item 7: reference numbers for a polynomial prediction interval (hand formula on v1 surrogate fields)
import numpy as np, warnings, scipy.stats as st
import active_subspaces as ac
from common import *
warnings.simplefilter("ignore", ac.ValidityWarning)
def pi(s, Yq, level=0.95, kind="prediction"):
    B = np.vander(s.Y[:, 0], 3, increasing=True); M, p = B.shape
    r = s.residuals; s2 = r @ r / (M - p)
    Bq = np.vander(Yq, 3, increasing=True); h = np.einsum('ij,jk,ik->i', Bq, np.linalg.inv(B.T @ B), Bq)
    t = st.t.ppf(0.5 + level / 2, M - p); g = s.evaluate(Yq[:, None])
    half = t * np.sqrt(s2 * ((1 + h) if kind == "prediction" else h))
    return g - half, g + half
X, f = hyshot30(); space = ac.Uniform(HYSHOT_XL, HYSHOT_XU)
sub = ac.active_subspace(X=X, f=f, space=space, estimator="ols", n=1)
s = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=2)
yb = np.abs(sub.W1[:, 0]).sum()
lo, hi = pi(s, np.array([-yb, 0.0, yb]))
print(f"HyShotII phi=0.30 (M=50) 95% PI at y=-{yb:.4f},0,+{yb:.4f}: lower {np.round(lo,4)} upper {np.round(hi,4)}")
lo, hi = pi(s, np.array([0.0]), kind="confidence"); print(f"   95% CI at y=0: [{lo[0]:.4f}, {hi[0]:.4f}]")
# Hydrology grass LH (hand-normalized Gaussian-identity workaround)
Xg, F = hydrology_grass(); Z = Xg.copy(); uni = [j for j in range(19) if j != 16]
Z[:, uni] = 2 * (Xg[:, uni] - YL) / (YU - YL) - 1; Z[:, 16] = (np.log(Xg[:, 16]) + 5.11) / .52
sp = ac.Gaussian(np.zeros(19), np.ones(19))
sub = ac.active_subspace(X=Z, f=F[:, 0], space=sp, estimator="ols", n=1)
s = ac.fit_surrogate(X=Z, f=F[:, 0], subspace=sub, kind="poly", degree=2)
lo, hi = pi(s, np.array([0.0])); lo2, hi2 = pi(s, np.array([0.0]), kind="confidence")
print(f"Hydrology grass LH 95% PI half-width at y=0: {(hi-lo)[0]/2:.4e}; CI half-width {(hi2-lo2)[0]/2:.4e}; in-sample R2 {1-np.sum(s.residuals**2)/np.sum((F[:,0]-F[:,0].mean())**2):.3f}")
