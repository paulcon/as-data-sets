# Item 7: prediction / confidence intervals for polynomial surrogates (Hydrology, HyShotII)
import numpy as np, warnings, dataclasses
import active_subspaces as ac
from common import *
warnings.simplefilter("ignore", ac.ValidityWarning)
X, f = hyshot30()
space = ac.Uniform(HYSHOT_XL, HYSHOT_XU)
sub = ac.active_subspace(X=X, f=f, space=space, estimator="ols", n=1)
s = ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=2)
print("PolynomialSurrogate fields:", [x.name for x in dataclasses.fields(s)])
print("public attrs:", [a for a in dir(s) if not a.startswith("_")])
print("surrogates module public:", [a for a in ac.surrogates.__all__])
in_sample = 1 - np.sum(s.residuals ** 2) / np.sum((f - f.mean()) ** 2)
print(f"HyShotII phi=0.30 (M={len(f)}): in-sample R2={in_sample:.4f}  loo_r_squared={s.loo_r_squared:.4f}")
