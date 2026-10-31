# Item 12: how many ValidityWarnings a 78-fit loop prints under the default warning filters
import numpy as np, active_subspaces as ac
from common import *
h, X, B, out = stomatal(); sp = ac.Uniform(B[:, 0], B[:, 1], names=h)
for k, (hh, F) in out.items():
    for i in range(F.shape[1]):
        ac.active_subspace(X=X[1:], f=F[1:, i], space=sp, estimator="ols", n=1)
