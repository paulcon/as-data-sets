"""Variance-based (Sobol') sensitivity indices with the estimators of Puy et al. (2022), Supplementary Materials.

First-order indices by the Saltelli et al. (2010) estimator (Supplementary Materials Eq. 14, p. 15), total-order
indices by the Jansen (1999) estimator (Eq. 15, p. 15), and second-order indices for chosen pairs (i, j) from the
closed index V[E(y | x_i, x_j)] estimated like Eq. 14, with A_B^(ij) (A with columns i and j from B) in place of
A_B^(i): S_ij = mean(f(B) (f(A_B^(ij)) - f(A))) / V(y) - S_i - S_j. V(y) is the variance of the outputs at the
rows of A and B. Written from the equations; imports numpy and scipy.
"""
import numpy as np
from scipy.stats import qmc


def base_matrices(N, k, seed):
    """A and B, each (N, k), from one scrambled Sobol' sequence in dimension 2k (A the left k columns, B the right)."""
    U = qmc.Sobol(d=2 * k, scramble=True, seed=seed).random(N)
    return U[:, :k], U[:, k:]


def mixed(A, B, cols):
    """A with the column(s) cols taken from B: A_B^(i) for cols = i, A_B^(ij) for cols = [i, j]."""
    C = A.copy()
    C[:, cols] = B[:, cols]
    return C


def evaluate(fun, A, B, pairs=()):
    """Model outputs at A, B, every A_B^(i) and A_B^(ij) for the given pairs: dict of arrays."""
    k = A.shape[1]
    out = {'A': fun(A), 'B': fun(B),
           'AB': np.column_stack([fun(mixed(A, B, i)) for i in range(k)])}
    if len(pairs):
        out['ABij'] = np.column_stack([fun(mixed(A, B, list(p))) for p in pairs])
    return out


def indices(y, pairs=(), rows=None):
    """First-order S, total-order T and second-order S_ij for the pairs (i, j) given to `evaluate`, in that order.

    y is the dict from `evaluate`; rows (optional) selects or resamples the rows, for the bootstrap."""
    fA, fB, fAB = y['A'], y['B'], y['AB']
    fABij = y.get('ABij', np.empty((len(fA), 0)))
    if rows is not None:
        fA, fB, fAB, fABij = fA[rows], fB[rows], fAB[rows], fABij[rows]
    V = np.var(np.r_[fA, fB])
    S = np.mean(fB[:, None] * (fAB - fA[:, None]), axis=0) / V              # Eq. 14
    T = 0.5 * np.mean((fA[:, None] - fAB) ** 2, axis=0) / V                  # Eq. 15
    Vc = np.mean(fB[:, None] * (fABij - fA[:, None]), axis=0) / V            # closed index of (x_i, x_j)
    S2 = np.array([Vc[p] - S[i] - S[j] for p, (i, j) in enumerate(pairs)])
    return S, T, S2


def bootstrap(y, pairs=(), R=1000, seed=0):
    """Indices and their bootstrap standard deviations over R resamples of the rows (rows resampled with replacement)."""
    rng = np.random.default_rng(seed)
    N = len(y['A'])
    est = indices(y, pairs)
    reps = [indices(y, pairs, rows=rng.integers(0, N, N)) for _ in range(R)]
    sd = tuple(np.std(np.array([r[q] for r in reps]), axis=0, ddof=1) for q in range(3))
    return est, sd


def g_function(U, a):
    """Sobol' G function on [0, 1]^k, prod_i (|4 u_i - 2| + a_i) / (1 + a_i)."""
    return np.prod((np.abs(4 * U - 2) + a) / (1 + a), axis=1)


def g_function_indices(a, pairs=()):
    """Exact S, T and S_ij of the G function: V_i = 1 / (3 (1 + a_i)^2), V = prod(1 + V_i) - 1."""
    a = np.asarray(a, dtype=float)
    Vi = 1.0 / (3.0 * (1.0 + a) ** 2)
    V = np.prod(1.0 + Vi) - 1.0
    S = Vi / V
    T = Vi * np.array([np.prod(np.delete(1.0 + Vi, i)) for i in range(len(a))]) / V
    S2 = np.array([Vi[i] * Vi[j] / V for i, j in pairs])
    return S, T, S2
