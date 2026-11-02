"""Lorenz-96 initial-state and forcing inference: forward model, data misfit and its exact gradient.

The model of Schuster, Constantine and Sullivan (2017), Sec. 5.2: the one-level Lorenz-96 system in
K = 36 dimensions,

    dX_k/dt = (X_{k+1} - X_{k-2}) X_{k-1} - X_k + F,   k = 1..K, indices mod K,

with the 37 inference variables x = (X_1(0), ..., X_36(0), F). The forward solve s(x) is the
trajectory at the times t_i = i dt, dt = 0.01, from t = 0. The data misfit (negative Gaussian
log-likelihood up to a constant, noise variance sigma^2 = 0.1) over the window [0, T] is

    Phi_T(x) = 1/(2 sigma^2) * sum_{t_i <= T} sum_k (X_k(t_i; x) - d_{ik})^2.

Time stepping is classical fourth-order Runge-Kutta with the fixed step dt = 0.01 (the paper used
an unspecified BDF method, see DATA.md). The gradient is the exact derivative of this discrete map,
computed by the discrete tangent-linear (forward-sensitivity) recursion of the Runge-Kutta scheme.
Every function is vectorized over a batch of samples. Imports only numpy.
"""
import numpy as np

K = 36                 # number of state variables
M_INPUTS = K + 1       # inference variables: X_1(0) .. X_36(0), F
DT = 0.01              # time step and observation spacing
T_END = 10.0           # end of the paper's observation window
N_STEPS = 1000         # steps of DT to T_END
NOISE_VAR = 0.1        # observation-noise variance sigma^2
PRIOR_VAR = 4.0        # prior variance of every inference variable (prior mean 0)
F_TRUE = 8.0           # forcing of the synthetic truth
NAMES = tuple([f"X{k}_0" for k in range(1, K + 1)] + ["F"])


def rhs(X, F):
    """Lorenz-96 right-hand side for states X (..., K) and forcing F (...) or scalar."""
    F = np.asarray(F)
    return (np.roll(X, -1, -1) - np.roll(X, 2, -1)) * np.roll(X, 1, -1) - X + F[..., None]


def _jvp(X, dX, dF):
    """Tangent of rhs at X: J_X dX + dF, with dX (..., K, p) and dF (..., p)."""
    Xs = X[..., :, None]
    xp1, xm1, xm2 = np.roll(Xs, -1, -2), np.roll(Xs, 1, -2), np.roll(Xs, 2, -2)
    return ((np.roll(dX, -1, -2) - np.roll(dX, 2, -2)) * xm1 + (xp1 - xm2) * np.roll(dX, 1, -2)
            - dX + dF[..., None, :])


def _step(X, F):
    h = DT
    k1 = rhs(X, F)
    k2 = rhs(X + .5 * h * k1, F)
    k3 = rhs(X + .5 * h * k2, F)
    k4 = rhs(X + h * k3, F)
    return X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def trajectory(x, n_steps=N_STEPS):
    """States at t_i = i DT, i = 0..n_steps, for inputs x (M, 37) or (37,): array (M, n_steps+1, K)."""
    x = np.atleast_2d(np.asarray(x))
    X, F = x[:, :K].copy(), x[:, K].copy()
    out = np.empty((x.shape[0], n_steps + 1, K), dtype=X.dtype)
    out[:, 0] = X
    for i in range(n_steps):
        X = _step(X, F)
        out[:, i + 1] = X
    return out


def synthetic_data(x_true, rng):
    """Data d = s(x_true) + N(0, NOISE_VAR) noise at all N_STEPS+1 times and K components."""
    s = trajectory(x_true)[0]
    return s + np.sqrt(NOISE_VAR) * rng.standard_normal(s.shape)


def misfit_and_grad(x, data, windows, grad=True):
    """Misfits Phi_T and their exact gradients for each T in `windows`.

    x       : (M, 37) inputs, physical units (initial state and F)
    data    : (N_STEPS+1, K) observations at t_i = i DT
    windows : increasing end times T (multiples of DT, <= T_END)
    Returns phi (M, len(windows)) and, if grad, G (M, len(windows), 37), dPhi_T/dx.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    M = x.shape[0]
    ends = [int(round(T / DT)) for T in windows]
    if any(e < 0 or e > data.shape[0] - 1 for e in ends) or ends != sorted(ends):
        raise ValueError("windows must be increasing multiples of DT within the data")
    p = M_INPUTS
    X, F = x[:, :K].copy(), x[:, K].copy()
    phi = np.zeros((M, len(ends)))
    G = np.zeros((M, len(ends), p)) if grad else None
    if grad:
        S = np.zeros((M, K, p))                    # dX/dx
        S[:, np.arange(K), np.arange(K)] = 1.0
        eF = np.zeros((M, p))
        eF[:, K] = 1.0                             # dF/dx
    acc_phi = np.zeros(M)
    acc_g = np.zeros((M, p)) if grad else None
    w = 0
    h = DT
    for i in range(ends[-1] + 1):
        r = X - data[i]
        acc_phi += 0.5 / NOISE_VAR * np.sum(r * r, axis=1)
        if grad:
            acc_g += np.einsum("mk,mkp->mp", r, S) / NOISE_VAR
        while w < len(ends) and ends[w] == i:
            phi[:, w] = acc_phi
            if grad:
                G[:, w] = acc_g
            w += 1
        if i == ends[-1]:
            break
        k1 = rhs(X, F)
        Y2 = X + .5 * h * k1
        k2 = rhs(Y2, F)
        Y3 = X + .5 * h * k2
        k3 = rhs(Y3, F)
        Y4 = X + h * k3
        k4 = rhs(Y4, F)
        if grad:
            d1 = _jvp(X, S, eF)
            d2 = _jvp(Y2, S + .5 * h * d1, eF)
            d3 = _jvp(Y3, S + .5 * h * d2, eF)
            d4 = _jvp(Y4, S + h * d3, eF)
            S = S + h / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
        X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if not np.all(np.isfinite(phi)) or (grad and not np.all(np.isfinite(G))):
        raise FloatingPointError("non-finite misfit or gradient (time stepping unstable)")
    return (phi, G) if grad else phi


def misfit(x, data, windows):
    """Misfits only, (M, len(windows))."""
    return misfit_and_grad(x, data, windows, grad=False)


def misfit_complex_step(x, data, T, j, h=1e-30):
    """Complex-step derivative dPhi_T/dx_j for inputs x (M, 37): an independent gradient check."""
    x = np.atleast_2d(np.asarray(x, dtype=complex))
    x[:, j] += 1j * h
    n = int(round(T / DT))
    s = trajectory(x, n)
    r = s - data[: n + 1]
    return (0.5 / NOISE_VAR * np.sum(r * r, axis=(1, 2))).imag / h


def leading_lyapunov(x, n_steps=N_STEPS, renorm=10, rng=None):
    """Leading Lyapunov exponent along the trajectory from one input x (37,), by the discrete
    tangent-linear map of the Runge-Kutta scheme with renormalization every `renorm` steps.
    The tangent starts from a random unit vector (rng: numpy Generator or seed)."""
    rng = np.random.default_rng(rng)
    X, F = np.asarray(x[:K], dtype=float)[None].copy(), np.array([float(x[K])])
    v = rng.standard_normal((1, K, 1))
    v /= np.linalg.norm(v)
    z = np.zeros((1, 1))
    h, log_growth = DT, 0.0
    for i in range(n_steps):
        k1 = rhs(X, F); Y2 = X + .5 * h * k1
        k2 = rhs(Y2, F); Y3 = X + .5 * h * k2
        k3 = rhs(Y3, F); Y4 = X + h * k3
        k4 = rhs(Y4, F)
        d1 = _jvp(X, v, z); d2 = _jvp(Y2, v + .5 * h * d1, z)
        d3 = _jvp(Y3, v + .5 * h * d2, z); d4 = _jvp(Y4, v + h * d3, z)
        v = v + h / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
        X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (i + 1) % renorm == 0:
            nv = np.linalg.norm(v)
            log_growth += np.log(nv)
            v /= nv
    return log_growth / (n_steps * DT)
