"""Metropolis-Hastings on all parameters, and on the active variables.

The samplers of Constantine, Kent & Bui-Thanh, "Accelerating Markov chain
Monte Carlo with active subspaces", SIAM J. Sci. Comput. 38(5):A2779-A2805,
2016, doi:10.1137/15M1042127, for a standard Gaussian prior on x:

- `random_walk_metropolis`: random-walk Metropolis-Hastings on x, the
  "vanilla" chain;
- `ActiveSubspaceMCMC.sample_active_variables`: random-walk
  Metropolis-Hastings on the active variables y = W1^T x with the Monte Carlo
  misfit estimate ghat(y). By default ghat is recomputed at the current state
  and at the proposal at every step, as the 2016 code that produced the
  paper's results did (an approximate, noisy Metropolis-Hastings);
  recompute=False keeps ghat with the state, the paper's Algorithm 1 as printed;
- `ActiveSubspaceMCMC.sample_original_variables`: the reconstruction (4.2),
  x = W1 y + W2 z with z drawn from the prior.

Acceptance tests are done in log space. Every random draw comes from the
`numpy.random.Generator` passed in, one step at a time and in a fixed order,
so a shorter run with the same seed reproduces the start of a longer one.
numpy only.
"""
import numpy as np


def random_walk_metropolis(misfit, x0, N, prop_std, rng, progress=None):
    """Random-walk Metropolis-Hastings for exp(-f(x)) times the N(0, I) prior.

    Parameters
    ----------
    misfit : callable
        x (m,) -> f(x), the data misfit (negative log-likelihood).
    x0 : array, shape (m,)
        Initial state.
    N : int
        Number of steps.
    prop_std : float
        Standard deviation of the proposal N(x_k, prop_std^2 I).
    rng : numpy.random.Generator
        Draws, per step: m standard normals for the proposal, then one uniform.
    progress : callable, optional
        Called as progress(k) every 5000 steps.

    Returns
    -------
    X : array, shape (N, m)
        States: X[0] = x0, and X[k] is the state at the start of step k.
    F : array, shape (N,)
        f(X[k]), cached: one misfit evaluation per step.
    accept : bool array, shape (N,)
        Whether the proposal made at step k was accepted, so that
        X[k+1] != X[k] exactly when accept[k] (for k < N-1).
    """
    x = np.array(x0, dtype=float)
    fx = misfit(x)
    X = np.empty((N, x.size))
    F = np.empty(N)
    accept = np.zeros(N, dtype=bool)
    for k in range(N):
        X[k], F[k] = x, fx
        xp = x + prop_std * rng.standard_normal(x.size)
        fp = misfit(xp)
        if np.log(rng.random()) <= (fx + 0.5 * x @ x) - (fp + 0.5 * xp @ xp):
            x, fx = xp, fp
            accept[k] = True
        if progress is not None and (k + 1) % 5000 == 0:
            progress(k + 1)
    return X, F, accept


def _step(prop, e):
    """Proposal increment: prop * e for a scalar standard deviation, prop @ e for a matrix."""
    return prop @ e if np.ndim(prop) == 2 else prop * e


class ActiveSubspaceMCMC:
    """Metropolis-Hastings on the active variables, for a misfit f(x) and a N(0, I) prior on x.

    Parameters
    ----------
    misfit : callable
        x (m,) -> f(x), the data misfit (negative log-likelihood).
    W1 : array, shape (m, n)
        Basis of the active subspace.
    W2 : array, shape (m, m-n)
        Basis of the inactive subspace; [W1 W2] is orthogonal.
    z_rule : (Z, w), optional
        A fixed quadrature rule for the conditional mean: nodes Z, shape
        (M, m-n), and weights w, shape (M,), for the N(0, I_{m-n}) density.
        Default: Monte Carlo with fresh draws, as in the paper's (2.16).
    evaluate : callable, optional
        X (k, m) -> f at each row, shape (k,). Default: misfit row by row.
        Pass a parallel map here; results must not depend on it.
    """

    def __init__(self, misfit, W1, W2, z_rule=None, evaluate=None):
        self.misfit = misfit
        self.W1 = np.asarray(W1, dtype=float)
        self.W2 = np.asarray(W2, dtype=float)
        self.m, self.n = self.W1.shape
        self.z_rule = z_rule
        self.evaluate = evaluate or (lambda X: np.array([misfit(x) for x in X]))

    def _points(self, y, Z):
        return self.W1 @ np.asarray(y, dtype=float) + Z @ self.W2.T

    def ghat(self, y, M, rng):
        """Estimate of the conditional mean of f given y: M misfit evaluations.

        Monte Carlo (2.16): draws M independent z ~ N(0, I_{m-n}) (an
        (M, m-n) block from rng) and averages f(W1 y + W2 z_i). With z_rule,
        the weighted sum over its nodes instead (M and rng are not used).
        """
        if self.z_rule is None:
            f = self.evaluate(self._points(y, rng.standard_normal((M, self.m - self.n))))
            return float(f.mean())
        Z, w = self.z_rule
        return float(w @ self.evaluate(self._points(y, Z)))

    def sample_active_variables(self, y0, N, M, prop, rng, recompute=True, progress=None):
        """Random-walk Metropolis-Hastings on y with the misfit estimate ghat.

        recompute=True (the default; the 2016 code, which produced the paper's
        results): at every step ghat is computed afresh, with new draws of z,
        at both the current state and the proposal, 2M misfit evaluations per
        step. Because ghat is noisy, this is an approximate (noisy)
        Metropolis-Hastings, not an exact sampler of a fixed density. Per step,
        rng gives n standard normals for the proposal, an (M, m-n) block for
        the current state, an (M, m-n) block for the proposal, then one uniform.

        recompute=False: the paper's Algorithm 1 as printed. ghat(y0) is
        computed once and ghat is kept with the state that it was computed
        for, so each step costs M evaluations (for the proposal). Per step,
        rng gives n standard normals, an (M, m-n) block for the proposal, then
        one uniform.

        With z_rule, ghat is deterministic and both settings give the same
        chain; the cached path is used.

        prop is the proposal: a scalar standard deviation (y' = y + prop * e)
        or an (n, n) matrix L (y' = y + L e), with e ~ N(0, I_n); the proposal
        covariance is then L L^T.

        Returns
        -------
        Y : array, shape (N, n)
            States of the active variables, Y[0] = y0.
        ghat : array, shape (N,)
            ghat at Y[k] as used in the acceptance test of step k.
        accept : bool array, shape (N,)
            Whether the proposal made at step k was accepted.
        """
        y = np.array(y0, dtype=float)
        Y = np.empty((N, self.n))
        G = np.empty(N)
        accept = np.zeros(N, dtype=bool)
        noisy = recompute and self.z_rule is None
        if not noisy:
            gy = self.ghat(y, M, rng)
        for k in range(N):
            yp = y + _step(prop, rng.standard_normal(self.n))
            if noisy:
                Zc = rng.standard_normal((M, self.m - self.n))
                Zp = rng.standard_normal((M, self.m - self.n))
                f = self.evaluate(np.vstack([self._points(y, Zc), self._points(yp, Zp)]))
                gy, gp = float(f[:M].mean()), float(f[M:].mean())
            else:
                gp = self.ghat(yp, M, rng)
            Y[k], G[k] = y, gy
            if np.log(rng.random()) <= (gy + 0.5 * y @ y) - (gp + 0.5 * yp @ yp):
                y, gy = yp, gp
                accept[k] = True
            if progress is not None and (k + 1) % 500 == 0:
                progress(k + 1)
        return Y, G, accept

    def sample_original_variables(self, Y, P, rng):
        """Reconstruction (4.2): P samples x = W1 y_k + W2 z_{k,l} per state y_k.

        Returns X, shape (N*P, m), with rows k*P ... k*P + P-1 built from Y[k].
        """
        Y = np.asarray(Y, dtype=float)
        Z = rng.standard_normal((Y.shape[0] * P, self.m - self.n))
        return np.repeat(Y @ self.W1.T, P, axis=0) + Z @ self.W2.T


def batch_means_se(x, theta=2/3):
    """Consistent batch-means standard error of the mean of a chain.

    Flegal, Haran & Jones (2008), Stat. Sci. 23(2), with batch size
    b = floor(N^theta) (theta = 2/3, as in the paper's section 5.6). x has
    shape (N,) or (N, k); the result has shape () or (k,).
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    b = int(np.floor(N**theta))
    a = N // b
    means = x[:a * b].reshape((a, b) + x.shape[1:]).mean(axis=1)
    return np.sqrt(b * means.var(axis=0, ddof=1) / (a * b))


def _autocov(x):
    """Autocovariance at lags 0..N-1 of each column of x, shape (N, k), via FFT."""
    N = x.shape[0]
    xc = x - x.mean(axis=0)
    f = np.fft.rfft(xc, n=2 * N, axis=0)
    return np.fft.irfft(f * np.conj(f), n=2 * N, axis=0)[:N] / N


def chain_diagnostics(chains):
    """Effective sample size, autocorrelation time and split R-hat of the mean.

    chains : array, shape (m, N) or (m, N, k): m chains of N steps (burn-in
    removed) of k scalar quantities.

    The autocorrelations are combined across chains as in Stan (Gelman et al.,
    Bayesian Data Analysis, 3rd ed., section 11.5) and summed with Geyer's
    initial monotone sequence estimator. Returns a dict with arrays of shape
    () or (k,): 'mean', 'se' (= sd / sqrt(ess)), 'ess', 'tau' (= m N / ess,
    in steps of one chain) and 'rhat' (split R-hat: each chain cut in halves;
    nan for a single chain shorter than 4 steps).
    """
    c = np.asarray(chains, dtype=float)
    squeeze = c.ndim == 2
    if squeeze:
        c = c[..., None]
    m, N, k = c.shape
    h = N // 2
    s = np.concatenate([c[:, :h], c[:, h:2 * h]], axis=0)
    W_s = s.var(axis=1, ddof=1).mean(axis=0)
    B_s = h * s.mean(axis=1).var(axis=0, ddof=1)
    rhat = np.sqrt(((h - 1) / h * W_s + B_s / h) / W_s)
    acov = np.stack([_autocov(ci) for ci in c])            # (m, N, k)
    W = c.var(axis=1, ddof=1).mean(axis=0)
    var_plus = (N - 1) / N * W
    if m > 1:
        var_plus = var_plus + c.mean(axis=1).var(axis=0, ddof=1)
    rho = 1 - (W - acov.mean(axis=0)) / var_plus             # (N, k)
    tau = np.empty(k)
    for j in range(k):
        pairs = rho[:N - N % 2, j].reshape(-1, 2).sum(axis=1)   # rho_2t + rho_2t+1
        n_pos = int(np.argmax(pairs <= 0)) if np.any(pairs <= 0) else len(pairs)
        g = np.minimum.accumulate(pairs[:n_pos])
        tau[j] = max(-1 + 2 * g.sum(), 1 / np.log10(m * N))
    ess = m * N / tau
    out = {'mean': c.mean(axis=(0, 1)), 'se': np.sqrt(var_plus / ess), 'ess': ess, 'tau': tau,
           'rhat': rhat}
    return {key: v[0] for key, v in out.items()} if squeeze else out
