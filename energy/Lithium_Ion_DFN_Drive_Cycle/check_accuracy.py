"""Recompute the accuracy numbers quoted in DATA.md for Lithium_Ion_DFN_Drive_Cycle.

    python check_accuracy.py               # all checks (checks 3-7 need PyBaMM 24.9.0)
    python check_accuracy.py --data-only   # checks 1-2 only: numpy, from the shipped files
    python check_accuracy.py --checks 5,7  # selected checks

Prints, beside the values DATA.md quotes:

1. Binned first-order indices of U_bar (time-averaged voltage, outputs.csv): Var(E[U_bar | bin])
   / Var(U_bar) over 10 and 16 equal-width bins of each input in the sampling coordinates
   (log10 for the log-uniform inputs); the four largest, and the small positive bias of the
   estimator, about (bins - 1) / M for an input without effect.
2. Polynomial chaos emulation of U_bar on the 650 runs: orthonormal bases (Legendre for the
   uniform inputs; for the seven log-uniform inputs either Legendre in log10 coordinates or
   polynomials orthonormal under the log-uniform density in the physical variable, built by a
   discretized Stieltjes procedure), fitted by least squares, total-order indices from the
   coefficients. Four rows: full degree 2 (325 terms) and additive degree 3 (73 terms), each
   in log10 and in physical coordinates; in-sample R^2 and the total indices of sigma_pos,
   Ds_pos and L_pos.
3. Reproduction: rows 0, 300 and 649 of inputs.csv rerun with the shipped settings; largest
   |U(t) - voltage.csv| (the file has 9 decimals).
4. Solver tolerance: rows 0, 1, 2 and 62 rerun with rtol = atol = 1e-9 against voltage.csv
   (rtol = atol = 1e-6): largest change of U(t), of U_bar and of U(600), and as a fraction of the
   standard deviation of U_bar and U(600) over the 650 runs.
5. Solver choice: PyBaMM's IDAKLU solver (default settings) against the CasADi solution at the
   box centre (dfn_us06_driver.nominal()).
6. Log-uniform bounds read as natural logarithms (the first possible cause in DATA.md): U_bar
   at the box centre with sigma_pos = e^(log10 5.2e-6) = 5.07e-3 S/m, and row 503 (the lowest
   U_bar) with its seven log-uniform inputs remapped x -> e^(log10 x): U_bar and min U(t).
7. One-at-a-time effects at the box centre (generate_data.oat): sigma_pos, Ds_pos, L_pos and
   R_pos each at its lower and upper bound.

Checks 1-2 need numpy only and read the files next to this script. Checks 3-7 need PyBaMM
24.9.0 with casadi 3.6.7 (pip install "pybamm==24.9.0" "casadi==3.6.7", Python 3.9-3.12), about
5-10 minutes on one core; without PyBaMM the script stops with an error that says what to
install (use --data-only to run checks 1-2 alone). Deterministic; writes nothing.
"""
import argparse
import os
import sys
import time
import warnings
from itertools import combinations

import numpy as np
from numpy.polynomial import legendre as L

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dfn_us06_driver as dfn  # noqa: E402
import generate_data as gd  # noqa: E402

LOG = dfn.LOG
Z_LO = np.where(LOG, np.log10(dfn.LOWER), dfn.LOWER)
Z_HI = np.where(LOG, np.log10(dfn.UPPER), dfn.UPPER)
REPORT = ["sigma_pos", "Ds_pos", "L_pos"]


def load():
    X = np.loadtxt(os.path.join(HERE, "inputs.csv"), delimiter=",")
    out = np.loadtxt(os.path.join(HERE, "outputs.csv"), delimiter=",")
    V = np.loadtxt(os.path.join(HERE, "voltage.csv"), delimiter=",")
    return X, out, V


def to_z(X):
    return np.where(LOG, np.log10(np.where(LOG, X, 1.0)), X)


# --- 1. binned first-order indices -------------------------------------------------------------
def binned_indices(Z, f, nbins):
    S = np.empty(Z.shape[1])
    for j in range(Z.shape[1]):
        edges = np.linspace(Z_LO[j], Z_HI[j], nbins + 1)
        b = np.clip(np.digitize(Z[:, j], edges) - 1, 0, nbins - 1)
        n = np.bincount(b, minlength=nbins)
        mu = np.bincount(b, weights=f, minlength=nbins) / np.maximum(n, 1)
        S[j] = np.sum(n * (mu - f.mean()) ** 2) / len(f) / f.var()
    return S


# --- 2. polynomial chaos emulation ---------------------------------------------------------------
def legendre_orthonormal(u, deg):
    """Orthonormal Legendre polynomials (uniform density on [-1, 1]) at u, degrees 0..deg."""
    return L.legvander(u, deg) * np.sqrt(2 * np.arange(deg + 1) + 1)


def loguniform_orthonormal(x, lo, hi, deg, nquad=4000):
    """Polynomials in the physical variable x, orthonormal under the log-uniform density on
    [lo, hi], degrees 0..deg: three-term recurrence from a discretized Stieltjes procedure
    (Gauss-Legendre quadrature in log10 x), evaluated at x scaled to s = x / hi."""
    t, w = L.leggauss(nquad)
    s_q = 10.0 ** (np.log10(lo) + (t + 1) / 2 * (np.log10(hi) - np.log10(lo))) / hi
    w = w / w.sum()
    s = x / hi
    a, b = np.zeros(deg), np.zeros(deg)
    p_prev_q, p_q = np.zeros_like(s_q), np.ones_like(s_q)
    norms = [1.0]
    P = [np.ones_like(s)]
    p_prev, p = np.zeros_like(s), np.ones_like(s)
    for k in range(deg):
        nk = np.sum(w * p_q ** 2)
        a[k] = np.sum(w * s_q * p_q ** 2) / nk
        b[k] = nk / np.sum(w * p_prev_q ** 2) if k > 0 else 0.0
        p_prev_q, p_q = p_q, (s_q - a[k]) * p_q - b[k] * p_prev_q
        p_prev, p = p, (s - a[k]) * p - b[k] * p_prev
        norms.append(np.sum(w * p_q ** 2))
        P.append(p)
    return np.column_stack(P) / np.sqrt(norms)


def univariate_bases(X, deg, physical):
    """List over inputs of (M, deg + 1) orthonormal univariate bases."""
    Z = to_z(X)
    out = []
    for j in range(X.shape[1]):
        if LOG[j] and physical:
            out.append(loguniform_orthonormal(X[:, j], dfn.LOWER[j], dfn.UPPER[j], deg))
        else:
            u = 2 * (Z[:, j] - Z_LO[j]) / (Z_HI[j] - Z_LO[j]) - 1
            out.append(legendre_orthonormal(u, deg))
    return out


def multi_indices(m, deg, additive):
    idx = [np.zeros(m, int)]
    if additive:
        for j in range(m):
            for d in range(1, deg + 1):
                a = np.zeros(m, int); a[j] = d; idx.append(a)
        return np.array(idx)
    assert deg == 2
    for j in range(m):
        a = np.zeros(m, int); a[j] = 1; idx.append(a)
    for j in range(m):
        a = np.zeros(m, int); a[j] = 2; idx.append(a)
    for i, j in combinations(range(m), 2):
        a = np.zeros(m, int); a[i] = a[j] = 1; idx.append(a)
    return np.array(idx)


def pc_emulation(X, f, deg, additive, physical):
    B = univariate_bases(X, deg, physical)
    alphas = multi_indices(X.shape[1], deg, additive)
    A = np.ones((len(f), len(alphas)))
    for j in range(X.shape[1]):
        A *= B[j][:, alphas[:, j]]
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    r2 = 1 - np.var(f - A @ c) / np.var(f)
    var = np.sum(c[1:] ** 2)
    total = np.array([np.sum(c[alphas[:, j] > 0] ** 2) / var for j in range(X.shape[1])])
    return len(alphas), r2, total


# --- 3-6. PyBaMM runs ----------------------------------------------------------------------------
def run(x, **kw):
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = dfn.voltage(x, **kw)
    return v, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-only", action="store_true", help="checks 1-2 only (numpy, no PyBaMM)")
    ap.add_argument("--checks", default="1,2,3,4,5,6,7", help="comma-separated checks to run")
    args = ap.parse_args()
    run_ = set(int(c) for c in args.checks.split(","))
    if args.data_only:
        run_ &= {1, 2}
    t_start = time.time()
    if run_ - {1, 2}:
        gd.require_pybamm()
    X, out, V = load()
    f, fend = out[:, 1], out[:, 2]
    Z = to_z(X)
    M = len(f)
    names = dfn.NAMES

    if 1 in run_:
        print("1. Binned first-order indices of U_bar (equal-width bins in the sampling coordinates)")
        for nb in (10, 16):
            S = binned_indices(Z, f, nb)
            top = np.argsort(-S)[:4]
            print(f"   {nb} bins: " + ", ".join(f"{names[j]} {S[j]:.2f}" for j in top)
                  + f"; bias for an input without effect about (bins - 1)/M = {(nb - 1) / M:.3f}")

    if 2 in run_:
        print("2. PC emulation of U_bar on the 650 runs (least squares, orthonormal bases)")
        print("   basis for the log-uniform inputs          terms  R^2 (in-sample)  total index " + " / ".join(REPORT))
        for deg, additive, label in ((2, False, "full degree 2"), (3, True, "additive degree 3")):
            for physical in (False, True):
                p, r2, T = pc_emulation(X, f, deg, additive, physical)
                coord = "physical" if physical else "log10"
                print(f"   {coord + ' coordinates, ' + label:42s} {p:5d}  {r2:15.2f}  "
                      + " / ".join(f"{T[names.index(n)]:.2f}" for n in REPORT))

    if not run_ - {1, 2}:
        print(f"done, {time.time() - t_start:.0f} s")
        return

    import pybamm
    print(f"   PyBaMM {pybamm.__version__}, casadi {__import__('casadi').__version__}, numpy {np.__version__}")
    if 3 in run_:
        print("3. Reproduction of voltage.csv (shipped settings)")
        for r in (0, 300, 649):
            v, s = run(X[r])
            print(f"   row {r}: max |dU| = {np.max(np.abs(v - V[r])):.1e} V ({s:.1f} s)", flush=True)

    if 4 in run_:
        print("4. Solver tolerance: rtol = atol = 1e-9 against the shipped 1e-6 runs")
        sd_mean, sd_end = f.std(), fend.std()
        for r in (0, 1, 2, 62):
            v, s = run(X[r], rtol=1e-9, atol=1e-9)
            vm, ve = gd.qois(v)
            dm, de = abs(vm - f[r]), abs(ve - fend[r])
            print(f"   row {r}: max |dU(t)| {np.max(np.abs(v - V[r])):.1e} V; |dU_bar| {dm:.1e} V "
                  f"({dm / sd_mean:.1e} of its sd {sd_mean:.3f} V); |dU(600)| {de:.1e} V "
                  f"({de / sd_end:.1e} of its sd {sd_end:.3f} V) ({s:.1f} s)", flush=True)

    x0 = dfn.nominal()
    if 5 in run_ or 6 in run_:
        v_c, _ = run(x0)
    if 5 in run_:
        print("5. Solver choice at the box centre: IDAKLU (default settings) against CasADi (shipped settings)")
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = pybamm.Simulation(pybamm.lithium_ion.DFN(), parameter_values=dfn.parameter_values(x0),
                                    solver=pybamm.IDAKLUSolver())
            sol = sim.solve(dfn.T_EVAL)
            v_i = np.asarray(sol["Voltage [V]"](dfn.T_EVAL), dtype=float)
        print(f"   U_bar (CasADi) {gd.qois(v_c)[0]:.4f} V; max |U_IDAKLU - U_CasADi| over t = 0..600 s = "
              f"{np.max(np.abs(v_i - v_c)):.1e} V ({time.time() - t0:.1f} s)", flush=True)

    if 6 in run_:
        print("6. Log-uniform bounds read as natural logarithms")
        js = names.index("sigma_pos")
        x = x0.copy()
        x[js] = np.exp(np.log10(dfn.LOWER[js]))
        v, _ = run(x)
        print(f"   centre: U_bar {gd.qois(v_c)[0]:.4f} V; with sigma_pos = {x[js]:.3g} S/m: U_bar {gd.qois(v)[0]:.4f} V "
              f"(change {1000 * (gd.qois(v)[0] - gd.qois(v_c)[0]):+.1f} mV)")
        r = int(np.argmin(f))
        x = X[r].copy()
        x[LOG] = np.exp(np.log10(x[LOG]))
        v, _ = run(x)
        print(f"   row {r} (lowest U_bar, {f[r]:.4f} V), log-uniform inputs remapped x -> e^(log10 x): "
              f"U_bar {gd.qois(v)[0]:.4f} V, min U(t) {v.min():.3f} V", flush=True)

    if 7 in run_:
        print("7. One-at-a-time effects at the box centre (generate_data.oat)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gd.oat()
    print(f"done, {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
