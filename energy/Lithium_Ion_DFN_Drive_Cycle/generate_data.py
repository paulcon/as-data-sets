"""Generate the stored runs of the PyBaMM DFN model under the scaled US06 drive cycle.

    python generate_data.py --n 650 --budget 1200            # writes inputs.csv, outputs.csv, voltage.csv
    python generate_data.py --n 650 --budget 1200 --resume   # continue from the rows already in --out
    python generate_data.py --rows 0,300,649 --out /tmp/dfn --check   # rerun some rows and compare
    python generate_data.py --oat                              # one-at-a-time effects quoted in DATA.md

Needs PyBaMM 24.9.0 (casadi 3.6.7), numpy and the model module dfn_us06.py next to this
script. The design is i.i.d. Monte Carlo (as in the paper): row i is the i-th draw of
numpy.random.default_rng(SEED).random((N_MAX, 24)), mapped to Table A.2 (uniform inputs
linearly, log-uniform inputs linearly in log10). Rows are run in order, so the first n rows
of any run are the same. `--budget` stops after the given wall time (s); the files then hold
the rows completed so far. `--resume` reads the files already in `--out`, checks that their inputs
are the first rows of the design (bit for bit), and continues after them, so a long design runs
in chunks. The shipped files were written in such chunks (see DATA.md).

`inputs.csv` is the source of truth for the inputs. `10.0 ** z` can differ in the last bit
between numpy builds (the files were written with numpy 1.26.4; numpy 2.5.4 gives 1116 entries
that differ by at most 2.2e-16 relative), so `--resume` and `--check` compare the design with the
stored inputs to a relative tolerance of 1e-15 and then use the stored values.

Every row is kept. A run that fails (solver error, or no solution up to t = 600 s) gets
status 0 and NaN outputs; nothing is dropped here.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
import dfn_us06 as dfn  # noqa: E402

SEED = 20261031
N_MAX = 2000
RTOL_INPUTS = 1e-15  # design vs. stored inputs: allows the last-bit differences of 10.0 ** z


def design(n):
    u = np.random.default_rng(SEED).random((N_MAX, len(dfn.NAMES)))[:n]
    lo = np.where(dfn.LOG, np.log10(dfn.LOWER), dfn.LOWER)
    hi = np.where(dfn.LOG, np.log10(dfn.UPPER), dfn.UPPER)
    z = lo + u * (hi - lo)
    return np.where(dfn.LOG, 10.0 ** z, z)


def qois(v):
    """V_mean: time average of V over [0, 600] s (trapezoid); V_end: V(600 s)."""
    t = dfn.T_EVAL
    vmean = np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(t)) / (t[-1] - t[0])
    return vmean, v[-1]


def write(out, X, status, V, Q, secs):
    M = len(status)
    names = ",".join(dfn.NAMES)
    units = ",".join(f"{r[0]} [{r[2]}]" for r in dfn.INPUTS)
    np.savetxt(os.path.join(out, "inputs.csv"), X[:M], delimiter=",", fmt="%.17g",
               header=f"Inputs of {M} i.i.d. Monte Carlo runs (seed {SEED}), physical units, Table A.2 of Zonta et al. 2026\n{units}")
    Q = np.asarray(Q, dtype=float).reshape(-1, 2)
    np.savetxt(os.path.join(out, "outputs.csv"), np.column_stack([status, Q, secs]), delimiter=",",
               fmt=["%d", "%.17g", "%.17g", "%.3f"],
               header="DFN (PyBaMM 24.9.0) under scaled US06: status (1 ok, 0 failed), time-averaged and final terminal voltage, run time\n"
                      "status [-],V_mean [V],V_end [V],run_time [s]")
    np.savetxt(os.path.join(out, "voltage.csv"), np.asarray(V), delimiter=",", fmt="%.9f",
               header="Terminal voltage [V] at t = 0, 1, ..., 600 s (columns); one row per run of inputs.csv; NaN for failed runs\n"
                      + ",".join(f"V(t={int(t)} s)" for t in dfn.T_EVAL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=float, default=np.inf, help="wall-time limit in s")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--resume", action="store_true", help="continue after the rows already in --out")
    ap.add_argument("--check", action="store_true", help="rerun --rows and compare with the files here")
    ap.add_argument("--rows", default="0,1,2", help="for --check: comma-separated row indices")
    ap.add_argument("--oat", action="store_true",
                    help="one-at-a-time runs at the box centre (each of four inputs at its bounds); prints, writes nothing")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.check:
        return check(a)
    if a.oat:
        return oat()
    X = design(a.n)
    status, V, Q, secs = [], [], [], []
    if a.resume and os.path.exists(os.path.join(a.out, "outputs.csv")):
        X_old = np.atleast_2d(np.loadtxt(os.path.join(a.out, "inputs.csv"), delimiter=","))
        O_old = np.atleast_2d(np.loadtxt(os.path.join(a.out, "outputs.csv"), delimiter=","))
        V_old = np.atleast_2d(np.loadtxt(os.path.join(a.out, "voltage.csv"), delimiter=","))
        k = len(O_old)
        if not np.allclose(X_old[:k], X[:k], rtol=RTOL_INPUTS, atol=0.0):
            raise SystemExit("the inputs in --out are not the first rows of this design")
        X[:k] = X_old[:k]  # the stored inputs are the source of truth
        # earlier rows are carried over as written (Q from the full-precision voltages of their run)
        status, V, Q, secs = [int(s) for s in O_old[:, 0]], list(V_old), list(O_old[:, 1:3]), list(O_old[:, 3])
        print(f"resuming after {k} rows", flush=True)
    t_start = time.time()
    for i in range(len(status), len(X)):
        x = X[i]
        t0 = time.time()
        try:
            v = dfn.voltage(x)
            s = 1
        except Exception as e:  # solver failure: keep the row, flag it
            print(f"row {i}: FAILED ({type(e).__name__}: {str(e)[:120]})", flush=True)
            v, s = np.full(dfn.T_EVAL.shape, np.nan), 0
        status.append(s); V.append(v); secs.append(time.time() - t0)
        Q.append(qois(v) if s else (np.nan, np.nan))
        if (i + 1) % 10 == 0:
            print(f"{i + 1} runs, {time.time() - t_start:.0f} s, {len(status) - sum(status)} failed", flush=True)
            write(a.out, X, status, V, Q, secs)
        if time.time() - t_start > a.budget:
            print(f"budget reached after {i + 1} runs", flush=True)
            break
    write(a.out, X, status, V, Q, secs)
    print(f"done: {len(status)} runs, {len(status) - sum(status)} failed, {time.time() - t_start:.0f} s")


def check(a):
    """Rerun the rows in --rows and compare with inputs.csv and voltage.csv next to this script."""
    rows = [int(r) for r in a.rows.split(",")]
    X = design(max(rows) + 1)
    X_old = np.loadtxt(os.path.join(HERE, "inputs.csv"), delimiter=",")
    V_old = np.loadtxt(os.path.join(HERE, "voltage.csv"), delimiter=",")
    worst = 0.0
    for r in rows:
        assert np.allclose(X[r], X_old[r], rtol=RTOL_INPUTS, atol=0.0), f"row {r}: inputs differ from the design"
        t0 = time.time()
        try:
            v = dfn.voltage(X_old[r])  # the stored inputs are the source of truth
        except Exception as e:
            v = np.full(dfn.T_EVAL.shape, np.nan)
            print(f"row {r}: FAILED ({type(e).__name__})")
        same_fail = np.isfinite(v).all() == np.isfinite(V_old[r]).all()
        d = np.nanmax(np.abs(v - V_old[r])) if np.isfinite(v).all() and np.isfinite(V_old[r]).all() else 0.0
        worst = max(worst, d)
        print(f"row {r}: max |dV| = {d:.3e} V, same failure status: {same_fail} ({time.time() - t0:.1f} s)", flush=True)
    print(f"check: {len(rows)} rows, largest |dV| = {worst:.3e} V")


OAT_INPUTS = ["sigma_pos", "Ds_pos", "L_pos", "R_pos"]


def oat():
    """Ū, U(600) and min U at the box centre (dfn_us06.nominal()) and with each of OAT_INPUTS
    moved alone to its lower and upper bound. About 9 runs of 5 s."""
    x0 = dfn.nominal()
    v = dfn.voltage(x0)
    vm, ve = qois(v)
    print(f"centre: V_mean {vm:.4f} V, V_end {ve:.4f} V, V_min {v.min():.4f} V", flush=True)
    for name in OAT_INPUTS:
        j = dfn.NAMES.index(name)
        res = []
        for val in (dfn.LOWER[j], dfn.UPPER[j]):
            x = x0.copy()
            x[j] = val
            v = dfn.voltage(x)
            vm_j, ve_j = qois(v)
            res.append(vm_j)
            print(f"{name} = {val:.3g}: V_mean {vm_j:.4f} V, V_end {ve_j:.4f} V, V_min {v.min():.4f} V", flush=True)
        print(f"{name}: V_mean(upper) - V_mean(lower) = {res[1] - res[0]:+.4f} V", flush=True)


if __name__ == "__main__":
    main()
