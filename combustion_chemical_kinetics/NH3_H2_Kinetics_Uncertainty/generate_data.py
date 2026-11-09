"""Regenerate the data files of NH3_H2_Kinetics_Uncertainty (needs Cantera 3.2.0 and the mechanism).

    python generate_data.py --out /tmp/nh3 --case idt                 # 1000 ignition samples (about 30 min of CPU)
    python generate_data.py --out /tmp/nh3 --case lfs                 # 200 flame samples (about 15 min of CPU)
    python generate_data.py --out /tmp/nh3 --case nominal             # the nominal model (about 15 s)
    python generate_data.py --out /tmp/nh3 --case idt --rows 0:10 --check   # a subset vs the shipped files

Cases (one Latin hypercube design each, in x = log10(A/A0) on [-f_ini, f_ini]^52):
  idt   1000 samples, seed 470001: idt_inputs.csv (x), idt_outputs.csv (tau, s),
        idt_gradients.csv (dtau/dx_i, s per log10 unit; forward differences, step 1e-3).
  lfs   200 samples, seed 470002: lfs_inputs.csv (x), lfs_outputs.csv (Su, cm/s, the number
        of grid points, and a flag for a flame solved from scratch), lfs_gradients.csv (dSu/dx_i, cm/s per log10 unit; Cantera's
        discrete adjoint on the converged grid).
  nominal  the nominal model (x = 0): nominal.csv, row 1 tau (s) and dtau/dx_i, row 2 Su (cm/s)
        and dSu/dx_i, computed as in the two cases above.

The Latin hypercube is drawn with numpy only (one random permutation and one uniform jitter per
column), so the design does not depend on the scipy version. Inputs are rounded to the digits
written before the model is run, so the files are consistent with each other.

Work is resumable: each finished row is appended to <out>/<case>_rows.txt, a rerun skips the
rows already there, and --max-seconds stops a run cleanly between rows (exit status 3 while
rows remain). The CSV files are written when every requested row is done. A row whose solve fails is
written as nan (none did in the shipped files).

Flames depend on the solver history. Each flame restarts from the converged nominal flame, but
the Sim1D object keeps state from earlier solves, so the refined grid, and with it Su, can
depend on which flames the same model instance solved before (up to about 3 % in Su at slow
and mid-range flames; DATA.md, "Path dependence"). The shipped flame rows were written in seven
runs that started at rows 0, 4, 51, 55, 92, 133 and 166 (LFS_SESSIONS), each with a new model
instance. The lfs case therefore starts a new instance at each of these rows, so a run that
starts at one of them and is not stopped by --max-seconds replays the original history and
reproduces the shipped rows to the tight tolerance below. A run that starts elsewhere (or is
resumed after a stop) solves the same flames from another history.

--out DIR is required and must be outside this directory, so the shipped files are never
overwritten. --check compares the new files with the shipped rows. Ignition delay and the
nominal model: relative max difference 1e-6 for x, tau and Su, and relative 2-norm 1e-3 per
gradient row. Flames: per row, the relative difference of Su and the relative 2-norm difference
of the gradient. The check reports how many rows agree to 1e-6 and 1e-3 (the same solver
history), and passes when every row is within 5e-2 in Su and 2e-1 in the gradient, the spread
that a different solver history produces (largest seen: 3.2 % in Su and 8.3e-2 in the gradient,
row 24), so a pass means the same flame to the accuracy of its grid, not the same digits.

Needs numpy, scipy and Cantera (the shipped files were written with Cantera 3.2.0:
pip install cantera==3.2.0) through nh3h2_kinetics_driver.py, and the mechanism file NH3_H2.yaml
from the paper's supplement (see the driver); without either the script stops with an error.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nh3h2_kinetics_driver as D  # noqa: E402

SEEDS = {"idt": 470001, "lfs": 470002}
SIZES = {"idt": 1000, "lfs": 200}
XFMT = "%.15e"
TOL = {"inputs": 1e-6, "outputs": 1e-6, "gradients": 1e-3}
TOL_LFS = {"outputs": 5e-2, "gradients": 2e-1}   # flames: the spread of a different solver history
LFS_SESSIONS = (0, 4, 51, 55, 92, 133, 166)      # first rows of the runs that wrote lfs_*.csv


def lhs(n, lower, upper, seed):
    """Latin hypercube: column j is a random permutation of the n strata with uniform jitter."""
    rng = np.random.default_rng(seed)
    m = len(lower)
    U = np.empty((n, m))
    for j in range(m):
        U[:, j] = (rng.permutation(n) + rng.random(n)) / n
    X = lower + (upper - lower) * U
    return np.array([[float(XFMT % v) for v in row] for row in X])


def design(case):
    return lhs(SIZES[case], D.LOWER, D.UPPER, SEEDS[case])


def run_row(model, case, x):
    """One sample: (value, extra, gradient). Raises on solver failure."""
    if case == "idt":
        tau, g = model.idt_grad(x)
        return tau, None, g
    n0 = model.n_fresh_starts
    Su, g = model.lfs_grad(x)
    return 100.0 * Su, [len(model._flame.grid), model.n_fresh_starts - n0], 100.0 * g


def read_done(path):
    done = {}
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                v = line.split(",")
                done[int(v[0])] = [float(s) for s in v[1:]]
    return done


def header(desc, cols):
    return desc + "\n" + ", ".join(cols)


def write_csv(case, X, done, rows, out):
    R = np.array([done[i] for i in rows])
    xcols = [f"{n} [log10 of the factor on A of entry {k + 1}]" for k, n in enumerate(D.NAMES)]
    np.savetxt(os.path.join(out, f"{case}_inputs.csv"), X[rows], fmt=XFMT, delimiter=",",
               header=header(f"NH3/H2 kinetics, 52 inputs x_i = log10(A_i/A_i0), Latin hypercube on "
                             f"[-f_ini, f_ini], seed {SEEDS[case]}, rows {rows[0]}:{rows[-1] + 1}", xcols))
    if case == "idt":
        np.savetxt(os.path.join(out, "idt_outputs.csv"), R[:, :1], fmt="%.15e", delimiter=",",
                   header=header("ignition delay tau (first time T = T5 + 400 K), 4.375% NH3/3.281% O2/"
                                 "92.344% Ar, T5 = 1587.982151 K, p5 = 10 atm, constant volume, Cantera 3.2.0",
                                 ["tau [s]"]))
        G = R[:, 1:]
        gdesc, gu = "dtau/dx_i, forward differences in x with step 1e-3", "s"
    else:
        np.savetxt(os.path.join(out, "lfs_outputs.csv"), R[:, :3], fmt=["%.12e", "%d", "%d"], delimiter=",",
                   header=header("laminar flame speed, NH3/H2 50/50 with air, phi = 1, 298 K, 1 atm, "
                                 "mixture-averaged transport, Cantera 3.2.0 FreeFlame",
                                 ["Su [cm/s]", "grid points [-]",
                                  "fresh start [1 if the restart from the nominal flame failed, else 0]"]))
        G = R[:, 3:]
        gdesc, gu = "dSu/dx_i, Cantera discrete adjoint on the converged grid", "cm/s"
    np.savetxt(os.path.join(out, f"{case}_gradients.csv"), G, fmt="%.10e", delimiter=",",
               header=header(gdesc, [f"d/d{n} [{gu} per log10 unit]" for n in D.NAMES]))


def check(case, out, rows):
    worst = {}
    for kind in ("inputs", "outputs", "gradients"):
        fn = f"{case}_{kind}.csv"
        new = np.loadtxt(os.path.join(out, fn), delimiter=",", ndmin=2)
        old = np.loadtxt(os.path.join(HERE, fn), delimiter=",", ndmin=2)[rows]
        if kind == "outputs":
            new, old = new[:, :1], old[:, :1]
        if kind == "gradients":       # relative 2-norm per row
            r = np.linalg.norm(new - old, axis=1) / np.linalg.norm(old, axis=1)
        elif case == "lfs" and kind == "outputs":   # relative difference per row
            r = np.abs(new[:, 0] - old[:, 0]) / np.abs(old[:, 0])
        else:
            r = np.atleast_1d(np.max(np.abs(new - old) / np.abs(old).max()))
        rel = float(np.max(r))
        tol = TOL_LFS[kind] if (case == "lfs" and kind != "inputs") else TOL[kind]
        worst[kind] = (rel, tol)
        ok = "ok" if rel <= tol else "ABOVE TOLERANCE"
        line = f"  {fn}: max relative difference {rel:.2e} (tolerance {tol:.0e}) {ok}"
        if case == "lfs" and kind != "inputs":
            same = int(np.sum(r <= TOL[kind]))
            line += f"; {same} of {len(r)} rows within {TOL[kind]:.0e} (same solver history)"
            far = [f"{rows[k]} ({r[k]:.1e})" for k in np.nonzero(r > TOL[kind])[0]]
            if far:
                line += "; other rows: " + ", ".join(far)
        print(line)
    if case == "lfs" and any(worst[k][0] > TOL[k] for k in ("outputs", "gradients")):
        print("  flame rows that differ by more than 1e-6 were solved after a different sequence of flames "
              "than when the shipped files were written; start the run at one of the rows "
              f"{', '.join(str(r) for r in LFS_SESSIONS)} to replay the original history")
    return all(v <= t for v, t in worst.values())


def run_nominal(model, out):
    tau, gt = model.idt_grad(D.NOMINAL)
    Su, gs = model.lfs_grad(D.NOMINAL)
    R = np.vstack([np.r_[tau, gt], np.r_[100.0 * Su, 100.0 * gs]])
    np.savetxt(os.path.join(out, "nominal.csv"), R, fmt="%.12e", delimiter=",",
               header=header("nominal model (x = 0): row 1 tau [s] then dtau/dx_i [s per log10 unit] "
                             "(forward differences, step 1e-3); row 2 Su [cm/s] then dSu/dx_i "
                             "[cm/s per log10 unit] (adjoint)", ["value"] + [f"d/d{n}" for n in D.NAMES]))
    print(f"nominal: tau = {tau:.6e} s, Su = {100 * Su:.5f} cm/s, {len(model._flame.grid)} grid points")
    return R


def check_nominal(out):
    new = np.loadtxt(os.path.join(out, "nominal.csv"), delimiter=",")
    old = np.loadtxt(os.path.join(HERE, "nominal.csv"), delimiter=",")
    ok = True
    for k, name in enumerate(("tau", "Su")):
        dv = abs(new[k, 0] / old[k, 0] - 1)
        dg = np.linalg.norm(new[k, 1:] - old[k, 1:]) / np.linalg.norm(old[k, 1:])
        ok &= dv <= TOL["outputs"] and dg <= TOL["gradients"]
        print(f"  nominal {name}: relative difference {dv:.2e}, gradient {dg:.2e}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True, choices=["idt", "lfs", "nominal"])
    p.add_argument("--out", required=True, help="output directory, outside this directory")
    p.add_argument("--rows", default=None, help="a:b, a subset of the rows")
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    p.add_argument("--max-seconds", type=float, default=None, help="stop between rows after this time")
    p.add_argument("--mech", default=None, help="path of NH3_H2.yaml (default: see the driver)")
    a = p.parse_args()
    out = os.path.abspath(a.out)
    here = os.path.realpath(HERE)
    if os.path.commonpath([os.path.realpath(out), here]) == here:
        sys.exit("--out must be a directory outside this one, so that the shipped files are not overwritten")
    if D.ct is None:
        sys.exit("Cantera is not installed. Install it (pip install cantera==3.2.0, the version that wrote "
                 "the shipped files) and rerun generate_data.py.")
    try:
        D.mech_path(a.mech)
    except FileNotFoundError as e:
        sys.exit(str(e))
    os.makedirs(out, exist_ok=True)
    if a.case == "nominal":
        run_nominal(D.KineticsModel(mech=a.mech), out)
        if a.check and not check_nominal(out):
            sys.exit(1)
        return
    X = design(a.case)
    if a.rows:
        i, j = (int(s) for s in a.rows.split(":"))
        rows = list(range(i, min(j, len(X))))
    else:
        rows = list(range(len(X)))
    log = os.path.join(out, f"{a.case}_rows.txt")
    done = read_done(log)
    todo = [i for i in rows if i not in done]
    model = D.KineticsModel(mech=a.mech)
    n_fresh = 0
    t0 = time.time()
    with open(log, "a") as fh:
        for k, i in enumerate(todo):
            if a.max_seconds is not None and time.time() - t0 > a.max_seconds:
                break
            if a.case == "lfs" and k > 0 and i in LFS_SESSIONS:   # a new instance, as when the files were written
                n_fresh += model.n_fresh_starts
                model = D.KineticsModel(mech=a.mech)
            ts = time.time()
            try:
                v, extra, g = run_row(model, a.case, X[i])
                vals = [v] + ([] if extra is None else extra) + list(g)
            except Exception as e:                  # recorded as nan; the notebook drops such rows
                print(f"{a.case} row {i} failed: {e}", flush=True)
                v = float("nan")
                vals = [v] * ((1 if a.case == "idt" else 3) + D.M_INPUTS)
                n_fresh += model.n_fresh_starts
                model = D.KineticsModel(mech=a.mech)
            fh.write(f"{i}," + ",".join(repr(float(s)) for s in vals) + "\n")
            fh.flush()
            done[i] = vals
            print(f"{a.case} row {i}: {v:.6e} ({time.time() - ts:.1f} s)", flush=True)
    left = [i for i in rows if i not in done]
    print(f"{a.case}: {len(rows) - len(left)} of {len(rows)} rows done; this run {time.time() - t0:.1f} s, "
          f"fresh flame starts {n_fresh + model.n_fresh_starts}", flush=True)
    if left:
        sys.exit(3)
    write_csv(a.case, X, done, rows, out)
    if a.check and not check(a.case, out, rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
