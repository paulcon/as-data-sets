"""Regenerate the data files of NH3_H2_Kinetics_Uncertainty (needs Cantera 3.2.0 and the mechanism).

    python generate_data.py --out /tmp/nh3 --case idt                 # 1000 ignition samples (about 30 min of CPU)
    python generate_data.py --out /tmp/nh3 --case lfs                 # 200 flame samples (about 15 min of CPU)
    python generate_data.py --out /tmp/nh3 --case nominal             # the nominal model (about 15 s)
    python generate_data.py --check                                   # all three cases in a temporary directory vs the shipped files
    python generate_data.py --case idt --rows 0:10 --check            # a subset vs the shipped files

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
rows already there, and --max-seconds stops a run cleanly between rows (exit status 4 while
rows remain). The CSV files are written when every requested row is done. A row whose solve fails is
written as nan (none did in the shipped files).

Flames depend on the solver history. Each flame restarts from the converged nominal flame, but
the Sim1D object keeps state from earlier solves, so the refined grid, and with it Su, can
depend on which flames the same model instance solved before (up to about 3 % in Su at slow
and mid-range flames; DATA.md, "Path dependence"). The shipped flame rows were written in seven
runs that started at rows 0, 4, 51, 55, 92, 133 and 166 (LFS_SESSIONS), each with a new model
instance. The lfs case therefore starts a new instance at each of these rows, so a run that
starts at one of them and is not stopped by --max-seconds replays the original history and
reproduces the shipped rows to round-off (DATA.md). A run that starts elsewhere (or is
resumed after a stop) solves the same flames from another history.

--out DIR is required without --check and must be outside this directory, so the shipped files are
never overwritten; with --check it defaults to a temporary directory (removed at the end) and --case
to all three cases in turn (nominal, idt, lfs). --check prints the largest differences from the shipped rows
(relative max difference for tau and Su, relative 2-norm per gradient row; for the flames, per row, with the
number of rows that agree to 1e-6 and 1e-3, as a run with the same solver history does; a different history
gives up to 3.2 % in Su and 8.3e-2 in the gradient, row 24) and fails only on structural mismatches (shapes,
nan entries, inputs). --max-seconds with --check needs --out DIR; a check stopped by --max-seconds exits 1
without comparing (without --check, a stopped run exits 4).

Needs numpy, scipy and Cantera (the shipped files were written with Cantera 3.2.0:
pip install cantera==3.2.0) through nh3h2_kinetics_driver.py, and the mechanism file NH3_H2.yaml
from the paper's supplement (see the driver); without either the script stops with an error.
"""
import argparse
import atexit
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nh3h2_kinetics_driver as D  # noqa: E402

SEEDS = {"idt": 470001, "lfs": 470002}
SIZES = {"idt": 1000, "lfs": 200}
XFMT = "%.15e"
INPUT_RTOL = 1e-8   # --check fails if a regenerated input (a seeded draw) differs by more (relative)
SAME_HISTORY = {"outputs": 1e-6, "gradients": 1e-3}   # flames: --check counts the rows within these (report only)
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
    """Print the largest differences; return (False on a structural mismatch, (difference, measure, file))."""
    ok, worst = True, (0.0, "", f"{case}_outputs.csv")
    for kind in ("inputs", "outputs", "gradients"):
        fn = f"{case}_{kind}.csv"
        new = np.loadtxt(os.path.join(out, fn), delimiter=",", ndmin=2)
        old = np.loadtxt(os.path.join(HERE, fn), delimiter=",", ndmin=2)[rows]
        if new.shape != old.shape or not np.array_equal(np.isnan(new), np.isnan(old)):
            print(f"  FAIL: {fn}: shape {new.shape} or nan entries differ from the shipped {old.shape}")
            ok = False
            continue
        if kind == "outputs":
            new, old = new[:, :1], old[:, :1]
        if kind == "gradients":       # relative 2-norm per row
            r, how = np.linalg.norm(new - old, axis=1) / np.linalg.norm(old, axis=1), "row 2-norm"
        elif case == "lfs" and kind == "outputs":   # relative difference per row
            r, how = np.abs(new[:, 0] - old[:, 0]) / np.abs(old[:, 0]), "elementwise"
        else:
            r, how = np.atleast_1d(np.max(np.abs(new - old) / np.abs(old).max())), "to the largest value"
        rel = float(np.nanmax(r, initial=0.0))
        line = f"  {fn}: max relative difference {rel:.2e} ({how})"
        if case == "lfs" and kind != "inputs":
            tight = SAME_HISTORY[kind]
            line += f"; {int(np.sum(r <= tight))} of {len(r)} rows within {tight:.0e} (same solver history)"
            far = [f"{rows[k]} ({r[k]:.1e})" for k in np.nonzero(r > tight)[0]]
            if far:
                line += "; other rows: " + ", ".join(far)
        print(line)
        if kind == "inputs":
            if rel > INPUT_RTOL:
                print(f"  FAIL: {fn} differs from the shipped inputs by {rel:.2e} > {INPUT_RTOL:g}")
                ok = False
            continue
        if rel >= worst[0]:
            worst = (rel, how, fn)
        if case == "lfs" and rel > SAME_HISTORY[kind]:
            print("  flame rows that differ by more than this were probably solved after a different sequence of "
                  "flames than when the shipped files were written; start the run at one of the rows "
                  f"{', '.join(str(r) for r in LFS_SESSIONS)} to replay the original history")
    return ok, worst


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
    """Print the nominal differences; return (False on a structural mismatch, (difference, measure, quantity))."""
    new = np.loadtxt(os.path.join(out, "nominal.csv"), delimiter=",", ndmin=2)
    old = np.loadtxt(os.path.join(HERE, "nominal.csv"), delimiter=",", ndmin=2)
    if new.shape != old.shape or not np.array_equal(np.isnan(new), np.isnan(old)):
        print(f"  FAIL: nominal.csv: shape {new.shape} or nan entries differ from the shipped {old.shape}")
        return False, (0.0, "", "nominal.csv")
    worst = (0.0, "", "nominal.csv")
    for k, name in enumerate(("tau", "Su")):
        dv = abs(new[k, 0] / old[k, 0] - 1)
        dg = np.linalg.norm(new[k, 1:] - old[k, 1:]) / np.linalg.norm(old[k, 1:])
        print(f"  nominal {name}: relative difference {dv:.2e}, gradient {dg:.2e} (relative 2-norm)")
        worst = max(worst, (dv, "elementwise", f"nominal.csv {name}"), (dg, "row 2-norm", f"nominal.csv d{name}/dx"))
    return True, worst


def run_case(case, a, out):
    """Run one case into out; return (0 done, 4 rows remain; with --check, the result of the comparison)."""
    if case == "nominal":
        run_nominal(D.KineticsModel(mech=a.mech), out)
        return 0, (check_nominal(out) if a.check else None)
    X = design(case)
    if a.rows:
        i, j = (int(s) for s in a.rows.split(":"))
        rows = list(range(i, min(j, len(X))))
    else:
        rows = list(range(len(X)))
    log = os.path.join(out, f"{case}_rows.txt")
    done = read_done(log)
    todo = [i for i in rows if i not in done]
    model = D.KineticsModel(mech=a.mech)
    n_fresh = 0
    t0 = time.time()
    with open(log, "a") as fh:
        for k, i in enumerate(todo):
            if a.max_seconds is not None and time.time() - t0 > a.max_seconds:
                break
            if case == "lfs" and k > 0 and i in LFS_SESSIONS:   # a new instance, as when the files were written
                n_fresh += model.n_fresh_starts
                model = D.KineticsModel(mech=a.mech)
            ts = time.time()
            try:
                v, extra, g = run_row(model, case, X[i])
                vals = [v] + ([] if extra is None else extra) + list(g)
            except Exception as e:                  # recorded as nan; the notebook drops such rows
                print(f"{case} row {i} failed: {e}", flush=True)
                v = float("nan")
                vals = [v] * ((1 if case == "idt" else 3) + D.M_INPUTS)
                n_fresh += model.n_fresh_starts
                model = D.KineticsModel(mech=a.mech)
            fh.write(f"{i}," + ",".join(repr(float(s)) for s in vals) + "\n")
            fh.flush()
            done[i] = vals
            print(f"{case} row {i}: {v:.6e} ({time.time() - ts:.1f} s)", flush=True)
    left = [i for i in rows if i not in done]
    print(f"{case}: {len(rows) - len(left)} of {len(rows)} rows done; this run {time.time() - t0:.1f} s, "
          f"fresh flame starts {n_fresh + model.n_fresh_starts}", flush=True)
    if left:
        return 4, None
    write_csv(case, X, done, rows, out)
    return 0, (check(case, out, rows) if a.check else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default=None, choices=["idt", "lfs", "nominal"],
                   help="required without --check; with --check, default all three in turn")
    p.add_argument("--out", default=None, help="output directory, outside this directory (required without "
                   "--check; with --check, default a temporary directory)")
    p.add_argument("--rows", default=None, help="a:b, a subset of the rows")
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    p.add_argument("--max-seconds", type=float, default=None, help="stop between rows after this time")
    p.add_argument("--mech", default=None, help="path of NH3_H2.yaml (default: see the driver)")
    a = p.parse_args()
    if a.case is None and not a.check:
        p.error("--case is required without --check")
    if a.out is None:
        if a.check:
            if a.max_seconds is not None:
                p.error("--max-seconds with --check needs --out DIR, so that a stopped run can be resumed")
            a.out = tempfile.mkdtemp(prefix="check_")
            atexit.register(shutil.rmtree, a.out, ignore_errors=True)
        else:
            p.error("--out DIR is required without --check")
    out = os.path.abspath(a.out)
    here = os.path.realpath(HERE)
    if os.path.commonpath([os.path.realpath(out), here]) == here:
        sys.exit("--out must be a directory outside this one, so that the shipped files are not overwritten")
    if D.ct is None:
        print("Cantera is not installed. Install it (pip install cantera==3.2.0, the version that wrote "
              "the shipped files) and rerun generate_data.py.", file=sys.stderr)
        sys.exit(3)
    try:
        D.mech_path(a.mech)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(3)
    os.makedirs(out, exist_ok=True)
    cases = [a.case] if a.case else ["nominal", "idt", "lfs"]
    failed, worst = [], (0.0, "", "")
    for case in cases:
        code, res = run_case(case, a, out)
        if code == 4:
            if a.check:
                print("check not done: stopped by --max-seconds before all rows were done; rerun with the same "
                      "--out to continue", file=sys.stderr)
                sys.exit(1)
            sys.exit(4)
        if res is not None:
            if not res[0]:
                failed.append(case)
            worst = max(worst, res[1])
    if a.check:
        print(f"largest difference from the shipped files: {worst[0]:.1e} (relative, {worst[1]}, {worst[2]})")
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
