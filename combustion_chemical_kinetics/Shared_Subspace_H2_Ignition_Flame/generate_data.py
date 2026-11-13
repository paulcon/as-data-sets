"""Regenerate the data files of Shared_Subspace_H2_Ignition_Flame (needs Cantera 3.2.0).

    python build_mechanism.py KONNOV_DIR                 # once: writes konnov2008_h2.yaml
    python generate_data.py --out DIR                    # all cases, resumable
    python generate_data.py --check                      # all cases in a temporary directory, vs shipped files
    python generate_data.py --case grad --rows 0:4 --check   # subset vs shipped files

Cases
  grad  350 samples of the 33 inputs (truncated normal, seed 2018033): the 21 outputs and
        their gradients; inputs.csv, outputs.csv, gradients.csv.
  mc    1000 further samples (seed 2018100), outputs only; mc_inputs.csv, mc_outputs.csv.
  line  the outputs along the shared direction v and along each output's own first
        eigenvector u_i, on 49 points xi in [-4, 4]; both directions are computed from
        gradients.csv (in DIR if present, else the shipped one) by
        h2_konnov_driver.shared_directions (v: its truncated-SVD solution); line_runs.csv, directions.csv.

Outputs (columns of outputs.csv): tau at the 16 ignition conditions of
h2_konnov_driver.IDT_CONDITIONS (the first is the paper's Fig. 1 condition, 1.59 bar and
1000 K), then S_L at the 5 pressures of SL_P. Gradients: forward differences
k_j -> (1 + 1e-3) k_j (Ji et al. Eq. 20), stored as df/dln k_j in the output's units.

Work is saved row by row under DIR/_parts, so an interrupted run continues where it
stopped; --max-seconds ends a run cleanly after that much wall time (rerun to continue).
The script never writes into its own directory: --out is required without --check and must be outside
it (not a subdirectory either; copy the finished CSVs in by hand); with --check it defaults to a temporary
directory, removed at the end (--max-seconds then needs --out DIR, and a --check run stopped by
--max-seconds exits 1 without comparing). --check prints the largest differences from the shipped files
(relative to each column's largest value) and fails only on structural mismatches (shapes, NaN pattern,
inputs).
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
import h2_konnov_driver as D  # noqa: E402

SEEDS = {"grad": 2018033, "mc": 2018100}
INPUT_RTOL = 1e-8   # --check fails if a regenerated input (a seeded draw) differs by more (relative)
SIZES = {"grad": 350, "mc": 1000}
XI = np.linspace(-4.0, 4.0, 49)


def draw(case):
    """Truncated standard normal x in [-3, 3]^33 (entries outside redrawn), returned as
    ln(k_j/k_j0) = sigma_j x_j rounded to the written digits."""
    rng = np.random.default_rng(SEEDS[case])
    X = rng.standard_normal((SIZES[case], D.M_IN))
    bad = np.abs(X) > 3.0
    while bad.any():
        X[bad] = rng.standard_normal(int(bad.sum()))
        bad = np.abs(X) > 3.0
    L = X * D.SIGMA
    return np.array([[float("%.15e" % v) for v in row] for row in L])


def output_names():
    n = [f"tau(P={p:.4g} atm, T0={t:.0f} K) [s]" for p, t in D.IDT_CONDITIONS]
    return n + [f"SL(P={p:.4g} atm, Tu=300 K) [m/s]" for p in D.SL_P]


def run_row(model, lnk, with_grad):
    x = lnk / D.SIGMA
    f, G = [], []
    for p, t in D.IDT_CONDITIONS:
        if with_grad:
            v, g = model.idt_grad(p, t, x)
            G.append(g)
        else:
            v = model.ignition_delay(p, t, x)
        f.append(v)
    for p in D.SL_P:
        if with_grad:
            v, g = model.sl_grad(p, x)
            G.append(g)
        else:
            v = model.flame_speed(p, x)
        f.append(v)
    out = np.array(f)
    return np.concatenate([out, np.ravel(G)]) if with_grad else out


def hdr(desc, cols):
    return desc + "\n" + ", ".join(cols)


def run_case(case, rows, out, tmax, t_start):
    L = draw(case)
    parts = os.path.join(out, "_parts")
    os.makedirs(parts, exist_ok=True)
    model = None
    done = 0
    for i in range(rows.start, rows.stop):
        fn = os.path.join(parts, f"{case}_{i:04d}.npy")
        if os.path.exists(fn):
            continue
        if time.time() - t_start > tmax:
            print(f"stopped after {time.time() - t_start:.0f} s (--max-seconds); rerun to continue")
            return False
        if model is None:
            model = D.H2Model()
        t0 = time.process_time()
        r = run_row(model, L[i], case == "grad")
        np.save(fn, r)
        done += 1
        with open(os.path.join(out, "generate.log"), "a") as fh:
            fh.write(f"{case} row {i} cpu {time.process_time() - t0:.2f} s\n")
    R = []
    for i in range(rows.start, rows.stop):
        R.append(np.load(os.path.join(parts, f"{case}_{i:04d}.npy")))
    R = np.array(R)
    X = L[rows]
    nout = len(D.IDT_CONDITIONS) + len(D.SL_P)
    pre = "mc_" if case == "mc" else ""
    desc = (f"H2/air, Konnov (2008) mechanism; ln(k_j/k_j0) for reactions R1..R33 (mechanism order), "
            f"x_j = ln(k_j/k_j0)/(ln(UF_j)/3) truncated standard normal on [-3, 3], seed {SEEDS[case]}, "
            f"rows {rows.start}:{rows.stop}")
    np.savetxt(os.path.join(out, f"{pre}inputs.csv"), X, fmt="%.15e", delimiter=",",
               header=hdr(desc, [f"ln(k{j + 1}/k{j + 1}_0) [-]" for j in range(D.M_IN)]))
    np.savetxt(os.path.join(out, f"{pre}outputs.csv"), R[:, :nout], fmt="%.15e", delimiter=",",
               header=hdr("ignition delay tau (first time T = T0 + 400 K, constant volume) at 16 "
                          "conditions, then laminar flame speed S_L at 5 pressures; Cantera 3.2.0",
                          output_names()))
    if case == "grad":
        cols = [f"d{n.split(' [')[0]}/dln(k{j + 1}) [{n.split('[')[1]}" for n in output_names()
                for j in range(D.M_IN)]
        np.savetxt(os.path.join(out, "gradients.csv"), R[:, nout:], fmt="%.10e", delimiter=",",
                   header=hdr("df_i/dln(k_j), forward differences k_j -> 1.001 k_j; columns: output i "
                              "(order of outputs.csv) major, reaction j minor", cols))
    print(f"case {case}: rows {rows.start}:{rows.stop} complete ({done} computed in this run)")
    return True


def run_line(out, tmax, t_start):
    gfile = os.path.join(out, "gradients.csv")
    ofile = os.path.join(out, "outputs.csv")
    if not os.path.exists(gfile) or np.loadtxt(gfile, delimiter=",", ndmin=2).shape[0] != SIZES["grad"]:
        gfile, ofile = os.path.join(HERE, "gradients.csv"), os.path.join(HERE, "outputs.csv")
    G = np.loadtxt(gfile, delimiter=",")
    F = np.loadtxt(ofile, delimiter=",")
    U, _, v = D.shared_directions(F, G)
    nout = U.shape[0]
    np.savetxt(os.path.join(out, "directions.csv"), np.vstack([U, v[None, :]]), fmt="%.15e",
               delimiter=",", header=hdr(
                   "rows 1-21: first eigenvector u_i of output i (normalized inputs x, R16 component "
                   "positive); row 22: shared direction v (truncated-SVD minimum-norm solution of "
                   "A^T v = 1, A = [u_2..u_21], max |u_i^T v - 1| <= 0.05; Ji et al. Eqs. 8-11), "
                   "the 20 training outputs; output 1, the Fig. 1 condition, left out",
                   [f"x{j + 1} [-]" for j in range(D.M_IN)]))
    parts = os.path.join(out, "_parts")
    os.makedirs(parts, exist_ok=True)
    model = D.H2Model()
    conds = [("idt", p, t) for p, t in D.IDT_CONDITIONS] + [("sl", p, None) for p in D.SL_P]
    for k, xi in enumerate(XI):
        fn = os.path.join(parts, f"line_{k:03d}.npy")
        if os.path.exists(fn):
            continue
        if time.time() - t_start > tmax:
            print("stopped (--max-seconds); rerun to continue")
            return False
        row = []
        for direc in ("v", "u"):
            for i, (kind, p, t) in enumerate(conds):
                x = xi * (v if direc == "v" else U[i])
                row.append(model.ignition_delay(p, t, x) if kind == "idt" else model.flame_speed(p, x))
        np.save(fn, np.array(row))
    R = np.array([np.load(os.path.join(parts, f"line_{k:03d}.npy")) for k in range(len(XI))])
    names = output_names()
    cols = ["xi [-]"] + [f"{n.split(' [')[0]} at x = xi v [{n.split('[')[1]}" for n in names] + \
        [f"{n.split(' [')[0]} at x = xi u_i [{n.split('[')[1]}" for n in names]
    np.savetxt(os.path.join(out, "line_runs.csv"), np.column_stack([XI, R]), fmt="%.15e",
               delimiter=",", header=hdr("outputs at x = xi v (shared direction) and x = xi u_i "
                                         "(own first eigenvector), directions from directions.csv",
                                         cols))
    print(f"case line: {len(XI)} points x {2 * nout} runs complete")
    return True


def check(case, out, rows):
    """Print max |new - shipped| / column max per file; return (False on a structural mismatch, the largest
    output difference and its file)."""
    files = {"grad": ["inputs.csv", "outputs.csv", "gradients.csv"],
             "mc": ["mc_inputs.csv", "mc_outputs.csv"],
             "line": ["directions.csv", "line_runs.csv"]}[case]
    ok, worst = True, (0.0, files[-1])
    for fn in files:
        new = np.loadtxt(os.path.join(out, fn), delimiter=",", ndmin=2)
        old = np.loadtxt(os.path.join(HERE, fn), delimiter=",", ndmin=2)
        if case != "line":
            old = old[rows]
        if new.shape != old.shape or not np.array_equal(np.isnan(new), np.isnan(old)):
            print(f"  FAIL: {fn}: shape {new.shape} or NaN pattern differs from the shipped {old.shape}")
            ok = False
            continue
        scale = np.nanmax(np.abs(old), axis=0)
        scale[scale == 0] = 1.0
        rel = np.nanmax(np.abs(new - old) / scale)
        print(f"  {fn}: max |new - shipped| / column max = {rel:.2e}")
        if fn.endswith("inputs.csv"):            # the inputs are seeded draws
            if rel > INPUT_RTOL:
                print(f"  FAIL: {fn} differs from the shipped inputs by {rel:.2e} > {INPUT_RTOL:g}")
                ok = False
        elif rel >= worst[0]:
            worst = (rel, fn)
    return ok, worst


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--case", default="all", choices=["all", "grad", "mc", "line"])
    p.add_argument("--rows", default=None, help="a:b, a subset of the rows (grad, mc)")
    p.add_argument("--out", default=None, help="output directory, not this directory (required without "
                   "--check; with --check, default a temporary directory)")
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    p.add_argument("--max-seconds", type=float, default=np.inf)
    a = p.parse_args()
    if a.out is None:
        if a.check:
            if a.max_seconds != np.inf:
                p.error("--max-seconds with --check needs --out DIR, so that a stopped run can be resumed")
            a.out = tempfile.mkdtemp(prefix="check_")
            atexit.register(shutil.rmtree, a.out, ignore_errors=True)
        else:
            p.error("--out DIR is required without --check")
    out_abs, here_abs = os.path.realpath(a.out), os.path.realpath(HERE)
    if os.path.commonpath([out_abs, here_abs]) == here_abs:
        sys.exit("--out must be a directory outside this one (not this directory or a subdirectory of it), "
                 "so that the shipped files are not overwritten")
    if D.ct is None:
        print("Cantera is not installed. Install it (pip install cantera==3.2.0, the version that wrote "
              "the shipped files) and rerun generate_data.py.", file=sys.stderr)
        sys.exit(3)
    if not os.path.exists(D.MECH):
        print("konnov2008_h2.yaml is missing: run build_mechanism.py on the supplementary files of "
              "Konnov (2008), doi:10.1016/j.combustflame.2007.10.024, then rerun generate_data.py.", file=sys.stderr)
        sys.exit(3)
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    cases = ["grad", "mc", "line"] if a.case == "all" else [a.case]
    good, worst = True, (0.0, "")
    for c in cases:
        if c == "line":
            if a.rows:
                sys.exit("--rows does not apply to the line case")
            complete = run_line(a.out, a.max_seconds, t_start)
            rows = None
        else:
            if a.rows:
                i, j = (int(s) for s in a.rows.split(":"))
                rows = slice(i, j)
            else:
                rows = slice(0, SIZES[c])
            complete = run_case(c, rows, a.out, a.max_seconds, t_start)
        if not complete:
            if a.check:
                sys.exit("check not done: stopped by --max-seconds before all rows were done; rerun with the "
                         "same --out to continue")
            break
        if a.check:
            ok, w = check(c, a.out, rows)
            good, worst = good and ok, max(worst, w)
    print(f"wall time {time.time() - t_start:.0f} s")
    if a.check:
        print(f"largest difference from the shipped files: {worst[0]:.1e} (relative to the column max, {worst[1]})")
        if not good:
            sys.exit(1)


if __name__ == "__main__":
    main()
