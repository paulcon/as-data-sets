"""Regenerate the data files of Ammonia_Syngas_Optimized_Kinetics (needs Cantera 3.2.0).

    python generate_data.py --out DIR                           # all 100 rows (about 45 min of CPU, one core)
    python generate_data.py --out DIR --max-minutes 10          # resumable: run for 10 min, then stop
    python generate_data.py --check                             # all 100 rows in a temporary directory, compared with the shipped files
    python generate_data.py --rows 0:3 --check                  # a subset, compared with the shipped files

Design: SIZE points uniform on the feasible region (ammonia_idt_driver.sample_feasible), Monte
Carlo with seed SEED; the first n rows of a larger design are the design of size n. Inputs are
rounded to the written digits before the model is run, so the files agree with each other.
For each row: tau at the three conditions A, B, C (outputs.csv) and forward-difference
gradients of log10 tau in the node coordinates with step 1e-3 of each input's half-range
(gradients_A.csv, gradients_B.csv, gradients_C.csv). Every finished row is saved as
DIR/rows/row_NNNN.npz and skipped on the next run; the CSV files are written when all
requested rows exist. Progress goes to DIR/generate_log.txt.

--out is required without --check and must lie outside this directory (not the directory or a folder in
it), so the shipped files are never overwritten (copy new files in by hand after comparing them); with
--check it defaults to a temporary directory, removed at the end (--max-minutes then needs --out DIR, and a
--check run stopped by --max-minutes exits 1 without comparing). --check prints the largest differences
from the shipped files (tau: elementwise relative; gradients: relative 2-norm per row in normalized
coordinates) and fails only on structural mismatches (shapes, NaN pattern, inputs). The gradients are
noisy: a rounding-level change of the inputs, x (1 +- 1e-13), changes the forward-difference gradients by
up to about 2e-2 at A and 6e-4 at B and C (check_accuracy.py, check 7), because the step of 1e-3 amplifies
the solver's noise in tau, so a rerun on another platform is not expected to match them more closely.

Needs Cantera through ammonia_idt_driver.py and the mechanism file, which is not part of this
data set (convert_mechanism.py makes it from the article's supplementary zip; DATA.md,
"Regenerating"). Without either the script stops with an error that says what to do.
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
import ammonia_idt_driver as M  # noqa: E402

SEED = 2023462
SIZE = 100
XFMT = "%.15e"
INPUT_RTOL = 1e-8     # --check fails if a regenerated input (a seeded draw) differs by more (relative)


def draw(n=SIZE):
    rng = np.random.default_rng(SEED)
    X, _ = M.sample_feasible(n, rng)
    return np.array([[float(XFMT % v) for v in row] for row in X])


def header(desc, cols):
    return desc + "\n" + ", ".join(cols)


def log(out, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(out, "generate_log.txt"), "a") as fh:
        fh.write(line + "\n")


def run_rows(rows, out, max_minutes):
    X = draw(max(SIZE, rows.stop))
    rdir = os.path.join(out, "rows")
    os.makedirs(rdir, exist_ok=True)
    model = M.IgnitionModel()
    t_start = time.time()
    for i in range(rows.start, rows.stop):
        fn = os.path.join(rdir, f"row_{i:04d}.npz")
        if os.path.exists(fn):
            continue
        if max_minutes is not None and time.time() - t_start > 60.0 * max_minutes:
            log(out, f"stopped after {max_minutes} min; rerun to continue")
            return False
        c0 = time.process_time()
        tau, G = model.log10_tau_and_grad(X[i])
        cpu = time.process_time() - c0
        np.savez(fn + ".tmp.npz", x=X[i], tau=tau, G=G, cpu=cpu)
        os.replace(fn + ".tmp.npz", fn)
        log(out, f"row {i}: tau = {', '.join(f'{v:.4e}' for v in tau)} s; cpu {cpu:.1f} s")
    return True


def assemble(rows, out):
    rdir = os.path.join(out, "rows")
    R = [np.load(os.path.join(rdir, f"row_{i:04d}.npz")) for i in range(rows.start, rows.stop)]
    X = np.array([r["x"] for r in R])
    tau = np.array([r["tau"] for r in R])
    G = np.array([r["G"] for r in R])            # (n, 3, 123)
    cpu = np.array([float(r["cpu"]) for r in R])
    cols = [f"{n} [log10 k/k*]" for n in M.NAMES]
    np.savetxt(os.path.join(out, "inputs.csv"), X, fmt=XFMT, delimiter=",",
               header=header(f"NH3 ignition, 123 inputs x_(r,T) = log10(k_r(T)/k*_r(T)) at T = 300, "
                             f"750, 3000 K; Monte Carlo uniform on the feasible region, seed {SEED}, "
                             f"rows {rows.start}:{rows.stop}", cols))
    np.savetxt(os.path.join(out, "outputs.csv"), tau, fmt="%.15e", delimiter=",",
               header=header("ignition delay times, Cantera 3.2.0, adiabatic constant volume, "
                             "mechanism ammonia_syngas_optimized.yaml; conditions A, B, C of "
                             "ammonia_idt_driver.CONDITIONS",
                             [f"tau_{k} [s]" for k in M.QOIS]))
    for q, k in enumerate(M.QOIS):
        np.savetxt(os.path.join(out, f"gradients_{k}.csv"), G[:, q, :], fmt="%.10e", delimiter=",",
                   header=header(f"d log10(tau_{k}) / d x_j, forward differences in the node "
                                 "coordinates (step 1e-3 of each half-range)",
                                 [f"dlog10tau_{k}/d{n} [-]" for n in M.NAMES]))
    log(out, f"wrote {len(X)} rows; cpu per row mean {cpu.mean():.1f} s, total {cpu.sum() / 60:.1f} min")


def check(rows, out):
    """Print the largest difference of each file; False on a structural mismatch."""
    ld = lambda d, n: np.loadtxt(os.path.join(d, n), delimiter=",", ndmin=2)  # noqa: E731
    ok, worst = True, (0.0, "elementwise", "outputs.csv")
    for fn in ["inputs.csv", "outputs.csv"] + [f"gradients_{k}.csv" for k in M.QOIS]:
        new, old = ld(out, fn), ld(HERE, fn)[rows]
        if new.shape != old.shape or not np.array_equal(np.isnan(new), np.isnan(old)):
            print(f"  FAIL: {fn}: shape {new.shape} or NaN pattern differs from the shipped {old.shape}")
            ok = False
            continue
        if fn == "inputs.csv":
            d = np.max(np.abs(new - old)) / np.max(np.abs(old))
            print(f"  {fn}: max |new - shipped| / max |shipped| = {d:.1e}")
            if d > INPUT_RTOL:
                print(f"  FAIL: {fn} differs from the shipped inputs by {d:.1e} > {INPUT_RTOL:g}")
                ok = False
        elif fn == "outputs.csv":
            d = np.nanmax(np.abs(new / old - 1))
            print(f"  {fn}: max relative difference {d:.1e}")
            worst = max(worst, (d, "elementwise", fn))
        else:
            g1, g0 = new * M.UPPER, old * M.UPPER
            d = np.nanmax(np.linalg.norm(g1 - g0, axis=1) / np.linalg.norm(g0, axis=1))
            print(f"  {fn}: max relative 2-norm difference {d:.1e}")
            worst = max(worst, (d, "row 2-norm", fn))
    print(f"largest difference from the shipped files: {worst[0]:.1e} (relative, {worst[1]}, {worst[2]})")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None,
                   help="output directory (required without --check; with --check, default a temporary directory)")
    p.add_argument("--rows", default=None, help="a:b, a subset of the rows")
    p.add_argument("--max-minutes", type=float, default=None, help="stop after this many minutes")
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    a = p.parse_args()
    if a.out is None:
        if a.check:
            if a.max_minutes is not None:
                p.error("--max-minutes with --check needs --out DIR, so that a stopped run can be resumed")
            a.out = tempfile.mkdtemp(prefix="check_")
            atexit.register(shutil.rmtree, a.out, ignore_errors=True)
        else:
            p.error("--out DIR is required without --check")
    out = os.path.abspath(a.out)
    here = os.path.realpath(HERE)
    if os.path.commonpath([os.path.realpath(out), here]) == here:
        sys.exit("--out must be outside this data set's directory (not the directory or a folder in it), "
                 "so that the shipped files are not overwritten")
    if M.ct is None:
        print("Cantera is not installed. Install it (pip install cantera==3.2.0, the version that "
              "wrote the shipped files) and rerun generate_data.py.", file=sys.stderr)
        sys.exit(3)
    if not os.path.exists(M.MECH):
        print(M.MECH_HELP + " generate_data.py.", file=sys.stderr)
        sys.exit(3)
    os.makedirs(out, exist_ok=True)
    if a.rows:
        i, j = (int(s) for s in a.rows.split(":"))
        rows = slice(i, j)
    else:
        rows = slice(0, SIZE)
    t0 = time.time()
    done = run_rows(rows, out, a.max_minutes)
    if not done:
        if a.check:
            print("check not done: stopped by --max-minutes before all rows existed; rerun with the same --out "
                  "to continue", file=sys.stderr)
            sys.exit(1)
        return
    assemble(rows, out)
    print(f"wall time {time.time() - t0:.0f} s")
    if a.check:
        sys.exit(0 if check(rows, out) else 1)


if __name__ == "__main__":
    main()
