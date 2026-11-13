"""Regenerate the data files of Minnesota_COVID19_Deaths (needs R and the article's supplement).

    python generate_data.py --code CODE_DIR --out DIR                 # all 100 samples (900 runs)
    python generate_data.py --code CODE_DIR --out DIR --chunk 60      # at most 60 runs, then stop;
                                                                      # rerun the same command to resume
    python generate_data.py --code CODE_DIR --rows 0:3 --check        # 3 samples vs the shipped files
    python generate_data.py --code CODE_DIR --check                   # all 100 samples vs the shipped files

CODE_DIR is the folder "code" of code.zip from the article's supplement
(https://doi.org/10.6084/m9.figshare.13611284). That R code (GPL-3.0, (c) Regents of the
University of Minnesota) is not part of this repository; mn_covid_eval.R only calls its
covid_sim(). R is needed: the shipped files were written with R 4.4.3 and plyr 1.8.9 (conda-forge).

Design: M = 100 Monte Carlo samples, independent uniform on [0.01, 0.99]^8 (the box of the
article's study script code/covid_sim.R), numpy default_rng(SEED). Inputs are rounded to the
written digits before the model runs, so each output belongs exactly to its stored input row.
Gradients: forward differences in the code's units with step 1e-7 (9 runs per sample).

Files written to --out: inputs.csv (M x 8), outputs.csv (M x 1, cumulative deaths at day 365),
gradients.csv (M x 8, deaths per unit of each input), and the run log evals.csv (row, deaths of
each of the 9 M runs; kept so that an interrupted run resumes). --out is required without --check
(--check without --out writes into a temporary directory and removes it) and may never be this
directory or a folder inside it, in any mode (full run, --chunk, --rows, --check), so the shipped
files are never overwritten. Without R or the supplement folder the script stops with an error
that says what to install (exit status 3).
--check prints the largest differences from the shipped rows (relative: |new - shipped| over the
largest |shipped| value of the same column among the checked rows; absolute for an all-zero
column, i.e. the gradients of inputs 4 and 5) and fails only on structural mismatches (shapes,
NaN pattern, inputs differing by more than INPUT_RTOL, or a run that did not finish).
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
import mn_covid_driver as D  # noqa: E402

SEED = 2021
M = 100
XFMT = "%.15e"


def draw():
    rng = np.random.default_rng(SEED)
    X = D.LOWER + (D.UPPER - D.LOWER) * rng.random((M, 8))
    return np.array([[float(XFMT % v) for v in row] for row in X])


def header(desc, cols):
    return desc + "\n" + ", ".join(cols)


def write(out, X, f, G, rows):
    cols = [f"{n} [{u}]" for n, u in zip(D.NAMES, D.UNITS)]
    np.savetxt(os.path.join(out, "inputs.csv"), X, fmt=XFMT, delimiter=",", header=header(
        f"Minnesota COVID-19 model (Wycoff et al. 2021, Sec. 4.3), 8 inputs, Monte Carlo uniform on "
        f"[0.01, 0.99]^8, seed {SEED}, rows {rows.start}:{rows.stop}; code units (fractions)", cols))
    np.savetxt(os.path.join(out, "outputs.csv"), f[:, None], fmt="%.17g", delimiter=",", header=header(
        "cumulative COVID-19 deaths in Minnesota at day 365, scenario 2 (covid_sim in covid_allinone.R, R)",
        ["deaths [persons]"]))
    np.savetxt(os.path.join(out, "gradients.csv"), G, fmt="%.10e", delimiter=",", header=header(
        f"d deaths / d x_j, forward differences in the code units, step {D.FD_STEP:g}",
        [f"d deaths/d {n} [persons per {u}]" for n, u in zip(D.NAMES, D.UNITS)]))


def inside_here(path):
    """True if path is this data set's directory or a folder inside it (symlinks resolved)."""
    out = os.path.realpath(path)
    here = os.path.realpath(HERE)
    return out == here or out.startswith(here + os.sep)


INPUT_RTOL = 1e-8  # --check fails if the regenerated inputs differ by more than this (relative)


def check(out, rows):
    """Print the largest differences from the shipped rows; exit 1 on a structural mismatch."""
    ok, worst = True, (0.0, "")
    for fn in ["inputs.csv", "outputs.csv", "gradients.csv"]:
        new = np.loadtxt(os.path.join(out, fn), delimiter=",", ndmin=2)
        old = np.loadtxt(os.path.join(HERE, fn), delimiter=",", ndmin=2)[rows]
        if new.shape != old.shape:
            print(f"  {fn}: shape {new.shape} differs from the shipped {old.shape}: FAIL")
            ok = False
            continue
        if not np.array_equal(np.isnan(new), np.isnan(old)):
            print(f"  {fn}: NaN pattern differs from the shipped file: FAIL")
            ok = False
        scale = np.nanmax(np.abs(old), axis=0)
        scale[scale == 0] = 1.0
        rel = np.nanmax(np.abs(new - old) / scale, axis=0)
        print(f"  {fn}: largest |new - shipped| / max |shipped column|, per column: "
              + " ".join(f"{r:.1e}" for r in rel))
        if fn == "inputs.csv" and np.max(np.abs(new - old) / np.abs(old)) > INPUT_RTOL:   # elementwise
            print(f"  {fn}: inputs differ by more than {INPUT_RTOL:g}: FAIL")
            ok = False
        worst = max(worst, (float(rel.max()), fn))
    print(f"largest difference from the shipped files: {worst[0]:.1e} (relative, {worst[1]})")
    if not ok:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", default=None, help="the supplement folder 'code' (or set MN_COVID_CODE)")
    p.add_argument("--out", default=None,
                   help="output directory (required without --check; with --check, default a temporary directory)")
    p.add_argument("--rows", default=None, help="a:b, a subset of the samples")
    p.add_argument("--chunk", type=int, default=None, help="at most this many model runs, then stop")
    p.add_argument("--rscript", default=None, help="the Rscript executable (default: on the PATH)")
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    a = p.parse_args()
    if a.out is None:
        if a.check:
            a.out = tempfile.mkdtemp(prefix="check_")
            atexit.register(shutil.rmtree, a.out, ignore_errors=True)
        else:
            p.error("--out DIR is required without --check")
    if inside_here(a.out):
        sys.exit("--out must be a directory outside this data set's directory (in every mode), so that "
                 "the shipped files are never overwritten")
    try:
        code = D.find_code(a.code)
        rscript = D.find_rscript(a.rscript)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(3)
    os.makedirs(a.out, exist_ok=True)
    rows = slice(0, M)
    if a.rows:
        i, j = (int(s) for s in a.rows.split(":"))
        rows = slice(i, j)
    X = draw()[rows]
    P = D.fd_points(X)
    log = os.path.join(a.out, "evals.csv")
    t0 = time.time()
    y = D.run_batch(P, log, code, rscript, max_new=a.chunk)
    dt = time.time() - t0
    done = int(np.sum(np.isfinite(y)))
    rv, pv = D.versions(rscript)
    print(f"{rv}, plyr {pv}; {done} of {len(P)} runs done ({dt:.0f} s in this call)")
    if done < len(P):
        print("incomplete: rerun the same command to continue")
        if a.check:
            print("--check did not finish the rows it was asked to check: FAIL")
            sys.exit(1)
        sys.exit(3)
    f, G = D.fd_assemble(y)
    write(a.out, X, f, G, rows)
    print(f"wrote inputs.csv, outputs.csv, gradients.csv ({len(X)} samples) to {a.out}")
    if a.check:
        check(a.out, rows)


if __name__ == "__main__":
    main()
