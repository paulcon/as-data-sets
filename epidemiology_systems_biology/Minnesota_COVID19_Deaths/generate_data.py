"""Regenerate the data files of Minnesota_COVID19_Deaths (needs R and the article's supplement).

    python generate_data.py --code CODE_DIR --out DIR                 # all 100 samples (900 runs)
    python generate_data.py --code CODE_DIR --out DIR --chunk 60      # at most 60 runs, then stop;
                                                                      # rerun the same command to resume
    python generate_data.py --code CODE_DIR --out DIR --rows 0:3 --check   # 3 samples vs the shipped files

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
each of the 9 M runs; kept so that an interrupted run resumes). --out is always required and may
never be this directory or a folder inside it, in any mode (full run, --chunk, --rows, --check),
so the shipped files are never overwritten.
--check compares entry by entry with the shipped rows: |new - shipped| <= 1e-6 x the largest
|shipped| value of the same column among the checked rows (an absolute 1e-6 for an all-zero
column, i.e. the gradients of inputs 4 and 5).
"""
import argparse
import os
import sys
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


RTOL = 1e-6


def check(out, rows):
    for fn in ["inputs.csv", "outputs.csv", "gradients.csv"]:
        new = np.loadtxt(os.path.join(out, fn), delimiter=",", ndmin=2)
        old = np.loadtxt(os.path.join(HERE, fn), delimiter=",", ndmin=2)[rows]
        scale = np.max(np.abs(old), axis=0)
        scale[scale == 0] = 1.0
        rel = np.max(np.abs(new - old) / scale, axis=0)
        print(f"  {fn}: max |new - shipped| / max |shipped column|, per column: "
              + " ".join(f"{r:.1e}" for r in rel) + f" (tolerance {RTOL:g})")
        if not np.all(rel <= RTOL):
            sys.exit(f"{fn} differs from the shipped file by more than the tolerance")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", default=None, help="the supplement folder 'code' (or set MN_COVID_CODE)")
    p.add_argument("--out", required=True)
    p.add_argument("--rows", default=None, help="a:b, a subset of the samples")
    p.add_argument("--chunk", type=int, default=None, help="at most this many model runs, then stop")
    p.add_argument("--rscript", default=None, help="the Rscript executable (default: on the PATH)")
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    a = p.parse_args()
    if inside_here(a.out):
        sys.exit("--out must be a directory outside this data set's directory (in every mode), so that "
                 "the shipped files are never overwritten")
    try:
        code = D.find_code(a.code)
        rscript = D.find_rscript(a.rscript)
    except RuntimeError as e:
        sys.exit(str(e))
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
        sys.exit(3)
    f, G = D.fd_assemble(y)
    write(a.out, X, f, G, rows)
    print(f"wrote inputs.csv, outputs.csv, gradients.csv ({len(X)} samples) to {a.out}")
    if a.check:
        check(a.out, rows)


if __name__ == "__main__":
    main()
