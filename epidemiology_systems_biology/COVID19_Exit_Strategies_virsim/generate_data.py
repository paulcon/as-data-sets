"""Regenerate the virsim runs of COVID19_Exit_Strategies_virsim (needs R and virsim 1.0.5).

    python generate_data.py --out DIR --design-only          # design files only (numpy + scipy)
    python generate_data.py --strategy FC --rows 0:3 --check           # rerun rows, compare
    python generate_data.py --check                          # all 2288 runs, compared with the shipped rows
    python generate_data.py --out DIR --jobs 16              # all 2288 runs (about 64 process-hours)

Step 1 (numpy + scipy) writes DIR/design/design_S.csv for S in FC, CT, IL, PO: 512 main
rows from a scrambled Sobol' sequence (scipy.stats.qmc.Sobol, seed 568 + strategy index)
mapped through the inverse CDFs of the paper's Table 1 (p. 4), each with its own seed of
virsim's random-number generator, drawn without replacement from 16384..65536 (p. 10);
then 6 replicate groups of 10 rows with new seeds (group 0 at the Table 1 means, group g
at main row g - 1). It compares the inputs and seeds with the shipped runs_S.csv
(inputs to a relative INPUT_RTOL, seeds exactly) and stops with --design-only.

Step 2 runs R/run_chunk.R (which sources R/strategies.R and calls virsim) on chunks of
at most 4 rows, --jobs processes at a time, writing DIR/results/S_rows_AAAA-BBBB.csv and
..._ts.csv. Finished chunks are skipped, so an interrupted run resumes.

Step 3 merges a strategy whose 572 rows are all done into DIR/runs_S.csv and
DIR/ic_prev_S.csv with convert_data.py. With --check, the rerun rows are compared with
the shipped rows: --check prints the largest differences of the QoIs (relative, with a floor of 1
on the shipped value) and fails only on structural mismatches (a missing chunk or row, a nonzero
run status, different seeds, or inputs differing by more than INPUT_RTOL). With the R and package
versions of the shipped run (R 4.3.3, virsim 1.0.5, data.table 1.14.10, Rcpp 1.0.12,
RcppArmadillo 0.12.8.1.0) the QoIs are expected to agree to the last digit; other builds may draw
another realization of the stochastic model for the same seed, which the check then reports.

virsim is an R package, not a Python model: it must be installed (R_LIBS pointing at
it) before step 2. Without it the script stops after step 1 and says what to install.
--out is required without --check and may not be this directory or inside it; --check
without --out writes into a temporary directory and removes it.
"""
import argparse
import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import convert_data as C  # noqa: E402

SEED = 568                       # design seed; strategy S uses SEED + index of S
SEED_LO, SEED_HI = 2 ** 14, 2 ** 16   # virsim seeds, inclusive (paper p. 10)
CHUNK = 4
VIRSIM_VERSION = "1.0.5"
INPUT_RTOL = 1e-8                # --check fails if the regenerated inputs differ by more than this (relative)

# Table 1, p. 4 (scipy parametrization: gamma(shape, scale=scale), beta(alpha, beta))
DIST = {
    "intervention_effect": stats.beta(38, 70),
    "uptake": stats.beta(16, 2),
    "trace_rate_I": stats.gamma(2, scale=0.2),
    "trace_prob_E": stats.beta(2, 6),
    "trace_contact_reduction": stats.beta(10, 2),
    "lock_length": stats.gamma(20, scale=2),
    "lift_length": stats.gamma(15, scale=1),
    "lock_effect": stats.beta(14, 42),
    "intervention_lift_interval": stats.gamma(25, scale=2),
    "pl_intervention_effect_hi": stats.beta(14, 42),
    "avg_duration_infectiousness": stats.gamma(25, scale=0.2),
    "R0": stats.gamma(100, scale=0.025),
    "intervention_effect_var_inv": stats.gamma(2, scale=0.05),
    "shape_exposed_time": stats.gamma(17.5, scale=1),
}

INSTALL_MSG = (
    "virsim {v} (an R package) is not installed or not found.\n"
    "Install R 4.3.3 and virsim {v} (on Ubuntu 24.04 x86-64: bash install_ubuntu2404.sh "
    "virsim-v1.0.5.tar.gz INSTALL_DIR, which builds virsim into INSTALL_DIR/Rlib; INSTALL_DIR outside "
    "this data set's directory), set R_LIBS to that library "
    "(export R_LIBS=INSTALL_DIR/Rlib OMP_NUM_THREADS=1) and rerun this script. The virsim source is "
    "at gitlab.com/luccoffeng/virsim, tag v1.0.5 (see DATA.md)."
).format(v=VIRSIM_VERSION)


def design(s):
    """Return the design rows (as an array) and column names of strategy s."""
    offset = C.STRATEGIES.index(s)
    names = C.inputs(s)
    m = len(names)
    sob = qmc.Sobol(d=m, scramble=True, seed=SEED + offset)
    u = sob.random_base2(int(np.log2(C.N_MAIN)))
    x_main = np.column_stack([DIST[n].ppf(u[:, j]) for j, n in enumerate(names)])
    x_mean = np.array([DIST[n].mean() for n in names])
    rng = np.random.default_rng(SEED + 100 + offset)
    seeds = rng.permutation(np.arange(SEED_LO, SEED_HI + 1))[: C.N_ROWS]
    rows = [[i, 0, -1, -1, seeds[i]] + list(x_main[i]) for i in range(C.N_MAIN)]
    r = C.N_MAIN
    for g in range(C.GROUPS):
        x, src = (x_mean, -1) if g == 0 else (x_main[g - 1], g - 1)
        for _ in range(C.REPS):
            rows.append([r, 1, g, src, seeds[r]] + list(x))
            r += 1
    return np.array(rows, dtype=float), ["row", "kind", "group", "source_row", "seed"] + names


def write_design(out, s):
    A, cols = design(s)
    names = cols[5:]
    units = ["-"] * 5 + [C.UNITS[n] for n in names]
    h1 = (f"virsim v1.0.5 design, strategy={s}, m={len(names)}, n_main={C.N_MAIN} (scrambled Sobol' "
          f"mapped through Table 1 inverse CDFs, seed {SEED + C.STRATEGIES.index(s)}), "
          f"replicates={C.GROUPS}x{C.REPS}; kind 0 = main, 1 = replicate; source_row = main row "
          f"repeated (-1 = Table 1 mean point or main row)")
    h2 = ", ".join(f"{c} [{u}]" for c, u in zip(cols, units))
    path = os.path.join(out, f"design_{s}.csv")
    np.savetxt(path, A, delimiter=",", fmt=["%d"] * 5 + ["%.17g"] * len(names),
               header=h1 + "\n" + h2, comments="# ")
    return path, A


def compare_design(s, A):
    shipped = np.loadtxt(os.path.join(HERE, f"runs_{s}.csv"), delimiter=",")
    m = len(C.inputs(s))
    if len(A) != len(shipped):
        print(f"{s}: design has {len(A)} rows, the shipped runs_{s}.csv {len(shipped)}: FAIL")
        return False
    same_int = np.array_equal(A[:, :5], shipped[:, :5])
    rel = np.max(np.abs(A[:, 5:5 + m] - shipped[:, 5:5 + m]) / np.abs(shipped[:, 5:5 + m]))
    ok = same_int and rel <= INPUT_RTOL   # scipy's inverse CDFs differ by a few ulp between platforms
    print(f"{s}: design vs shipped runs_{s}.csv: rows/kind/group/source_row/seed "
          f"{'equal' if same_int else 'DIFFER'}; inputs largest relative difference {rel:.1e}"
          f"{'' if ok else ': FAIL'}")
    return ok


def virsim_available(env):
    rs = shutil.which("Rscript")
    if rs is None:
        return False, "Rscript not found on PATH"
    try:
        p = subprocess.run([rs, "-e", 'cat(as.character(packageVersion("virsim")))'],
                           capture_output=True, text=True, env=env, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    if p.returncode != 0:
        return False, "R cannot load virsim"
    if p.stdout.strip() != VIRSIM_VERSION:
        return False, f"virsim {p.stdout.strip()} found, {VIRSIM_VERSION} needed"
    return True, rs


def run_chunks(rs, env, out, jobs):
    res = os.path.join(out, "results")
    os.makedirs(res, exist_ok=True)

    def one(job):
        s, i, j = job
        cmd = [rs, os.path.join(HERE, "R", "run_chunk.R"),
               os.path.join(out, "design", f"design_{s}.csv"), str(i), str(j), res]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        with open(os.path.join(res, "run.log"), "a") as fh:
            fh.write(p.stdout + p.stderr)
        return p.returncode

    return one, res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None,
                    help="output folder, outside this directory (required without --check; with --check, "
                         "default a temporary directory)")
    ap.add_argument("--strategy", choices=C.STRATEGIES, nargs="*", default=C.STRATEGIES)
    ap.add_argument("--rows", default=None, help="a:b, rows a..b inclusive (default all 0:571)")
    ap.add_argument("--jobs", type=int, default=1, help="R processes at a time")
    ap.add_argument("--check", action="store_true", help="compare rerun rows with the shipped files")
    ap.add_argument("--design-only", action="store_true", help="write and check the design only")
    a = ap.parse_args()
    if a.out is None:
        if a.check:
            a.out = tempfile.mkdtemp(prefix="check_")
            atexit.register(shutil.rmtree, a.out, ignore_errors=True)
        else:
            ap.error("--out DIR is required without --check")
    out = os.path.realpath(a.out)
    here = os.path.realpath(HERE)
    if os.path.commonpath([out, here]) == here:
        sys.exit("--out must be outside this data set's directory (not the directory or a folder in it); "
                 "the shipped files are never overwritten")
    first, last = (0, C.N_ROWS - 1) if a.rows is None else map(int, a.rows.split(":"))
    if not 0 <= first <= last < C.N_ROWS:
        sys.exit(f"--rows must lie in 0:{C.N_ROWS - 1}")

    os.makedirs(os.path.join(out, "design"), exist_ok=True)
    ok = True
    for s in a.strategy:
        _, A = write_design(os.path.join(out, "design"), s)
        ok &= compare_design(s, A)
    if a.design_only:
        sys.exit(0 if ok else 1)
    if not ok:
        sys.exit("the design differs from the shipped inputs; not running virsim")

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    found, rs = virsim_available(env)
    if not found:
        print(f"{rs}.\n{INSTALL_MSG}", file=sys.stderr)
        sys.exit(3)

    jobs = [(s, i, min(i + CHUNK - 1, last)) for s in a.strategy for i in range(first, last + 1, CHUNK)]
    one, res = run_chunks(rs, env, out, a.jobs)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        codes = list(ex.map(one, jobs))
    print(f"{len(jobs)} chunks in {time.time() - t0:.0f} s on {a.jobs} processes; "
          f"{sum(c != 0 for c in codes)} chunks exited with an error (see {res}/run.log)")

    worst = (0.0, "")
    for s in a.strategy:
        if first == 0 and last == C.N_ROWS - 1:
            table, series, _ = C.merge(res, s)
            C.write(out, s, table, series)
        if a.check:
            d, same = check_rows(res, s, first, last)
            ok &= same
            worst = max(worst, (d, f"runs_{s}.csv"))
    if a.check:
        print(f"largest difference from the shipped files: {worst[0]:.1e} (relative, {worst[1]})")
        sys.exit(0 if ok else 1)


def check_rows(res, s, first, last):
    """Print the largest QoI difference of the rerun rows; return (difference, False on a structural mismatch)."""
    shipped = np.loadtxt(os.path.join(HERE, f"runs_{s}.csv"), delimiter=",")
    cols, _ = C.out_columns(s)
    ci = {c: i for i, c in enumerate(cols)}
    new = {}
    for i in range(first, last + 1, CHUNK):
        j = min(i + CHUNK - 1, last)
        path = os.path.join(res, f"{s}_rows_{i:04d}-{j:04d}.csv")
        if not os.path.exists(path):
            print(f"  {s}: {os.path.basename(path)} missing: FAIL")
            return 0.0, False
        rc, rows = C.read_text_rows(path)
        for r in rows:
            new[int(r[0])] = {c: float(v) for c, v in zip(rc, r)}
    ok, worst = True, 0.0
    for k in range(first, last + 1):
        if k not in new:
            print(f"  {s} row {k}: missing from the rerun: FAIL")
            ok = False
            continue
        r, old = new[k], shipped[k]
        if r["status"] != 0:
            print(f"  {s} row {k}: status {r['status']:.0f}: FAIL")
            ok = False
            continue
        if r["seed"] != old[ci["seed"]] or any(
                abs(r[n] - old[ci[n]]) > INPUT_RTOL * abs(old[ci[n]]) for n in C.inputs(s)):
            print(f"  {s} row {k}: seed or inputs differ from the shipped row: FAIL")
            ok = False
        for q in ("IC_prev_avg_max", "IC_excess_days"):
            worst = max(worst, abs(r[q] - old[ci[q]]) / max(1.0, abs(old[ci[q]])))
    print(f"  {s} rows {first}-{last}: QoIs vs shipped, largest relative difference {worst:.1e} "
          "(|new - shipped| / max(1, |shipped|))")
    return worst, ok

if __name__ == "__main__":
    main()
