"""Regenerate the data files of Pulmonary_Pulse_Wave_Network (needs git and g++).

    python generate_data.py --work /tmp/pulmwk --out /tmp/pulmwk/out           # all 160 rows, about 1.1 CPU-hours
    python generate_data.py --work /tmp/pulmwk --rows 0:5 --out /tmp/pulmwk/out --check   # subset vs shipped

The solver is the authors' C++ program sor06 (repository Colebank_Identifiability_PDE,
WK_Simulator, commit pinned in pulmonary_wk_driver.COMMIT). It is licensed CC BY-NC 4.0 and is
not part of this data set: the script clones it into --work, compiles it there with g++, and runs
it as a command-line program (pass --source DIR to use an existing clone instead of cloning).

Design: a Latin hypercube of 160 points (seed 2025130, numpy default_rng) on the box of
pulmonary_wk_driver.LOWER/UPPER in the code-unit coordinates (ln k1, k2, ln k3, Rp1, Rp2, Rd1,
Rd2, CT1, CT2). Inputs are rounded to the 15 digits written before the solver runs.
Gradients: forward differences, step 1e-3 of each input's half-range (backward where the forward
point would leave the box), 9 extra solver runs per row.

Files written to --out: inputs.csv (160 x 9), outputs.csv (160 x 4: p_mean, p_sys, p_dia in
mmHg, status), gradients.csv (160 x 27: d p_mean/dx_1..9, d p_sys/dx_1..9, d p_dia/dx_1..9).
status: 0 forward differences, 1 at least one backward difference, 2 the base run failed
(outputs and gradients NaN), 3 a perturbed run failed (gradients NaN).

Each finished row is appended to --work/rows.jsonl, so an interrupted run resumes where it stopped.
--check and --rows write only to --out DIR outside this directory; --check compares with the
shipped files: outputs to 1e-9 relative (max |new - shipped| / max |shipped|), gradients to 1e-6
relative per column block (the same compiler gives identical numbers; another compiler or
platform can change the last digits of each run, which the differences amplify).
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import pulmonary_wk_driver as D  # noqa: E402


def inside(path):
    """True if path is this directory or below it (symbolic links resolved)."""
    return os.path.commonpath([os.path.realpath(path), HERE]) == HERE


SEED = 2025130
M = 160
XFMT = "%.15e"
TOL_OUT = 1e-9
TOL_GRAD = 1e-6


def design():
    rng = np.random.default_rng(SEED)
    m = len(D.NAMES)
    U = np.empty((M, m))
    for j in range(m):
        U[:, j] = (rng.permutation(M) + rng.random(M)) / M
    X = D.LOWER + (D.UPPER - D.LOWER) * U
    return np.array([[float(XFMT % v) for v in row] for row in X])


def child_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def run_rows(X, rows, exe, cache):
    done = {}
    if os.path.exists(cache):
        with open(cache) as fh:
            for line in fh:
                rec = json.loads(line)
                done[rec["row"]] = rec
    for i in rows:
        if i in done:
            continue
        c0, t0 = child_cpu(), time.time()
        f0 = D.qois(X[i], exe)
        if f0 is None:
            rec = {"row": i, "f": [np.nan] * 3, "g": [np.nan] * 27, "status": 2}
        else:
            G, flag = D.forward_gradient(X[i], f0, exe)
            rec = {"row": i, "f": f0.tolist(), "g": G.ravel().tolist(), "status": flag}
        rec["cpu_s"] = child_cpu() - c0
        rec["wall_s"] = time.time() - t0
        with open(cache, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[i] = rec
        print(f"row {i}: status {rec['status']}, {rec['cpu_s']:.1f} CPU-s", flush=True)
    return [done[i] for i in rows]


def write(out, X, recs, rows):
    F = np.array([r["f"] + [r["status"]] for r in recs], dtype=float)
    G = np.array([r["g"] for r in recs], dtype=float)
    sl = f"rows {rows.start}:{rows.stop}"
    cols = [f"{n} [{u}]" for n, u in zip(D.NAMES, D.UNITS)]
    np.savetxt(os.path.join(out, "inputs.csv"), X[rows], fmt=XFMT, delimiter=",",
               header=f"1-D pulmonary pulse-wave model with Windkessel outlets (Colebank 2025), "
                      f"Latin hypercube, seed {SEED}, {sl}; code units (see DATA.md)\n"
                      + ", ".join(cols))
    np.savetxt(os.path.join(out, "outputs.csv"), F, fmt=["%.10e"] * 3 + ["%d"], delimiter=",",
               header="MPA midpoint pressure over the last cycle: time average, maximum, minimum; "
                      "status 0/1 ok (1: a backward difference), 2 base run failed, 3 a "
                      "perturbed run failed\np_mean [mmHg], p_sys [mmHg], p_dia [mmHg], status [-]")
    gcols = [f"d{q}/d{n} [mmHg per {u}]" for q in D.QOI_NAMES for n, u in zip(D.NAMES, D.UNITS)]
    np.savetxt(os.path.join(out, "gradients.csv"), G, fmt="%.8e", delimiter=",",
               header="forward differences of (p_mean, p_sys, p_dia) in the code-unit inputs, "
                      "step 1e-3 of each half-range\n" + ", ".join(gcols))


def check(out, rows):
    for fn, tol in (("inputs.csv", TOL_OUT), ("outputs.csv", TOL_OUT), ("gradients.csv", TOL_GRAD)):
        new = np.loadtxt(os.path.join(out, fn), delimiter=",", ndmin=2)
        old = np.loadtxt(os.path.join(HERE, fn), delimiter=",", ndmin=2)[rows]
        ok = np.isfinite(old) & np.isfinite(new)
        same_nan = np.array_equal(np.isnan(old), np.isnan(new))
        if fn == "gradients.csv":
            rel = max(np.max(np.abs(new[:, b][ok[:, b]] - old[:, b][ok[:, b]]), initial=0.0)
                      / max(np.max(np.abs(old[:, b][ok[:, b]]), initial=0.0), 1e-300)
                      for b in [slice(k * 9, (k + 1) * 9) for k in range(3)])
        else:
            rel = np.max(np.abs(new[ok] - old[ok])) / np.max(np.abs(old[ok]))
        verdict = "OK" if (rel <= tol and same_nan) else "FAIL"
        print(f"  {fn}: max rel. difference {rel:.2e} (tolerance {tol:.0e}), "
              f"same failed rows: {same_nan} -> {verdict}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="directory for the solver build and row cache")
    p.add_argument("--source", default=None, help="an existing clone of the authors' repository")
    p.add_argument("--rows", default=None, help="a:b, a subset of the rows")
    p.add_argument("--out", required=True, help="output directory (pass this directory itself, explicitly, to rewrite the shipped files)")
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    a = p.parse_args()
    if (a.check or a.rows) and inside(a.out):
        sys.exit("--check and --rows need --out DIR outside this directory, so that the shipped "
                 "files are not overwritten")
    if inside(a.work):
        sys.exit("--work must be outside this directory")
    try:
        exe = D.build(a.work, a.source)
    except D.SolverMissing as e:
        print(f"{e}. The solver is the authors' sor06 (C++, cloned from {D.REPO_URL}); "
              "install git and g++ and rerun generate_data.py.", file=sys.stderr)
        sys.exit(3)
    os.makedirs(a.out, exist_ok=True)
    rows = slice(0, M)
    if a.rows:
        i, j = (int(s) for s in a.rows.split(":"))
        rows = slice(i, j)
    X = design()
    t0 = time.time()
    recs = run_rows(X, range(rows.start, rows.stop), exe, os.path.join(a.work, "rows.jsonl"))
    write(a.out, X, recs, rows)
    cpu = sum(r["cpu_s"] for r in recs)
    print(f"{len(recs)} rows: solver CPU {cpu:.0f} s (sum over the cached rows), "
          f"this call {time.time() - t0:.0f} s wall")
    if a.check:
        check(a.out, rows)


if __name__ == "__main__":
    main()
