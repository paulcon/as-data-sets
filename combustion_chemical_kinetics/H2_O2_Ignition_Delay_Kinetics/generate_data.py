"""Regenerate the data files of H2_O2_Ignition_Delay_Kinetics (needs Cantera 3.2.0).

    python generate_data.py                          # all files, 2 processes (about 15 min)
    python generate_data.py --check                  # all files in a temporary directory, vs shipped files
    python generate_data.py --case 19 --rows 0:10 --check   # subset vs shipped files

Cases
  19    1000 Monte Carlo samples of the 19 lnA_i, uniform on [0.97, 1.03] lnA*_i (paper Sec. 4);
        inputs_19.csv, outputs_19.csv (tau), gradients_19.csv (dtau/dlnA_i).
  36    1000 Monte Carlo samples of the 36 inputs, uniform on +/- 2 % (paper Sec. 5);
        inputs_36.csv, outputs_36.csv, gradients_36.csv.
  mc36  10 000 further Monte Carlo samples of the 36 inputs, tau only (the paper's
        cross-validation set, Table 2); mc_inputs_36.csv, mc_outputs_36.csv.

Gradients: central differences in the physical inputs with step 1e-3 of each input's
half-range. Inputs are rounded to the digits written before the model is run, so the files
are consistent with each other.

Needs Cantera (the shipped files were written with Cantera 3.2.0: pip install cantera==3.2.0)
through h2o2_ignition_driver.py; without it the script stops with an error. --check and --rows
write only to --out DIR outside this directory, so the shipped files are never overwritten (with
--check, a temporary directory by default). --check prints the largest differences from the shipped files
(max |new - shipped| / max |shipped| per file) and fails only on structural mismatches (shapes, NaN pattern,
inputs).
"""
import argparse
import atexit
import os
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h2o2_ignition_driver as M  # noqa: E402

SEEDS = {"19": 2019019, "36": 2019036, "mc36": 2019100}
SIZES = {"19": 1000, "36": 1000, "mc36": 10000}
XFMT = {"19": "%.15e", "36": "%.15e", "mc36": "%.12e"}


def space(case):
    if case == "19":
        return M.NAMES19, M.LOWER19, M.UPPER19, ["ln(cm-mol-s)"] * 19
    return M.NAMES36, M.LOWER36, M.UPPER36, M.UNITS36


def draw(case):
    names, lo, up, _ = space(case)
    rng = np.random.default_rng(SEEDS[case])
    X = lo + (up - lo) * rng.random((SIZES[case], len(names)))
    # round to the written digits, so the outputs belong to the stored inputs exactly
    return np.array([[float(XFMT[case] % v) for v in row] for row in X])


_model = None


def _init():
    global _model
    _model = M.IgnitionModel()


def _work(args):
    case, x, with_grad = args
    f = _model.tau19 if case == "19" else _model.tau36
    steps = M.STEPS19 if case == "19" else M.STEPS36
    tau = f(x)
    g = _model.grad(f, x, steps) if with_grad else None
    return tau, g


def header(desc, cols):
    return desc + "\n" + ", ".join(cols)


def run_case(case, rows, out, workers):
    names, lo, up, units = space(case)
    X = draw(case)[rows]
    with_grad = case != "mc36"
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        res = pool.map(_work, [(case, x, with_grad) for x in X], chunksize=4)
    dt = time.time() - t0
    tau = np.array([r[0] for r in res])
    cols = [f"{n} [{u}]" for n, u in zip(names, units)]
    pre = "mc_" if case == "mc36" else ""
    suf = "36" if case == "mc36" else case
    np.savetxt(os.path.join(out, f"{pre}inputs_{suf}.csv"), X, fmt=XFMT[case], delimiter=",",
               header=header(f"H2/O2 ignition, {len(names)} inputs, Monte Carlo uniform, seed "
                             f"{SEEDS[case]}, rows {rows.start}:{rows.stop}; physical units", cols))
    np.savetxt(os.path.join(out, f"{pre}outputs_{suf}.csv"), tau[:, None], fmt="%.15e",
               delimiter=",", header=header(
                   "ignition delay tau (time at which T = T0 + 400 K), Cantera, "
                   "mechanism h2o2_li2004.yaml", ["tau [s]"]))
    if with_grad:
        G = np.array([r[1] for r in res])
        np.savetxt(os.path.join(out, f"gradients_{suf}.csv"), G, fmt="%.10e", delimiter=",",
                   header=header("dtau/dx_j, central differences in the physical inputs "
                                 "(step 1e-3 of each half-range)",
                                 [f"dtau/d{n} [s per {u}]" for n, u in zip(names, units)]))
    print(f"case {case}: {len(X)} samples in {dt:.1f} s on {workers} processes")
    return dt


INPUT_RTOL = 1e-8   # --check fails if a regenerated input (a seeded draw) differs by more (relative)


def check(case, out, rows):
    """Print max |new - shipped| / max |shipped| per file; return (False on a structural mismatch, the largest
    output difference and its file)."""
    pre = "mc_" if case == "mc36" else ""
    suf = "36" if case == "mc36" else case
    files = [f"{pre}inputs_{suf}.csv", f"{pre}outputs_{suf}.csv"]
    if case != "mc36":
        files.append(f"gradients_{suf}.csv")
    ok, worst = True, (0.0, files[1])
    for fn in files:
        new = np.loadtxt(os.path.join(out, fn), delimiter=",", ndmin=2)
        old = np.loadtxt(os.path.join(HERE, fn), delimiter=",", ndmin=2)[rows]
        if new.shape != old.shape or not np.array_equal(np.isnan(new), np.isnan(old)):
            print(f"  FAIL: {fn}: shape {new.shape} or NaN pattern differs from the shipped {old.shape}")
            ok = False
            continue
        rel = np.nanmax(np.abs(new - old)) / np.nanmax(np.abs(old))
        print(f"  {fn}: max |new - shipped| / max |shipped| = {rel:.2e}")
        if "inputs" in fn:
            if rel > INPUT_RTOL:
                print(f"  FAIL: {fn} differs from the shipped inputs by {rel:.2e} > {INPUT_RTOL:g}")
                ok = False
        elif rel >= worst[0]:
            worst = (rel, fn)
    return ok, worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="all", choices=["all", "19", "36", "mc36"])
    p.add_argument("--rows", default=None, help="a:b, a subset of the rows")
    p.add_argument("--out", default=None,
                   help="output directory (default: this directory; with --check, a temporary directory)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    a = p.parse_args()
    if a.out is None:
        if a.check:
            a.out = tempfile.mkdtemp(prefix="check_")
            atexit.register(shutil.rmtree, a.out, ignore_errors=True)
        else:
            a.out = HERE
    if (a.check or a.rows) and os.path.abspath(a.out) == os.path.abspath(HERE):
        sys.exit('--check and --rows need --out DIR outside this directory, so that the shipped files are not overwritten')
    if M.ct is None:
        print('Cantera is not installed. Install it (pip install cantera==3.2.0, the version that wrote '
              'the shipped files) and rerun generate_data.py.', file=sys.stderr)
        sys.exit(3)
    os.makedirs(a.out, exist_ok=True)
    cases = ["19", "36", "mc36"] if a.case == "all" else [a.case]
    total = 0.0
    ok, worst = True, (0.0, "")
    for c in cases:
        if a.rows:
            i, j = (int(s) for s in a.rows.split(":"))
            rows = slice(i, j)
        else:
            rows = slice(0, SIZES[c])
        total += run_case(c, rows, a.out, a.workers)
        if a.check:
            ok_c, worst_c = check(c, a.out, rows)
            ok = ok and ok_c
            worst = max(worst, worst_c)
    print(f"total {total:.1f} s")
    if a.check:
        print(f"largest difference from the shipped files: {worst[0]:.1e} (relative to the largest value, {worst[1]})")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
