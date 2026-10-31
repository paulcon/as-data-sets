"""Regenerate the data files of H2_O2_Ignition_Delay_Kinetics (needs Cantera; tested 3.2.0).

    python generate_data.py                          # all files, 2 processes (about 15 min)
    python generate_data.py --case 19 --rows 0:10 --out /tmp/h2 --check   # subset vs shipped files

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
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h2o2_ignition_model as M  # noqa: E402

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


def check(case, out, rows):
    pre = "mc_" if case == "mc36" else ""
    suf = "36" if case == "mc36" else case
    files = [f"{pre}inputs_{suf}.csv", f"{pre}outputs_{suf}.csv"]
    if case != "mc36":
        files.append(f"gradients_{suf}.csv")
    for fn in files:
        new = np.loadtxt(os.path.join(out, fn), delimiter=",", ndmin=2)
        old = np.loadtxt(os.path.join(HERE, fn), delimiter=",", ndmin=2)[rows]
        rel = np.max(np.abs(new - old)) / np.max(np.abs(old))
        print(f"  {fn}: max |new - shipped| / max |shipped| = {rel:.2e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="all", choices=["all", "19", "36", "mc36"])
    p.add_argument("--rows", default=None, help="a:b, a subset of the rows")
    p.add_argument("--out", default=HERE)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cases = ["19", "36", "mc36"] if a.case == "all" else [a.case]
    total = 0.0
    for c in cases:
        if a.rows:
            i, j = (int(s) for s in a.rows.split(":"))
            rows = slice(i, j)
        else:
            rows = slice(0, SIZES[c])
        total += run_case(c, rows, a.out, a.workers)
        if a.check:
            check(c, a.out, rows)
    print(f"total {total:.1f} s")


if __name__ == "__main__":
    main()
