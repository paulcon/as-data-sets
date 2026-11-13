"""Regenerate the data files of European_LV_Feeder_Probabilistic_Power_Flow (needs OpenDSSDirect.py).

    python generate_data.py --out DIR                                   # all files into DIR
    python generate_data.py --check                                     # all files vs shipped files
    python generate_data.py --case 55 --rows 0:20 --check               # subset vs shipped files

The script never writes into this directory: --out DIR is required without --check and must be
outside this directory (not the directory or a folder in it); --check without --out writes into
a temporary directory and removes it.

Cases (IEEE European LV Test Feeder at 09:26, opendss_driver.py)
  55     1000 Monte Carlo samples of the 55 load active powers, P_i ~ N(P_Li, 0.1^2) kW,
         power factor 0.95 (the paper's base scenario); inputs_55.csv, outputs_55.csv and
         grad_<QoI>_55.csv for the QoIs V899A, I78A, PLOSS.
  110    1000 Monte Carlo samples of the 55 active powers (as above) and the 55 power factors,
         pf_i ~ U(0.85, 0.95); inputs_110.csv, outputs_110.csv, grad_<QoI>_110.csv.
  lhs55  10 000 Latin hypercube samples of the 55-input case (uniform LHS mapped through the
         normal inverse CDF), QoIs only; lhs_inputs_55.csv, lhs_outputs_55.csv.
  nominal  the QoIs and the three phase voltages at bus 899 at P = P_L, with pf 0.95 (row 1)
         and pf 0.90 (row 2, the mean of U(0.85, 0.95)); nominal.csv.

Gradients: central differences in the physical inputs with step 0.01 of each input's standard
deviation (1e-3 kW for P, 2.887e-4 for pf). Inputs are rounded to the written digits (10
decimals) before the model runs, so the outputs belong to the stored inputs.

--check prints the largest differences from the shipped files (relative: max |new - shipped| /
max |shipped| of each file; input files elementwise) and fails only on structural mismatches (shapes, NaN pattern, inputs
differing by more than INPUT_RTOL).

Needs OpenDSSDirect.py (the shipped files were written with OpenDSSDirect.py 0.9.4:
pip install opendssdirect.py==0.9.4), and scipy for the lhs55 case.
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
import opendss_driver as M  # noqa: E402

SEEDS = {"55": 2017055, "110": 2017110, "lhs55": 2017100}
SIZES = {"55": 1000, "110": 1000, "lhs55": 10000, "nominal": 2}
XFMT = "%.10f"
INPUT_RTOL = 1e-8  # --check fails if the regenerated inputs differ by more than this (relative)
INPUT_FILES = ("inputs_55.csv", "inputs_110.csv", "lhs_inputs_55.csv")

def draw(case):
    rng = np.random.default_rng(SEEDS[case])
    n = SIZES[case]
    if case == "55":
        X = M.MEAN55 + M.SD55 * rng.standard_normal((n, M.NLOAD))
    elif case == "110":
        P = M.P_REF + M.SD_P * rng.standard_normal((n, M.NLOAD))
        pf = M.PF_LO + (M.PF_HI - M.PF_LO) * rng.random((n, M.NLOAD))
        X = np.hstack([P, pf])
    else:
        from scipy.stats import norm, qmc
        U = qmc.LatinHypercube(d=M.NLOAD, rng=rng).random(n)
        X = M.MEAN55 + M.SD55 * norm.ppf(U)
    return np.array([[float(XFMT % v) for v in row] for row in X])


def names_units(case):
    if case == "110":
        return M.NAMES110, M.UNITS110
    return M.NAMES55, M.UNITS55


def files(case):
    if case == "nominal":
        return ["nominal.csv"]
    if case == "lhs55":
        return ["lhs_inputs_55.csv", "lhs_outputs_55.csv"]
    return ([f"inputs_{case}.csv", f"outputs_{case}.csv"]
            + [f"grad_{q}_{case}.csv" for q in M.QOI_NAMES])


def run_nominal(out, log):
    model = M.FeederModel()
    F = []
    for pf in (M.PF_NOM, 0.5 * (M.PF_LO + M.PF_HI)):
        f = model.solve(M.P_REF, np.full(M.NLOAD, pf))
        M.dss.Circuit.SetActiveBus("899")
        vm = M.dss.Bus.VMagAngle()[0::2]
        nodes = M.dss.Bus.Nodes()
        F.append([pf, f[0], vm[nodes.index(2)], vm[nodes.index(3)], f[1], f[2]])
    np.savetxt(os.path.join(out, "nominal.csv"), np.array(F), fmt="%.12e", delimiter=",",
               header="IEEE European LV Test Feeder at 09:26 (OpenDSS), every load at P = P_Li; row 1 pf 0.95, "
                      "row 2 pf 0.90\npf [-], V899A [V], V899B [V], V899C [V], I78A [A], PLOSS [kW]")
    log("case nominal: 2 points")
    return 0.0


def run_case(case, rows, out, log):
    if case == "nominal":
        return run_nominal(out, log)
    model = M.FeederModel()
    names, units = names_units(case)
    X = draw(case)[rows]
    fun = model.qoi110 if case == "110" else model.qoi55
    steps = M.STEPS110 if case == "110" else M.STEPS55
    t0 = time.time()
    F = np.empty((len(X), 3))
    G = np.empty((3, len(X), len(names)))
    for k, x in enumerate(X):
        F[k] = fun(x)
        if case != "lhs55":
            G[:, k, :] = model.grad(fun, x, steps)
        if (k + 1) % 100 == 0:
            log(f"  case {case}: {k + 1}/{len(X)} samples, {time.time() - t0:.1f} s")
    dt = time.time() - t0
    desc = {"55": "1000 Monte Carlo samples, P_i ~ N(P_Li, 0.1^2) kW, pf 0.95",
            "110": "1000 Monte Carlo samples, P_i ~ N(P_Li, 0.1^2) kW, pf_i ~ U(0.85, 0.95)",
            "lhs55": "10000 Latin hypercube samples, P_i ~ N(P_Li, 0.1^2) kW, pf 0.95"}[case]
    tag = f"IEEE European LV Test Feeder at 09:26 (OpenDSS); {desc}; seed {SEEDS[case]}; rows {rows.start}:{rows.stop}"
    fn = files(case)
    np.savetxt(os.path.join(out, fn[0]), X, fmt=XFMT, delimiter=",",
               header=tag + "; physical units\n" + ", ".join(f"{n} [{u}]" for n, u in zip(names, units)))
    np.savetxt(os.path.join(out, fn[1]), F, fmt="%.12e", delimiter=",",
               header="QoIs: phase-A voltage magnitude at bus 899; phase-A current in LINE78 at bus 72; "
                      "total circuit active power loss\n"
               + ", ".join(f"{q} [{u}]" for q, u in zip(M.QOI_NAMES, M.QOI_UNITS)))
    if case != "lhs55":
        for q, (qn, qu) in enumerate(zip(M.QOI_NAMES, M.QOI_UNITS)):
            np.savetxt(os.path.join(out, fn[2 + q]), G[q], fmt="%.10e", delimiter=",",
                       header=f"d{qn}/dx_j, central differences in the physical inputs (step 0.01 sd); "
                              f"rows as {fn[0]}\n"
                       + ", ".join(f"d{qn}/d{n} [{qu} per {u}]" for n, u in zip(names, units)))
    log(f"case {case}: {len(X)} samples in {dt:.1f} s")
    return dt


def check(case, out, rows, log):
    """Print the largest differences of a case's new files from the shipped rows.

    Returns (largest difference, its file, False on a structural mismatch)."""
    ok, worst = True, (0.0, "")
    for fn in files(case):
        new = np.loadtxt(os.path.join(out, fn), delimiter=",", ndmin=2)
        old = np.loadtxt(os.path.join(HERE, fn), delimiter=",", ndmin=2)[rows]
        if new.shape != old.shape:
            log(f"  {fn}: shape {new.shape} differs from the shipped {old.shape} FAIL")
            ok = False
            continue
        if not np.array_equal(np.isnan(new), np.isnan(old)):
            log(f"  {fn}: NaN pattern differs from the shipped file FAIL")
            ok = False
        if fn in INPUT_FILES:
            rel = float(np.max(np.abs(new - old) / np.maximum(np.abs(old), 1e-300)))
            log(f"  {fn}: largest relative difference (elementwise) = {rel:.2e}")
        else:
            rel = float(np.nanmax(np.abs(new - old), initial=0.0) / np.nanmax(np.abs(old)))
            log(f"  {fn}: largest difference, max |new - shipped| / max |shipped| = {rel:.2e}")
        if fn in INPUT_FILES and rel > INPUT_RTOL:
            log(f"  {fn}: inputs differ by more than {INPUT_RTOL:g} FAIL")
            ok = False
        worst = max(worst, (rel, fn))
    return worst[0], worst[1], ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="all", choices=["all", "55", "110", "lhs55", "nominal"])
    p.add_argument("--rows", default=None, help="a:b, a subset of the rows")
    p.add_argument("--out", default=None,
                   help="output directory (required without --check; with --check, default a temporary directory)")
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    a = p.parse_args()
    if a.out is None:
        if a.check:
            a.out = tempfile.mkdtemp(prefix="check_")
            atexit.register(shutil.rmtree, a.out, ignore_errors=True)
        else:
            p.error("--out DIR is required without --check")
    out = os.path.abspath(a.out)
    here = os.path.realpath(HERE)
    if os.path.commonpath([os.path.realpath(out), here]) == here:
        sys.exit("--out must be outside this data set's directory (not the directory or a folder in it), "
                 "so that the shipped files are not overwritten")
    if M.dss is None:
        print(f"OpenDSSDirect.py is not installed. Install it ({M.INSTALL}, the version that "
              "wrote the shipped files; scipy is also needed) and rerun generate_data.py.", file=sys.stderr)
        sys.exit(3)
    os.makedirs(a.out, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    cases = ["55", "110", "lhs55", "nominal"] if a.case == "all" else [a.case]
    total, ok, worst = 0.0, True, (0.0, "")
    for c in cases:
        if a.rows and c != "nominal":
            i, j = (int(s) for s in a.rows.split(":"))
            rows = slice(i, j)
        else:
            rows = slice(0, SIZES[c])
        total += run_case(c, rows, a.out, log)
        if a.check:
            d, fn, same = check(c, a.out, rows, log)
            ok &= same
            worst = max(worst, (d, fn))
    log(f"total {total:.1f} s")
    if a.check:
        log(f"largest difference from the shipped files: {worst[0]:.1e} (relative, {worst[1]})")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
