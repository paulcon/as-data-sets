"""Regenerate the data files of Parcel_Droplet_Activation (needs pyrcel 2.0.0).

    python generate_data.py --out DIR                       # all 300 rows (about 1.4 CPU-hours)
    python generate_data.py --check                         # all 300 rows in a temporary directory, compared with the shipped files
    python generate_data.py --rows 0:10 --check             # a subset, compared with the shipped files

Design: a Latin hypercube of M = 300 points in the Table 1 box of Rothenberg and Wang (2016,
p. 1259), in the paper's coordinates (log10 N, log10 mu_g, sigma_g, kappa, log10 V, T, P,
alpha_c), drawn with numpy (seed SEED): in each column a random permutation of the 300 strata
and a uniform position inside each stratum. Inputs are rounded to the 16 significant digits
written before the model runs, so the files are consistent with each other.

Each row with log10 mu_g >= MU_GRAD = -2.5 (mu_g >= 3.16 nm) is one pyrcel run with
forward-mode automatic-differentiation gradients (pyrcel_parcel_driver.run_with_grad). Below
MU_GRAD only the value is computed (pyrcel_parcel_driver.run): there the smallest bins have dry
radii below 0.3 nm, the derivative solve failed (a non-finite linear solve) in 4 of 5 trial
rows, and a run costs 10 to 100 times more. If the derivative solve fails above
MU_GRAD, the gradient is computed by central differences of the value
(pyrcel_parcel_driver.fd_gradient, step 1e-3 of each half-range; grad_method = 2). Rows
without gradients have nan in the gradient columns and grad_ok = 0. Rows are appended to
DIR/rows.jsonl as they finish, so an interrupted run resumes where it stopped; the CSV files
are written when every requested row is done.

The script never writes into its own directory: --out is required without --check and must be outside
it (not a subdirectory either; copy the finished CSVs in by hand); with --check it defaults to a temporary
directory, removed at the end. --check prints the largest differences from the shipped files (absolute on
log10 S_max and the activated fraction, relative 2-norm per row on the gradients scaled to [-1, 1]) and fails
only on structural mismatches (shapes, nan entries, flags, inputs); the shipped files are the record.

Needs pyrcel 2.0.0 (pip install pyrcel==2.0.0, Python >= 3.11); without it the script stops
with an error naming what to install.
"""
import argparse
import atexit
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pyrcel_parcel_driver as D  # noqa: E402

SEED = 448_2016
M = 300
MU_GRAD = -2.5     # gradients only for log10 mu_g >= MU_GRAD (mu_g >= 3.16 nm); below, values only
OUT_COLS = ["log10_smax [log10 (decimal supersaturation)]", "act_frac [-]", "peaked [1/0]",
            "grad_ok [1/0]", "grad_method [0 none, 1 AD, 2 FD]", "z_smax [m]", "S_400m [decimal]",
            "solver_steps [-]"]


def design():
    rng = np.random.default_rng(SEED)
    U = np.empty((M, 8))
    for j in range(8):
        U[:, j] = (rng.permutation(M) + rng.random(M)) / M
    X = D.LOWER + (D.UPPER - D.LOWER) * U
    return np.array([[float("%.15e" % v) for v in row] for row in X])


def one(x):
    t0 = time.process_time()
    if x[1] < MU_GRAD:
        r = D.run(x)
        if not r["ok"]:
            raise D.ParcelError(f"pyrcel run failed at x = {list(x)}")
        nan = [float("nan")] * 8
        return {"log10_smax": r["log10_smax"], "act_frac": r["act_frac"], "peaked": r["peaked"],
                "grad_ok": 0, "grad_method": 0, "z_smax": r["z_smax"], "S_end": r["S_end"], "steps": r["steps"],
                "grad": nan, "grad_af": nan, "cpu_s": time.process_time() - t0}
    method = 1
    try:
        r = D.run_with_grad(x)
        g, ga = r["grad_log10_smax"], r["grad_act_frac"]
        if not (r["ok"] and np.all(np.isfinite(g)) and np.all(np.isfinite(ga))):
            raise D.ParcelError("non-finite derivative")
    except Exception:  # derivative solve failed: central finite differences of the value
        Gfd, r = D.fd_gradient(x)
        g, ga = Gfd["log10_smax"], Gfd["act_frac"]
        method = 2
    if not r["ok"]:
        raise D.ParcelError(f"pyrcel run failed at x = {list(x)}")
    gok = int(np.all(np.isfinite(g)) and np.all(np.isfinite(ga)))
    if not gok:
        g = ga = np.full(8, np.nan)
        method = 0
    return {"log10_smax": r["log10_smax"], "act_frac": r["act_frac"], "peaked": r["peaked"],
            "grad_ok": gok, "grad_method": method, "z_smax": r["z_smax"], "S_end": r["S_end"], "steps": r["steps"],
            "grad": [float(v) for v in g], "grad_af": [float(v) for v in ga],
            "cpu_s": time.process_time() - t0}


def header(desc, cols):
    return desc + "\n" + ", ".join(cols)


def write(out, X, rows, res):
    cols = [f"{n} [{u}]" for n, u in zip(D.NAMES, D.UNITS)]
    sl = f"rows {rows.start}:{rows.stop}"
    np.savetxt(os.path.join(out, "inputs.csv"), X[rows], fmt="%.15e", delimiter=",",
               header=header(f"Parcel droplet activation, 8 inputs (Rothenberg and Wang 2016, Table 1), "
                             f"Latin hypercube M = {M}, seed {SEED}, {sl}; paper coordinates", cols))
    O = np.array([[r["log10_smax"], r["act_frac"], r["peaked"], r["grad_ok"], r["grad_method"],
                   r["z_smax"], r["S_end"], r["steps"]] for r in res])
    np.savetxt(os.path.join(out, "outputs.csv"), O, fmt=["%.15e", "%.15e", "%d", "%d", "%d", "%.10e", "%.15e", "%d"],
               delimiter=",", header=header(
                   f"pyrcel {D.PYRCEL_VERSION} adiabatic parcel, 200 bins, S0 = 0; S_max = max S over the first "
                   f"400 m of ascent; act_frac from Eq. 10; {sl}", OUT_COLS))
    for key, fn, what in (("grad", "gradients.csv", "dlog10_smax"), ("grad_af", "act_frac_gradients.csv", "dact_frac")):
        G = np.array([r[key] for r in res])
        np.savetxt(os.path.join(out, fn), G, fmt="%.10e", delimiter=",", header=header(
            f"{what}/dx_j in the paper coordinates: forward-mode automatic differentiation (JAX), or central "
            f"differences where grad_method = 2; nan where grad_ok = 0; {sl}",
            [f"{what}/d{n} [per {u}]" for n, u in zip(D.NAMES, D.UNITS)]))


INPUT_RTOL = 1e-8   # --check fails if a regenerated input (a seeded draw) differs by more (relative)


def check(out, rows):
    """Print the largest difference of each file; False on a structural mismatch."""
    ld = lambda d, n: np.loadtxt(os.path.join(d, n), delimiter=",", ndmin=2)  # noqa: E731
    hr = (D.UPPER - D.LOWER) / 2
    passed, worst = True, (0.0, "absolute", "outputs.csv")
    for fn in ("inputs.csv", "outputs.csv", "gradients.csv", "act_frac_gradients.csv"):
        new, old = ld(out, fn), ld(HERE, fn)[rows]
        if new.shape != old.shape:
            print(f"  FAIL: {fn}: shape {new.shape} differs from the shipped {old.shape}")
            passed = False
            continue
        if fn == "inputs.csv":       # the design is a seeded draw
            d = np.max(np.abs(new - old) / np.abs(old).max(axis=0))
            print(f"  {fn}: max |new - shipped| / max |shipped| per column = {d:.2e}")
            if d > INPUT_RTOL:
                print(f"  FAIL: {fn} differs from the shipped inputs by {d:.2e} > {INPUT_RTOL:g}")
                passed = False
            continue
        if fn == "outputs.csv":
            if not np.array_equal(np.isnan(new[:, :2]), np.isnan(old[:, :2])):
                print(f"  FAIL: {fn}: nan entries of log10 S_max or act_frac differ from the shipped file")
                passed = False
                continue
            d = np.nanmax(np.abs(new[:, :2] - old[:, :2]))
            flags = bool(np.all(new[:, 2:5] == old[:, 2:5]))
            print(f"  {fn}: max |new - shipped| = {d:.2e}; flags equal: {flags}")
            if not flags:
                print(f"  FAIL: {fn}: the flags peaked, grad_ok or grad_method differ from the shipped file")
                passed = False
            if d >= worst[0]:
                worst = (d, "absolute", fn)
            continue
        new, old = new * hr, old * hr
        if not np.array_equal(np.isfinite(new), np.isfinite(old)):
            print(f"  FAIL: {fn}: nan entries differ from the shipped file")
            passed = False
            continue
        ok = np.all(np.isfinite(old), axis=1)
        e = np.linalg.norm(new[ok] - old[ok], axis=1) / np.maximum(np.linalg.norm(old[ok], axis=1), 1e-300)
        emax = e.max() if ok.any() else 0.0
        print(f"  {fn}: max relative 2-norm difference per row = {emax:.2e} ({ok.sum()} rows with gradients)")
        if emax >= worst[0]:
            worst = (emax, "relative, row 2-norm", fn)
    print(f"largest difference from the shipped files: {worst[0]:.1e} ({worst[1]}, {worst[2]})")
    return passed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None,
                   help="output directory (required without --check; with --check, default a temporary directory)")
    p.add_argument("--rows", default=None, help="a:b, a subset of the rows")
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    a = p.parse_args()
    if a.out is None:
        if a.check:
            a.out = tempfile.mkdtemp(prefix="check_")
            atexit.register(shutil.rmtree, a.out, ignore_errors=True)
        else:
            p.error("--out DIR is required without --check")
    out = os.path.abspath(a.out)
    if os.path.commonpath([os.path.realpath(out), os.path.realpath(HERE)]) == os.path.realpath(HERE):
        sys.exit("--out must be a directory outside this one (not this directory or a subdirectory of it), "
                 "so that the shipped files are not overwritten")
    if D.pm is None:
        print("pyrcel is not installed. Install it (pip install pyrcel==2.0.0, the version that wrote "
              "the shipped files) and rerun generate_data.py.", file=sys.stderr)
        sys.exit(3)
    os.makedirs(out, exist_ok=True)
    import jax
    jax.config.update("jax_compilation_cache_dir", os.path.join(out, "jax_cache"))
    if a.rows:
        i, j = (int(s) for s in a.rows.split(":"))
        rows = slice(i, j)
    else:
        rows = slice(0, M)
    X = design()
    log = os.path.join(out, "rows.jsonl")
    done = {}
    if os.path.exists(log):
        with open(log) as fh:
            for line in fh:
                d = json.loads(line)
                done[d["row"]] = d
    t0 = time.process_time()
    for k in range(rows.start, rows.stop):
        if k in done:
            continue
        d = one(X[k])
        d["row"] = k
        with open(log, "a") as fh:
            fh.write(json.dumps(d) + "\n")
        done[k] = d
        print(f"row {k}: log10 S_max = {d['log10_smax']:.4f}, grad_ok = {d['grad_ok']}, "
              f"{d['cpu_s']:.1f} s CPU", flush=True)
    res = [done[k] for k in range(rows.start, rows.stop)]
    write(out, X, rows, res)
    print(f"{len(res)} rows; CPU time of these rows {sum(r['cpu_s'] for r in res):.0f} s "
          f"(this call {time.process_time() - t0:.0f} s)")
    if a.check and not check(out, rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
