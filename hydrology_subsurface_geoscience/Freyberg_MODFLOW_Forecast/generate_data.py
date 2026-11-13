"""Regenerate the data files of Freyberg_MODFLOW_Forecast (needs MODFLOW-2005 1.12.00, mf2005dbl).

    python generate_data.py --out DIR                      # all 500 rows (about 1 h on one core)
    python generate_data.py --check                        # all rows, in a temporary directory, compared
    python generate_data.py --out DIR --rows 0:5 --check   # a subset, compared with the shipped files

Design: 500 Monte Carlo samples, independent and uniform in log10 of the 71 inputs between the
bounds of the PEST control file (recharge 5e-5 to 5e-4 m/d; the 70 pilot-point K 0.5 to 50 m/d),
numpy default_rng seed 602988. Inputs are rounded to the written digits before the model runs.

For each sample: one MODFLOW run (calibration heads at 16 targets, forecast heads at 6 pumping
wells after the river conductance is cut 100-fold, dry-cell count) and 71 more for forward
differences of the 6 forecasts in log10 of each input (step 1e-4; freyberg_driver.FD_STEP).
A sample where MODFLOW does not converge (at the point or in its stencil) is stored as nan with
dry-cell counts -1.

Then the calibrated model of the release (the PEST control file's parameter values): its 16
calibration heads, river flux and 6 forecasts, and their first-order (linear) uncertainty with
the prior standard deviation (log10 upper - log10 lower bound)/4 for every input and observation
standard deviations 1/weight (calibrated.csv; 143 runs, central differences with step 1e-3).

Files written to DIR: inputs.csv, outputs.csv, gradients.csv, calibrated.csv (and chunks/ with
the partial results, so an interrupted run resumes where it stopped). DIR must be outside this directory,
so the shipped files are never overwritten; --check without --out uses a temporary directory and removes
it. --check compares the rows computed with the shipped rows and prints the largest differences from the
shipped files (heads absolute in m; gradients in relative 2-norm per row; calibrated.csv relative). It
fails (exit status 1) only on structural mismatches: shapes, NaN pattern, dry-cell counts, or inputs
differing by more than INPUT_RTOL of their largest magnitude.

Needs numpy and the double-precision MODFLOW-2005 1.12.00 executable (mf2005dbl on the PATH or
MF2005DBL=/path/to/mf2005dbl); without it the script exits with status 3 and says what to install.
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
import freyberg_driver as M  # noqa: E402

SEED = 602988
N = 500
CHUNK = 25
XFMT = "%.17e"
INPUT_RTOL = 1e-8   # inputs are deterministic draws: a larger difference means a different seed or range


def draw():
    rng = np.random.default_rng(SEED)
    X = M.LOWER_LOG10 + (M.UPPER_LOG10 - M.LOWER_LOG10) * rng.random((N, len(M.NAMES)))
    return np.array([[float(XFMT % v) for v in row] for row in X])


def run_rows(X, i, j, chunkdir):
    fn = os.path.join(chunkdir, "rows_%04d_%04d.npz" % (i, j))
    if os.path.exists(fn):
        return fn
    wd = tempfile.mkdtemp(prefix="frey_gen_")
    C, F, D, DS, G = [], [], [], [], []
    t0 = time.time()
    try:
        for r in range(i, j):
            try:
                c, f, nd, ndmax, g = M.sample(X[r], workdir=wd)
            except RuntimeError as e:      # MODFLOW did not converge at the sample or in its stencil
                print("row %d: %s; stored as nan with dry-cell counts -1" % (r, e), flush=True)
                c, f, nd, ndmax = np.full(16, np.nan), np.full(6, np.nan), -1, -1
                g = np.full((6, len(M.NAMES)), np.nan)
            C.append(c)
            F.append(f)
            D.append(nd)
            DS.append(ndmax)
            G.append(g.reshape(-1))
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    np.savez(fn + ".tmp.npz", C=np.array(C), F=np.array(F), D=np.array(D), DS=np.array(DS),
             G=np.array(G), rows=np.arange(i, j))
    os.replace(fn + ".tmp.npz", fn)
    print("rows %d:%d done in %.1f s" % (i, j, time.time() - t0), flush=True)
    return fn


def pst_obs():
    """Observation values and weights of the PEST control file, in targpest order."""
    import re
    pst = os.path.join(M.FILES, "figure5b_from_model_of_table6a_pilot_points_heads_too_highBEST.pst")
    obs = {}
    with open(pst) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 4 and p[3] in ("calib_head", "fore_head", "calib_flux"):
                obs[p[0].lower()] = (float(p[1]), float(p[2]))
    names = [re.sub(r"r0", "r", M.TARGET_NAMES[k].lower()) for k in M.CALIB] + ["flux_obs"]
    names += M.FORECAST_NAMES
    return names, np.array([obs[n] for n in names])


def river_flux(d):
    """Net river leakage (m3/d) in stress period 1 from the double-precision budget file."""
    raw = open(os.path.join(d, M.NAME + ".cbb"), "rb").read()
    off, n = 0, M.NROW * M.NCOL
    while off < len(raw):
        kper = int(np.frombuffer(raw, "<i4", 2, off)[1])
        text = raw[off + 8:off + 24].decode().strip()
        if kper == 1 and text == "RIVER LEAKAGE":
            return float(np.frombuffer(raw, "<f8", n, off + 36).sum())
        off += 36 + 8 * n
    raise RuntimeError("no river leakage record in the budget file")


def calibrated(out, h=1e-3):
    """The calibrated model of the release and its linear (FOSM) forecast uncertainty."""
    wd = tempfile.mkdtemp(prefix="frey_cal_")
    try:
        def obsvec(x):
            c, f, _ = M.run(x, workdir=wd)
            return np.r_[c, river_flux(wd), f]
        xb = M.X_BEST_LOG10
        y0 = obsvec(xb)
        J = np.empty((len(y0), len(xb)))
        for i in range(len(xb)):
            e = np.zeros(len(xb))
            e[i] = h
            J[:, i] = (obsvec(xb + e) - obsvec(xb - e)) / (2 * h)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    names, ov = pst_obs()
    sd_prior = (M.UPPER_LOG10 - M.LOWER_LOG10) / 4.0
    Cp = np.diag(sd_prior ** 2)
    Jo = J[:17]
    Ce = np.diag(1.0 / ov[:17, 1] ** 2)
    Cpost = Cp - Cp @ Jo.T @ np.linalg.solve(Jo @ Cp @ Jo.T + Ce, Jo @ Cp)
    pri = np.sqrt(np.einsum("ij,jk,ik->i", J, Cp, J))
    post = np.sqrt(np.einsum("ij,jk,ik->i", J, Cpost, J))
    np.savetxt(os.path.join(out, "calibrated.csv"), np.c_[ov[:, 0], ov[:, 1], y0, pri, post],
               fmt="%.12e", delimiter=",",
               header=header("calibrated pilot-point model (PEST control file values); rows: %s; "
                             "linear uncertainty with prior sd (log10 upper - log10 lower)/4, central "
                             "differences step %g, observation sd 1/weight (16 heads, flux)"
                             % (" ".join(names), h),
                             ["observed or true value [m or m3/d]", "PEST weight [1/m or d/m3]",
                              "model value [m or m3/d]", "prior sd [m or m3/d]",
                              "posterior sd [m or m3/d]"]))
    print("calibrated.csv written", flush=True)


def header(desc, cols):
    return desc + "\n" + ", ".join(cols)


def write(out, X, parts, i0, i1):
    C = np.concatenate([p["C"] for p in parts])
    F = np.concatenate([p["F"] for p in parts])
    D = np.concatenate([p["D"] for p in parts])
    DS = np.concatenate([p["DS"] for p in parts])
    G = np.concatenate([p["G"] for p in parts])
    in_cols = ["log10_R [log10(m/d)]"] + ["log10_%s [log10(m/d)]" % n for n in M.NAMES[1:]]
    np.savetxt(os.path.join(out, "inputs.csv"), X[i0:i1], fmt=XFMT, delimiter=",",
               header=header("Freyberg pilot-point MODFLOW-2005 model, 71 inputs: log10 recharge and "
                             "log10 K at 70 pilot points; Monte Carlo, uniform in log10 between the PEST "
                             "bounds, seed %d, rows %d:%d" % (SEED, i0, i1), in_cols))
    fc = ["h_%s [m]" % n for n in M.FORECAST_NAMES]
    cc = ["h_%s [m]" % M.TARGET_NAMES[k] for k in M.CALIB]
    np.savetxt(os.path.join(out, "outputs.csv"), np.c_[F, C, D, DS], fmt="%.12e", delimiter=",",
               header=header("forecast heads at the 6 pumping wells (stress period 2, river conductance "
                             "/100), calibration heads at the 16 targets (stress period 1), dry active "
                             "cells at the sample and the largest count over its difference stencil; "
                             "nan = next to a dry cell; counts -1 = MODFLOW did not converge",
                             fc + cc + ["n_dry [cells]", "n_dry_stencil_max [cells]"]))
    gc = ["dh_%s/dlog10_%s [m]" % (f, n) for f in M.FORECAST_NAMES for n in M.NAMES]
    np.savetxt(os.path.join(out, "gradients.csv"), G, fmt="%.10e", delimiter=",",
               header=header("gradients of the 6 forecast heads with respect to log10 of the 71 inputs "
                             "(m per log10 unit), forward differences, step %g; forecast-major order "
                             "(71 columns per forecast)" % M.FD_STEP, gc))


def check(out, i0, i1, ship=HERE):
    """Print the largest differences from the shipped rows; False on a structural mismatch."""
    ok, diffs = True, []
    new_o = np.loadtxt(os.path.join(out, "outputs.csv"), delimiter=",", ndmin=2)
    old_o = np.loadtxt(os.path.join(ship, "outputs.csv"), delimiter=",", ndmin=2)[i0:i1]
    if new_o.shape != old_o.shape:
        print("  FAIL: outputs.csv: shape %s differs from the shipped rows %s" % (new_o.shape, old_o.shape))
        return False
    same_nan = np.array_equal(np.isnan(new_o), np.isnan(old_o))
    d = np.nanmax(np.abs(new_o[:, :22] - old_o[:, :22])) if same_nan and not np.isnan(old_o[:, :22]).all() else 0.0
    same_dry = np.array_equal(new_o[:, 22:], old_o[:, 22:])
    print("  outputs.csv: largest |new - shipped| head = %.2e m; nan pattern equal: %s; dry counts equal: %s"
          % (d, same_nan, same_dry))
    if not (same_nan and same_dry):
        print("  FAIL: outputs.csv: the NaN pattern or the dry-cell counts differ from the shipped rows")
        ok = False
    diffs.append((d, "absolute in m", "outputs.csv"))
    new_g = np.loadtxt(os.path.join(out, "gradients.csv"), delimiter=",", ndmin=2)
    old_g = np.loadtxt(os.path.join(ship, "gradients.csv"), delimiter=",", ndmin=2)[i0:i1]
    if new_g.shape != old_g.shape or not np.array_equal(np.isnan(new_g), np.isnan(old_g)):
        print("  FAIL: gradients.csv: shape or nan pattern differs from the shipped rows")
        return False
    fin = np.all(np.isfinite(old_g), axis=1) & np.all(np.isfinite(new_g), axis=1)
    if fin.any():
        rel = np.max(np.linalg.norm(new_g[fin] - old_g[fin], axis=1)
                     / np.linalg.norm(old_g[fin], axis=1))
        print("  gradients.csv: largest relative 2-norm difference per row = %.2e" % rel)
        diffs.append((rel, "relative", "gradients.csv"))
    new_x = np.loadtxt(os.path.join(out, "inputs.csv"), delimiter=",", ndmin=2)
    old_x = np.loadtxt(os.path.join(ship, "inputs.csv"), delimiter=",", ndmin=2)[i0:i1]
    if new_x.shape != old_x.shape:
        print("  FAIL: inputs.csv: shape %s differs from the shipped rows %s" % (new_x.shape, old_x.shape))
        return False
    dx = np.max(np.abs(new_x - old_x)) / np.max(np.abs(old_x))
    print("  inputs.csv: largest |new - shipped| / largest |shipped| = %.1e" % dx)
    if dx > INPUT_RTOL:
        print("  FAIL: inputs.csv: the inputs differ from the shipped ones by more than %g (relative)" % INPUT_RTOL)
        ok = False
    diffs.append((dx, "relative", "inputs.csv"))
    new_c = np.loadtxt(os.path.join(out, "calibrated.csv"), delimiter=",")
    old_c = np.loadtxt(os.path.join(ship, "calibrated.csv"), delimiter=",")
    if new_c.shape != old_c.shape or not np.array_equal(np.isnan(new_c), np.isnan(old_c)):
        print("  FAIL: calibrated.csv: shape or nan pattern differs from the shipped file")
        return False
    rc = np.nanmax(np.abs(new_c - old_c) / np.maximum(np.abs(old_c), 1e-12))
    print("  calibrated.csv: largest elementwise relative difference = %.2e" % rc)
    diffs.append((rc, "relative", "calibrated.csv"))
    d, measure, name = max(diffs, key=lambda t: t[0])
    print("largest difference from the shipped files: %.1e (%s, %s)" % (d, measure, name))
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="output directory, outside this directory "
                   "(required without --check; with --check, default: a temporary directory)")
    p.add_argument("--rows", default=None, help="a:b, a subset of the rows")
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    a = p.parse_args()
    if a.out is None:
        if a.check:
            a.out = tempfile.mkdtemp(prefix="check_")
            atexit.register(shutil.rmtree, a.out, ignore_errors=True)
        else:
            p.error("--out DIR is required without --check")
    here, out = os.path.realpath(HERE), os.path.realpath(a.out)
    if os.path.commonpath([here, out]) == here:      # this directory, a subdirectory, or a link into it
        sys.exit("--out must be a directory outside this data set, so the shipped files are not "
                 "overwritten")
    try:
        M.find_mf2005()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(3)
    i0, i1 = (0, N) if a.rows is None else (int(s) for s in a.rows.split(":"))
    chunkdir = os.path.join(out, "chunks")
    os.makedirs(chunkdir, exist_ok=True)
    X = draw()
    t0 = time.time()
    files = []
    for i in range(i0, i1, CHUNK):
        files.append(run_rows(X, i, min(i + CHUNK, i1), chunkdir))
    parts = [np.load(f) for f in files]
    write(out, X, parts, i0, i1)
    if a.rows is None or a.check:
        calibrated(out)
    print("rows %d:%d written to %s in %.1f s" % (i0, i1, out, time.time() - t0))
    if a.check and not check(out, i0, i1):
        sys.exit(1)


if __name__ == "__main__":
    main()
