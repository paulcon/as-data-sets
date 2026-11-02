"""Generate inputs.csv, weights.csv, outputs.csv and fd_gradients_meanQ.csv by running Raven.

Usage (see DATA.md):
  python generate_data.py --raven /path/Raven.exe --obs /path/xSSA/examples/data_in/data_obs/08KC001 \
      --cache /tmp/raven_cache --rows 0:1024  # run (or resume) a block of rows
  python generate_data.py ... --assemble    # write the CSVs here when every row is done
  python generate_data.py ... --rows 0:8 --check   # rerun a subset, compare with the CSVs here (writes nothing)
  python generate_data.py ... --fd          # rewrite fd_gradients_meanQ.csv (rows 0, 1, 2)

--cache defaults to a folder in the system temp directory, outside the repository.
Without Raven (--raven / RAVEN_EXE) and the xSSA streamflow folder (--obs / XSSA_OBS_DIR)
every mode except --assemble exits with an install message.

The Raven input deck is filled from the xSSA template file (LGPL-3.0), which
raven_salmon_driver.fetch_templates downloads once from GitHub at a pinned commit into
--templates (default: <cache>/templates) and checks against its SHA-256. It is
not part of this repository.

The design is deterministic (scrambled Sobol', seed below), so every run draws
the same 4096 inputs. Rows are run one at a time in one process; each finished
block of BLOCK rows is cached as an .npz, so a stopped run resumes.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import raven_salmon_driver as rs  # noqa: E402

SEED = 20201202          # scrambled Sobol' seed
N_SAMPLES = 4096         # 2^12
BLOCK = 128              # rows per cache file
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
OUT_NAMES = ["mean_Q", "NSE", "KGE"] + [f"Q_{m}" for m in MONTHS]
FD_ROWS = (0, 1, 2)      # design rows of the finite-difference check
FD_STEP = 1e-3           # step in normalized units (fraction of each half-range)
INSTALL = ("Raven is not available. Build commit cb745e1 of github.com/CSHS-CWRA/RavenHydroFramework as "
           "described in DATA.md (\"Raven build\"), clone github.com/julemai/xSSA for the forcing and "
           "streamflow files, set RAVEN_EXE and XSSA_OBS_DIR (or pass --raven and --obs), and rerun this script.")


def design():
    u = qmc.Sobol(d=rs.M_INPUTS, scramble=True, seed=SEED).random(N_SAMPLES)
    return rs.LOWER + u * (rs.UPPER - rs.LOWER)


def one_run(x, raven, obs, workdir, templates):
    """All outputs for one row: mean_Q, NSE, KGE and the 12 monthly mean flows."""
    tmp = tempfile.mkdtemp(prefix="raven_", dir=workdir)
    try:
        stem = rs.write_deck(x, tmp, templates)
        os.symlink(os.path.abspath(obs), os.path.join(tmp, rs.BASIN["id"]))
        out = os.path.join(tmp, "output")
        os.makedirs(out)
        p = subprocess.run([os.path.abspath(raven), stem, "-o", out + "/"], cwd=tmp,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        hyd = os.path.join(out, "Hydrographs.csv")
        if p.returncode != 0 or not os.path.exists(hyd):
            raise RuntimeError("Raven failed: " + p.stdout.decode()[-2000:])
        _, dates, qsim, qobs = rs.read_hydrograph(hyd)
        s = rs.summaries(dates, qsim, qobs)
        diag = rs.read_diagnostics(os.path.join(out, "Diagnostics.csv"))
        if abs(diag["DIAG_NASH_SUTCLIFFE"] - s["NSE"]) > 1e-5 * max(1.0, abs(s["NSE"])):
            raise RuntimeError("own NSE disagrees with Raven's Diagnostics.csv")
        i0, i1 = dates.index(rs.EVAL_START), dates.index(rs.END) + 1
        mon = np.array([int(d[5:7]) for d in dates[i0:i1]])     # month of the row date
        qm = [qsim[i0:i1][mon == k].mean() for k in range(1, 13)]
        warn = 0
        err = os.path.join(out, "Raven_errors.txt")
        if os.path.exists(err):
            warn = sum(1 for ln in open(err) if ln.startswith("WARNING"))
        return np.array([s["mean_Q"], s["NSE"], s["KGE"]] + qm), warn
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_rows(X, rows, args, templates):
    os.makedirs(args.cache, exist_ok=True)
    work = os.path.join(args.cache, "tmp")
    os.makedirs(work, exist_ok=True)
    for b0 in range(rows.start - rows.start % BLOCK, rows.stop, BLOCK):
        path = os.path.join(args.cache, f"block_{b0:05d}.npz")
        if os.path.exists(path):
            continue
        t = time.time()
        Y = np.empty((BLOCK, len(OUT_NAMES)))
        W = np.zeros(BLOCK, dtype=int)
        for k in range(BLOCK):
            Y[k], W[k] = one_run(X[b0 + k], args.raven, args.obs, work, templates)
        np.savez(path, Y=Y, warnings=W, seconds=time.time() - t)
        print(f"block {b0}:{b0 + BLOCK} done in {time.time() - t:.0f} s", flush=True)


def fmt_header(desc, names):
    return desc + "\n" + ",".join(names)


def assemble(X, args):
    Y = np.empty((N_SAMPLES, len(OUT_NAMES)))
    warn = np.zeros(N_SAMPLES, dtype=int)
    secs = 0.0
    for b0 in range(0, N_SAMPLES, BLOCK):
        d = np.load(os.path.join(args.cache, f"block_{b0:05d}.npz"))
        Y[b0:b0 + BLOCK], warn[b0:b0 + BLOCK] = d["Y"], d["warnings"]
        secs += float(d["seconds"])
    Wt = np.array([rs.weights(x) for x in X])
    in_names = [f"{n} [{u}]" for n, u in zip(rs.NAMES, rs.UNITS)]
    np.savetxt(os.path.join(HERE, "inputs.csv"), X, delimiter=",", fmt="%.17g",
               header=fmt_header(f"Raven Salmon River 08KC001: {N_SAMPLES} scrambled Sobol' points (seed {SEED}) "
                                 "on the 43-input box of Mai et al. (2020) Table C2; physical units", in_names))
    np.savetxt(os.path.join(HERE, "weights.csv"), Wt, delimiter=",", fmt="%.17g",
               header=fmt_header("13 process-option weights derived from r1..r8 of inputs.csv by the "
                                 "Moeini et al. (2011) transform (paper Eq. A1); not independent inputs",
                                 rs.WEIGHT_NAMES))
    out_names = ["mean_Q [m3/s]", "NSE [-]", "KGE [-]"] + [f"Q_{m} [m3/s]" for m in MONTHS]
    np.savetxt(os.path.join(HERE, "outputs.csv"), Y, delimiter=",", fmt="%.10g",
               header=fmt_header("Raven v3.0 (commit cb745e1) simulated streamflow at 08KC001, 1991-01-01..2010-12-31: "
                                 "mean daily flow, NSE and KGE vs. observed, mean flow per calendar month",
                                 out_names))
    print(f"wrote CSVs; Raven warnings in {np.count_nonzero(warn)} runs; total block time {secs:.0f} s")


def fd_gradients(X, args, templates, h=FD_STEP):
    """Central differences of mean_Q in normalized units (one-sided where the
    step would leave the box), 86 Raven runs per row."""
    work = os.path.join(args.cache, "tmp")
    os.makedirs(work, exist_ok=True)
    half = (rs.UPPER - rs.LOWER) / 2
    G = np.empty((len(FD_ROWS), rs.M_INPUTS))
    t = time.time()
    for a, row in enumerate(FD_ROWS):
        x0 = X[row]
        for j in range(rs.M_INPUTS):
            up = min(h, (rs.UPPER[j] - x0[j]) / half[j])
            dn = min(h, (x0[j] - rs.LOWER[j]) / half[j])
            xp, xm = x0.copy(), x0.copy()
            xp[j] += up * half[j]
            xm[j] -= dn * half[j]
            yp, _ = one_run(xp, args.raven, args.obs, work, templates)
            ym, _ = one_run(xm, args.raven, args.obs, work, templates)
            G[a, j] = (yp[0] - ym[0]) / (up + dn)
        print(f"row {row}: |grad| = {np.linalg.norm(G[a]):.6g} ({time.time() - t:.0f} s)", flush=True)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raven", default=os.environ.get("RAVEN_EXE"))
    ap.add_argument("--obs", default=os.environ.get("XSSA_OBS_DIR"))
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "raven_salmon_cache"))
    ap.add_argument("--templates", default=None,
                    help="folder for the downloaded xSSA template file (default <cache>/templates)")
    ap.add_argument("--rows", default=f"0:{N_SAMPLES}")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fd", action="store_true", help="write fd_gradients_meanQ.csv (rows 0, 1, 2)")
    args = ap.parse_args()
    X = design()
    if args.assemble:
        assemble(X, args)
        return
    if not args.raven or not os.path.exists(args.raven) or not args.obs or not os.path.isdir(args.obs):
        sys.exit(INSTALL)
    templates = rs.load_templates(args.templates or os.path.join(args.cache, "templates"))
    print(f"deck templates: {rs.XSSA_REPO}@{rs.XSSA_COMMIT} {rs.XSSA_TEMPLATE_FILE} (SHA-256 verified)")
    if args.fd:
        G = fd_gradients(X, args, templates)
        out = np.hstack([np.array(FD_ROWS, dtype=float)[:, None], G])
        np.savetxt(os.path.join(HERE, "fd_gradients_meanQ.csv"), out, delimiter=",", fmt="%.10g",
                   header=fmt_header(f"Finite-difference gradient of mean_Q (m3/s per normalized unit, each input "
                                     f"scaled to [-1, 1] by its half-range) at design rows {FD_ROWS} of inputs.csv; "
                                     f"central differences, step {FD_STEP} (one-sided at the box edge)",
                                     ["row [-]"] + [f"d mean_Q / d z_{n} [m3/s]" for n in rs.NAMES]))
        return
    a, b = (int(v) for v in args.rows.split(":"))
    if args.check:
        Xf = np.loadtxt(os.path.join(HERE, "inputs.csv"), delimiter=",")
        Yf = np.loadtxt(os.path.join(HERE, "outputs.csv"), delimiter=",")
        print("design reproduces inputs.csv bit for bit:", np.array_equal(Xf, X))
        work = os.path.join(args.cache, "tmp")
        os.makedirs(work, exist_ok=True)
        worst = 0.0
        for i in range(a, b):
            y, _ = one_run(Xf[i], args.raven, args.obs, work, templates)
            rel = np.max(np.abs(y - Yf[i]) / np.maximum(np.abs(Yf[i]), 1e-12))
            worst = max(worst, rel)
            print(f"row {i}: mean_Q {y[0]:.6g} (file {Yf[i, 0]:.6g}), NSE {y[1]:.6g} (file {Yf[i, 1]:.6g})")
        print(f"largest relative difference over rows {a}:{b} and 15 outputs: {worst:.2e}")
        return
    run_rows(X, range(a, b), args, templates)


if __name__ == "__main__":
    main()
