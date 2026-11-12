"""Recompute the numbers of the Accuracy section of DATA.md (Raven_Salmon_River_Blended_Model).

    python check_accuracy.py --skip-raven            check 1 only (numpy/scipy; about 1 s)
    python check_accuracy.py --raven RAVEN_EXE --obs XSSA_OBS_DIR --cache DIR [--rows 0:32]
                                                     all checks (needs Raven; a few minutes)

RAVEN_EXE and XSSA_OBS_DIR may also be given as environment variables, as for generate_data.py (see
DATA.md, "Regenerating", for the Raven build and the xSSA files). The xSSA deck templates are downloaded
into <cache>/templates on first use and checked against their SHA-256. Run it from any directory.

Checks:
 1. The scrambled Sobol' design of generate_data.py reproduces inputs.csv, and the Eq. (A1) transform
    of raven_salmon_driver.weights reproduces weights.csv (largest absolute difference; 0 expected).
 2. Rerun Raven at the given rows (default 0:32) with raven_salmon_driver.run: largest relative difference of
    mean_Q, NSE and KGE from outputs.csv, and the largest differences between our NSE and KGE (computed
    from Hydrographs.csv) and Raven's own DIAG_NASH_SUTCLIFFE and DIAG_KLING_GUPTA (6 significant digits).
 3. Central-difference gradients of mean_Q and NSE at design rows 0, 1, 2 in normalized coordinates with
    steps 1e-2 and 1e-3 of each half-range (one-sided at the box edge, 86 runs per row and step): the
    relative difference of the two gradients and the cosine of the angle between them.

Imports numpy, scipy and the standard library; Raven is run as an outside executable.
"""
import argparse
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import raven_salmon_driver as rs  # noqa: E402
import generate_data as gd  # noqa: E402

INSTALL = ("Raven is not available. Build commit cb745e1 of github.com/CSHS-CWRA/RavenHydroFramework as "
           "described in DATA.md (\"Raven build\"), clone github.com/julemai/xSSA for the forcing and "
           "streamflow files, set RAVEN_EXE and XSSA_OBS_DIR (or pass --raven and --obs), and rerun this "
           "script; or pass --skip-raven for check 1 only.")


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=",")


def outputs(x, args, templates, work):
    r = rs.run(x, args.raven, args.obs, templates, workdir=work)
    return np.array([r["mean_Q"], r["NSE"], r["KGE"], r["DIAG_NSE"], r["DIAG_KGE"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raven", default=os.environ.get("RAVEN_EXE"))
    ap.add_argument("--obs", default=os.environ.get("XSSA_OBS_DIR"))
    ap.add_argument("--cache", default=None, help="folder for the downloaded templates (default: a temp folder)")
    ap.add_argument("--rows", default="0:32")
    ap.add_argument("--skip-raven", action="store_true")
    args = ap.parse_args()

    X, Wt, F = load("inputs.csv"), load("weights.csv"), load("outputs.csv")
    Xd = gd.design()
    W = np.array([rs.weights(x) for x in Xd])
    print(f"1. design vs inputs.csv: max |diff| {np.abs(Xd - X).max():.1e}; "
          f"Eq. (A1) weights vs weights.csv: max |diff| {np.abs(W - Wt).max():.1e}")
    if args.skip_raven:
        return
    if not args.raven or not os.path.exists(args.raven) or not args.obs or not os.path.isdir(args.obs):
        print(INSTALL, file=sys.stderr)
        sys.exit(3)

    cache = args.cache or tempfile.mkdtemp(prefix="raven_check_")
    templates = rs.load_templates(os.path.join(cache, "templates"))
    work = tempfile.mkdtemp(prefix="raven_runs_", dir=cache)

    a, b = (int(v) for v in args.rows.split(":"))
    Y = np.array([outputs(X[i], args, templates, work) for i in range(a, b)])
    rel = np.abs(Y[:, :3] - F[a:b, :3]) / np.abs(F[a:b, :3])
    print(f"2. rows {a}:{b}: largest relative difference from outputs.csv: mean_Q {rel[:, 0].max():.1e}, "
          f"NSE {rel[:, 1].max():.1e}, KGE {rel[:, 2].max():.1e}")
    print(f"   ours vs Raven's Diagnostics.csv: max |NSE - DIAG_NSE| {np.abs(Y[:, 1] - Y[:, 3]).max():.1e}, "
          f"max |KGE - DIAG_KGE| {np.abs(Y[:, 2] - Y[:, 4]).max():.1e}")

    half = (rs.UPPER - rs.LOWER) / 2
    print("3. finite-difference gradients of mean_Q and NSE, steps 1e-2 and 1e-3 (normalized units)")
    for row in (0, 1, 2):
        x0 = X[row]
        G = {}
        for h in (1e-2, 1e-3):
            g = np.empty((2, rs.M_INPUTS))
            for j in range(rs.M_INPUTS):
                up = min(h, (rs.UPPER[j] - x0[j]) / half[j])
                dn = min(h, (x0[j] - rs.LOWER[j]) / half[j])
                xp, xm = x0.copy(), x0.copy()
                xp[j] += up * half[j]
                xm[j] -= dn * half[j]
                g[:, j] = (outputs(xp, args, templates, work)[:2] - outputs(xm, args, templates, work)[:2]) / (up + dn)
            G[h] = g
        for k, q in enumerate(("mean_Q", "NSE")):
            g2, g3 = G[1e-2][k], G[1e-3][k]
            print(f"   row {row} {q:6s}: relative difference {np.linalg.norm(g2 - g3) / np.linalg.norm(g3):.2%}, "
                  f"cosine {g2 @ g3 / np.linalg.norm(g2) / np.linalg.norm(g3):.5f}")
        if row == 0:
            fd = load("fd_gradients_meanQ.csv")
            print(f"   row 0 mean_Q, step 1e-3 vs fd_gradients_meanQ.csv: max |diff| {np.abs(G[1e-3][0] - fd[0, 1:]).max():.1e}")


if __name__ == "__main__":
    main()
