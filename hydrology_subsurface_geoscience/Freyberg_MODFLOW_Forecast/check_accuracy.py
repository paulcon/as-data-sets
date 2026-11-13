"""Recompute the accuracy numbers quoted in DATA.md for Freyberg_MODFLOW_Forecast (needs MODFLOW-2005).

    python check_accuracy.py            # all checks, about 6.5 min on one core
    python check_accuracy.py --quick    # check 1 only (the calibrated model)
    python check_accuracy.py --only 5   # check 5 only (slope kinks), about 1.5 min
    python check_accuracy.py --only 6   # check 6 only (prior of the linear analysis), about 12 s

Prints, beside each value DATA.md quotes:

1. The calibrated pilot-point model of the release (Figure5b directory; parameter values of its
   PEST control file): K statistics of the 70 pilot-point values (paper Table 6a) and of the
   kriged field over the 705 active cells, the RMSE of the 16 calibration heads and of the 6 forecast heads against the PEST observation values (Table
   6a: 0.000 m and 0.30 m), and the river flux against the flux observation.
2. Linearity: the forecast heads at the calibrated point for recharge x 0.5, 1, 2. For a model
   that is affine in recharge (confined flow, fixed transmissivity) the two increments have the
   ratio 2.
3. Finite-difference step: gradients of the forecast fr20c14 at the first shipped row with no dry
   cell (row 1) by central
   differences with steps 1e-2, 1e-3, 1e-5 and forward differences, against central 1e-4.
4. Shipped gradients: at the first 10 shipped rows with no dry cell in the stencil, central
   differences with step 1e-4 against the stored forward differences (relative 2-norm error per
   row, fr20c14 and all six forecasts).
5. Slope kinks from the river package. At shipped rows 30, 316 and 389: the stored forward
   differences against central differences with steps 1e-3 and 1e-4 (relative 2-norm per
   forecast). At row 389: a line scan of 21 points over +-2e-3 in log10 R, the slopes of the
   fr9c16 head between neighbouring points, and the river cells whose stress-period-2 head crosses
   the cell's river bottom inside the scan (MODFLOW's RIV leakage C (stage - h) becomes
   C (stage - bottom) below the bottom, so the head has a slope discontinuity there). Over all
   333 kept rows: how often a forward step of 1e-4 in log10 R moves a river-cell head across its
   river bottom.
6. The assumed prior of the linear analysis (calibrated.csv). The Jacobian at the calibrated point
   (central differences, step 1e-3, as generate_data.py) and the posterior peak densities
   1/(sd sqrt(2 pi)) of the six forecasts for a prior sd of (log10 range)/2, /4 (calibrated.csv)
   and /6, and for /4 with an observation sd of 0.1 m instead of 1/weight = 1 m for the heads.

The computation is deterministic and writes nothing in this directory. Needs numpy and MODFLOW-2005
1.12.00 (mf2005dbl on the PATH or MF2005DBL=/path/to/mf2005dbl); without it the script exits with
status 3 and says what to install.
"""
import argparse
import os
import re
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import freyberg_driver as M  # noqa: E402

PST = os.path.join(M.FILES, "figure5b_from_model_of_table6a_pilot_points_heads_too_highBEST.pst")
IQ = M.FORECAST_NAMES.index("fr20c14")


def pst_obs():
    obs = {}
    with open(PST) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 4 and p[3] in ("calib_head", "fore_head", "calib_flux"):
                obs[p[0].lower()] = (float(p[1]), float(p[2]))
    return obs


def river_flux(d):
    """Net river leakage (m3/d) in stress period 1 from the double-precision budget file."""
    raw = open(os.path.join(d, M.NAME + ".cbb"), "rb").read()
    off, n = 0, M.NROW * M.NCOL
    while off < len(raw):
        kstp, kper = np.frombuffer(raw, "<i4", 2, off)
        text = raw[off + 8:off + 24].decode().strip()
        a = np.frombuffer(raw, "<f8", n, off + 36)
        if kper == 1 and text == "RIVER LEAKAGE":
            return float(a.sum())
        off += 36 + 8 * n
    raise RuntimeError("no river leakage record")


def river_cells():
    """(row, column, bottom) of the stress-period-2 river cells (0-based row/column)."""
    lines = open(os.path.join(M.FILES, M.NAME + ".riv")).read().splitlines()
    out = []
    for ln in lines[4:]:
        p = ln.split()
        if len(p) < 6:
            break
        out.append((int(p[1]) - 1, int(p[2]) - 1, float(p[5])))
    return out


def river_margin(x, d, cells):
    """Forecast heads and (stress-period-2 head - river bottom) at every river cell."""
    f = M.run(x, workdir=d)[1]
    h2 = M._read_heads(os.path.join(d, M.NAME + ".hds"))[1]
    return f, np.array([h2[i, j] - b for i, j, b in cells])


def run_obs(x, d):
    c, f, nd = M.run(x, workdir=d)
    return np.r_[c, river_flux(d)], f, nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", type=int, default=None, help="run only this check (5 or 6)")
    a = ap.parse_args()
    try:
        M.find_mf2005()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(3)
    d = tempfile.mkdtemp(prefix="frey_chk_")
    try:
        if a.only == 5:
            check5(d)
            return
        if a.only == 6:
            check6(d)
            return
        obs = pst_obs()
        cnames = [re.sub(r"r0", "r", M.TARGET_NAMES[k].lower()) for k in M.CALIB]
        co = np.array([obs[n][0] for n in cnames])
        fo = np.array([obs[n][0] for n in M.FORECAST_NAMES])
        xb = M.X_BEST_LOG10
        print("1. Calibrated model (Figure5b PEST control file values)")
        kp = 10 ** xb[1:]
        print("   K at the 70 pilot points: mean %.2f, median %.2f, sd %.2f, max %.2f, min %.2f m/d "
              "(paper Table 6a: 5.52, 5.45, 1.35, 8.57, 2.70)"
              % (kp.mean(), np.median(kp), kp.std(ddof=1), kp.max(), kp.min()))
        K = M.kfield(kp)
        act = M.IBOUND != 0
        print("   K over the %d active cells (kriged field): mean %.2f, median %.2f, sd %.2f, max %.2f, "
              "min %.2f m/d" % (act.sum(), K[act].mean(), np.median(K[act]), K[act].std(ddof=1),
                                K[act].max(), K[act].min()))
        y, f, nd = run_obs(xb, d)
        print("   recharge %.4e m/d; dry cells %d" % (10 ** xb[0], nd))
        print("   RMSE calibration heads %.1e m (Table 6a: 0.000); RMSE forecasts %.3f m (Table 6a: 0.30)"
              % (np.sqrt(np.mean((y[:16] - co) ** 2)), np.sqrt(np.mean((f - fo) ** 2))))
        print("   river flux %.2f m3/d (PEST observation %.2f)" % (y[16], obs["flux_obs"][0]))
        for n, v, t in zip(M.FORECAST_NAMES, f, fo):
            print("   forecast %-8s model %.3f m, true %.3f m" % (n, v, t))
        if a.quick:
            return
        print("2. Linearity in recharge at the calibrated point (ratio 2 if affine)")
        hs = []
        for s in (0.5, 1.0, 2.0):
            x = xb.copy()
            x[0] += np.log10(s)
            hs.append(M.run(x, workdir=d)[1])
        hs = np.array(hs)
        for k, n in enumerate(M.FORECAST_NAMES):
            print("   %-8s h(R/2) %.3f, h(R) %.3f, h(2R) %.3f m; increment ratio %.3f"
                  % (n, hs[0, k], hs[1, k], hs[2, k], (hs[2, k] - hs[1, k]) / (hs[1, k] - hs[0, k])))
        X = np.loadtxt(os.path.join(HERE, "inputs.csv"), delimiter=",")
        O = np.loadtxt(os.path.join(HERE, "outputs.csv"), delimiter=",")
        G = np.loadtxt(os.path.join(HERE, "gradients.csv"), delimiter=",").reshape(len(X), 6, 71)
        r0 = int(np.flatnonzero(O[:, 23] == 0)[0])
        print("3. Step study at shipped row %d (the first with no dry cell), forecast fr20c14 "
              "(reference: central, step 1e-4)" % r0)
        ref = M.sample(X[r0], h=1e-4, central=True, workdir=d)[4]
        for h in (1e-2, 1e-3, 1e-5):
            gc = M.sample(X[r0], h=h, central=True, workdir=d)[4]
            gf = M.sample(X[r0], h=h, central=False, workdir=d)[4]
            print("   step %.0e: central %.1e, forward %.1e (relative 2-norm difference)"
                  % (h, np.linalg.norm(gc[IQ] - ref[IQ]) / np.linalg.norm(ref[IQ]),
                     np.linalg.norm(gf[IQ] - ref[IQ]) / np.linalg.norm(ref[IQ])))
        print("4. Shipped forward differences (step %g) against central differences (step 1e-4)" % M.FD_STEP)
        rows = [r for r in range(len(X)) if O[r, 23] == 0][:10]
        e1, e6 = [], []
        for r in rows:
            gc = M.sample(X[r], h=1e-4, central=True, workdir=d)[4]
            e1.append(np.linalg.norm(G[r, IQ] - gc[IQ]) / np.linalg.norm(gc[IQ]))
            e6.append(np.linalg.norm(G[r] - gc) / np.linalg.norm(gc))
        print("   rows %s" % rows)
        print("   fr20c14: max %.1e, median %.1e; all six: max %.1e" % (max(e1), np.median(e1), max(e6)))
        check5(d)
        check6(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def check5(d):
    X = np.loadtxt(os.path.join(HERE, "inputs.csv"), delimiter=",")
    O = np.loadtxt(os.path.join(HERE, "outputs.csv"), delimiter=",")
    G = np.loadtxt(os.path.join(HERE, "gradients.csv"), delimiter=",").reshape(len(X), 6, 71)
    print("5. Slope kinks (river package): stored forward differences (step %g) against central "
          "differences, relative 2-norm per forecast" % M.FD_STEP)
    for r in (30, 316, 389):
        for h in (1e-3, 1e-4):
            gc = M.sample(X[r], h=h, central=True, workdir=d)[4]
            e = np.linalg.norm(G[r] - gc, axis=1) / np.linalg.norm(gc, axis=1)
            print("   row %d, central step %.0e: " % (r, h)
                  + ", ".join("%s %.1e" % (n, v) for n, v in zip(M.FORECAST_NAMES, e)))
    cells = river_cells()
    r = 389
    t = np.linspace(-2e-3, 2e-3, 21)
    F, D = [], []
    for s in t:
        x = X[r].copy()
        x[0] += s
        f, m = river_margin(x, d, cells)
        F.append(f)
        D.append(m)
    F, D = np.array(F), np.array(D)
    slope = np.diff(F[:, 0]) / np.diff(t)
    print("   row 389, line scan in log10 R (offsets -2e-3 .. 2e-3, 21 points): slopes of h_fr9c16 "
          "(m per log10 unit) between neighbouring points:")
    print("   " + " ".join("%.2f" % v for v in slope))
    k = int(np.argmax(np.abs(np.diff(slope))))
    print("   largest slope change between the intervals [%.1e, %.1e] and [%.1e, %.1e]: %.2f -> %.2f"
          % (t[k], t[k + 1], t[k + 1], t[k + 2], slope[k], slope[k + 1]))
    for c in range(len(cells)):
        sg = np.sign(D[:, c])
        if np.any(sg != sg[0]):
            i = int(np.flatnonzero(sg != sg[0])[0])
            print("   river cell row %d, column %d (bottom %.2f m): head - bottom changes sign between "
                  "offsets %.1e and %.1e (%.2e m to %.2e m)"
                  % (cells[c][0] + 1, cells[c][1] + 1, cells[c][2], t[i - 1], t[i], D[i - 1, c], D[i, c]))
    keep = np.flatnonzero(O[:, 23] == 0)
    xr, near = [], []
    for r in keep:
        x = X[r].copy()
        _, m0 = river_margin(x, d, cells)
        x[0] += M.FD_STEP
        _, m1 = river_margin(x, d, cells)
        if np.any(np.sign(m0) != np.sign(m1)):
            xr.append(int(r))
        near.append(np.min(np.abs(m0)))
    near = np.array(near)
    print("   kept rows: %d; a forward step of %g in log10 R moves a river-cell head across its "
          "bottom at %d of them (rows %s)" % (len(keep), M.FD_STEP, len(xr), xr))
    print("   smallest |head - river bottom| over the river cells at the kept samples: median %.2e m; "
          "below 1e-3 m at %d rows, below 1e-2 m at %d rows"
          % (np.median(near), int(np.sum(near < 1e-3)), int(np.sum(near < 1e-2))))


def check6(d, h=1e-3):
    import generate_data as GD
    xb = M.X_BEST_LOG10

    def obsvec(x):
        c, f, _ = M.run(x, workdir=d)
        return np.r_[c, river_flux(d), f]

    J = np.empty((23, len(xb)))
    for i in range(len(xb)):
        e = np.zeros(len(xb))
        e[i] = h
        J[:, i] = (obsvec(xb + e) - obsvec(xb - e)) / (2 * h)
    _, ov = GD.pst_obs()
    Jo, Jf = J[:17], J[17:]
    print("6. Linear analysis at the calibrated point: posterior peak density of the forecasts %s"
          % ", ".join(M.FORECAST_NAMES))
    for div, osd in ((2, None), (4, None), (6, None), (4, 0.1)):
        Cp = np.diag(((M.UPPER_LOG10 - M.LOWER_LOG10) / div) ** 2)
        sde = 1.0 / ov[:17, 1]
        if osd is not None:
            sde = sde.copy()
            sde[:16] = osd
        Ce = np.diag(sde ** 2)
        Cpost = Cp - Cp @ Jo.T @ np.linalg.solve(Jo @ Cp @ Jo.T + Ce, Jo @ Cp)
        post = np.sqrt(np.einsum("ij,jk,ik->i", Jf, Cpost, Jf))
        lab = "prior sd = log10 range/%d, head obs sd %s" % (div, "1/weight" if osd is None else "%g m" % osd)
        print("   %-45s %s" % (lab, " ".join("%.2f" % v for v in 1 / (post * np.sqrt(2 * np.pi)))))


if __name__ == "__main__":
    main()
