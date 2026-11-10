"""Drive MODFLOW-2005 for the Freyberg (1988) pilot-point model of Hunt, Fienen & White (2020).

The model is the steady-state, single-layer, water-table (LAYTYP = 1) MODFLOW-2005 model of the
USGS data release Hunt, Haserodt, Fienen & White (2020), https://doi.org/10.5066/P96A7ZC9,
directory model/Figure5b_pilot_points_flat_bottom (70 pilot-point K parameters, one recharge
parameter, flat aquifer base 3.3409 m except at the 22 well and target cells). The release runs
it on Windows with PEST utilities; this module replaces those utilities by Python code:

* fac2real (PEST groundwater utilities): K in every cell from the 70 pilot-point values with
  the kriging factors of factors1.dat, interpolated in log10 K (the factor file's transform
  flag), then bounded by 1e-10 and 2e7 m/d as in the release's fac2real1.in.
* targpest (PEST groundwater utilities): heads at the 16 calibration targets and the 6
  forecast wells by bilinear interpolation between the four surrounding cell centres, using
  the coordinates of targpest.dat.
* PEST's template step: the recharge array (one value R) and the K array are written in full
  precision in free format.

Two settings differ from the release, both for finite-difference gradients: the PCG solver is
tightened (HCLOSE 1e-5 -> 1e-10 m, RCLOSE 0.1 -> 1e-6 m3/d, MXITER 100 -> 200, ITER1 25 -> 50), and
the double-precision MODFLOW-2005 build (mf2005dbl) writes the head file in double precision.

Stress period 1 is the calibration (pre-lining) state; stress period 2 is the forecast state,
with the river conductance reduced from 4320 to 43.2 m2/d (two orders of magnitude, paper
p. 169). Both are steady state.

Needs MODFLOW-2005 1.12.00 (double-precision build, executable name mf2005dbl) on the PATH or in
the environment variable MF2005DBL, and numpy. See DATA.md for the install recipe.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, "modflow_files")
NROW, NCOL, DELR = 40, 20, 250.0
NAME = "freyberg_revisit_pp"
MF_VERSION = "1.12.00"

# Parameter bounds of the PEST control file
# figure5b_from_model_of_table6a_pilot_points_heads_too_highBEST.pst, "* parameter data":
# line 20 r1 (recharge) 5e-5 to 5e-4 m/d, log-transformed; lines 22-91 kpkp775 ... kpkp844
# (pilot-point K) 0.5 to 50 m/d, log-transformed.
R_BOUNDS = (5.0e-5, 5.0e-4)
K_BOUNDS = (0.5, 50.0)


def _read_points():
    names, xy, kbest = [], [], []
    with open(os.path.join(FILES, "points1.dat")) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 5:
                names.append(p[0].upper())
                xy.append((float(p[1]), float(p[2])))
                kbest.append(float(p[4]))
    return names, np.array(xy), np.array(kbest)


PP_NAMES, PP_XY, PP_BEST = _read_points()
NPP = len(PP_NAMES)
NAMES = ["R"] + ["K_" + n for n in PP_NAMES]
LOWER_LOG10 = np.log10(np.array([R_BOUNDS[0]] + [K_BOUNDS[0]] * NPP))
UPPER_LOG10 = np.log10(np.array([R_BOUNDS[1]] + [K_BOUNDS[1]] * NPP))


def _read_best_r():
    pst = os.path.join(FILES, "figure5b_from_model_of_table6a_pilot_points_heads_too_highBEST.pst")
    with open(pst) as fh:
        for line in fh:
            p = line.split()
            if p and p[0] == "r1":
                return float(p[3])
    raise RuntimeError("r1 not found in the PEST control file")


R_BEST = _read_best_r()
X_BEST_LOG10 = np.log10(np.concatenate([[R_BEST], PP_BEST]))


def _read_factors():
    with open(os.path.join(FILES, "factors1.dat")) as fh:
        lines = fh.read().split("\n")
    ncol, nrow = (int(v) for v in lines[2].split())
    npp = int(lines[3])
    names = [s.strip().upper() for s in lines[4:4 + npp]]
    assert (ncol, nrow, npp) == (NCOL, NROW, NPP) and names == PP_NAMES
    W = np.zeros((nrow * ncol, npp))
    logflag = np.zeros(nrow * ncol, dtype=bool)
    offset = np.zeros(nrow * ncol)
    for line in lines[4 + npp:]:
        p = line.split()
        if not p:
            continue
        cell, flag, n, off = int(p[0]) - 1, int(p[1]), int(p[2]), float(p[3])
        logflag[cell] = flag == 1
        offset[cell] = off
        for k in range(n):
            W[cell, int(p[4 + 2 * k]) - 1] = float(p[5 + 2 * k])
    return W, logflag, offset


FAC_W, FAC_LOG, FAC_OFF = _read_factors()


def kfield(k_pp):
    """K (m/d) in every cell (row-major, 40 x 20) from the 70 pilot-point K values (fac2real)."""
    k_pp = np.asarray(k_pp, dtype=float)
    lin = FAC_W @ k_pp + FAC_OFF
    lg = FAC_W @ np.log10(k_pp) + FAC_OFF
    k = np.where(FAC_LOG, 10.0 ** lg, lin)
    return np.clip(k, 1e-10, 2e7).reshape(NROW, NCOL)


def _read_targets():
    names, xy = [], []
    with open(os.path.join(FILES, "targpest.dat")) as fh:
        lines = fh.read().split("\n")
    for line in lines:
        p = line.split()
        if len(p) == 11 and (p[0].startswith("cr") or p[0].startswith("fr")):
            names.append(p[0])
            xy.append((float(p[1]), float(p[2])))
    return names, np.array(xy)


TARGET_NAMES, TARGET_XY = _read_targets()
CALIB = [i for i, n in enumerate(TARGET_NAMES) if n.startswith("cr")]
FORE = [i for i, n in enumerate(TARGET_NAMES) if n.startswith("fr")]
FORECAST_NAMES = [TARGET_NAMES[i] for i in FORE]


def _read_ibound():
    vals = []
    with open(os.path.join(FILES, NAME + ".bas")) as fh:
        lines = fh.read().split("\n")
    for line in lines[5:]:
        p = line.split()
        if len(vals) >= NROW * NCOL:
            break
        vals.extend(int(v) for v in p)
    return np.array(vals[:NROW * NCOL]).reshape(NROW, NCOL)


IBOUND = _read_ibound()


def interp(h, xy, strict=True):
    """Bilinear interpolation of the head array h (40 x 20) between cell centres (targpest).

    Cells with zero weight are ignored. If a cell with nonzero weight is inactive or dry, the
    function raises ValueError (strict=True) or returns NaN for that target (strict=False)."""
    out = np.empty(len(xy))
    for k, (x, y) in enumerate(xy):
        cf = x / DELR - 0.5                       # column coordinate of cell centres
        rf = (NROW * DELR - y) / DELR - 0.5       # row coordinate (rows run north to south)
        j0 = int(np.clip(np.floor(cf), 0, NCOL - 2))
        i0 = int(np.clip(np.floor(rf), 0, NROW - 2))
        tx, ty = cf - j0, rf - i0
        blk = h[i0:i0 + 2, j0:j0 + 2]
        wts = np.array([[(1 - ty) * (1 - tx), (1 - ty) * tx], [ty * (1 - tx), ty * tx]])
        bad = (np.abs(blk) > 1e20) | (IBOUND[i0:i0 + 2, j0:j0 + 2] == 0)
        if np.any(bad & (wts > 1e-12)):
            if strict:
                raise ValueError("dry or inactive cell next to target %d" % k)
            out[k] = np.nan
            continue
        out[k] = float(np.sum(np.where(wts > 1e-12, wts * blk, 0.0)))
    return out


def find_mf2005():
    exe = os.environ.get("MF2005DBL") or shutil.which("mf2005dbl")
    if exe is None or not os.path.exists(exe):
        raise RuntimeError(
            "MODFLOW-2005 %s (double-precision build mf2005dbl) not found. Install it (DATA.md, "
            "'Regenerating'), put it on the PATH or set MF2005DBL=/path/to/mf2005dbl, and rerun."
            % MF_VERSION)
    return os.path.abspath(exe)


def _write_arrays(d, R, K):
    with open(os.path.join(d, NAME + ".rch"), "w") as fh:
        fh.write("# MODFLOW2000 Recharge Package\nPARAMETER  0\n 3 50\n 1 1\n")
        fh.write("        19     1.000(FREE)                  -1     RECHARGE\n")
        for i in range(NROW):
            fh.write(" ".join("%.16e" % R for _ in range(NCOL)) + "\n")
        fh.write(" -1 1\n")       # stress period 2 reuses the recharge array
    with open(os.path.join(d, NAME + "1._kx"), "w") as fh:
        for i in range(NROW):
            fh.write(" ".join("%.16e" % v for v in K[i]) + "\n")
    with open(os.path.join(d, NAME + ".lpf"), "w") as fh:
        fh.write("# MODFLOW2000 Layer Property Flow (LPF) Package\n 50  -1.000000e+030 0\n"
                 "  1\n  0\n  1.000000e+000\n 0\n 0\n"
                 "        73   1.00000(FREE)                    -1     HYDRAULIC CONDUCTIVITY Layer   1\n"
                 "         0   1.00000(FREE)                    -1     VERTICAL HYDRAULIC CONDUCTIVITY Layer   1\n")
    with open(os.path.join(d, NAME + ".pcg"), "w") as fh:
        fh.write(" 200 50 1 0 9999\n 1.000000e-010 1.000000e-006 1.000000e+000 2 5 0 7.000000e-001 7.000000e-001\n")
    with open(os.path.join(d, NAME + ".nam"), "w") as fh:
        fh.write("LIST 7 %s.lst\nBAS6  1 %s.bas\nDIS  11 %s.dis\nLPF 14 %s.lpf\nWEL 20 %s.wel\n"
                 "RIV 18 %s.riv\nRCH 19 %s.rch\nOC  22 %s.oc\nPCG 32 %s.pcg\n"
                 "DATA(BINARY) 50 %s.cbb\nDATA 73 %s1._kx\nDATA(BINARY) 30 %s.hds\n"
                 % ((NAME,) * 12))


def _read_heads(path):
    """Heads of both stress periods from a MODFLOW binary head file (double precision)."""
    raw = open(path, "rb").read()
    # record: kstp kper (i4) pertim totim (r8) text (16 chars) ncol nrow ilay (i4), data r8
    hdr = np.dtype([("kstp", "<i4"), ("kper", "<i4"), ("pertim", "<f8"), ("totim", "<f8"),
                    ("text", "S16"), ("ncol", "<i4"), ("nrow", "<i4"), ("ilay", "<i4")])
    rec = hdr.itemsize + 8 * NROW * NCOL
    if len(raw) != 2 * rec:
        raise RuntimeError("unexpected head file size %d (expected double precision)" % len(raw))
    out = {}
    for k in range(2):
        h = np.frombuffer(raw, dtype=hdr, count=1, offset=k * rec)[0]
        a = np.frombuffer(raw, dtype="<f8", count=NROW * NCOL, offset=k * rec + hdr.itemsize)
        out[int(h["kper"])] = a.reshape(NROW, NCOL).copy()
    return out[1], out[2]


def run(x_log10, workdir=None):
    """One MODFLOW run. x_log10 = [log10 R (m/d), log10 K at the 70 pilot points (m/d)].

    Returns (calibration heads at the 16 targets in stress period 1, forecast heads at the 6
    pumping wells in stress period 2, number of dry active cells at the end of stress period 2),
    heads in m. A head next to a dry cell is NaN (MODFLOW-2005 without rewetting: a dry cell
    stays inactive, and a dry well cell stops pumping). Raises RuntimeError if MODFLOW fails or
    does not converge."""
    x = np.asarray(x_log10, dtype=float)
    R, k_pp = 10.0 ** x[0], 10.0 ** x[1:]
    exe = find_mf2005()
    d = workdir or tempfile.mkdtemp(prefix="frey_")
    try:
        for f in (".bas", ".dis", ".wel", ".riv", ".oc"):
            shutil.copy(os.path.join(FILES, NAME + f), d)
        _write_arrays(d, R, kfield(k_pp))
        p = subprocess.run([exe, NAME + ".nam"], cwd=d, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=120)
        lst = open(os.path.join(d, NAME + ".lst"), errors="replace").read()
        if ("FAILED TO MEET SOLVER CONVERGENCE" in lst
                or "Normal termination" not in p.stdout.decode(errors="replace")):
            raise RuntimeError("MODFLOW did not converge or failed")
        h1, h2 = _read_heads(os.path.join(d, NAME + ".hds"))
        ndry = int(np.sum((np.abs(h2) > 1e20) & (IBOUND != 0)))
        return (interp(h1, TARGET_XY[CALIB], strict=False),
                interp(h2, TARGET_XY[FORE], strict=False), ndry)
    finally:
        if workdir is None:
            shutil.rmtree(d, ignore_errors=True)


FD_STEP = 1.0e-4     # forward-difference step in log10 units (checked in check_accuracy.py)


def sample(x_log10, h=FD_STEP, central=False, workdir=None):
    """Model outputs and finite-difference gradients at one input point.

    Returns calibration heads (16), forecast heads (6), dry cells at x, the largest number of
    dry cells over the difference stencil, and the gradients (6 x 71) of the forecast heads with
    respect to log10 of the 71 inputs (m per log10 unit). Forward differences with step h by
    default; central=True uses central differences."""
    x = np.asarray(x_log10, dtype=float)
    c0, f0, nd0 = run(x, workdir)
    ndmax = nd0
    G = np.empty((len(f0), len(x)))
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = h
        _, fp, ndp = run(x + e, workdir)
        if central:
            _, fm, ndm = run(x - e, workdir)
            G[:, i] = (fp - fm) / (2 * h)
            ndmax = max(ndmax, ndp, ndm)
        else:
            G[:, i] = (fp - f0) / h
            ndmax = max(ndmax, ndp)
    return c0, f0, nd0, ndmax, G
