"""Generate the SIPNET_Ecosystem_Carbon_Flux data with the compiled SIPNET model.

This is the provenance script for the shipped files. It needs a compiled SIPNET binary
(C code, github.com/PecanProject/sipnet, BSD 3-clause; built here from commit
c3479bd681ad092e6be33fcf14272e23e0b3fcbf, version v2.2.0-11) and numpy + scipy. The
notebook does not need it.

For each of M Latin hypercube samples of the 8 SIPNET parameters calibrated by
Fer et al. (2018) (uniform priors of their Table A2), it runs SIPNET on the shipped
two-year driver `niwot_2004_2005.clim` with the parameter file `sipnet_template.param`
and records three end-of-period totals, plus their gradients with respect to the
physical parameters by central finite differences.

    python generate_data.py --sipnet /path/to/sipnet               # all M samples
    python generate_data.py --sipnet /path/to/sipnet --rows 0:10 --out /tmp/x --check
    python generate_data.py --sipnet /path/to/sipnet --fd-check [--out /tmp/x --check]

Outputs (CSV, two '#' header lines, np.loadtxt(path, delimiter=',')):
    inputs.csv     M x 8   physical parameters, columns in the order of NAMES
    outputs.csv    M x 3   QOIS (see QOI_UNITS)
    gradients.csv  M x 24  column 8*i + j = d QOIS[i] / d NAMES[j] (physical units)
    fd_check_soilWHC.csv   (--fd-check) step check of d QOIS / d soilWHC at the samples with
                   small soilWHC (normalized soilWHC < FD_CHECK_Z), steps FD_CHECK_STEPS
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 514                     # Latin hypercube seed (spreadsheet ID of the paper)
M = 1000                       # number of samples
H = 1e-3                       # finite-difference step in the normalized inputs [-1, 1]
FD_CHECK_STEPS = (3e-5, 1e-5)  # smaller steps of the soilWHC check (normalized inputs)
FD_CHECK_Z = -0.8              # check samples with normalized soilWHC below this (< 3.69 cm)

# SIPNET name, PEcAn trait name (Fer et al. 2018), units, lower, upper (Table A2 priors)
INPUTS = [
    ("baseSoilResp",     "som_respiration_rate", "1/yr",                        0.001, 0.3),
    ("soilRespQ10",      "soil_respiration_Q10", "1",                         1.4,   3.0),
    ("soilWHC",          "soilWHC",              "cm",                        0.1,   36.0),
    ("halfSatPar",       "half_saturation_PAR",  "Einsteins m-2 day-1",       4.0,   27.0),
    ("dVpdSlope",        "dVPDSlope",            "1/kPa",                     0.01,  0.25),
    ("leafGrowth",       "leafGrowth",           "g C m-2",                   0.0,   252.0),
    ("psnTOpt",          "psnTOpt",              "degC",                      5.0,   40.0),
    ("leafTurnoverRate", "leaf_turnover_rate",   "1/yr",                      0.03,  6.0),
]
NAMES = [r[0] for r in INPUTS]
LOWER = np.array([r[3] for r in INPUTS])
UPPER = np.array([r[4] for r in INPUTS])
# Values of the 8 parameters in PEcAn's template.param_v2 (used as the nominal point)
NOMINAL = np.array([0.06, 2.9, 12.0, 17.0, 0.05, 126.0, 24.0, 0.13])

QOIS = ["cum_nee", "total_et", "total_gpp"]
QOI_UNITS = ["g C m-2", "mm", "g C m-2"]

TEMPLATE = os.path.join(HERE, "sipnet_template.param")
CLIM = os.path.join(HERE, "niwot_2004_2005.clim")

SIPNET_IN = """RUNTYPE = standard
FILENAME = sipnet
DO_MAIN_OUTPUT = 1
PRINT_HEADER = 0
DO_SINGLE_OUTPUTS = 1
DUMP_CONFIG = 0
"""
# Columns of sipnet.out (SIPNET v2.2.0, no header)
OUT_COLS = ("year day time plantWoodC plantLeafC woodCreation soil coarseRootC fineRootC "
            "litter soilWater soilWetnessFrac snow npp nee cumNEE gpp rAboveground rSoil "
            "rRoot ra rh rtot evapotranspiration fluxestranspiration minN soilOrgN litterN "
            "plantStorageN n2o nLeaching nFixation nUptake ch4 nppStorage lai").split()
N_STEPS = 1462                 # half-day steps, 2004-01-01 to 2005-12-31


def write_param_file(path, x):
    """The template with the value column of the 8 study parameters replaced by x."""
    values = dict(zip(NAMES, x))
    seen, lines = set(), []
    with open(TEMPLATE) as fh:
        for line in fh:
            tok = line.split()
            if tok and tok[0] in values:
                tok[1] = repr(float(values[tok[0]]))
                line = "\t".join(tok) + "\n"
                seen.add(tok[0])
            lines.append(line)
    if seen != set(NAMES):
        raise KeyError(f"not in template: {sorted(set(NAMES) - seen)}")
    with open(path, "w") as fh:
        fh.writelines(lines)


def run_sipnet(x, binary):
    """One SIPNET run at physical parameters x; returns the 3 QoIs."""
    d = tempfile.mkdtemp(prefix="sipnet_")
    try:
        write_param_file(os.path.join(d, "sipnet.param"), x)
        shutil.copyfile(CLIM, os.path.join(d, "sipnet.clim"))
        with open(os.path.join(d, "sipnet.in"), "w") as fh:
            fh.write(SIPNET_IN)
        # An empty events file, with events on (the SIPNET default). With --no-events,
        # SIPNET v2.2.0-11 segfaults when it logs a computed leaf-on event.
        open(os.path.join(d, "events.in"), "w").close()
        r = subprocess.run([binary, "-i", "sipnet.in", "--quiet"], cwd=d,
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"SIPNET exit {r.returncode} at x = {list(x)}: {r.stderr[-500:]}")
        out = np.loadtxt(os.path.join(d, "sipnet.out"))
        nee_cum = np.loadtxt(os.path.join(d, "sipnet.NEE_cum"))   # printed with 6 decimals
        gpp_cum = np.loadtxt(os.path.join(d, "sipnet.GPP_cum"))
        if out.shape != (N_STEPS, len(OUT_COLS)) or nee_cum.shape != (N_STEPS,):
            raise RuntimeError(f"unexpected output shape {out.shape} at x = {list(x)}")
        c = {n: i for i, n in enumerate(OUT_COLS)}
        q = np.array([
            nee_cum[-1],                                    # g C m-2, positive = source
            10.0 * out[:, c["evapotranspiration"]].sum(),   # cm -> mm
            gpp_cum[-1],                                    # g C m-2
        ])
        if not np.all(np.isfinite(q)):
            raise RuntimeError(f"non-finite output at x = {list(x)}")
        return q
    finally:
        shutil.rmtree(d, ignore_errors=True)


def stencil(xj, j, h):
    """Offsets (in normalized units) and weights of a second-order difference for input j.

    Central (-h, +h) when both points stay in [lower, upper]; otherwise the second-order
    one-sided formula (0, +-h, +-2h), so no run leaves the prior box.
    """
    t = 2 * (xj - LOWER[j]) / (UPPER[j] - LOWER[j]) - 1
    if t - h >= -1 and t + h <= 1:
        return np.array([-h, h]), np.array([-0.5, 0.5]) / h
    s = 1.0 if t - h < -1 else -1.0
    return s * np.array([0.0, h, 2 * h]), s * np.array([-1.5, 2.0, -0.5]) / h


def value_and_gradient(args):
    """QoIs at x and their gradients with respect to the physical inputs."""
    x, binary, h = args
    f0 = run_sipnet(x, binary)
    G = np.empty((len(QOIS), len(NAMES)))
    half = (UPPER - LOWER) / 2
    for j in range(len(NAMES)):
        offs, w = stencil(x[j], j, h)
        acc = np.zeros(len(QOIS))
        for o, wk in zip(offs, w):
            if o == 0.0:
                acc += wk * f0
            else:
                xp = x.copy()
                xp[j] += o * half[j]
                acc += wk * run_sipnet(xp, binary)
        G[:, j] = acc / half[j]                 # d f / d x_physical
    return f0, G


def soilwhc_derivatives(args):
    """d QOIS / d soilWHC (physical units) at x with each step of FD_CHECK_STEPS, concatenated."""
    x, binary = args
    j = NAMES.index("soilWHC")
    half = (UPPER[j] - LOWER[j]) / 2
    f0, out = None, []
    for h in FD_CHECK_STEPS:
        offs, w = stencil(x[j], j, h)
        acc = np.zeros(len(QOIS))
        for o, wk in zip(offs, w):
            if o == 0.0:
                f0 = run_sipnet(x, binary) if f0 is None else f0
                acc += wk * f0
            else:
                xp = x.copy()
                xp[j] += o * half
                acc += wk * run_sipnet(xp, binary)
        out.append(acc / half)
    return np.concatenate(out)


def fd_check(X, binary, workers, out_dir, check):
    """Write fd_check_soilWHC.csv: soilWHC derivatives with smaller steps at small soilWHC."""
    j = NAMES.index("soilWHC")
    z = 2 * (X[:, j] - LOWER[j]) / (UPPER[j] - LOWER[j]) - 1
    rows = np.where(z < FD_CHECK_Z)[0]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        D = np.array(list(ex.map(soilwhc_derivatives, [(X[i], binary) for i in rows], chunksize=4)))
    print(f"fd check: {len(rows)} samples, {workers} workers: {time.perf_counter() - t0:.1f} s")
    A = np.column_stack([rows, X[rows, j], D])
    cols = ["row (0-based, in inputs.csv)", "soilWHC [cm]"] + [
        f"d{q}/dsoilWHC h={h:g} [{u}/cm]" for h in FD_CHECK_STEPS for q, u in zip(QOIS, QOI_UNITS)]
    write_csv(os.path.join(out_dir, "fd_check_soilWHC.csv"), A,
              f"Step check: d output / d soilWHC (physical units) by central differences with steps "
              f"{', '.join(f'{h:g}' for h in FD_CHECK_STEPS)} in normalized inputs (one-sided near the "
              f"lower bound), at the samples with normalized soilWHC < {FD_CHECK_Z:g}; SIPNET v2.2.0-11, "
              f"Latin hypercube seed {SEED}. Compare with gradients.csv (step {H:g})", cols)
    if check:
        ref = read_csv(os.path.join(HERE, "fd_check_soilWHC.csv"))
        err = np.abs(A - ref).max() / np.abs(ref).max()
        print(f"fd_check_soilWHC.csv: max abs difference / max abs value = {err:.2e}")


def sample_inputs(m=M, seed=SEED):
    """Latin hypercube (scipy qmc, seeded) mapped to the uniform prior box."""
    u = qmc.LatinHypercube(d=len(NAMES), seed=seed).random(m)
    return LOWER + (UPPER - LOWER) * u


def write_csv(path, A, description, columns, fmt="%.10g"):
    np.savetxt(path, A, delimiter=",", fmt=fmt,
               header=description + "\n" + ",".join(columns), comments="# ")


def read_csv(path):
    return np.loadtxt(path, delimiter=",")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sipnet", required=True, help="path to the compiled SIPNET binary")
    ap.add_argument("--rows", default=None, help="subset a:b of the M samples")
    ap.add_argument("--out", default=HERE, help="output directory")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--h", type=float, default=H)
    ap.add_argument("--check", action="store_true", help="compare with the shipped files")
    ap.add_argument("--fd-check", action="store_true",
                    help="write only fd_check_soilWHC.csv (smaller-step soilWHC derivatives)")
    a = ap.parse_args()
    binary = os.path.abspath(a.sipnet)
    X = sample_inputs()
    if a.fd_check:
        os.makedirs(a.out, exist_ok=True)
        return fd_check(X, binary, a.workers, a.out, a.check)
    rows = slice(*map(int, a.rows.split(":"))) if a.rows else slice(None)
    Xs = X[rows]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(value_and_gradient, [(x, binary, a.h) for x in Xs], chunksize=4))
    F = np.array([r[0] for r in res])
    G = np.array([r[1].reshape(-1) for r in res])
    dt = time.perf_counter() - t0
    print(f"{len(Xs)} samples, {a.workers} workers: {dt:.1f} s")
    os.makedirs(a.out, exist_ok=True)
    tag = (f"SIPNET v2.2.0-11 on niwot_2004_2005.clim, Latin hypercube seed {SEED}, "
           f"rows {a.rows or '0:' + str(M)}")
    write_csv(os.path.join(a.out, "inputs.csv"), Xs,
              f"SIPNET parameters (physical units), uniform priors of Fer et al. 2018 Table A2; {tag}",
              [f"{n} [{u}]" for n, _, u, _, _ in INPUTS])
    write_csv(os.path.join(a.out, "outputs.csv"), F,
              f"Two-year totals 2004-2005 (cum_nee positive = carbon source); {tag}",
              [f"{q} [{u}]" for q, u in zip(QOIS, QOI_UNITS)])
    write_csv(os.path.join(a.out, "gradients.csv"), G,
              f"d output / d input (physical units), column 8*i+j; central differences h = {a.h:g} "
              f"in normalized inputs (one-sided near bounds); {tag}",
              [f"d{q}/d{n}" for q in QOIS for n in NAMES])
    if a.check:
        for name, A in [("inputs.csv", Xs), ("outputs.csv", F), ("gradients.csv", G)]:
            ref = read_csv(os.path.join(HERE, name))[rows]
            err = np.abs(A - ref).max() / np.abs(ref).max()
            print(f"{name}: max abs difference / max abs value = {err:.2e}")


if __name__ == "__main__":
    sys.exit(main())
