"""Python driver for the Hector simple climate model run through Matilda's set_params.

Hector (C++ with an R interface, GPL-3) and Matilda (R, MIT) are outside codes: this file
does not contain or port them. It maps the six scaled inputs z (standard normal) to Hector's
parameters with Matilda's prior distributions (Brown et al. 2024, Table 1, p. 5), writes
them to a CSV, calls ``Rscript hector_worker.R`` and reads the results back.

Needed: R with the packages hector 3.1.1 and matilda (GitHub tag v1.0.0). Rscript is taken
from the environment variable HECTOR_RSCRIPT, else from PATH; R packages from R_LIBS.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "hector_worker.R")

# Inputs in the order of Table 1 (p. 5). kind "normal": x = mean + sd*z.
# kind "lognormal": mean and sd are the arithmetic mean and sd of x (Matilda's lognorm()):
# x = exp(mu + s*z), mu = log(mean^2/sqrt(sd^2+mean^2)), s = sqrt(log(1+sd^2/mean^2)).
PARAMS = [
    # (Hector/Matilda name, symbol, units, mean, sd, kind)
    ("AERO_SCALE", "alpha", "1", 1.0, 0.23, "normal"),
    ("BETA", "beta", "1", 0.55, 0.10, "normal"),
    ("ECS", "ECS", "degC", 3.0, 0.65, "lognormal"),
    ("DIFFUSIVITY", "Keff", "cm2/s", 1.16, 0.118, "normal"),
    ("NPP_FLUX0", "NPP0", "PgC/yr", 56.2, 14.3, "normal"),
    ("Q10_RH", "Q10", "1", 2.2, 1.0, "lognormal"),
]
NAMES = [p[0] for p in PARAMS]
M_IN = len(PARAMS)
OUTPUTS = ["warming", "co2_2100", "gmst_1850_1900"]
HIST_YEARS = np.arange(1950, 2024)
SCENARIOS = ("ssp119", "ssp126", "ssp245", "ssp370", "ssp585")


def lognormal_params(mean, sd):
    """Matilda's lognorm(): log-space mean and sd for a given arithmetic mean and sd."""
    mu = np.log(mean**2 / np.sqrt(sd**2 + mean**2))
    s = np.sqrt(np.log(1.0 + sd**2 / mean**2))
    return mu, s


def z_to_x(Z):
    """Scaled inputs Z (N x 6, standard normal) -> physical parameters X (N x 6)."""
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    X = np.empty_like(Z)
    for j, (_, _, _, mean, sd, kind) in enumerate(PARAMS):
        if kind == "normal":
            X[:, j] = mean + sd * Z[:, j]
        else:
            mu, s = lognormal_params(mean, sd)
            X[:, j] = np.exp(mu + s * Z[:, j])
    return X


def x_to_z(X):
    """Inverse of z_to_x."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Z = np.empty_like(X)
    for j, (_, _, _, mean, sd, kind) in enumerate(PARAMS):
        if kind == "normal":
            Z[:, j] = (X[:, j] - mean) / sd
        else:
            mu, s = lognormal_params(mean, sd)
            Z[:, j] = (np.log(X[:, j]) - mu) / s
    return Z


def rscript():
    exe = os.environ.get("HECTOR_RSCRIPT") or shutil.which("Rscript")
    if not exe:
        raise RuntimeError("Rscript not found: install R and set HECTOR_RSCRIPT or PATH")
    return exe


def tool_versions():
    """Return (R version, hector version, matilda version) or raise RuntimeError."""
    code = ('suppressPackageStartupMessages({library(hector); library(matilda)});'
            'cat(paste(R.version$major, R.version$minor, sep="."),'
            'as.character(packageVersion("hector")),'
            'as.character(packageVersion("matilda")))')
    r = subprocess.run([rscript(), "-e", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("R packages hector/matilda not loadable: " + r.stderr.strip()[-300:])
    return tuple(r.stdout.split())


def run_x(X, scenario="ssp245", hist=False, timeout=None):
    """Run Hector for physical parameter rows X (N x 6).

    Returns an array N x 3 (warming degC, co2_2100 ppmv, gmst_1850_1900 degC), or N x (3+74) with
    hist=True (raw Hector gmst for 1950-2023 appended). Failed runs are NaN.
    """
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario " + str(scenario))
    X = np.atleast_2d(np.asarray(X, dtype=float))
    with tempfile.TemporaryDirectory() as d:
        pin = os.path.join(d, "params.csv")
        pout = os.path.join(d, "out.csv")
        with open(pin, "w") as f:
            f.write(",".join(NAMES) + "\n")
            for row in X:
                f.write(",".join(repr(float(v)) for v in row) + "\n")
        cmd = [rscript(), WORKER, scenario, pin, pout] + (["hist"] if hist else [])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError("hector_worker.R failed: " + r.stderr.strip()[-500:])
        out = np.genfromtxt(pout, delimiter=",", skip_header=1)
    return np.atleast_2d(out)


def run_z(Z, scenario="ssp245", hist=False, timeout=None):
    """Run Hector at scaled inputs Z (N x 6)."""
    return run_x(z_to_x(Z), scenario=scenario, hist=hist, timeout=timeout)


def fd_stencil(Z, h, central=True):
    """Points for finite-difference gradients in z: for each row, the base point (forward
    only) and +-h (central) or +h (forward) along each coordinate."""
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    pts = []
    for z in Z:
        if not central:
            pts.append(z.copy())
        for j in range(M_IN):
            e = np.zeros(M_IN)
            e[j] = h
            pts.append(z + e)
            if central:
                pts.append(z - e)
    return np.array(pts)


def fd_gradients(F, h, central=True):
    """Gradients (N x 6 for each output column) from outputs F of the points fd_stencil made.

    F has shape (N*k, nout). Returns array (nout, N, 6)."""
    k = 2 * M_IN if central else M_IN + 1
    F = np.asarray(F).reshape(-1, k, F.shape[-1])
    if central:
        G = (F[:, 0::2, :] - F[:, 1::2, :]) / (2.0 * h)
    else:
        G = (F[:, 1:, :] - F[:, :1, :]) / h
    return np.transpose(G, (2, 0, 1))
