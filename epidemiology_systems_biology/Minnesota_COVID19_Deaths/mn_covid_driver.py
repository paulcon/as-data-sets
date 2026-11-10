"""Driver for the Minnesota COVID-19 model as run by Wycoff, Binois and Wild (2021).

N. Wycoff, M. Binois and S. M. Wild, "Sequential learning of active subspaces", Journal of
Computational and Graphical Statistics 30(4) (2021) 1224-1237, doi:10.1080/10618600.2021.1874962,
Sec. 4.3. The model is the age-structured compartmental model of the University of Minnesota
COVID-19 Modeling Consortium (Enns et al. 2020), in the version shipped with the article's
supplement (code.zip on figshare, doi:10.6084/m9.figshare.13611284): file
code/helper/covid_allinone.R, function covid_sim(x), which returns the cumulative number of
COVID-19 deaths in Minnesota at the end of the 365-day horizon of the model's "scenario 2"
(social distancing from day 1 to day 41, shelter in place from day 6 to day 20).

That R code is (c) 2020 Regents of the University of Minnesota, licensed under the GNU GPL 3.0,
and is NOT part of this repository. This module only calls it: it writes the input points to a
CSV file, runs ``Rscript mn_covid_eval.R CODE_DIR in.csv out.csv``, and reads the results. Because
the model is an outside (R) code, this module is a driver, not a numpy/scipy model.

Inputs (the eight arguments x[1..8] of covid_sim, in the code's own units, fractions):
  init_cases_detected                  fraction of cases detected at day 0 (sets the number of
                                       undetected initial infections, 233 (1 - x1) / x1)
  sip_contact_reduction                contact reduction during shelter in place
  social_distancing_contact_reduction  contact reduction during social distancing
  sixty_plus_contact_reduction         contact reduction of people aged 60+ (targeted distancing)
  behavior_change_contact_reduction    contact reduction from lasting behaviour change
  prop_asymptomatic                    proportion of infections that are asymptomatic or mild
  p_h_80                               probability that a symptomatic person aged 80+ is
                                       hospitalized (scales the hospitalization of every age group)
  p_dying_home_80                      probability that a symptomatic, non-hospitalized person
                                       aged 70+ dies at home
Ranges: the article's study script (code/covid_sim.R) maps the unit cube to [0.01, 0.99]^8 by
x = 0.98 u + 0.01; LOWER and UPPER below are that box.

In scenario 2 the start times of targeted 60+ distancing and of lasting behaviour change are
infinite, so inputs 4 and 5 never enter the computation: the output does not depend on them.

The constants need only numpy. Running the model needs R (the shipped data were written with
R 4.4.3 and plyr 1.8.9 from conda-forge) and the unpacked supplement folder ``code``; pass its
path as ``code_dir`` or set the environment variable MN_COVID_CODE.
"""
import hashlib
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_R = os.path.join(HERE, "mn_covid_eval.R")

NAMES = ["init_cases_detected", "sip_contact_reduction", "social_distancing_contact_reduction",
         "sixty_plus_contact_reduction", "behavior_change_contact_reduction", "prop_asymptomatic",
         "p_h_80", "p_dying_home_80"]
SHORT = ["detected", "SIP", "SD", "60+", "behavior", "asympt.", "p_h_80", "p_home_80"]
DESCRIPTIONS = [
    "fraction of cases detected at day 0",
    "contact reduction, shelter in place",
    "contact reduction, social distancing",
    "contact reduction, people aged 60+",
    "contact reduction, behaviour change",
    "proportion asymptomatic or mild",
    "P(hospitalized | symptomatic, age 80+)",
    "P(death at home | symptomatic, not hospitalized, age 70+)",
]
UNITS = ["fraction"] * 5 + ["fraction", "probability", "probability"]
# Defaults of parameters() in covid_allinone.R (marked there "calibrated value 5/11/2020",
# except sixty_plus_contact_reduction and behavior_change_contact_reduction).
NOMINAL = np.array([0.02143008, 0.5511887, 0.3755247, 0.5, 0.1, 0.4116304, 0.1032184, 0.1392806])
EPS = 0.01                                  # code/covid_sim.R: x = (1 - 2 eps) u + eps
LOWER = np.full(8, EPS)
UPPER = np.full(8, 1.0 - EPS)
INACTIVE = [3, 4]                           # 0-based: inputs never read in scenario 2
FD_STEP = 1e-7                              # forward-difference step in the code's units
INITIAL_CASES = 233                         # detected cases on 2020-03-23 in the model's data file
# sha256 of the files used (code.zip from figshare 13611284, as downloaded 2026-11-09)
SHA256_CODE_ZIP = "585279ae0eaf56012b526b106b6897901eeefc836d6088a8c803e1b171f8e081"


def from_unit(u):
    """Map the unit cube to the input box, as code/covid_sim.R does."""
    return (1.0 - 2.0 * EPS) * np.asarray(u, dtype=float) + EPS


def find_code(code_dir=None):
    """Return the supplement's ``code`` folder; raise with instructions if it is missing."""
    code_dir = code_dir or os.environ.get("MN_COVID_CODE")
    msg = ("The Minnesota COVID-19 model is not in this repository. Download code.zip from the "
           "article's supplement (https://doi.org/10.6084/m9.figshare.13611284; sha256 "
           f"{SHA256_CODE_ZIP}), unzip it, and pass the folder 'code' with --code DIR "
           "(or set MN_COVID_CODE).")
    if not code_dir:
        raise RuntimeError(msg)
    if not os.path.isfile(os.path.join(code_dir, "helper", "covid_allinone.R")):
        raise RuntimeError(f"{code_dir} has no helper/covid_allinone.R. " + msg)
    return os.path.abspath(code_dir)


def find_rscript(rscript=None):
    """Return the Rscript executable; raise with instructions if R or plyr is missing."""
    rscript = rscript or os.environ.get("RSCRIPT") or shutil.which("Rscript")
    msg = ("R is needed (the shipped files were written with R 4.4.3 and the R package plyr 1.8.9, "
           "e.g. micromamba create -p renv -c conda-forge r-base=4.4.3 r-plyr=1.8.9). Install it and "
           "rerun, passing --rscript PATH or setting RSCRIPT if Rscript is not on the PATH.")
    if not rscript or not os.path.exists(rscript) and shutil.which(rscript) is None:
        raise RuntimeError(msg)
    r = subprocess.run([rscript, "-e", "suppressPackageStartupMessages(library(plyr)); "
                        "cat(R.version.string, as.character(packageVersion('plyr')), sep='|')"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("R runs but the package plyr is missing. " + msg)
    return rscript


def versions(rscript):
    r = subprocess.run([rscript, "-e", "suppressPackageStartupMessages(library(plyr)); "
                        "cat(R.version.string, as.character(packageVersion('plyr')), sep='|')"],
                       capture_output=True, text=True, check=True)
    rv, pv = r.stdout.strip().split("|")
    return rv, pv


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_batch(X, out_csv, code_dir, rscript, max_new=None):
    """Evaluate covid_sim at the rows of X with R; resumable through out_csv.

    Rows already present in out_csv are skipped. Returns the deaths for every row (NaN where a
    row is not yet done) after the R process exits.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    in_csv = out_csv + ".in"
    np.savetxt(in_csv, X, delimiter=",", fmt="%.17g")
    cmd = [rscript, EVAL_R, code_dir, in_csv, out_csv]
    if max_new is not None:
        cmd.append(str(int(max_new)))
    subprocess.run(cmd, check=True)
    return read_batch(out_csv, len(X))


def read_batch(out_csv, n):
    y = np.full(n, np.nan)
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        a = np.loadtxt(out_csv, delimiter=",", ndmin=2)
        y[a[:, 0].astype(int) - 1] = a[:, 1]
    return y


def deaths(X, code_dir=None, rscript=None):
    """Cumulative deaths at day 365 (scenario 2) at each row of X (code units)."""
    code_dir, rscript = find_code(code_dir), find_rscript(rscript)
    with tempfile.TemporaryDirectory() as tmp:
        return run_batch(X, os.path.join(tmp, "y.csv"), code_dir, rscript)


def fd_points(X, h=FD_STEP):
    """The 9 evaluation points per sample of the forward-difference gradient: x, x + h e_j."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    P = np.repeat(X, 9, axis=0)
    for j in range(8):
        P[j + 1::9, j] += h
    return P


def fd_assemble(y, h=FD_STEP):
    """Values and forward-difference gradients from the outputs at fd_points (in order)."""
    Y = np.asarray(y).reshape(-1, 9)
    return Y[:, 0], (Y[:, 1:] - Y[:, :1]) / h


def deaths_and_grad(X, h=FD_STEP, code_dir=None, rscript=None):
    """Deaths and forward-difference gradients (deaths per unit of each input) at the rows of X."""
    return fd_assemble(deaths(fd_points(X, h), code_dir, rscript), h)
