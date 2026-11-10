"""Driver for the 1-D pulmonary pulse-wave solver of Colebank (2025), Windkessel outlets.

The solver (C++ program ``sor06``, M.S. Olufsen, M.U. Qureshi, M.J. Colebank) is an outside
code. It is NOT part of this data set: its repository is licensed CC BY-NC 4.0, so this module
fetches it from GitHub at a pinned commit, compiles it in a directory you choose, and runs it as
a command-line program. Nothing from that repository is copied into this directory.

    import pulmonary_wk_driver as D
    exe = D.build("/tmp/pulmwk")          # git clone + g++ (needs git and g++)
    out = D.run(D.NOMINAL, exe)           # dict of the three MPA pressure QoIs, or None

Inputs (code units, the order of the solver's command line and of the authors'
WK_Simulator/sample_3ves.m):
    ln k1, k2, ln k3   wall stiffness Eh/r0 = k1 exp(k2 r0) + k3, k1 and k3 in g/(cm s^2),
                       k2 in 1/cm; k1 and k3 are sampled log-uniformly by the authors, so the
                       coordinates here are their natural logarithms
    Rp1, Rp2, Rd1, Rd2 proximal and distal Windkessel resistances, LPA (1) and RPA (2), in the
                       solver's nondimensional units (1 unit = R_SCALE mmHg s/mL)
    CT1, CT2           Windkessel compliances, nondimensional (1 unit = C_SCALE mL/mmHg)

Quantities of interest (this data set's choice; the paper uses the whole time series): the
time average, maximum and minimum over the last computed cardiac cycle of the pressure at the
midpoint of the main pulmonary artery (MPA), in mmHg.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

REPO_URL = "https://github.com/mjcolebank/Colebank_Identifiability_PDE.git"
COMMIT = "9b2741f89faf1c12127c2a05a5c229be32c9905a"   # 2025-06-04, the version used here
SUBDIR = "WK_Simulator"
SOURCES = ["tools.c", "junction.c", "sor06.c", "arteries.c"]
RUNFILES = ["Qin.dat", "connectivity.txt"]

NAMES = ["ln_k1", "k2", "ln_k3", "Rp1", "Rp2", "Rd1", "Rd2", "CT1", "CT2"]
UNITS = ["ln(g/(cm s^2))", "1/cm", "ln(g/(cm s^2))", "nondim", "nondim", "nondim", "nondim",
         "nondim", "nondim"]
# Bounds from WK_Simulator/sample_3ves.m (low/upp; k1 and k3 bounds are on their logarithms there)
LOWER = np.array([np.log(1e6), -30.0, np.log(3e4), 0.5, 0.5, 0.8, 0.8, 1e-5, 1e-5])
UPPER = np.array([np.log(1e8), -10.0, np.log(5e5), 10.0, 10.0, 20.0, 20.0, 1.2, 1.2])
NOMINAL = 0.5 * (LOWER + UPPER)

# Unit conversions of the solver (sor06.h: rho = 1.055 g/cm^3, g = 981 cm/s^2, Lr = 1 cm,
# q = 10 cm^3/s, conv = 1333.22 dyn/cm^2 per mmHg)
RHO, GRAV, LR, QSC, CONV = 1.055, 981.0, 1.0, 10.0, 1333.220
P_SCALE = RHO * GRAV * LR / CONV          # mmHg per nondimensional pressure unit
R_SCALE = P_SCALE / QSC                   # mmHg s/mL per nondimensional resistance unit
C_SCALE = LR ** 3 / P_SCALE               # mL/mmHg per nondimensional compliance unit
K_TO_MMHG = 1.0 / CONV                    # mmHg per g/(cm s^2)

QOI_NAMES = ["p_mean", "p_sys", "p_dia"]
QOI_LONG = ["time-averaged MPA midpoint pressure", "maximum (systolic) MPA midpoint pressure",
            "minimum (diastolic) MPA midpoint pressure"]
NT = 512          # printed time points per cardiac cycle (sor06.h: plts)
NROW = 9          # rows per printed time: proximal, midpoint, distal x (MPA, LPA, RPA)
MPA_MID = 3       # row of the MPA midpoint
TIMEOUT_S = 120


class SolverMissing(RuntimeError):
    pass


def build(workdir, source=None):
    """Clone the authors' repository at COMMIT (or use ``source``, a clone) and compile sor06.

    Returns the directory that holds the executable and its two input files.
    """
    for tool in ("g++",) + (() if source else ("git",)):
        if shutil.which(tool) is None:
            raise SolverMissing(f"{tool} not found; install it (the shipped files used g++ 13.3.0)")
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    if source is None:
        source = os.path.join(workdir, "Colebank_Identifiability_PDE")
        if not os.path.isdir(source):
            subprocess.run(["git", "clone", "-q", REPO_URL, source], check=True)
        subprocess.run(["git", "-C", source, "checkout", "-q", COMMIT], check=True)
    src = os.path.join(source, SUBDIR)
    exe_dir = os.path.join(workdir, "bin")
    os.makedirs(exe_dir, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith((".c", ".h")) or f in RUNFILES:
            shutil.copy(os.path.join(src, f), exe_dir)
    objs = []
    for s in SOURCES:
        o = s.replace(".c", ".o")
        subprocess.run(["g++", "-c", "-O2", "-D_REENTRANT", s, "-o", o], cwd=exe_dir, check=True,
                       stderr=subprocess.DEVNULL)
        objs.append(o)
    subprocess.run(["g++", "-O2", "-o", "sor06"] + objs + ["-lm"], cwd=exe_dir, check=True)
    return exe_dir


def solver_dir(default=None):
    """The compiled solver directory: argument, else $PULMWK_SOLVER, else None."""
    d = default or os.environ.get("PULMWK_SOLVER")
    if d and os.path.isfile(os.path.join(d, "sor06")):
        return d
    return None


def to_solver_args(x):
    """Code-unit inputs (NAMES order) -> the nine numbers on sor06's command line."""
    x = np.asarray(x, dtype=float)
    a = x.copy()
    a[0] = np.exp(x[0])
    a[2] = np.exp(x[2])
    return a


def waveforms(x, exe_dir, tag=1, timeout=TIMEOUT_S):
    """Run sor06 once. Returns the (512, 9, 6) printed array or None if the run failed.

    Columns: t [s], x [cm], p [mmHg], q [mL/s], A [cm^2], wave speed; rows as NROW above.
    """
    a = to_solver_args(x)
    with tempfile.TemporaryDirectory(prefix="pulmwk_") as tmp:
        for f in RUNFILES:
            os.symlink(os.path.join(exe_dir, f), os.path.join(tmp, f))
        cmd = [os.path.join(exe_dir, "sor06")] + ["%.17g" % v for v in a] + [str(int(tag))]
        try:
            r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0 or "ERROR" in r.stdout:
            return None
        fn = os.path.join(tmp, f"output_{int(tag)}.2d")
        try:
            d = np.loadtxt(fn)
        except (OSError, ValueError):
            return None
    if d.shape != (NT * NROW, 6) or not np.all(np.isfinite(d)):
        return None
    return d.reshape(NT, NROW, 6)


def qois(x, exe_dir, tag=1, timeout=TIMEOUT_S):
    """[p_mean, p_sys, p_dia] in mmHg at the MPA midpoint, or None if the run failed."""
    w = waveforms(x, exe_dir, tag, timeout)
    if w is None:
        return None
    p = w[:, MPA_MID, 2]
    return np.array([p.mean(), p.max(), p.min()])


REL_STEP = 1e-3   # finite-difference step as a fraction of each input's half-range


def forward_gradient(x, f0, exe_dir, rel_step=REL_STEP):
    """Forward differences of the three QoIs in the code-unit inputs (9 extra runs).

    Step h_j = rel_step * (UPPER_j - LOWER_j) / 2. Where x_j + h_j would leave the box the
    difference is backward. Returns (G, flag): G is (3, 9); flag 0 = all forward,
    1 = at least one backward difference, 3 = a perturbed run failed (G is NaN).
    """
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    hw = 0.5 * (UPPER - LOWER)
    G = np.full((3, len(x)), np.nan)
    flag = 0
    for j in range(len(x)):
        h = rel_step * hw[j]
        xs = x.copy()
        sgn = 1.0
        if x[j] + h > UPPER[j]:
            sgn = -1.0
            flag = 1
        xs[j] += sgn * h
        fs = qois(xs, exe_dir)
        if fs is None:
            return np.full((3, len(x)), np.nan), 3
        G[:, j] = sgn * (fs - f0) / h
    return G, flag


def fd_gradient(x, exe_dir, rel_step=1e-3, f0=None):
    """Central differences of the three QoIs in the code-unit inputs (18 extra runs).

    Used by check_accuracy.py. Step h_j = rel_step * (UPPER_j - LOWER_j) / 2; where x_j +- h_j
    would leave the box the step is one-sided (inward), first order. Returns (G, flag): G is
    (3, 9), flag 0 = all central, 1 = some one-sided, -1 = a perturbed run failed (G is NaN).
    """
    x = np.asarray(x, dtype=float)
    hw = 0.5 * (UPPER - LOWER)
    G = np.full((3, len(x)), np.nan)
    flag = 0
    for j in range(len(x)):
        h = rel_step * hw[j]
        xp, xm = x.copy(), x.copy()
        if x[j] + h > UPPER[j]:
            if f0 is None:
                return G, -1
            xm[j] -= h
            fm = qois(xm, exe_dir)
            if fm is None:
                return np.full_like(G, np.nan), -1
            G[:, j] = (f0 - fm) / h
            flag = 1
        elif x[j] - h < LOWER[j]:
            if f0 is None:
                return G, -1
            xp[j] += h
            fp = qois(xp, exe_dir)
            if fp is None:
                return np.full_like(G, np.nan), -1
            G[:, j] = (fp - f0) / h
            flag = 1
        else:
            xp[j] += h
            xm[j] -= h
            fp = qois(xp, exe_dir)
            fm = qois(xm, exe_dir) if fp is not None else None
            if fp is None or fm is None:
                return np.full_like(G, np.nan), -1
            G[:, j] = (fp - fm) / (2 * h)
    return G, flag
