"""Write inputs.csv and outputs.csv for Pyloric_Circuit_Energy_Consumption.

The model is the pyloric-network simulator of Deistler, Macke and Goncalves (PNAS 2022),
github.com/mackelab/pyloric (MIT licence), commit 89dd27d769fa2652d21bca1dbbc098c85e40244a.
It is NOT part of this repository and is not importable from the repository's .venv:
clone it, then pass its directory with --pyloric (see DATA.md, "Regenerating").

pyloric's package __init__ imports torch and sbi, which are used only by create_prior()
(to wrap a sbi BoxUniform). This script never calls create_prior(); it builds the same
box from pyloric.prior.prior_bounds() and draws its own seeded samples. To avoid a
multi-gigabyte torch install, empty stand-in modules named `torch` and `sbi.utils` are put
in sys.modules before pyloric is imported. pyloric's simulate() and summary_stats() then
run unchanged.

Usage
  python generate_data.py --pyloric /path/to/pyloric --rows 0:100 --out CHUNKDIR
  ...                                                  (one call per block of rows)
  python generate_data.py --merge CHUNKDIR            # writes inputs.csv, outputs.csv here
  python generate_data.py --pyloric ... --check      # all rows, in a temporary directory, compared
  python generate_data.py --pyloric ... --rows 0:5 --out OTHERDIR --check

CHUNKDIR and OTHERDIR must be outside this directory (with --check, --out defaults to a temporary
directory that is removed at the end). Without pyloric (or without pandas or
Cython, which it needs) the script exits with status 3 and says what to install.

--check prints the largest differences of the computed rows from the shipped CSVs and fails only on
structural mismatches (shapes, NaN pattern, spike counts and flags, inputs that differ by more than
INPUT_RTOL).
"""
import argparse
import atexit
import glob
import os
import shutil
import sys
import tempfile
import time
import types

import numpy as np

SEED_X = 401          # Monte Carlo design of the 31 inputs
NOISE_SEED = 1        # one noise realization shared by every run (common random numbers)
M = 1100              # number of samples
DT = 0.025            # ms (paper Methods; pyloric default)
T_MAX = 11000         # ms; the first T_BURN ms are discarded, leaving 10 s (pyloric defaults)
T_BURN = 1000
INPUT_RTOL = 1e-8     # --check fails if the regenerated inputs differ by more (relative)
N_E = 4               # columns of outputs.csv that are energies; the others are counts and 0/1 flags
NOISE_STD = 0.001     # pyloric's noise_std (default): SD of a Gaussian current added at every step,
                      # in the units of the ionic currents (see DATA.md); the paper writes 0.001 mV ms^-0.5

NEURONS = ["AB/PD", "LP", "PY"]
CHANNELS = ["Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "Leak"]
SYNAPSES = ["AB-LP", "PD-LP", "AB-PY", "PD-PY", "LP-PD", "LP-PY", "PY-LP"]
NAMES = [f"g_{c}_{n.replace('/', '')}" for n in NEURONS for c in CHANNELS] + \
        [f"g_syn_{s.replace('-', '_')}" for s in SYNAPSES]
OUT_COLS = ["E_ABPD", "E_LP", "E_PY", "E_total", "spikes_ABPD", "spikes_LP", "spikes_PY",
            "pyloric_like", "finite"]

HERE = os.path.dirname(os.path.abspath(__file__))
PYLORIC_COMMIT = "89dd27d769fa2652d21bca1dbbc098c85e40244a"
INSTALL = (f"{os.path.basename(sys.argv[0])} needs the pyloric simulator (github.com/mackelab/pyloric, commit "
           f"{PYLORIC_COMMIT[:7]}) with numpy, pandas, scipy, Cython, setuptools and a C compiler. Clone it "
           f"(git clone https://github.com/mackelab/pyloric && git -C pyloric checkout {PYLORIC_COMMIT[:7]}), "
           "install the packages (DATA.md, 'Regenerating'), and rerun "
           f"{os.path.basename(sys.argv[0])} with --pyloric PATH.")


def import_pyloric(path):
    """Import pyloric from a clone at `path`, with stand-ins for torch and sbi. Exits with
    status 3 and the install message if pyloric, pandas or Cython cannot be imported."""
    if not path or not os.path.isfile(os.path.join(path, "pyloric", "__init__.py")):
        print(f"pyloric not found{'' if not path else ' at ' + path}. " + INSTALL, file=sys.stderr)
        sys.exit(3)
    try:
        import pandas  # noqa: F401
        import Cython  # noqa: F401
    except ImportError as err:
        print(f"{err.name} is not installed. " + INSTALL, file=sys.stderr)
        sys.exit(3)
    t = types.ModuleType("torch")
    t.Tensor = type("Tensor", (), {})          # only used in an isinstance() check
    sbi_utils = types.ModuleType("sbi.utils")
    sbi_utils.BoxUniform = None                # only used by create_prior()
    sys.modules.setdefault("torch", t)
    sys.modules.setdefault("sbi", types.ModuleType("sbi"))
    sys.modules.setdefault("sbi.utils", sbi_utils)
    sys.path.insert(0, os.path.abspath(path))
    try:
        import pyloric
        from pyloric import simulate, summary_stats
        from pyloric.prior import prior_bounds
        from pyloric.utils import select_names
    except ImportError as err:
        print(f"pyloric could not be imported ({err}). " + INSTALL, file=sys.stderr)
        sys.exit(3)
    return pyloric, simulate, summary_stats, prior_bounds, select_names


def pyloric_setups():
    return {"membrane_gbar": np.ones((3, 8), bool), "Q10_gbar_mem": np.zeros(8, bool),
            "Q10_gbar_syn": np.zeros(2, bool), "Q10_tau_m": np.zeros(1, bool),
            "Q10_tau_h": np.zeros(1, bool), "Q10_tau_CaBuff": np.zeros(1, bool),
            "Q10_tau_syn": np.zeros(2, bool)}


def design(lo, hi):
    """M x 31 samples in pyloric's coordinates (mS/cm^2; synapses ln(g / mS))."""
    u = np.random.default_rng(SEED_X).random((M, 31))
    return lo + (hi - lo) * u


def to_physical(theta):
    """pyloric coordinates -> stored units (mS/cm^2; synapses nS)."""
    X = theta.copy()
    X[:, 24:] = np.exp(theta[:, 24:]) * 1e6
    return X


def run_one(theta, simulate, summary_stats, tn, cn, dt=DT, noise_std=NOISE_STD, seed=NOISE_SEED):
    """One pyloric run at theta (pyloric coordinates); returns the 9 outputs of outputs.csv.
    dt, noise_std and seed default to the settings of the shipped files (check_accuracy.py
    varies them)."""
    import pandas as pd
    df = pd.DataFrame([theta], columns=pd.MultiIndex.from_arrays([tn, cn]))
    out = simulate(df.loc[0], dt=dt, t_max=T_MAX, noise_std=noise_std,
                   track_energy=True, seed=seed)
    finite = bool(np.all(np.isfinite(out["voltage"])) and np.all(np.isfinite(out["energy"])))
    ss = summary_stats(out, stats_customization={"energies": True, "num_spikes": True,
                                                 "pyloric_like": True}, t_burn_in=T_BURN)
    # pyloric's "energies": sum of the power trace (nW/cm^2) over the 10 s, times dt/1000,
    # i.e. nJ/cm^2 per neuron. Divide by 10 s and by 1000: microJ s^-1 cm^-2.
    e = np.array([ss["energies"][k].iloc[0] for k in ["PM", "LP", "PY"]], float)
    e_check = out["energy"][:, int(round(T_BURN / dt)):int(round(T_MAX / dt))].sum(axis=1) * dt / 1000
    assert np.allclose(e, e_check, rtol=1e-12, equal_nan=True)
    e = e / ((T_MAX - T_BURN) / 1000) / 1000
    spikes = [ss["num_spikes"][k].iloc[0] for k in ["PM", "LP", "PY"]]
    pyl = float(bool(ss["pyloric_like"].iloc[0, 0]))
    return np.r_[e, e.sum(), spikes, pyl, float(finite)]


def write_csv(path, A, header1, cols):
    np.savetxt(path, A, delimiter=",", fmt="%.17g",
               header=header1 + "\n" + ",".join(cols), comments="# ")


def merge(chunkdir, outdir):
    files = sorted(glob.glob(os.path.join(chunkdir, "chunk_*.npz")))
    rows, Y, cpu = [], [], 0.0
    for f in files:
        z = np.load(f)
        rows.append(z["rows"]); Y.append(z["Y"]); cpu += float(z["cpu_s"])
    rows = np.concatenate(rows); Y = np.vstack(Y)
    order = np.argsort(rows)
    assert np.array_equal(rows[order], np.arange(M)), "missing or duplicate rows"
    Y = Y[order]
    X = np.loadtxt(os.path.join(chunkdir, "inputs_all.csv"), delimiter=",")
    write_csv(os.path.join(outdir, "inputs.csv"), X,
              f"Pyloric network (pyloric 89dd27d), {M} Monte Carlo samples of the prior box, "
              f"seed {SEED_X}; membrane conductances mS/cm^2, synaptic conductances nS",
              [n + (" [nS]" if n.startswith("g_syn") else " [mS/cm^2]") for n in NAMES])
    write_csv(os.path.join(outdir, "outputs.csv"), Y,
              f"Energy per second over 10 s (after 1 s burn-in), dt {DT} ms, noise seed "
              f"{NOISE_SEED}; energies in microJ s^-1 cm^-2; spikes = counts in 10 s; "
              "pyloric_like and finite are 0/1 flags",
              [c + (" [uJ/s/cm^2]" if c.startswith("E_") else "") for c in OUT_COLS])
    print(f"merged {len(files)} chunks, {M} rows; total simulation CPU {cpu:.0f} s")


def compare(X, Y, rows):
    """Compare inputs X and outputs Y of `rows` with the shipped CSVs; print the largest differences;
    return True on a structural mismatch."""
    X_all = np.loadtxt(os.path.join(HERE, "inputs.csv"), delimiter=",", ndmin=2)
    Y_all = np.loadtxt(os.path.join(HERE, "outputs.csv"), delimiter=",", ndmin=2)
    X_ship, Y_ship = X_all[rows], Y_all[rows]
    if len(X_all) != M or len(Y_all) != M or X.shape != X_ship.shape or Y.shape != Y_ship.shape:
        print(f"FAIL: shipped rows inputs {len(X_all)}, outputs {len(Y_all)} (of {M}); shapes of the {len(rows)} "
              f"rows checked: inputs {X.shape} vs {X_ship.shape}, outputs {Y.shape} vs {Y_ship.shape}")
        return True
    dx = np.max(np.abs(X - X_ship) / np.abs(X_ship).clip(1e-300))
    de = np.nanmax(np.abs(Y[:, :N_E] - Y_ship[:, :N_E]) / np.abs(Y_ship[:, :N_E]).clip(1e-300), initial=0.0)
    print(f"check: largest relative difference (elementwise) inputs {dx:.1e}, energies {de:.1e}")
    fail = False
    for what, bad in (("inputs differ by more than %g (relative)" % INPUT_RTOL, dx > INPUT_RTOL),
                      ("NaN pattern differs", not np.array_equal(np.isnan(Y), np.isnan(Y_ship))),
                      ("spike counts or flags differ", not np.array_equal(Y[:, N_E:], Y_ship[:, N_E:], equal_nan=True))):
        if bad:
            print(f"FAIL: {what}")
            fail = True
    print(f"largest difference from the shipped files: {de:.1e} (relative, energies in outputs.csv)")
    return fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyloric", help="path to a clone of github.com/mackelab/pyloric")
    ap.add_argument("--rows", default=f"0:{M}")
    ap.add_argument("--out", default=None,
                    help="directory for the chunk files, outside this directory (required without --check; "
                         "with --check, default a temporary directory)")
    ap.add_argument("--merge", metavar="CHUNKDIR")
    ap.add_argument("--check", action="store_true",
                    help="compare the rows computed with the shipped CSVs")
    a = ap.parse_args()
    if a.out is None and a.check and not a.merge:
        a.out = tempfile.mkdtemp(prefix="check_")
        atexit.register(shutil.rmtree, a.out, ignore_errors=True)
    if a.merge:
        merge(a.merge, HERE)
        return
    if not a.out or os.path.abspath(a.out) == HERE:
        ap.error("--out DIR is required and must be outside this directory (the chunk files are not shipped)")
    _, simulate, summary_stats, prior_bounds, select_names = import_pyloric(a.pyloric)
    lo, hi = prior_bounds(pyloric_setups())
    tn, cn = select_names(pyloric_setups())
    theta = design(lo, hi)
    os.makedirs(a.out, exist_ok=True)
    np.savetxt(os.path.join(a.out, "inputs_all.csv"), to_physical(theta), delimiter=",",
               fmt="%.17g")
    r0, r1 = (int(s) for s in a.rows.split(":"))
    rows = np.arange(r0, r1)
    Y = np.empty((len(rows), len(OUT_COLS)))
    c0, w0 = time.process_time(), time.time()
    for k, i in enumerate(rows):
        Y[k] = run_one(theta[i], simulate, summary_stats, tn, cn)
    cpu, wall = time.process_time() - c0, time.time() - w0
    np.savez(os.path.join(a.out, f"chunk_{r0:05d}_{r1:05d}.npz"), rows=rows, Y=Y, cpu_s=cpu,
             wall_s=wall)
    print(f"rows {r0}:{r1}: CPU {cpu:.1f} s, wall {wall:.1f} s")
    if a.check and compare(to_physical(theta)[rows], Y, rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
