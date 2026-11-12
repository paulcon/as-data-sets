"""Generate demo_runs.csv: PROCESS v3.4.2 optimisations of an EU H-mode DEMO-like tokamak
over the six input intervals of Miralles-Dolz et al. (2022), Table I.

Kind: stored data. PROCESS (UKAEA, MIT licence, https://github.com/ukaea/PROCESS, tag v3.4.2)
must be installed in the Python that runs this script; see DATA.md ("Regenerating"). Without it,
run, check and baseline stop with an install message (assemble needs only numpy and scipy). The
shipped demo_runs.csv is the record: rerun PROCESS only if a problem is found in it.

Design: the first N points of one scrambled Sobol' sequence (scipy.stats.qmc.Sobol, d = 6,
seed SEED) mapped linearly onto the Table I intervals. Each point is one PROCESS optimisation
(minimise R0 subject to P_net >= 400 MW, t_burn >= 7200 s and the deck's other constraints),
warm-started from the optimum of the paper's baseline point, read from the shipped file
baseline_warmstart.json (PROCESS v3.4.2 output, MIT; R0 = 8.356952988650034 m). Every run is
kept; a run that PROCESS does not declare converged (ifail != 1) stays in the file with
converged = 0.

Usage (one process; resumable; each call stops after --cpu-budget CPU seconds). --work is required:
PROCESS writes its scratch files (results.jsonl, MFILEs, process.log) there, never in this directory.
    python generate_data.py run --n 128 --cpu-budget 1080 --work <scratch dir>
    python generate_data.py assemble --work <scratch dir>        # writes demo_runs.csv
    python generate_data.py check --rows 0,5 --work <scratch dir> # rerun rows, compare
    python generate_data.py baseline --work <scratch dir>         # recompute the warm start
Add --recompute-baseline to run/check to warm-start from a recomputed baseline (written to
<scratch dir>/baseline.json) instead of the shipped baseline_warmstart.json; in another
numpy/scipy the recomputed one need not be bitwise identical, and R0 depends on the start.
"""
import argparse, importlib.util, json, os, sys, time, platform

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from process_driver import run_one, NAMES  # noqa: E402

INSTALL = ("PROCESS is not installed. Install PROCESS v3.4.2 (https://github.com/ukaea/PROCESS, tag v3.4.2; "
           "the recipe is in DATA.md, \"Regenerating\") in the Python that runs this script, and rerun "
           "generate_data.py.")

SEED = 675
N_MAX = 128          # Sobol' sequence length drawn (2^7); runs use its first N points
MAXCAL = 50          # VMCON iterations per attempt (PROCESS retries a failed solve twice)

# Table I of the paper (p. 4441): lower, upper, baseline
LOWER = np.array([1.0, 8.7, 0.6, 1e-5, 1.75, 0.06])
UPPER = np.array([1.3, 9.5, 0.8, 1e-4, 1.90, 0.12])
BASE = np.array([1.1, 9.2, 0.75, 5e-5, 1.85, 0.069])   # W baseline printed "50^-5", read 5e-5

OUTCOLS = ["ifail", "rmajor", "p_plant_electric_net_mw", "t_plant_pulse_burn",
           "p_fusion_total_mw", "b_plasma_toroidal_on_axis", "q95",
           "p_div_bt_q_aspect_rmajor_mw", "f_nd_impurity_electrons(13)", "capcost",
           "sqsumsq", "n_solver_iterations", "cpu_s"]
HEADER2 = ("index,H_factor [-],PsepBt_qAR_max [MW T/m],core_radius [-],f_W [-],"
           "kappa_sep [-],f_He [-],converged [1/0],ifail [-],R0 [m],P_net [MW],t_burn [s],"
           "P_fusion [MW],B_T [T],q95 [-],PsepBt_qAR [MW T/m],f_Xe [-],capcost [M$],"
           "sqsumsq [-],n_iter [-],cpu [s]")


def design(n=N_MAX):
    s = qmc.Sobol(d=6, scramble=True, seed=SEED).random_base2(int(np.log2(N_MAX)))
    return (LOWER + (UPPER - LOWER) * s)[:n]


def template():
    return open(os.path.join(HERE, "demo_template_IN.DAT"), encoding="utf-8").read()


BASELINE_FILE = os.path.join(HERE, "baseline_warmstart.json")   # the warm start of every row


def baseline(work, recompute=False):
    """Warm start: the shipped baseline_warmstart.json, unless recompute is True.

    Recomputing solves the Table I baseline point from the deck's starting values with
    R0 = 9.0 m (the stock deck has rmajor = 8.0 m) and writes <work>/baseline.json.
    """
    if not recompute:
        return json.load(open(BASELINE_FILE))
    path = os.path.join(work, "baseline.json")
    res, itv = run_one(template(), BASE, os.path.join(work, "tmp"), "baseline",
                       rmajor0=9.0, maxcal=200)
    if res["ifail"] != 1:
        raise RuntimeError(f"baseline did not converge: {res}")
    out = {"x": BASE.tolist(), "res": res, "itv": itv}
    json.dump(out, open(path, "w"), indent=1)
    return out


def env_info():
    import importlib.metadata as md
    pk = {p: md.version(p) for p in ["process", "numpy", "scipy", "numba", "cvxpy",
                                     "pyvmcon", "clarabel", "coolprop"]}
    return {"python": platform.python_version(), "platform": platform.platform(), **pk}


def do_run(a):
    os.makedirs(a.work, exist_ok=True)
    meta = os.path.join(a.work, "meta.json")
    if not os.path.exists(meta):
        json.dump({"env": env_info(), "seed": SEED, "maxcal": MAXCAL,
                   "date": time.strftime("%Y-%m-%d")}, open(meta, "w"), indent=1)
    t_start = time.process_time()
    base = baseline(a.work, a.recompute_baseline)
    warm = {k: v for k, v in base["itv"].items() if k != "rmajor"}
    X = design(a.n)
    resf = os.path.join(a.work, "results.jsonl")
    done = set()
    if os.path.exists(resf):
        done = {json.loads(l)["index"] for l in open(resf) if l.strip()}
    mf = os.path.join(a.work, "mfiles"); os.makedirs(mf, exist_ok=True)
    for i, x in enumerate(X):
        if i in done:
            continue
        if time.process_time() - t_start > a.cpu_budget:
            print(f"cpu budget reached; {len(done)} of {len(X)} done", flush=True)
            return
        res, _ = run_one(template(), x, os.path.join(a.work, "tmp"), f"s{i:04d}",
                         rmajor0=base["itv"]["rmajor"], maxcal=MAXCAL, warm=warm,
                         keep_mfile_gz=os.path.join(mf, f"s{i:04d}_MFILE.DAT.gz"))
        rec = {"index": i, "x": x.tolist(), **res,
               "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(resf, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done.add(i)
        print(f"{i:4d} ifail={res['ifail']:.0f} R0={res['rmajor']:.4f} "
              f"iter={res['n_solver_iterations']:.0f} cpu={res['cpu_s']:.1f}s {res['error']}",
              flush=True)
    print(f"all {len(X)} done", flush=True)


def do_assemble(a):
    recs = sorted((json.loads(l) for l in open(os.path.join(a.work, "results.jsonl"))
                   if l.strip()), key=lambda r: r["index"])
    idx = [r["index"] for r in recs]
    if idx != list(range(len(idx))):
        raise SystemExit("results are not a contiguous prefix of the design")
    X = design(len(idx))
    rows = []
    for r, x in zip(recs, X):
        if not np.array_equal(np.array(r["x"]), x):
            raise SystemExit(f"row {r['index']}: stored inputs differ from the design")
        ifail = r["ifail"] if not r["error"] else -1.0
        conv = 1.0 if ifail == 1 else 0.0
        rows.append([r["index"], *x, conv, *[ifail if c == "ifail" else r[c] for c in OUTCOLS]])
    rows = np.array(rows, dtype=float)
    meta = json.load(open(os.path.join(a.work, "meta.json")))
    h1 = (f"PROCESS v{meta['env']['process']} optimisations (min R0, P_net>=400 MW, t_burn>=2 h) "
          f"of an EU H-mode DEMO-like tokamak over Miralles-Dolz et al. 2022 Table I; first "
          f"{len(rows)} points of a scrambled Sobol' sequence (seed {SEED}); converged=0 rows are "
          f"PROCESS failures kept as returned")
    np.savetxt(os.path.join(HERE, "demo_runs.csv"), rows, delimiter=",",
               fmt="%.17g", header=h1 + "\n" + HEADER2, comments="# ")
    print(f"wrote demo_runs.csv: {rows.shape}, converged {int(rows[:, 7].sum())}")


def do_check(a):
    data = np.loadtxt(os.path.join(HERE, "demo_runs.csv"), delimiter=",")
    base = baseline(a.work, a.recompute_baseline)
    warm = {k: v for k, v in base["itv"].items() if k != "rmajor"}
    for i in [int(s) for s in a.rows.split(",")]:
        x = data[i, 1:7]
        res, _ = run_one(template(), x, os.path.join(a.work, "tmp"), f"chk{i:04d}",
                         rmajor0=base["itv"]["rmajor"], maxcal=MAXCAL, warm=warm)
        print(f"row {i}: stored R0 {data[i, 9]:.10f} conv {data[i, 7]:.0f} | rerun R0 "
              f"{res['rmajor']:.10f} ifail {res['ifail']:.0f} | rel diff "
              f"{abs(res['rmajor'] - data[i, 9]) / data[i, 9]:.2e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "assemble", "check", "baseline"])
    ap.add_argument("--n", type=int, default=N_MAX)
    ap.add_argument("--cpu-budget", type=float, default=1080.0)
    ap.add_argument("--work", required=True,
                    help="scratch directory for PROCESS's files (results.jsonl, MFILEs, process.log), "
                         "outside this directory")
    ap.add_argument("--rows", default="0")
    ap.add_argument("--recompute-baseline", action="store_true",
                    help="warm-start from a recomputed baseline instead of baseline_warmstart.json")
    a = ap.parse_args()
    a.work = os.path.abspath(a.work)
    if os.path.commonpath([a.work, HERE]) == HERE:
        ap.error("--work must be a scratch directory outside the data-set directory")
    if a.cmd != "assemble" and importlib.util.find_spec("process") is None:
        print(INSTALL, file=sys.stderr)
        sys.exit(3)
    if a.cmd != "assemble":          # PROCESS writes process.log into the working directory
        os.makedirs(a.work, exist_ok=True)
        os.chdir(a.work)
    if a.cmd == "baseline":
        b = baseline(a.work, recompute=True)
        print(f"baseline R0 {b['res']['rmajor']!r} m (shipped file: "
              f"{json.load(open(BASELINE_FILE))['itv']['rmajor']!r} m)")
    else:
        {"run": do_run, "assemble": do_assemble, "check": do_check}[a.cmd](a)
