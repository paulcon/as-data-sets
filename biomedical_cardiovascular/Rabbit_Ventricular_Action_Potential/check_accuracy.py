"""Recompute the accuracy numbers quoted in DATA.md for Rabbit_Ventricular_Action_Potential (needs Myokit).

    python check_accuracy.py                  # checks 1-5 with Myokit's CVODES simulation (the solver of
                                              # the shipped data); about 3 min of CPU plus compiling
    python check_accuracy.py --checks 1,2     # only some checks
    python check_accuracy.py --scipy          # checks 1-3 with Myokit's parser and scipy's LSODA instead of
                                              # CVODES (no SUNDIALS needed); 6-12 min of CPU per 1000-beat
                                              # run, about 1 h in all
    python check_accuracy.py --authors        # check 6 only: numpy, no Myokit; downloads 1.8 MB

Prints, beside the values DATA.md quotes:

1. Row 0 of rabbit_ap_samples.csv rerun (1000 beats from the model file's initial state): the
   biomarkers of beats 500 and 1000 against the CSV, as the maximum relative difference. This checks
   the integrator (with --scipy, a different one from the shipped data's).
2. pNaK one at a time: pNaK = 0.1, 0.5 and 1.0, the other 14 factors at 1 (the paper's Fig 2(a)
   sweep): Vmax, A90, A30 and [Na]i at the end of beats 5, 20, 100, 500 and 1000. The paper's
   Fig 2(a) shows A90 of about 310, 245 and 205 ms at these pNaK.
3. The fixed subsarcolemmal Na reversal potential: at the nominal point (all factors 1) and at
   row 0, the biomarkers of beat 1000 with the file's E_Na_SL = -15 mV and with the Nernst form
   E_Na_SL = RT/F log([Na]o/[Na]_SL) that the file comments out (Shannon et al. 2004, Eq 10).
4. (CVODES only) The near-threshold stimulus at the nominal point: A90 of beats 499-501 after one
   498-beat pre-pacing call; A90 of beat 500 after 250 + 249 beats; A90 of beat 1000 for three
   splits of the train (the shipped sequence 499 + log 500 + 498 + log 999-1000; one 998-beat
   call + log 999-1000; 249 + log 250 + 748 + log 999-1000). Then the first 10 normal rows of the
   CSV with the second and third splits: A90 of beat 1000 against the CSV.
5. (CVODES only) Solver tolerance at the nominal point (one 998-beat call): A90 of beat 1000 with
   CVODES abs/rel tolerances 1e-6/1e-4 (as shipped) and 1e-5/1e-3.
6. (--authors; numpy only, needs the network) The authors' stored full-model runs, read from their
   repository at commit f11c5e6 (github.com/ZhechaoYanggla/Sobol_paper): the median and 5-95 % range of
   the 2000 runs in `fig060/reduced model.zip` (`1/apd90b.npy`, ...; checked identical in all 15
   folders), the share of A90 values at the 499.9 ms ceiling, and the R^2 of their six-factor reduced
   model against the full model for A90 (`fig080/apd90_NS.npy` against `fig080/apd90_R.npy`), with and
   without fig080.py's filter of values >= 499 ms. The arrays are loaded with allow_pickle=False and kept
   in memory.

Checks 1-5 run each logged beat as a separate integration call after the unlogged beats (check 1 in the
default mode uses the shipped sequence of shannon_driver.run instead). Deterministic. Checks 1-5 need numpy, scipy and Myokit (pip install myokit; the shipped data used Myokit
1.39.2). The default mode also needs SUNDIALS (CVODES) and a C compiler, which Myokit uses to
compile the model; --scipy needs neither. Without Myokit the script stops with an error that says
what to install. It reads the files next to it and writes nothing.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shannon_driver as sm  # noqa: E402
import generate_data as gd  # noqa: E402

BCL, T0, DT = sm.BCL, sm.T0, sm.DT_LOG
NB = int(round(BCL / DT))                       # logged points per beat
TB = np.arange(NB) * DT
STIM = (int(round(T0 / DT)), int(round((T0 + 5.0) / DT)))   # stimulus [76, 81) ms, from the .mmt protocol
NERNST = "model_parameters.R * model_parameters.T / model_parameters.F * log(model_parameters.Nao / Na_buffer.Na_SL)"
LOG = ["cell.V", "Ca_buffer.Cai", "Na_buffer.Nai"]
NOMINAL = np.ones(15)


def need_myokit():
    try:
        import myokit  # noqa: F401
    except ImportError:
        print("Myokit is not installed. Install it (pip install myokit; the shipped data used Myokit 1.39.2; "
              "the default mode also needs SUNDIALS and a C compiler, see DATA.md), then rerun "
              "check_accuracy.py, or rerun it with --scipy, which needs Myokit but not SUNDIALS.", file=sys.stderr)
        sys.exit(3)
    return myokit


def load_model(nernst=False):
    myokit = need_myokit()
    model, protocol, _ = myokit.load(sm.MMT)
    if nernst:
        model.get("reversal_potentials.E_Na_SL").set_rhs(NERNST)
    return model, protocol


# ---------------------------------------------------------------- CVODES (Myokit's Simulation) -----
class CvodesRun:
    """One 1000-beat train with Myokit's Simulation, logged at chosen beats."""
    _sims = {}

    def __init__(self, p, nernst=False, abs_tol=1e-6, rel_tol=1e-4):
        myokit = need_myokit()
        key = (nernst, abs_tol, rel_tol)
        if key not in self._sims:
            model, protocol = load_model(nernst)
            sim = myokit.Simulation(model, protocol)
            sim.set_tolerance(abs_tol, rel_tol)
            inai = [v.qname() for v in model.states()].index("Na_buffer.Nai")
            self._sims[key] = (sim, list(sim.default_state()), inai)
        self.sim, s0, self.inai = self._sims[key]
        self.sim.set_default_state(s0)
        self.sim.reset()
        for (name, var, lo, hi), val in zip(sm.INPUTS, p):
            self.sim.set_constant(var, float(val))
        self.sim.set_constant("cell.stim_amplitude", 9.5)
        self.beat = 0

    def skip(self, n):
        if n > 0:
            self.sim.pre(n * BCL)
            self.beat += n

    def log(self, n=1):
        rel = np.arange(0.0, n * BCL, DT)
        d = self.sim.run(n * BCL, log=LOG, log_times=self.sim.time() + rel)
        self.beat += n
        return [{k: np.asarray(d[k])[i * NB:(i + 1) * NB] for k in LOG} for i in range(n)], \
            float(self.sim.state()[self.inai])


# ---------------------------------------------------------------- scipy LSODA on Myokit's parse -----
def python_rhs(p, nernst=False):
    """The model as a flat Python function rhs(y, t, pace), generated from Myokit's parse of the .mmt."""
    myokit = need_myokit()
    from myokit.formats.python import PythonExpressionWriter
    model, _ = load_model(nernst)
    for (name, var, lo, hi), val in zip(sm.INPUTS, p):
        model.get(var).set_rhs(float(val))
    states = list(model.states())
    tvar, pvar = model.binding("time"), model.binding("pace")

    def nm(var):
        return "v_" + var.qname().replace(".", "__")

    def lhs(x):
        if isinstance(x, myokit.Derivative):
            return "d_" + nm(x.var())
        return nm(x.var() if isinstance(x, myokit.Name) else x)

    w = PythonExpressionWriter()
    w.set_lhs_function(lhs)
    const, body = [], []
    for eqs in model.solvable_order().values():
        for eq in eqs:
            var = eq.lhs.var()
            if var is tvar or var is pvar:
                continue
            (const if var.is_constant() else body).append(w.eq(eq))
    src = ["import math", "from math import exp, log, sqrt, floor", "def make():"]
    src += ["    " + c for c in const]
    src += ["    def rhs(y, t, pace):", f"        {nm(tvar)} = t", f"        {nm(pvar)} = pace"]
    src += [f"        {nm(v)} = y[{i}]" for i, v in enumerate(states)]
    src += ["        " + b for b in body]
    src += ["        return [" + ", ".join("d_" + nm(v) for v in states) + "]", "    return rhs"]
    ns = {}
    exec("\n".join(src), ns)          # code generated here from the model file, not read from elsewhere
    names = [v.qname() for v in states]
    return ns["make"](), np.array(model.initial_values(as_floats=True)), names


class ScipyRun:
    """One train integrated with scipy's LSODA (rtol 1e-6, atol 1e-8), restarted at the stimulus edges."""

    def __init__(self, p, nernst=False):
        self.rhs, self.y, names = python_rhs(p, nernst)
        self.iv, self.ica, self.ina = (names.index(n) for n in LOG)
        self.beat = 0

    def _piece(self, a, c, pace, grid=None):
        from scipy.integrate import odeint
        T = self.beat * BCL
        t = np.r_[T + a, T + c] if grid is None else np.r_[T + grid, T + c]
        Y = odeint(self.rhs, self.y, t, args=(pace,), rtol=1e-6, atol=1e-8, mxstep=100000)
        self.y = Y[-1]
        return Y[:-1]

    def _beat(self, logged):
        out = []
        for (i0, i1), pace in (((0, STIM[0]), 0.0), (STIM, -1.0), ((STIM[1], NB), 0.0)):
            grid = np.arange(i0, i1) / 10.0 if logged else None
            Y = self._piece(i0 / 10.0, i1 / 10.0, pace, grid)
            if logged:
                out.append(Y)
        self.beat += 1
        return np.vstack(out) if logged else None

    def skip(self, n):
        for _ in range(n):
            self._beat(False)

    def log(self, n=1):
        res = []
        for _ in range(n):
            Y = self._beat(True)
            res.append({"cell.V": Y[:, self.iv], "Ca_buffer.Cai": Y[:, self.ica], "Na_buffer.Nai": Y[:, self.ina]})
        return res, float(self.y[self.ina])


# ---------------------------------------------------------------- helpers -----------------------------
def train(make, p, beats, **kw):
    """Pace from the initial state; return {beat: (biomarkers, [Na]i at the end of the beat)}."""
    r = make(p, **kw)
    out = {}
    for b in sorted(beats):
        r.skip(b - 1 - r.beat)
        (d,), nai = r.log(1)
        out[b] = (sm.biomarkers(TB, d["cell.V"], d["Ca_buffer.Cai"]), nai)
    return out


def fmt(bm, keys=("Vmax", "A90", "A30")):
    return ", ".join(f"{k} {bm[k]:.1f}" for k in keys)


def check1(make, D):
    print("\n1. Row 0 of the CSV rerun from the model file's initial state")
    col = {c: j for j, c in enumerate(gd.COLS)}
    t = time.process_time()
    if make is CvodesRun:          # the shipped pacing sequence (shannon_driver.run), as generate_data.py
        r = sm.run(D[0, :15], n_beats=gd.N_BEATS, check_beat=gd.CHECK_BEAT)
        res = {500: ({q: r["b500_" + q] for q in gd.QOIS},), 1000: ({q: r[q] for q in gd.QOIS},)}
    else:
        res = train(make, D[0, :15], [500, 1000])
    for b, pre in ((500, "b500_"), (1000, "")):
        bm = res[b][0]
        rel = max(abs(bm[q] - D[0, col[pre + q]]) / abs(D[0, col[pre + q]]) for q in bm)
        print(f"   beat {b}: {fmt(bm)} (CSV: Vmax {D[0, col[pre + 'Vmax']]:.1f}, A90 {D[0, col[pre + 'A90']]:.1f}, "
              f"A30 {D[0, col[pre + 'A30']]:.1f}); max relative difference over the 8 biomarkers {rel:.1e}")
    print(f"   ({time.process_time() - t:.0f} s CPU)")


def check2(make):
    print("\n2. pNaK one at a time (others 1): A90 (ms), Vmax (mV), A30 (ms), [Na]i at the end of the beat (mM)")
    beats = [5, 20, 100, 500, 1000]
    for pnak in (0.1, 0.5, 1.0):
        t = time.process_time()
        p = NOMINAL.copy()
        p[sm.NAMES.index("pNaK")] = pnak
        res = train(make, p, beats)
        for b in beats:
            bm, nai = res[b]
            ap = "" if bm["Vmax"] > 0 else "  (no AP: Vmax <= 0)"
            print(f"   pNaK {pnak:.1f} beat {b:4d}: A90 {bm['A90']:6.1f}, Vmax {bm['Vmax']:6.1f}, A30 {bm['A30']:6.1f}, "
                  f"[Na]i {nai:5.2f}{ap}")
        print(f"   ({time.process_time() - t:.0f} s CPU)")


def check3(make, D):
    print("\n3. E_Na_SL fixed at -15 mV (the file) against the Nernst form, beat 1000")
    for lab, p in (("nominal (all factors 1)", NOMINAL), ("row 0", D[0, :15])):
        for nernst in (False, True):
            t = time.process_time()
            bm, nai = train(make, p, [1000], nernst=nernst)[1000]
            print(f"   {lab:24s} {'Nernst E_Na_SL' if nernst else 'file, -15 mV  '}: {fmt(bm)}, [Na]i {nai:.2f} "
                  f"({time.process_time() - t:.0f} s CPU)")


def check4(D):
    print("\n4. Near-threshold stimulus (CVODES): how the train is split")
    r = CvodesRun(NOMINAL)
    r.skip(498)
    ds, _ = r.log(3)
    print("   nominal, one 498-beat call, then beats 499-501: A90 "
          + ", ".join(f"{sm.biomarkers(TB, d['cell.V'], d['Ca_buffer.Cai'])['A90']:.1f}" for d in ds) + " ms")
    r = CvodesRun(NOMINAL)
    r.skip(250)
    r.skip(249)
    (d,), _ = r.log(1)
    print(f"   nominal, 250 + 249 beats, then beat 500: A90 {sm.biomarkers(TB, d['cell.V'], d['Ca_buffer.Cai'])['A90']:.1f} ms")

    def split(p, how):
        r = CvodesRun(p)
        if how == "shipped":
            r.skip(499); r.log(1); r.skip(498)
        elif how == "one call":
            r.skip(998)
        else:
            r.skip(249); r.log(1); r.skip(748)
        ds, _ = r.log(2)
        return sm.biomarkers(TB, ds[1]["cell.V"], ds[1]["Ca_buffer.Cai"])["A90"]

    hows = ("shipped", "one call", "check at 250")
    print("   nominal, A90 of beat 1000: " + ", ".join(f"{h} {split(NOMINAL, h):.2f}" for h in hows) + " ms")
    col = {c: j for j, c in enumerate(gd.COLS)}
    rows = np.flatnonzero(D[:, col["response"]] == 0)[:10]
    worst = []
    for i in rows:
        a = [split(D[i, :15], h) for h in hows[1:]]
        c = D[i, col["A90"]]
        worst.append(max(abs(x - c) / c for x in a))
        print(f"   row {i:3d}: A90 CSV {c:7.2f}; one call {a[0]:7.2f}; check at 250 {a[1]:7.2f} ms")
    w = np.array(worst)
    print(f"   rows within 0.03 %: {int(np.sum(w < 3e-4))} of {len(w)}; largest relative difference {w.max():.1%}")


def check5():
    print("\n5. Solver tolerance (CVODES), nominal point, one 998-beat call, A90 of beat 1000")
    for at, rt in ((1e-6, 1e-4), (1e-5, 1e-3)):
        r = CvodesRun(NOMINAL, abs_tol=at, rel_tol=rt)
        r.skip(998)
        ds, _ = r.log(2)
        print(f"   abs/rel {at:g}/{rt:g}: {sm.biomarkers(TB, ds[1]['cell.V'], ds[1]['Ca_buffer.Cai'])['A90']:.2f} ms")


AUTHORS = ("https://raw.githubusercontent.com/ZhechaoYanggla/Sobol_paper/"
           "f11c5e627c33b63479baaa4dae821eb342a82e68/")


def check6():
    import io
    import urllib.request
    import zipfile
    print("\n6. The authors' stored runs (repository commit f11c5e6)")

    def get(path):
        with urllib.request.urlopen(AUTHORS + path.replace(" ", "%20"), timeout=60) as r:
            return r.read()

    def npy(b):
        return np.load(io.BytesIO(b), allow_pickle=False)

    z = zipfile.ZipFile(io.BytesIO(get("fig060/reduced model.zip")))
    names = [f"{i}/{{}}.npy" for i in range(1, 16)]
    for key, lab in (("apd90b", "A90 (ms)"), ("apd30b", "A30 (ms)"), ("vmaxb", "Vmax (mV)"), ("vpltb", "Vplt (mV)"),
                     ("vrestb", "Vrest (mV)")):
        arrs = [npy(z.read(n.format(key))) for n in names]
        same = all(np.array_equal(arrs[0], b) for b in arrs[1:])
        p50, p5, p95 = np.percentile(arrs[0], [50, 5, 95])
        print(f"   {lab:10s} median {p50:6.1f} (5-95 %: {p5:.1f} to {p95:.1f}); n = {len(arrs[0])}; "
              f"identical in the 15 folders: {same}")
    a90 = npy(z.read("1/apd90b.npy"))
    print(f"   A90 values at the 499.9 ms ceiling: {np.mean(a90 > 499.8):.1%}; minimum {a90.min():.1f} ms")
    R, NS = npy(get("fig080/apd90_R.npy")), npy(get("fig080/apd90_NS.npy"))

    def r2(y, yh):
        return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)

    m = (R < 499) & (NS < 499)
    print(f"   fig080, six-factor model against the full model for A90: R^2 = {r2(R, NS):.3f} ({len(R)} runs); "
          f"{r2(R[m], NS[m]):.3f} after removing values >= 499 ms ({m.sum()} runs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scipy", action="store_true", help="checks 1-3 with scipy LSODA instead of CVODES")
    ap.add_argument("--authors", action="store_true", help="check 6 only (numpy, network)")
    ap.add_argument("--checks", default=None, help="comma-separated list, e.g. 1,2")
    a = ap.parse_args()
    if a.authors:
        check6()
        return
    myokit = need_myokit()
    todo = [int(c) for c in a.checks.split(",")] if a.checks else ([1, 2, 3] if a.scipy else [1, 2, 3, 4, 5])
    if a.scipy and any(c > 3 for c in todo):
        sys.exit("checks 4 and 5 are about CVODES and need the default mode")
    if not a.scipy:
        try:
            myokit.Simulation(*load_model())
        except Exception as e:  # noqa: BLE001
            print(f"Myokit cannot compile a simulation ({type(e).__name__}: {str(e).splitlines()[0]}). Install SUNDIALS and a C "
                  "compiler (see DATA.md), then rerun check_accuracy.py, or rerun it with --scipy.", file=sys.stderr)
            sys.exit(3)
    make = ScipyRun if a.scipy else CvodesRun
    print(f"Myokit {myokit.__version__}, numpy {np.__version__}; integrator: "
          + ("scipy odeint (LSODA), rtol 1e-6, atol 1e-8" if a.scipy else "CVODES, abs/rel tolerance 1e-6/1e-4"))
    D = np.loadtxt(os.path.join(HERE, "rabbit_ap_samples.csv"), delimiter=",")
    t0 = time.time()
    for c in todo:
        {1: lambda: check1(make, D), 2: lambda: check2(make), 3: lambda: check3(make, D),
         4: lambda: check4(D), 5: check5}[c]()
    print(f"\nwall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
