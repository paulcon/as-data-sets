"""Shannon et al. (2004) rabbit ventricular myocyte, run with Myokit, with the 15
current scaling factors of Yang, Gao, Smith & Simitev (2025)."""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MMT = os.path.join(HERE, "shannon_wang_puglisi_weber_bers_2004_a_p.mmt")

# (name, Myokit variable, lower, upper) -- ranges from Table 4, p. 12 of the paper.
INPUTS = [
    ("pNaK",  "INaK.p",   1e-4, 1.07),
    ("pK1",   "IK1.p",    0.03, 1.01),
    ("pCaL",  "ICaL.p",   0.7,  1.6),
    ("pCap",  "ICap.p",   1e-4, 4.5),
    ("pNa",   "INa.p",    1.0,  10.0),
    ("pKr",   "IKr.p",    1e-4, 3.2),
    ("pNaCa", "INaCa.p",  0.6,  3.0),
    ("pKs",   "IKs.p",    1e-4, 10.0),
    ("pClCa", "ICl_Ca.p", 1e-4, 10.0),
    ("pNab",  "INab.p",   1e-4, 10.0),
    ("ptos",  "Itos.p",   1e-4, 2.5),
    ("pClb",  "IClb.p",   1e-4, 10.0),
    ("pCab",  "ICab.p",   0.34, 10.0),
    ("pKp",   "IKp.p",    1e-4, 10.0),
    ("ptof",  "Itof.p",   1e-4, 4.0),
]
NAMES = [r[0] for r in INPUTS]
LOWER = np.array([r[2] for r in INPUTS])
UPPER = np.array([r[3] for r in INPUTS])

BCL = 500.0      # ms
T0 = 76.0        # ms, stimulus onset within each beat (protocol in the .mmt file)
DT_LOG = 0.1     # ms, logging step of the last two beats

_cache = {}


def _sim(abs_tol=1e-6, rel_tol=1e-4):
    """Compiled simulation (cached per process) and the model file's initial state."""
    key = (abs_tol, rel_tol)
    if key not in _cache:
        import myokit            # imported here, so the constants above need only numpy
        model, protocol, _ = myokit.load(MMT)
        sim = myokit.Simulation(model, protocol)
        sim.set_tolerance(abs_tol, rel_tol)
        _cache[key] = (sim, list(sim.default_state()))
    return _cache[key]


def biomarkers(t, V, Cai):
    """Biomarkers of one beat (t in [0, BCL), stimulus at T0), per Table 3 of the paper."""
    vmax = V.max()
    ipk = int(V.argmax())
    vrest = V[-1]                       # V(t_BCL): last logged point of the beat
    amp = vmax - vrest

    def apd(frac):
        thr = vmax - frac * abs(amp)
        after = np.where(V[ipk:] <= thr)[0]
        if len(after) == 0:
            return np.nan
        j = ipk + after[0]
        # linear interpolation between samples j-1 and j
        t1, t2, v1, v2 = t[j - 1], t[j], V[j - 1], V[j]
        tc = t1 + (thr - v1) * (t2 - t1) / (v2 - v1) if v2 != v1 else t2
        return tc - T0

    a90 = apd(0.9)
    a30 = apd(0.3)
    vplt = np.interp(T0 + a90 / 2.0, t, V) if np.isfinite(a90) else np.nan
    bulk = np.trapezoid(np.abs(V - vrest), t)
    return dict(Vmax=vmax, Vrest=vrest, A90=a90, A30=a30, Vplt=vplt, B=bulk,
                Cai_ED=Cai[int(np.searchsorted(t, T0))], Cai_PSV=Cai.max())


def _log_beats(sim, n):
    """Run n beats from the current state, logging V and [Ca]i every DT_LOG ms."""
    rel = np.arange(0.0, n * BCL, DT_LOG)
    t_start = sim.time()
    d = sim.run(n * BCL, log=["cell.V", "Ca_buffer.Cai"], log_times=t_start + rel)
    return np.asarray(d["cell.V"]), np.asarray(d["Ca_buffer.Cai"])


def run(p, n_beats=1000, abs_tol=1e-6, rel_tol=1e-4, stim_amplitude=9.5, check_beat=500,
        return_traces=False):
    """Pace n_beats beats at BCL 500 ms from the model file's initial state.

    Returns the biomarkers of the last beat, of the second-to-last beat (prefix 'prev_'),
    of beat `check_beat` (prefix 'b500_' for the default; None skips it), and a response
    class: 'normal' (1:1), 'no_AP', 'alternans_or_2to1'.

    Note: Myokit's Simulation.pre() replaces the default state, so the initial state is
    restored explicitly here; every call starts from the same state (stateless).
    """
    sim, s0 = _sim(abs_tol, rel_tol)
    sim.set_default_state(s0)
    sim.reset()                       # state = model file's initial state, time = 0
    for (name, var, lo, hi), val in zip(INPUTS, p):
        sim.set_constant(var, float(val))
    sim.set_constant("cell.stim_amplitude", float(stim_amplitude))   # A/F
    nb = int(round(BCL / DT_LOG))
    tb = np.arange(nb) * DT_LOG
    out = {}
    done = 0
    if check_beat:
        assert 1 <= check_beat <= n_beats - 2
        sim.pre((check_beat - 1) * BCL)
        V, C = _log_beats(sim, 1)
        out.update({"b500_" + k: v for k, v in biomarkers(tb, V[:nb], C[:nb]).items()})
        done = check_beat
    sim.pre((n_beats - 2 - done) * BCL)
    V, C = _log_beats(sim, 2)
    prev = biomarkers(tb, V[:nb], C[:nb])
    last = biomarkers(tb, V[nb:2 * nb], C[nb:2 * nb])
    out.update(last)
    out.update({"prev_" + k: v for k, v in prev.items()})
    out["cls"] = classify(prev, last)
    if return_traces:
        out["trace"] = (tb, V[nb:2 * nb], C[nb:2 * nb])
    return out


def classify(prev, last, vmax_thr=0.0, rel_tol_alt=0.02):
    if not (last["Vmax"] > vmax_thr and prev["Vmax"] > vmax_thr):
        return "no_AP"
    if not (np.isfinite(last["A90"]) and np.isfinite(prev["A90"])):
        return "no_AP"
    if abs(last["A90"] - prev["A90"]) > rel_tol_alt * max(abs(last["A90"]), 1.0):
        return "alternans_or_2to1"
    return "normal"
