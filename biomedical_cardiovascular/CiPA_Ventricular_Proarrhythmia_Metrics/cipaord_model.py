"""CiPAORdv1.0 human ventricular endocardial myocyte (Myokit) with the nine virtual-drug
inputs of Parikh, Di Achille, Kozloski & Gurev (2019), Front. Pharmacol. 10:1054.

The model file cipaord_endo.mmt is a Myokit re-expression of CiPAORdv1.0: its ORd core
follows the O'Hara-Rudy reference implementation (BSD-3, copyright T. O'Hara and Y. Rudy;
notice in the .mmt header and in LICENSE-ORd), and its hERG/drug-binding section is written
in flux form from Li et al. (2017); it was checked against the FDA C code (see DATA.md).
This module is MIT. The constants below need only numpy;
running the model needs Myokit (which compiles the model to C and integrates it with
SUNDIALS CVODES).
"""
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MMT = os.path.join(HERE, "cipaord_endo.mmt")

# (name, Myokit variable, lower, upper, units) -- Table 2, p. 3 of the paper.
# Ku: Table 2 prints 0-1 /ms; the text on p. 3 gives 1e-5 to 1 (bounds of the 28 CiPA
# drugs). We use the text's lower bound, so every virtual drug binds hERG at a nonzero rate.
INPUTS = [
    ("bINa",  "drug.bINa",  0.0,    80.0, "%"),
    ("bINaL", "drug.bINaL", 0.0,    80.0, "%"),
    ("bIto",  "drug.bIto",  0.0,    80.0, "%"),
    ("bIKs",  "drug.bIKs",  0.0,    80.0, "%"),
    ("bICaL", "drug.bICaL", 0.0,    80.0, "%"),
    ("bIK1",  "drug.bIK1",  0.0,    80.0, "%"),
    ("sbIKr", "drug.sbIKr", 0.0,     4.0, "-"),
    ("Vhalf", "drug.Vhalf", -200.0, -1.0, "mV"),
    ("Ku",    "drug.Ku",    1e-5,    1.0, "1/ms"),
]
NAMES = [r[0] for r in INPUTS]
LOWER = np.array([r[2] for r in INPUTS])
UPPER = np.array([r[3] for r in INPUTS])
UNITS = [r[4] for r in INPUTS]

CL = 2000.0        # ms, pacing cycle length (paper p. 4)
N_BEATS = 1000     # beats to steady state (paper p. 4)
N_LAST = 5         # last beats analysed (paper p. 4)
T_FINE = 20.0      # ms, length of the finely logged start of each analysed beat
DT_FINE = 0.01     # ms, logging step in [0, T_FINE)
DT_COARSE = 0.25   # ms, logging step in [T_FINE, CL)
LOG_VARS = ["engine.time", "membrane.V", "dot(membrane.V)", "conc.cai", "qnet.q"]

QOI_NAMES = ["qNet", "APD90", "APD50", "peakVm", "diastolicCa", "peakCa", "CaTD50", "CaTD90"]
QOI_UNITS = ["nC/uF", "ms", "ms", "mV", "nM", "nM", "ms", "ms"]

_cache = {}


def _sim(abs_tol, rel_tol):
    """Compiled simulation, cached per process, and the model file's initial state."""
    key = (abs_tol, rel_tol)
    if key not in _cache:
        import myokit            # imported here, so the constants above need only numpy
        model, protocol, _ = myokit.load(MMT)
        sim = myokit.Simulation(model, protocol)
        sim.set_tolerance(abs_tol=abs_tol, rel_tol=rel_tol)
        iq = [v.qname() for v in model.states()].index("qnet.q")
        _cache[key] = (sim, list(model.initial_values(as_floats=True)), iq)
    return _cache[key]


def _crossing(t, x, i0, level):
    """First time after index i0 at which x falls to `level` (linear interpolation)."""
    k = np.nonzero(x[i0:] <= level)[0]
    if len(k) == 0:
        return np.nan
    j = i0 + k[0]
    if j == 0 or x[j - 1] == x[j]:
        return t[j]
    return t[j - 1] + (level - x[j - 1]) * (t[j] - t[j - 1]) / (x[j] - x[j - 1])


def beat_metrics(t, V, dV, cai, qnet):
    """Derived features of one beat (t from 0 at the stimulus), per Table 3 of the paper and
    the definitions of the FDA CiPA code (AP_simulation.R, metric_funs.R), with crossing
    times interpolated on the logging grid instead of read off a 1 ms grid.

    APDx: from the time of max dV/dt to the first time after the peak at which
    V <= Vrest + (1 - x)(Vmax - Vrest), Vrest = min V before the peak. CaTDx: from the time
    [Ca]i first exceeds 90 % of its upstroke to the first time after the peak at which
    [Ca]i <= Ca_rest + (1 - x)(Ca_max - Ca_rest). max_dv: largest dV/dt between 30 %
    repolarisation and APD90 (> 0 marks an EAD, as in the FDA code).
    """
    out = dict(qNet=qnet)
    iup = int(np.argmax(dV))
    t_up = t[iup]
    imax = int(np.argmax(V))
    vmax = V[imax]
    vrest = V[:imax + 1].min()
    out["peakVm"] = vmax
    out["dVdt_max"] = dV[iup]
    apd50 = apd90 = max_dv = np.nan
    if vmax >= 0 and vrest < -40:
        amp = vmax - vrest
        t30 = _crossing(t, V, imax, vrest + 0.7 * amp)
        if np.isfinite(t30):
            t50 = _crossing(t, V, imax, vrest + 0.5 * amp)
            if np.isfinite(t50):
                apd50 = t50 - t_up
                t90 = _crossing(t, V, imax, vrest + 0.1 * amp)
                if np.isfinite(t90):
                    apd90 = t90 - t_up
                    sel = (t > t30) & (t < t90)
                    if t90 - t30 > 1 and sel.any():
                        max_dv = dV[sel].max()
    out["APD90"], out["APD50"], out["max_dv"] = apd90, apd50, max_dv
    jmax = int(np.argmax(cai))
    camax = cai[jmax]
    carest = cai[:jmax + 1].min()
    out["peakCa"] = camax * 1e6          # mM -> nM
    out["diastolicCa"] = carest * 1e6
    up = np.nonzero(cai > carest + 0.9 * (camax - carest))[0]
    catd50 = catd90 = np.nan
    if len(up):
        j = up[0]
        lev = carest + 0.9 * (camax - carest)
        t_caup = t[j] if j == 0 else t[j - 1] + (lev - cai[j - 1]) * (t[j] - t[j - 1]) / (cai[j] - cai[j - 1])
        c50 = _crossing(t, cai, jmax, carest + 0.5 * (camax - carest))
        if np.isfinite(c50):
            catd50 = c50 - t_caup
            c90 = _crossing(t, cai, jmax, carest + 0.1 * (camax - carest))
            if np.isfinite(c90):
                catd90 = c90 - t_caup
    out["CaTD50"], out["CaTD90"] = catd50, catd90
    return out


def _log_beat(sim, iq):
    """Run one beat from the current state (time = k*CL), q reset to 0, logged finely."""
    st = sim.state()
    st[iq] = 0.0
    sim.set_state(st)
    t0 = sim.time()
    d1 = sim.run(T_FINE, log=LOG_VARS, log_interval=DT_FINE)
    d2 = sim.run(CL - T_FINE, log=LOG_VARS, log_interval=DT_COARSE)
    q_end = sim.state()[iq]
    arr = {k: np.concatenate([np.asarray(d1[k]), np.asarray(d2[k])]) for k in LOG_VARS}
    t = arr["engine.time"] - t0
    return beat_metrics(t, arr["membrane.V"], arr["dot(membrane.V)"], arr["conc.cai"], q_end), \
        (t, arr["membrane.V"], arr["conc.cai"])


def run(x, n_beats=N_BEATS, n_last=N_LAST, check_beat=500, abs_tol=1e-6, rel_tol=1e-4,
        return_traces=False):
    """Pace n_beats beats at CL = 2000 ms from the control steady state with virtual drug x
    (9 inputs in the order of NAMES, physical units).

    Returns a dict: the features of the last beat (QOI_NAMES, plus max_dv, dVdt_max), the
    number of EAD beats and of beats without full repolarisation among the last n_last beats,
    the spread (max - min) of qNet and APD90 over those beats, qNet and APD90 of beat
    `check_beat` (convergence check; None skips it), and the CPU time.
    Stateless: every call restores the model file's initial state and time 0.
    The default tolerances (abs 1e-6, rel 1e-4) are those of the shipped data.
    """
    import myokit
    c0 = time.process_time()
    sim, s0, iq = _sim(abs_tol, rel_tol)
    sim.reset()
    sim.set_time(0)
    sim.set_state(s0)
    for (name, var, lo, hi, u), val in zip(INPUTS, x):
        sim.set_constant(var, float(val))
    out = {}
    done = 0
    if check_beat:
        assert 1 <= check_beat < n_beats - n_last
        sim.run((check_beat - 1) * CL, log=myokit.LOG_NONE)
        mb, _ = _log_beat(sim, iq)
        out["qNet_b500"], out["APD90_b500"] = mb["qNet"], mb["APD90"]
        done = check_beat
    sim.run((n_beats - n_last - done) * CL, log=myokit.LOG_NONE)
    beats = []
    for k in range(n_last):
        mb, tr = _log_beat(sim, iq)
        beats.append(mb)
    last = beats[-1]
    out.update({k: last[k] for k in QOI_NAMES + ["max_dv", "dVdt_max"]})
    mdv = np.array([b["max_dv"] for b in beats])
    out["n_EAD"] = int(np.sum(mdv[np.isfinite(mdv)] > 0))
    out["n_norepol"] = int(np.sum(~np.isfinite([b["APD90"] for b in beats])))
    q = np.array([b["qNet"] for b in beats]); a = np.array([b["APD90"] for b in beats])
    out["qNet_spread"] = float(q.max() - q.min())
    out["APD90_spread"] = float(a.max() - a.min()) if np.all(np.isfinite(a)) else np.nan
    out["cpu_s"] = time.process_time() - c0
    if return_traces:
        out["trace"] = tr
    return out


def response_class(r):
    """0 normal; 1 EAD in any of the last beats; 2 no full repolarisation or no AP."""
    if r["n_norepol"] > 0 or not np.isfinite(r["APD90"]):
        return 2
    if r["n_EAD"] > 0:
        return 1
    return 0
