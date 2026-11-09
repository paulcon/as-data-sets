"""OpenDSS driver for the IEEE European LV Test Feeder at the on-peak minute (09:26).

This module drives OpenDSS (through OpenDSSDirect.py) and is not a numpy/scipy model. Its
constants (input names, nominal values, distributions, finite-difference steps) need only numpy,
so the notebook can import them without OpenDSS. Solving needs OpenDSSDirect.py; the shipped
files were written with OpenDSSDirect.py 0.9.4 (DSS C-API 0.14.5, OpenDSS SVN 3723):

    pip install opendssdirect.py==0.9.4

The feeder files in feeder/ are the IEEE European LV Test Feeder as distributed in
dss-extensions/electricdss-tst (Version8/Distrib/IEEETestCases/LVTestCase, commit 3b20839),
unchanged. The model:

* unbalanced three-phase power flow (OpenDSS snapshot solution), 50 Hz, 11 kV source at
  1.05 p.u. behind its short-circuit impedance, 800 kVA 11/0.416 kV Dyn transformer, 905 lines,
  55 single-phase loads (feeder/Master.dss as distributed);
* the 55 load demands are set to the values of their 1-min load shapes at point 566
  (09:26, the on-peak minute), and the load shapes are detached (snapshot solution);
* loads use the files' OpenDSS model 1 (constant P and Q) with the default Vminpu = 0.95 and
  Vmaxpu = 1.05, so a load whose voltage is above 1.05 p.u. (of 230 V) draws the power of a
  constant impedance sized at 1.05 p.u.;
* the solution tolerance is tightened from OpenDSS's default 1e-4 to 1e-12 (p.u. voltage change
  between iterations), so the finite differences are not polluted by the iteration error.

Quantities of interest (one solve gives all three):
  V899A  phase-A (node 1) voltage magnitude at bus 899, V (phase to ground)
  I78A   phase-A current magnitude in Line.LINE78 at its bus-1 end (bus 72), A
  PLOSS  total active power loss of the circuit (lines and transformer), kW
"""
import os

import numpy as np

try:
    import opendssdirect as dss
except ImportError:  # constants still import
    dss = None

HERE = os.path.dirname(os.path.abspath(__file__))
FEEDER = os.path.join(HERE, "feeder")
MASTER = os.path.join(FEEDER, "Master.dss")
INSTALL = "pip install opendssdirect.py==0.9.4"

T_INDEX = 565            # 0-based index of load-shape point 566, minute 09:26 (on-peak)
SD_P = 0.1               # kW, sd of each load's active power, N(P_Li, 0.1^2) (paper p. 700)
PF_NOM = 0.95            # constant power factor of the base scenario (paper p. 700)
PF_LO, PF_HI = 0.85, 0.95  # U(0.85, 0.95) power factors of the 110-input case (paper p. 701)
SD_PF = (PF_HI - PF_LO) / np.sqrt(12.0)
TOL = 1e-12
MAXITER = 100
QOI_NAMES = ("V899A", "I78A", "PLOSS")
QOI_UNITS = ("V", "A", "kW")


def _read_loads():
    """Load names, buses, phases and load-shape files from feeder/Loads.txt and LoadShapes.txt."""
    shapes = {}
    with open(os.path.join(FEEDER, "LoadShapes.txt")) as fh:
        for line in fh:
            tok = line.split()
            if len(tok) > 1 and tok[0].lower() == "new":
                name = tok[1].split(".", 1)[1].lower()
                fn = line.split("file=")[1].split(")")[0]
                shapes[name] = fn
    names, buses, phases, files = [], [], [], []
    with open(os.path.join(FEEDER, "Loads.txt")) as fh:
        for line in fh:
            tok = line.split()
            if len(tok) < 2 or tok[0].lower() != "new":
                continue
            kv = dict(t.split("=", 1) for t in tok[2:])
            names.append(tok[1].split(".", 1)[1])
            bus, ph = kv["Bus1"].split(".")
            buses.append(bus)
            phases.append(int(ph))
            files.append(shapes[kv["Yearly"].lower()])
    return names, buses, np.array(phases), files


LOAD_NAMES, LOAD_BUSES, LOAD_PHASES, _SHAPE_FILES = _read_loads()
NLOAD = len(LOAD_NAMES)
P_REF = np.array([np.loadtxt(os.path.join(FEEDER, f))[T_INDEX] for f in _SHAPE_FILES])  # kW

NAMES55 = tuple(f"P_{n}" for n in LOAD_NAMES)
NAMES110 = NAMES55 + tuple(f"pf_{n}" for n in LOAD_NAMES)
UNITS55 = ("kW",) * NLOAD
UNITS110 = UNITS55 + ("-",) * NLOAD
MEAN55 = P_REF.copy()
SD55 = np.full(NLOAD, SD_P)
# 110 inputs: P Gaussian (as above), pf uniform on [0.85, 0.95]; "MEAN"/"SD" are the moments
MEAN110 = np.concatenate([P_REF, np.full(NLOAD, 0.5 * (PF_LO + PF_HI))])
SD110 = np.concatenate([SD55, np.full(NLOAD, SD_PF)])
NOMINAL55 = P_REF.copy()
NOMINAL110 = np.concatenate([P_REF, np.full(NLOAD, PF_NOM)])
# central-difference steps: 1 % of each input's standard deviation
STEPS55 = 0.01 * SD55
STEPS110 = 0.01 * SD110


class FeederModel:
    """The feeder in OpenDSS. One instance per process (OpenDSSDirect uses one global engine)."""

    def __init__(self, tol=TOL, maxiter=MAXITER):
        if dss is None:
            raise ImportError(f"OpenDSSDirect.py is not installed ({INSTALL})")
        dss.Basic.AllowForms(False)
        dss.Text.Command("clear")
        cwd = os.getcwd()
        try:
            dss.Text.Command(f'redirect "{MASTER}"')
        finally:
            os.chdir(cwd)  # OpenDSS changes the working directory on redirect
        dss.Text.Command("set mode=snapshot")
        dss.Solution.Convergence(tol)
        dss.Solution.MaxIterations(maxiter)
        names = [n.lower() for n in dss.Loads.AllNames()]
        if names != [n.lower() for n in LOAD_NAMES]:
            raise RuntimeError("load order in OpenDSS differs from feeder/Loads.txt")
        for n, p in zip(LOAD_NAMES, P_REF):
            dss.Loads.Name(n)
            dss.Loads.Yearly("")
            dss.Loads.Daily("")
            dss.Loads.kW(p)
            dss.Loads.PF(PF_NOM)
        dss.Circuit.SetActiveBus("899")
        self._node899 = dss.Bus.Nodes().index(1)
        self.solve(P_REF)

    def solve(self, P, pf=None):
        """Return (V899A [V], I78A [A], PLOSS [kW]) for load powers P (kW) and power factors pf."""
        P = np.asarray(P, float)
        pf = np.full(NLOAD, PF_NOM) if pf is None else np.asarray(pf, float)
        for n, p, c in zip(LOAD_NAMES, P, pf):
            dss.Loads.Name(n)
            dss.Loads.kW(p)
            dss.Loads.PF(c)
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            raise RuntimeError("OpenDSS power flow did not converge")
        dss.Circuit.SetActiveBus("899")
        v = dss.Bus.VMagAngle()[2 * self._node899]
        dss.Circuit.SetActiveElement("Line.LINE78")
        i = dss.CktElement.CurrentsMagAng()[0]
        loss = dss.Circuit.Losses()[0] / 1e3
        return np.array([v, i, loss])

    def load_voltages_pu(self):
        """Voltage magnitude at each load's node, p.u. of 230 V (after the last solve)."""
        out = np.empty(NLOAD)
        for k, (b, ph) in enumerate(zip(LOAD_BUSES, LOAD_PHASES)):
            dss.Circuit.SetActiveBus(b)
            nodes = dss.Bus.Nodes()
            out[k] = dss.Bus.VMagAngle()[2 * nodes.index(int(ph))] / 230.0
        return out

    def qoi55(self, x):
        return self.solve(x)

    def qoi110(self, x):
        x = np.asarray(x, float)
        return self.solve(x[:NLOAD], x[NLOAD:])

    def grad(self, fun, x, steps):
        """Central differences of the three QoIs; returns an array (3, m) in physical units."""
        x = np.asarray(x, float)
        G = np.empty((3, len(x)))
        for j, h in enumerate(steps):
            xp = x.copy()
            xm = x.copy()
            xp[j] += h
            xm[j] -= h
            G[:, j] = (fun(xp) - fun(xm)) / (2 * h)
        return G
