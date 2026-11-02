"""Run XFOIL 6.99 for the flapped NACA 2412 airfoil of the second AIAA UQ challenge.

Written for the as-data-sets repository (MIT licence). Standard library only.

XFOIL (M. Drela and H. Youngren, MIT, GPL) is an outside program: this module writes its
keyboard input and reads its polar file; no XFOIL code is copied here. The command sequence
follows the challenge workflow of Davis and Renganathan (github.com/csdlpsu/uqchallenge2,
uqlib.py, class XFOIL, commit 73eaec9):

    NACA 2412; GDES FLAP, hinge x/c = 0.70, hinge y at 0.5 of the local thickness,
    deflection beta (deg, + trailing edge down); PPAR N = 100 panel nodes;
    OPER, MACH 0, VISC Re, VPAR N = Ncrit (0.01 in the challenge), PACC, ALFA alpha.

Convergence protocol, each attempt in a fresh XFOIL process. Attempts 1-3 are those of the
authors' driver (xfoil_training.py) plus one negative-alpha sweep, with XFOIL's default limit
of 20 viscous iterations; attempts 4-5 are a retry ladder used only when 1-3 all fail, so a
point that converges in attempts 1-3 gets exactly the same numbers as without the ladder.
  1. ALFA alpha directly.                                      -> status 0
  2. if that fails: ALFA 0, 1, 2, ..., ceil(alpha)-1, then alpha
     (numpy.arange(0, alpha) then alpha, as the authors do).    -> status 1
  3. our addition, used only if 2 fails: the same unit-step
     continuation from 0 towards alpha for alpha < 0
     (the authors' sweep is empty for alpha <= 0).              -> status 2
  4. ITER 100 (100 viscous iterations), ALFA alpha directly.    -> status 3
  5. ITER 100 continuation ("sequenced"):                       -> status 4
     start at the integer angle nearest to alpha that converges directly with ITER 100,
     trying first the integers on the side of alpha towards 0 (below alpha when alpha > 0,
     as on the authors' 0, 1, 2, ... path; nearest first, at most 5) and then those on the
     other side;
     from the start, step towards alpha in steps of 0.5 deg (0.25 deg when alpha > 7 deg
     and flap > 10 deg). If a step does not converge, the path is cut back to its last
     converged angle and the rest is redone in a new process with half the step (down to
     1/16 deg); if that also fails, the next start angle is tried. The new process repeats
     the converged prefix exactly, so it resumes from the same converged state.
Only the target angle is read as the answer: attempts 1-4 switch PACC on just before the last
ALFA; attempt 5 accumulates every angle of its path and requires every one to be in the polar
file (XFOIL writes only converged points), in order, ending at alpha. A point that fails all
attempts raises XfoilFailure; no number is returned for it.

XFOIL prints CL and CM with 4 decimals and CD with 5 decimals in its polar file; these are the
precision of every value returned here.
"""
import math
import os
import shutil
import subprocess
import tempfile

DEFAULT_EXE = os.environ.get("XFOIL_EXE", "xfoil")
RETRY_ITER = 100          # viscous iterations in attempts 4 and 5 (XFOIL's default is 20)
SEQ_STEP = 0.5            # continuation step, deg
SEQ_STEP_HIGH = 0.25      # ... when alpha > 7 deg and flap > 10 deg
SEQ_MIN_STEP = 0.0625     # smallest step after halving, deg
SEQ_MAX_STARTS = 5        # integer start angles tried on each side of alpha

STATUS_NOTE = ("status 0 = converged directly, 1 = in the authors' 0,1,2,... sweep, "
               "2 = in our sweep towards negative alpha, 3 = directly with ITER 100, "
               "4 = in an ITER 100 continuation (sequenced), -1 = no convergence (nan)")


class XfoilFailure(RuntimeError):
    """XFOIL did not converge (or did not run) at the requested point."""


def _num(x):
    return repr(float(x))


def _deck(alphas, flap, re, ncrit, xtr_top, xtr_bot, panels, hinge_x, hinge_yt, mach,
          itmax=None, pacc_all=False):
    lines = [
        "PLOP", "G F", "",                        # graphics off
        "NACA 2412",
        "GDES", "FLAP", _num(hinge_x), "999", _num(hinge_yt), _num(flap), "EXEC", "",
        "PPAR", "N", str(int(panels)), "", "",
        "OPER",
        "MACH", _num(mach),
        "VISC", _num(re),
        "VPAR", "N", _num(ncrit), "XTR", _num(xtr_top), _num(xtr_bot), "",
    ]
    if itmax is not None:                         # attempts 4-5 only
        lines += ["ITER", str(int(itmax))]
    if pacc_all:                                  # attempt 5: keep every converged angle
        lines += ["PACC", "polar.dat", ""]
        for a in alphas:
            lines += ["ALFA", _num(a)]
        lines += ["", "QUIT", ""]
        return "\n".join(lines)
    for a in alphas[:-1]:                         # continuation points, not stored
        lines += ["ALFA", _num(a)]
    lines += ["PACC", "polar.dat", "", "ALFA", _num(alphas[-1]), "", "QUIT", ""]
    return "\n".join(lines)


def _polar_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                vals = [float(p) for p in parts[:7]]
            except ValueError:
                continue
            rows.append(vals)
    return rows


def _as_result(row, alpha):
    a, cl, cd, cdp, cm, xt, xb = row
    if abs(a - alpha) > 6e-4:                     # polar prints alpha with 3 decimals
        return None
    vals = dict(cl=cl, cd=cd, cdp=cdp, cm=cm, xtr_top_out=xt, xtr_bot_out=xb)
    if not all(math.isfinite(v) for v in vals.values()):
        return None
    return vals


def _read_polar(path, alpha):
    """Return the last data row of an XFOIL polar file if its alpha matches."""
    rows = _polar_rows(path)
    if not rows:
        return None
    return _as_result(rows[-1], alpha)


def _xfoil(exe, deck, timeout):
    """Run one XFOIL process on a deck; return the rows of its polar file."""
    work = tempfile.mkdtemp(prefix="xfoil_")
    try:
        try:
            subprocess.run([exe], input=deck, cwd=work, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        return _polar_rows(os.path.join(work, "polar.dat"))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _attempt(exe, alphas, timeout, itmax=None, **kw):
    rows = _xfoil(exe, _deck(alphas, itmax=itmax, **kw), timeout)
    if not rows:
        return None
    return _as_result(rows[-1], alphas[-1])


def _converged_prefix(exe, path, timeout, **kw):
    """Run an ITER 100 path with every angle accumulated; return (number of leading angles
    that converged in order, polar row of the last of them)."""
    rows = _xfoil(exe, _deck(path, itmax=RETRY_ITER, pacc_all=True, **kw), timeout)
    if not rows:
        return 0, None
    k = 0
    for a, row in zip(path, rows):
        if abs(row[0] - a) > 6e-4 or not all(math.isfinite(v) for v in row):
            break
        k += 1
    return k, (rows[k - 1] if k else None)


def _plan(start, target, step):
    """Angles from start (excluded) to target (included) in equal-or-shorter steps."""
    n = max(1, int(math.ceil(abs(target - start) / step - 1e-9)))
    return [start + (target - start) * (i + 1) / n for i in range(n)]


def _sequenced(exe, alpha, flap, timeout, kw):
    """Attempt 5. Returns (result, info) or (None, None)."""
    step0 = SEQ_STEP_HIGH if (alpha > 7.0 and flap > 10.0) else SEQ_STEP
    below = [float(k) for k in range(int(math.ceil(alpha)) - 1,
                                     int(math.ceil(alpha)) - 1 - SEQ_MAX_STARTS, -1)]
    above = [float(k) for k in range(int(math.floor(alpha)) + 1,
                                     int(math.floor(alpha)) + 1 + SEQ_MAX_STARTS)]
    sides = [("below", below), ("above", above)]
    if alpha < 0:                                 # the side towards alpha = 0 first
        sides.reverse()
    for side, starts in sides:
        for a0 in starts:
            k, _ = _converged_prefix(exe, [a0], timeout, **kw)
            if k != 1:
                continue                          # this start does not converge: next one
            done, step, runs = [a0], step0, 1
            while step >= SEQ_MIN_STEP - 1e-12:
                path = done + _plan(done[-1], alpha, step)
                k, row = _converged_prefix(exe, path, timeout, **kw)
                runs += 1
                if k == len(path):
                    res = _as_result(row, alpha)
                    if res is None:
                        break
                    info = dict(side=side, start=a0, step_min=step, n_steps=len(path) - 1,
                                n_runs=runs)
                    return res, info
                if k > len(done):                 # keep the converged part of the path
                    done = path[:k]
                step /= 2.0
            # this start failed even at the smallest step: try the next start
    return None, None


def run(alpha, flap, re, ncrit=0.01, xtr_top=1.0, xtr_bot=1.0, panels=100,
        hinge_x=0.70, hinge_yt=0.5, mach=0.0, exe=None, timeout=60.0, retry=True):
    """One XFOIL analysis. Returns a dict with cl, cd, cdp, cm (dimensionless),
    xtr_top_out, xtr_bot_out (transition x/c on each side), status (0-4; see the module
    docstring) and, for status 4, seq = dict(side, start, step_min, n_steps, n_runs).
    retry=False stops after attempt 3 (the protocol before the retry ladder).
    Raises XfoilFailure if no attempt converges."""
    found = shutil.which(exe or DEFAULT_EXE)
    if found is None:
        raise XfoilFailure("XFOIL executable not found: %r (set XFOIL_EXE)"
                           % (exe or DEFAULT_EXE))
    exe = os.path.abspath(found)
    alpha = float(alpha)
    flap = float(flap)
    kw = dict(flap=flap, re=re, ncrit=ncrit, xtr_top=xtr_top, xtr_bot=xtr_bot,
              panels=panels, hinge_x=hinge_x, hinge_yt=hinge_yt, mach=mach)
    out = _attempt(exe, [alpha], timeout, **kw)
    if out is not None:
        out["status"] = 0
        return out
    # the authors' sweep: numpy.arange(0, alpha) followed by alpha
    sweep = [float(k) for k in range(0, int(math.ceil(alpha)))] + [alpha]
    if len(sweep) > 1:
        out = _attempt(exe, sweep, timeout, **kw)
        if out is not None:
            out["status"] = 1
            return out
    if alpha < 0:
        neg = [float(-k) for k in range(0, int(math.ceil(-alpha)))] + [alpha]
        if len(neg) > 1:
            out = _attempt(exe, neg, timeout, **kw)
            if out is not None:
                out["status"] = 2
                return out
    if retry:
        out = _attempt(exe, [alpha], timeout, itmax=RETRY_ITER, **kw)
        if out is not None:
            out["status"] = 3
            return out
        out, info = _sequenced(exe, alpha, flap, timeout, kw)
        if out is not None:
            out["status"] = 4
            out["seq"] = info
            return out
    raise XfoilFailure("XFOIL did not converge at alpha=%r, flap=%r, Re=%r, Ncrit=%r, "
                       "xtr=(%r, %r)" % (alpha, flap, re, ncrit, xtr_top, xtr_bot))
