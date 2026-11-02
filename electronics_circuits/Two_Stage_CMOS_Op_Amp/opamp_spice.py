"""Two-stage CMOS op-amp of Settaluri et al. (DATE 2020), simulated with ngspice.

Writes the ngspice deck from `two_stage_opamp.cir.template`, runs `ngspice -b` in a
working directory, and measures four scalars from the open-loop AC sweep and the DC
operating point:

    gain     |v(out)/v(in)| at 1 Hz (V/V), the paper's "gain" (AutoCkt: |H| at the first
             sweep point)
    ugbw     unity-gain frequency (Hz): the frequency where |H| crosses 1
    pm       phase margin (deg): 180 + phase(H) at ugbw, phase unwrapped from 1 Hz
    i_supply total current drawn from the 1.2 V supply (A), the paper's "bias current"
             (AutoCkt reads -i(vdd) at the operating point)

Inputs are the AutoCkt design parameters: the parallel multipliers mp1, mn1, mp3, mn3,
mn4, mn5 of 0.5 um / 90 nm devices (integers 1..100 in the paper; any real value >= 1
here) and the compensation capacitor cc_pF (pF). Held settings (ibias, load, common
mode, supply, solver tolerance, sweep density) are keyword arguments with AutoCkt's
values.

Every way a measurement can be undefined raises `MeasurementError` instead of returning
a number: ngspice failing or writing no output, |H(1 Hz)| <= 1 (no unity-gain crossing),
|H| still >= 1 at 10 GHz, more than one crossing, or a low-frequency phase that is not
near 0 deg (an inverting or broken amplifier). AutoCkt itself returns freq[0] or -180 deg
in such cases; this module does not.

Needs numpy, scipy and the `ngspice` program (checked with ngspice-42, Ubuntu package
42+ds-3build1). Stateless: every call works in its own directory.
"""
import os
import shutil
import string
import subprocess
import tempfile

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, 'two_stage_opamp.cir.template')
MODEL_CARD = os.path.join(HERE, '45nm_bulk.txt')

NAMES = ('mp1', 'mn1', 'mp3', 'mn3', 'mn4', 'mn5', 'cc_pF')
UNITS = ('1', '1', '1', '1', '1', '1', 'pF')
LOWER = np.array([1., 1., 1., 1., 1., 1., 0.1])       # AutoCkt YAML / paper Sec. III.B
UPPER = np.array([100., 100., 100., 100., 100., 100., 10.0])
QOI_NAMES = ('gain', 'ugbw', 'pm', 'i_supply')
QOI_UNITS = ('V/V', 'Hz', 'deg', 'A')

HELD = dict(ibias_uA=30.0, cload_pF=10.0, vcm=0.6, vdd=1.2,   # AutoCkt netlist values
            reltol=1e-9,                                       # AutoCkt: ngspice default 1e-3
            points_per_decade=100)                             # AutoCkt: 10

NGSPICE = os.environ.get('NGSPICE', 'ngspice')


class MeasurementError(RuntimeError):
    """A simulation or measurement that has no defined value at this design."""


def ngspice_version():
    out = subprocess.run([NGSPICE, '--version'], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if 'ngspice-' in line:
            return line.strip(' *')
    return 'unknown'


def write_deck(path, design, **held):
    """Write the ngspice deck for `design` (mapping name -> value) to `path`."""
    values = dict(HELD)
    values.update(held)
    values.update({k: float(design[k]) for k in NAMES})
    text = string.Template(open(TEMPLATE).read()).substitute(
        {k: repr(float(v)) if k != 'points_per_decade' else str(int(v)) for k, v in values.items()})
    with open(path, 'w') as fh:
        fh.write(text)


def read_outputs(workdir):
    """Read ac.txt and op.txt written by the deck: (freq, H, i_supply)."""
    ac_path, op_path = os.path.join(workdir, 'ac.txt'), os.path.join(workdir, 'op.txt')
    if not (os.path.isfile(ac_path) and os.path.isfile(op_path)):
        raise MeasurementError('ngspice wrote no ac.txt/op.txt')
    ac = np.loadtxt(ac_path, skiprows=1, ndmin=2)
    with open(op_path) as fh:
        header = fh.readline().split()
    op = np.loadtxt(op_path, skiprows=1, ndmin=2)
    if 'i(vdd)' not in header or op.shape != (1, len(header)):
        raise MeasurementError(f'unexpected op.txt: header {header}, values {op.shape}')
    i_vdd = op[0, header.index('i(vdd)')]
    if ac.shape[1] != 3 or ac.shape[0] < 10 or not np.all(np.isfinite(ac)) or not np.isfinite(i_vdd):
        raise MeasurementError(f'unexpected ac.txt: {ac.shape}')
    return ac[:, 0], ac[:, 1] + 1j * ac[:, 2], -i_vdd


def measure_unity_gain(freq, H):
    """(ugbw, pm) from the AC sweep; raises MeasurementError where either is undefined."""
    mag = np.abs(H)
    if not mag[0] > 1.0:
        raise MeasurementError(f'|H(1 Hz)| = {mag[0]:.4g} <= 1: no unity-gain crossing')
    below = np.flatnonzero(mag < 1.0)
    if below.size == 0:
        raise MeasurementError(f'|H| >= 1 up to {freq[-1]:.3g} Hz: unity-gain frequency outside the sweep')
    k = below[0]
    if np.any(mag[k:] >= 1.0):
        raise MeasurementError('|H| crosses 1 more than once: unity-gain frequency ambiguous')
    phase = np.rad2deg(np.unwrap(np.angle(H)))
    if abs(phase[0]) > 10.0:
        raise MeasurementError(f'phase at 1 Hz = {phase[0]:.3g} deg, not near 0: inverting or broken amplifier')
    # cubic splines in log10(f) through the sweep points; the crossing is bracketed by k-1, k
    lf = np.log10(freq)
    x = brentq(CubicSpline(lf, np.log10(mag)), lf[k - 1], lf[k], xtol=1e-14, rtol=1e-14)
    ugbw, pm = 10.0 ** x, 180.0 + float(CubicSpline(lf, phase)(x))
    if not (np.isfinite(ugbw) and np.isfinite(pm)):
        raise MeasurementError('non-finite unity-gain frequency or phase margin')
    return float(ugbw), pm


def measure(freq, H, i_supply, strict=True):
    """gain, ugbw, pm, i_supply from the AC sweep and the supply current.

    gain and i_supply are defined at every design that simulates. ugbw and pm are not: with
    strict=True an undefined one raises MeasurementError; with strict=False both are NaN and
    the reason is returned under 'error'.
    """
    if not i_supply > 0:
        raise MeasurementError(f'supply current {i_supply!r} A is not positive')
    out = dict(gain=float(np.abs(H[0])), i_supply=float(i_supply))
    try:
        out['ugbw'], out['pm'] = measure_unity_gain(freq, H)
    except MeasurementError as e:
        if strict:
            raise
        out.update(ugbw=np.nan, pm=np.nan, error=str(e))
    return out


SIM_FAILURE_WORDS = ('error', 'abort', 'singular', 'too small', 'no convergence')


def simulate(design, workdir=None, keep=False, timeout=60, strict=True, **held):
    """Run ngspice at one design and return {'gain', 'ugbw', 'pm', 'i_supply'}.

    design: mapping with the keys of NAMES, or a sequence in that order (physical units).
    strict: see `measure`. A failed simulation always raises MeasurementError.
    """
    if not isinstance(design, dict):
        design = dict(zip(NAMES, np.asarray(design, dtype=float)))
    for k in NAMES:
        v = float(design[k])
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f'{k} = {v!r}: must be finite and positive')
    tmp = workdir or tempfile.mkdtemp(prefix='opamp_')
    try:
        if not os.path.exists(os.path.join(tmp, '45nm_bulk.txt')):
            shutil.copy(MODEL_CARD, tmp)
        write_deck(os.path.join(tmp, 'opamp.cir'), design, **held)
        for name in ('ac.txt', 'op.txt'):
            if os.path.exists(os.path.join(tmp, name)):
                os.remove(os.path.join(tmp, name))
        # one passive OpenMP thread: ngspice-42 evaluates BSIM4 devices with OpenMP, and its
        # spinning threads slowed concurrent runs tenfold here; results are unchanged
        env = dict(os.environ, OMP_NUM_THREADS='1', OMP_WAIT_POLICY='passive')
        proc = subprocess.run([NGSPICE, '-b', 'opamp.cir'], cwd=tmp, capture_output=True,
                              text=True, timeout=timeout, env=env)
        # the BPTM card's "ckappas/ckappad too small" and "dtox ignored" warnings are expected
        bad = [l.strip() for l in (proc.stdout + proc.stderr).splitlines()
               if any(w in l.lower() for w in SIM_FAILURE_WORDS) and 'ckappa' not in l.lower()]
        if proc.returncode != 0:
            raise MeasurementError(f'ngspice exited with code {proc.returncode}'
                                   + (f': {bad[0]}' if bad else ''))
        if bad:
            raise MeasurementError('ngspice reported: ' + bad[0])
        return measure(*read_outputs(tmp), strict=strict)
    finally:
        if workdir is None and not keep:
            shutil.rmtree(tmp, ignore_errors=True)


def qoi_vector(x, strict=True, **kw):
    """The four QoIs at the physical point x (length 7), as an array in QOI_NAMES order."""
    r = simulate(x, strict=strict, **kw)
    return np.array([r[q] for q in QOI_NAMES])
