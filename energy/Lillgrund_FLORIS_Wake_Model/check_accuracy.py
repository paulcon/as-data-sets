"""Recompute the accuracy and smoothness numbers quoted in DATA.md for Lillgrund_FLORIS_Wake_Model.

    python check_accuracy.py                  # all checks; check 1 needs FLORIS 2.0.1
    python check_accuracy.py --no-floris      # checks 2-5 only (numpy only; 4-6 min)
    python check_accuracy.py --floris-only    # check 1 only (25-45 s)

Checks, each printed beside the value DATA.md quotes:

1. The port against FLORIS 2.0.1 ("Accuracy"). FLORIS is set up through its own input dictionary
   with the layout, the turbine table of `lillgrund_model.turbine_table()`, the held settings and
   the six parameters (set in both the `gauss_legacy` velocity model and the `gauss` deflection
   model). On 29 flow cases it compares each turbine's power and final turbulence intensity with
   `lillgrund_model.turbine_powers`. The cases: the defaults at 180 deg and 8 and 11 m/s; TI 0.10
   with no shear; TI 0.08 with shear 0.2 at 270 deg; sample 448 at 120 deg (the pair with NaN
   rotor points); three random parameter sets (numpy default_rng(0)) at each of the six directions
   and at 213 and 270 deg. DATA.md: largest power difference of order 1e-16 of the largest turbine
   power, turbulence intensities to order 1e-17, 0.4-1.1 s per FLORIS solve and 0.05-0.14 s per
   port solve.
2. Gradient step study ("Accuracy"), at samples 0-11 and 180, 120 and 90 deg: the shipped gradients
   (frozen differences, step 1e-4 of the half-range) against frozen differences with steps 1e-3,
   1e-5 and 1e-6, and against plain central differences (nothing frozen) with step 1e-6; relative
   2-norm differences. DATA.md: 1e-8 to 4e-4 at 1e-3, 1e-10 to 8e-8 at 1e-5 and 1e-6, and the same
   for the plain differences.
3. Plain central differences with the shipped step 1e-4 against the shipped gradients at samples
   0-99, 180 deg ("Accuracy"). DATA.md: none differs by more than 1e-4.
4. Smoothness sweeps ("Smoothness"): 101 points along each input's range, the other inputs at the
   FLORIS defaults, 180 deg. Where a discrete choice (effective yaw angles, overlap counts) changes
   between two points, the jump is the power minus the previous piece continued (the solve with
   the previous point's choices frozen). DATA.md: each jump at most 0.047 MW, the jumps of one
   input add up to at most 0.64 MW (b_d), ranges 3-14 MW for the five inputs other than a_d,
   and a_d has no jumps and a range of 0.15 MW.
5. Sample 448 at 120 deg ("Smoothness"): the NaN rotor points, and the shipped gradient against
   frozen differences with steps 1e-5, 1e-6 and 1e-7. DATA.md: three points; a relative 1.6e-5,
   in the alpha, beta and b_d components.

The notebook's check cell prints the other numbers of "Accuracy" (three samples re-solved, plain
differences at two), and `generate_data.py --check` compares a rerun with the shipped files; they
are not repeated here.

Checks 2-5 import numpy and lillgrund_model.py only. Check 1 also imports FLORIS 2.0.1, which is
not in the repository's .venv. It was run in a separate Python 3.11 venv. FLORIS 2.0.1 declares
dependencies that no longer install (the withdrawn `sklearn` package), so it is installed without
them:

    python3.11 -m venv venv-floris
    venv-floris/bin/pip install numpy==1.26.4 scipy==1.11.4 matplotlib==3.8.4 pandas==2.1.4 \\
        coloredlogs==15.0.1 scikit-learn==1.3.2 pyproj==3.6.1
    venv-floris/bin/pip install --no-deps seaborn==0.13.2 floris==2.0.1
    venv-floris/bin/python check_accuracy.py --floris-only

Without FLORIS, the script runs checks 2-5, then prints this message and exits with status 3
(with --floris-only it stops at once; --no-floris never needs FLORIS). Deterministic
(fixed rows and seed); it reads the files next to it and writes nothing.
"""
import argparse
import importlib.util
import os
import platform
import sys
import time
import warnings

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lillgrund_model as lm  # noqa: E402

GRAD_ROWS = range(12)            # check 2
GRAD_DIRS = (180.0, 120.0, 90.0)
PLAIN_ROWS = range(100)          # check 3
N_SWEEP = 101                    # check 4
NAN_ROW, NAN_WD = 448, 120.0     # check 5

INSTALL = ('FLORIS 2.0.1 is not installed. Install it in a separate Python 3.11 venv (not .venv):\n'
           '  python3.11 -m venv venv-floris\n'
           '  venv-floris/bin/pip install numpy==1.26.4 scipy==1.11.4 matplotlib==3.8.4 pandas==2.1.4 '
           'coloredlogs==15.0.1 scikit-learn==1.3.2 pyproj==3.6.1\n'
           '  venv-floris/bin/pip install --no-deps seaborn==0.13.2 floris==2.0.1\n'
           'and rerun with venv-floris/bin/python check_accuracy.py, '
           'or run the numpy-only checks with --no-floris.')


def load():
    ld = lambda n: np.loadtxt(os.path.join(HERE, n), delimiter=',')  # noqa: E731
    return ld('inputs.csv'), ld('gradients.csv').reshape(-1, len(lm.DIRECTIONS), len(lm.NAMES))


def rel(g, g0):
    return np.linalg.norm(g - g0)/np.linalg.norm(g0)


# ---- check 1 ----------------------------------------------------------------------------------

def floris_input(x, wd, ws, ti, shear):
    """FLORIS 2.0.1 input dictionary for the port's set-up and parameters x."""
    alpha, beta, ka, kb, ad, bd = (float(v) for v in x)
    ws_tab, cp_tab, ct_tab = lm.turbine_table()
    shared = {'alpha': alpha, 'beta': beta, 'ka': ka, 'kb': kb, 'eps_gain': lm.EPS_GAIN}
    velocity = dict(shared, calculate_VW_velocities=True, use_yaw_added_recovery=True,
                    yaw_recovery_alpha=lm.YAW_RECOVERY_ALPHA)
    deflection = dict(shared, ad=ad, bd=bd, dm=1.0, use_secondary_steering=True)
    turbulence = dict(zip(('initial', 'constant', 'ai', 'downstream'), lm.CRESPO_HERNANDEZ))
    turbine = {'rotor_diameter': lm.D, 'hub_height': lm.HUB_HEIGHT, 'blade_count': 3,
               'pP': lm.PP, 'pT': lm.PP, 'generator_efficiency': 1.0, 'yaw_angle': 0.0,
               'tilt_angle': 0.0, 'TSR': lm.TSR,
               'power_thrust_table': {'power': cp_tab.tolist(), 'thrust': ct_tab.tolist(),
                                      'wind_speed': ws_tab.tolist()}}
    farm = {'wind_speed': [float(ws)], 'wind_direction': [float(wd)],
            'turbulence_intensity': [float(ti)], 'wind_shear': float(shear), 'wind_veer': 0.0,
            'specified_wind_height': lm.HUB_HEIGHT, 'air_density': lm.RHO,
            'layout_x': lm.LAYOUT_X.tolist(), 'layout_y': lm.LAYOUT_Y.tolist(),
            'wind_x': [0], 'wind_y': [0]}
    off = {'enable': False, 'level': 'INFO'}
    return {'type': 'floris_input', 'name': 'lillgrund', 'description': 'port check',
            'floris_version': 'v2.0.0', 'logging': {'console': off, 'file': off},
            'turbine': {'type': 'turbine', 'name': 'swt-2.3-93', 'description': '',
                        'properties': turbine},
            'wake': {'type': 'wake', 'name': 'gch', 'description': '', 'properties': {
                'velocity_model': 'gauss_legacy', 'deflection_model': 'gauss',
                'turbulence_model': 'crespo_hernandez', 'combination_model': 'sosfs',
                'parameters': {'wake_velocity_parameters': {'gauss_legacy': velocity},
                               'wake_deflection_parameters': {'gauss': deflection},
                               'wake_turbulence_parameters': {'crespo_hernandez': turbulence}}}},
            'farm': {'type': 'farm', 'name': 'lillgrund', 'description': '', 'properties': farm}}


def check_floris(X):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # FLORIS 2.0.1's own SyntaxWarnings on import
        try:
            import floris
            from floris.tools.floris_interface import FlorisInterface
        except ImportError:
            print(INSTALL, file=sys.stderr)
            sys.exit(3)
    import scipy
    version = getattr(floris, '__version__', None)
    if version is None:
        from importlib.metadata import version as pkg_version
        version = pkg_version('floris')
    print(f'1. port against FLORIS {version} (Python {platform.python_version()}, numpy {np.__version__}, '
          f'scipy {scipy.__version__}, {platform.system()} {platform.machine()})')
    cases = [(lm.NOMINAL, 180.0, 8.0, lm.TI, lm.SHEAR), (lm.NOMINAL, 180.0, 11.0, lm.TI, lm.SHEAR),
             (lm.NOMINAL, 180.0, 8.0, 0.10, 0.0), (lm.NOMINAL, 270.0, 8.0, 0.08, 0.2),
             (X[NAN_ROW], NAN_WD, 8.0, lm.TI, lm.SHEAR)]
    rng = np.random.default_rng(0)
    for wd in lm.DIRECTIONS + (213.0, 270.0):
        for _ in range(3):
            cases.append((lm.LOWER + (lm.UPPER - lm.LOWER)*rng.random(6), wd, 8.0, lm.TI, lm.SHEAR))
    dp, dti, t_fl, t_port = [], [], [], []
    for x, wd, ws, ti, shear in cases:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')      # FLORIS's sqrt of a negative at sample 448's NaN points
            fi = FlorisInterface(input_dict=floris_input(x, wd, ws, ti, shear))
            fi.calculate_wake()
        P_fl = np.array(fi.get_turbine_power())
        TI_fl = np.array([t.current_turbulence_intensity for t in fi.floris.farm.turbines])
        t1 = time.perf_counter()
        P, st = lm.turbine_powers(x, wd, ws, ti, shear, return_state=True)
        t2 = time.perf_counter()
        t_fl.append(t1 - t0)
        t_port.append(t2 - t1)
        dp.append(np.max(np.abs(P - P_fl))/np.max(np.abs(P_fl)))
        dti.append(np.max(np.abs(st['ti'] - TI_fl)))
    print(f'   {len(cases)} flow cases: largest difference in turbine power {max(dp):.1e} of the largest '
          f'turbine power (DATA.md: of order 1e-16); largest difference in final turbulence intensity '
          f'{max(dti):.1e} (DATA.md: of order 1e-17)')
    print(f'   per solve: FLORIS {min(t_fl):.2f}-{max(t_fl):.2f} s (median {np.median(t_fl):.2f}), port '
          f'{min(t_port):.3f}-{max(t_port):.3f} s (median {np.median(t_port):.3f}) '
          f'(DATA.md: 0.4-1.1 s and 0.05-0.14 s)')


# ---- checks 2-5 -------------------------------------------------------------------------------

def plain_grad(x, wd, h):
    """Central differences with step h of the half-range, every discrete choice recomputed (MW)."""
    half = (lm.UPPER - lm.LOWER)/2
    g = np.empty(6)
    for j in range(6):
        dx = np.zeros(6)
        dx[j] = h*half[j]
        g[j] = (lm.farm_power(x + dx, wd) - lm.farm_power(x - dx, wd))/(2*dx[j])
    return g/1e6


def frozen_grad(x, wd, h):
    return lm.farm_power_and_grad(x, wd, h=h)[1]/1e6


def check_steps(X, G):
    print(f'\n2. gradient step study, samples {GRAD_ROWS[0]}-{GRAD_ROWS[-1]} at '
          + ', '.join(f'{wd:g}' for wd in GRAD_DIRS) + ' deg: relative 2-norm difference from the shipped '
          'gradients (frozen, step 1e-4)')
    quoted = {'frozen 1e-3': '1e-8 to 4e-4', 'frozen 1e-5': '1e-10 to 8e-8', 'frozen 1e-6': '1e-10 to 8e-8',
              'plain 1e-6': '1e-10 to 8e-8'}
    err = {k: [] for k in quoted}
    for wd in GRAD_DIRS:
        k = lm.DIRECTIONS.index(wd)
        for i in GRAD_ROWS:
            for h in (1e-3, 1e-5, 1e-6):
                err[f'frozen {h:.0e}'.replace('e-0', 'e-')].append(rel(frozen_grad(X[i], wd, h), G[i, k]))
            err['plain 1e-6'].append(rel(plain_grad(X[i], wd, 1e-6), G[i, k]))
    for name, e in err.items():
        print(f'   {name:12s}: {min(e):.1e} to {max(e):.1e} over {len(e)} sample-direction pairs '
              f'(DATA.md: {quoted[name]})')


def check_plain(X, G):
    k = lm.DIRECTIONS.index(180.0)
    e = np.array([rel(plain_grad(X[i], 180.0, 1e-4), G[i, k]) for i in PLAIN_ROWS])
    print(f'\n3. plain central differences, step 1e-4, samples {PLAIN_ROWS[0]}-{PLAIN_ROWS[-1]} at 180 deg: '
          f'{np.sum(e > 1e-4)} of {len(e)} differ from the shipped gradients by more than 1e-4 '
          f'(largest {e.max():.1e}; DATA.md: none)')


def sweep(j):
    """Powers (MW) along input j through the defaults, and the jumps where a discrete choice changes."""
    t = np.linspace(lm.LOWER[j], lm.UPPER[j], N_SWEEP)
    P, jumps, prev = np.empty(N_SWEEP), [], None
    for n, v in enumerate(t):
        x = lm.NOMINAL.copy()
        x[j] = v
        p, st = lm.turbine_powers(x, return_state=True)
        P[n] = p.sum()/1e6
        if prev is not None and (st['yaw'] != prev['yaw'] or st['count'] != prev['count']):
            jumps.append(P[n] - lm.turbine_powers(x, frozen=prev).sum()/1e6)
        prev = st
    return P, np.abs(jumps)


def check_sweeps():
    print(f'\n4. {N_SWEEP}-point sweeps through the FLORIS defaults at 180 deg (MW)')
    biggest, total = (0.0, ''), (0.0, '')
    ranges = {}
    for j, name in enumerate(lm.NAMES):
        P, jumps = sweep(j)
        ranges[name] = P.max() - P.min()
        top = jumps.max() if jumps.size else 0.0
        print(f'   {name:5s}: range {ranges[name]:6.3f}; {jumps.size:2d} jumps, largest {top:.3f}, '
              f'sum {jumps.sum():.3f}')
        biggest = max(biggest, (top, name))
        total = max(total, (jumps.sum(), name))
    others = [ranges[n] for n in lm.NAMES if n != 'ad']
    print(f'   largest jump {biggest[0]:.3f} ({biggest[1]}; DATA.md: at most 0.047); largest sum '
          f'{total[0]:.2f} ({total[1]}; DATA.md: 0.64, bd); ranges of the five inputs other than ad '
          f'{min(others):.2f}-{max(others):.2f} (DATA.md: 3-14); ad range {ranges["ad"]:.2f} (DATA.md: 0.15, no jumps)')


def check_nan_sample(X, G):
    x, k = X[NAN_ROW], lm.DIRECTIONS.index(NAN_WD)
    st = lm.turbine_powers(x, NAN_WD, return_state=True)[1]
    print(f'\n5. sample {NAN_ROW} at {NAN_WD:g} deg: {st["n_nan"]} rotor points with NaN '
          f'(DATA.md: three)')
    g0 = G[NAN_ROW, k]
    for h in (1e-5, 1e-6, 1e-7):
        g = frozen_grad(x, NAN_WD, h)
        comp = ', '.join(f'{n} {abs(a - b)/np.linalg.norm(g):.0e}' for n, a, b in zip(lm.NAMES, g0, g))
        print(f'   frozen step {h:.0e}: relative 2-norm difference from the shipped gradient '
              f'{rel(g0, g):.1e} (DATA.md: 1.6e-5); by component, |difference| / |gradient|: {comp}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='\n'.join(__doc__.split('\n')[1:]))
    which = ap.add_mutually_exclusive_group()
    which.add_argument('--no-floris', action='store_true', help='run checks 2-5 only (numpy only)')
    which.add_argument('--floris-only', action='store_true', help='run check 1 only (needs FLORIS 2.0.1)')
    a = ap.parse_args()
    t0 = time.perf_counter()
    X, G = load()
    missing = not a.no_floris and not a.floris_only and importlib.util.find_spec('floris') is None
    if not a.no_floris and not missing:
        check_floris(X)
    if not a.floris_only:
        check_steps(X, G)
        check_plain(X, G)
        check_sweeps()
        check_nan_sample(X, G)
    print(f'\nrun time {time.perf_counter() - t0:.0f} s')
    if missing:
        print(INSTALL, file=sys.stderr)
        sys.exit(3)


if __name__ == '__main__':
    main()
