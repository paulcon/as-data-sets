"""Set up and run one fdfault rupture simulation of Daub et al. (2021).

Adapted from fabmogp v1.0 (https://github.com/alan-turing-institute/fabmogp, files
earthquake.py and utils.py at tag v1.0, which the paper's Dockerfile checks out; MIT
licence, Copyright (c) 2020 Alan Turing Institute, Eric Daub, Derek Groen, Hamid Arabnejad;
full text in LICENSE-fabmogp). The problem set-up (grid, friction, barriers, nucleation,
fault profile seed 18749) is theirs, unchanged at the default keywords. Changes: np.complex
-> complex (numpy 2); scipy.integrate.simps -> simpson (identical for 401 points); absolute
paths; the friction constants, profile seed and grid refinement as keywords; the moment
returned with the final-slip profile; and the Latin hypercube of mogp_emulator 0.4.0
(LatinHypercubeDesign._draw_samples, MIT) written out with numpy.

Needs the fdfault Python package and the compiled fdfault binary (see DATA.md); lhs_mogp
needs numpy only.
"""
import os
import subprocess
import numpy as np
from scipy.integrate import simpson

LOWER = np.array([-120., 0.1, 0.9])
UPPER = np.array([-80., 0.4, 1.1])


def lhs_mogp(n, seed):
    """Latin hypercube exactly as mogp_emulator 0.4.0 draws it after np.random.seed(seed)."""
    np.random.seed(seed)
    r = np.reshape(np.tile(np.arange(n, dtype=float) / float(n), 3), (3, n))
    for row in r:
        np.random.shuffle(row)
    r = np.transpose(r) + np.random.random((n, 3)) / float(n)
    return LOWER + (UPPER - LOWER) * r


# ---- fabmogp utils.py (unchanged except np.complex) ------------------------------------
def generate_profile(npoints, length, alpha, window, h=1., seed=None):
    npoints = int(npoints); window = int(window); length = float(length)
    alpha = float(alpha); h = float(h)
    if seed is not None:
        seed = int(seed)
    prng = np.random.RandomState(seed)
    phase = 2. * np.pi * prng.rand(npoints)
    k = 2. * np.pi * np.fft.fftfreq(npoints, length / float(npoints - 1))
    amp = np.zeros(npoints)
    nflt = npoints // window
    amp[1:] = (alpha * (2. * np.pi / np.abs(k[1:])) ** (0.5 * (1. + 2. * h))
               * np.sqrt(np.pi / length) / 2. * float(npoints))
    amp[nflt + 1:-nflt] = 0.
    f = amp * np.exp(complex(0., 1.) * phase)
    fund = np.fft.fft(prng.choice([-1., 1.]) * alpha * length
                      * np.sin(np.linspace(0., length, npoints) * np.pi / length))
    f = np.real(np.fft.ifft(f + fund))
    return f - f[0] - (f[-1] - f[0]) / length * np.linspace(0., length, npoints)


def calc_diff(f, dx):
    df = (np.roll(f, -3) / 60. - np.roll(f, -2) * 3. / 20. + np.roll(f, -1) * 3. / 4.
          - np.roll(f, 1) * 3. / 4. + np.roll(f, 2) * 3. / 20. - np.roll(f, 3) / 60.) / dx
    df[0] = (-21600./13649.*f[0]+81763./40947.*f[1]+131./27298.*f[2]-9143./13649.*f[3]+20539./81894.*f[4])/dx
    df[1] = (-81763./180195.*f[0]+7357./36039.*f[2]+30637./72078.*f[3]-2328./12013.*f[4]+6611./360390.*f[5])/dx
    df[2] = (-131./54220.*f[0]-7357./16266.*f[1]+645./2711.*f[3]+11237./32532.*f[4]-3487./27110.*f[5])/dx
    df[3] = (9143./53590.*f[0]-30637./64308.*f[1]-645./5359.*f[2]+13733./32154.*f[4]-67./4660.*f[5]+72./5359.*f[6])/dx
    df[4] = (-20539./236310.*f[0]+2328./7877.*f[1]-11237./47262.*f[2]-13733./23631.*f[3]+89387./118155.*f[5]-1296./7877.*f[6]+144./7877.*f[7])/dx
    df[5] = (-6611./262806.*f[1]+3487./43801.*f[2]+1541./87602.*f[3]-89387./131403.*f[4]+32400./43801.*f[6]-6480./43801.*f[7]+720./43801.*f[8])/dx
    df[-1] = -(-21600./13649.*f[-1]+81763./40947.*f[-2]+131./27298.*f[-3]-9143./13649.*f[-4]+20539./81894.*f[-5])/dx
    df[-2] = -(-81763./180195.*f[-1]+7357./36039.*f[-3]+30637./72078.*f[-4]-2328./12013.*f[-5]+6611./360390.*f[-6])/dx
    df[-3] = -(-131./54220.*f[-1]-7357./16266.*f[-2]+645./2711.*f[-4]+11237./32532.*f[-5]-3487./27110.*f[-6])/dx
    df[-4] = -(9143./53590.*f[-1]-30637./64308.*f[-2]-645./5359.*f[-3]+13733./32154.*f[-5]-67./4660.*f[-6]+72./5359.*f[-7])/dx
    df[-5] = -(-20539./236310.*f[-1]+2328./7877.*f[-2]-11237./47262.*f[-3]-13733./23631.*f[-4]+89387./118155.*f[-6]-1296./7877.*f[-7]+144./7877.*f[-8])/dx
    df[-6] = -(-6611./262806.*f[-2]+3487./43801.*f[-3]+1541./87602.*f[-4]-89387./131403.*f[-5]+32400./43801.*f[-7]-6480./43801.*f[-8]+720./43801.*f[-9])/dx
    return df


def generate_normals_2d(x, y):
    dx = x[2] - x[1]
    m = calc_diff(y, dx)
    return -m / np.sqrt(1. + m ** 2), 1. / np.sqrt(1. + m ** 2)


def rotate_xy2nt_2d(sxx, sxy, syy, n):
    m = np.array([n[1], -n[0]]) / np.sqrt(n[0] ** 2 + n[1] ** 2)   # "y" orientation
    sn = n[0] ** 2 * sxx + 2. * n[0] * n[1] * sxy + n[1] ** 2 * syy
    st = n[0] * m[0] * sxx + (m[0] * n[1] + n[0] * m[1]) * sxy + n[1] * m[1] * syy
    return sn, st


# ---- fabmogp earthquake.py ------------------------------------------------------------
def create_problem(arg, name, output_dir, refine=1, profile_seed=18749, mus=0.7, mud=0.2, dc=0.8):
    """Write the fdfault input file. arg = (syy [MPa], sxy/syy ratio, sxx/syy ratio).

    The keywords are held at the paper's values (Table 1; fabmogp); refine scales the grid and
    the number of time steps.
    """
    import fdfault
    syy, ston, sxtosy = arg
    p = fdfault.problem(name)
    p.set_rkorder(4)
    p.set_sbporder(4)
    nt = 800 * refine + 1
    nx = 400 * refine + 1
    ny = 150 * refine + 1
    lx, ly = 32., 12.
    p.set_nt(nt)
    p.set_cfl(0.3)
    p.set_ninfo((nt - 1) // 4)
    p.set_nblocks((1, 2, 1))
    p.set_nx_block(([nx], [ny, ny], [1]))
    p.set_block_lx((0, 0, 0), (lx, ly))
    p.set_block_lx((0, 1, 0), (lx, ly))
    p.set_bounds((0, 0, 0), ['absorbing', 'absorbing', 'absorbing', 'none'])
    p.set_bounds((0, 1, 0), ['absorbing', 'absorbing', 'none', 'absorbing'])
    sxx = sxtosy * syy
    sxy = -syy * ston
    x = np.linspace(0., lx, nx)
    y = ly * np.ones(nx) + generate_profile(nx, lx, 1.e-2, 20 * refine, 1., profile_seed)
    surf = fdfault.curve(nx, 'y', x, y)
    norm_x, norm_y = generate_normals_2d(x, y)
    sn = np.zeros(nx); st = np.zeros(nx)
    for i in range(nx):
        sn[i], st[i] = rotate_xy2nt_2d(sxx, sxy, syy, (norm_x[i], norm_y[i]))
    if not np.all(st + mus * sn < 0.):
        raise ValueError("initial shear stress exceeds the static strength somewhere on the fault")
    p.set_block_surf((0, 0, 0), 3, surf)
    p.set_block_surf((0, 1, 0), 2, surf)
    p.set_stress((sxx, sxy, 0., syy, 0., 0.))
    p.set_iftype(0, 'slipweak')
    p.add_pert(fdfault.swparam('constant', dc=dc, mus=mus, mud=mud), 0)
    p.add_pert(fdfault.swparam('boxcar', x0=2., dx=2., mus=10000.), 0)
    p.add_pert(fdfault.swparam('boxcar', x0=30., dx=2., mus=10000.), 0)
    nuc_pert = np.zeros((nx, 1))
    idx = (np.abs(x - lx / 2.) < 2.)
    nuc_pert[idx, 0] = (-mus * sn[idx] - st[idx]) + 0.1
    p.set_loadfile(0, fdfault.loadfile(nx, 1, np.zeros((nx, 1)), nuc_pert, np.zeros((nx, 1))))
    datadir = os.path.join(output_dir, "data")
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "problems"), exist_ok=True)
    p.set_datadir(datadir + os.sep)
    p.add_output(fdfault.output("ufault", 'U', nt, nt, 1, 0, nx - 1, 1, ny, ny, 1, 0, 0, 1))
    p.write_input(directory=os.path.join(output_dir, "problems"))
    return os.path.join(output_dir, "problems", name + ".in")


def run(arg, workdir, fdfault_dir, n_proc=1, name="sim", keep=False, **problem_kw):
    """Run one simulation. Returns (integrated final slip [m km], x [km], slip [m]).

    problem_kw go to create_problem (refine, profile_seed, mus, mud, dc).
    """
    import fdfault.analysis
    infile = create_problem(arg, name, workdir, **problem_kw)
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
               OMPI_MCA_rmaps_base_oversubscribe="1")
    res = subprocess.run(["mpiexec", "-n", str(int(n_proc)), os.path.join(fdfault_dir, "fdfault"),
                          infile], cwd=fdfault_dir, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"fdfault failed ({res.returncode}): {res.stderr[-2000:]}")
    U = fdfault.analysis.output(name, "ufault", os.path.join(workdir, "data") + os.sep)
    U.load()
    slip, x = np.array(U.U, dtype=float), np.array(U.x, dtype=float)
    if not np.all(np.isfinite(slip)):
        raise RuntimeError("non-finite slip")
    moment = float(simpson(slip, x=x))
    if not keep:
        for d in ("data", "problems"):
            for fn in os.listdir(os.path.join(workdir, d)):
                if fn.startswith(name):
                    os.remove(os.path.join(workdir, d, fn))
    return moment, x, slip
