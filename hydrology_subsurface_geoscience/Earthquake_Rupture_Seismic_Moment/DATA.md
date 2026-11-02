# Earthquake_Rupture_Seismic_Moment: data

Kind: **stored data**. The outputs come from an outside compiled code, the finite-difference rupture
simulator fdfault, as run by the authors of Daub et al. (2021) and printed in their Tables 2 and 3.
Five of the 30 runs were repeated here with fdfault v1.0.1 (`fdfault_reruns.csv`). The full-precision
inputs (`design_inputs.csv`) are redrawn by `generate_data.py` from the seeds in the authors' workflow.
No value in `paper_tables.csv` has been changed.

## Files

| file | contents | role |
|---|---|---|
| `Earthquake_Rupture_Seismic_Moment.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 604; outputs committed |
| `paper_tables.csv` | 30 × 6: table (2 or 3), row in table, the three inputs and the integrated final slip (m km), **as printed** | stored data (the analysis uses its last column) |
| `design_inputs.csv` | 30 × 6: table, row, row in the Latin hypercube draw, and the three inputs at full precision, same row order | inputs, regenerated |
| `fdfault_reruns.csv` | 5 × 7: row of `paper_tables.csv`, the three inputs, integrated final slip (m km) and maximum slip (m) from fdfault, wall time (s) | check runs, stored data |
| `generate_data.py` | writes `design_inputs.csv` (numpy and scipy: it imports `fdfault_problem`, which imports `scipy.integrate`); `--rerun` runs fdfault and writes `fdfault_reruns.csv` | regenerates the files |
| `fdfault_problem.py` | the fdfault problem set-up, Latin hypercube, run and Simpson integral; adapted from fabmogp (MIT) | helper module for `generate_data.py` |
| `LICENSE-fabmogp` | the MIT licence of fabmogp, kept beside the adapted code | licence |

The CSVs are plain comma-separated numbers after two `#` lines (a description and the column names), so
`np.loadtxt(path, delimiter=',')` reads them. Inputs are in physical units: normal stress σ_yy in MPa
(negative in compression), the two ratios dimensionless.

## Model

Citation: E. G. Daub, H. Arabnejad, I. Mahmood and D. Groen, "Uncertainty quantification of dynamic
earthquake rupture simulations", *Phil. Trans. R. Soc. A* 379(2197):20200076, 2021,
doi:10.1098/rsta.2020.0076 (publisher PDF, CC BY 4.0).

- **Simulator**: fdfault (E. G. Daub; https://github.com/edaub/fdfault, v1.0.1, MIT), a parallel
  high-order summation-by-parts finite-difference code for frictional failure and elastic wave
  propagation (Kozdon, Dunham and Nordström 2012, the paper's ref. 29). Two-dimensional plane strain,
  velocity–stress form of the elastic wave equation (Eq. (3.1), p. 7).
- **Geometry and grid** (Sec. 3(b), pp. 7–9; fabmogp `earthquake.py`): a 32 km fault between two
  blocks 12 km thick; 401 × 302 grid points (two blocks of 401 × 151), 4th-order Runge–Kutta and
  4th-order SBP operators, 800 time steps at CFL 0.3, absorbing outer boundaries. Fault profile:
  self-similar random roughness (spectral exponent `h = 1` in the code, random phases) with RMS deviation from a plane 10⁻² of the length (p. 8), cut off below
  20 grid spacings, plus a half-sine of the same amplitude ratio, one realization (seed 18749).
- **Friction**: slip weakening (Eq. (3.2), p. 7; Ida 1972), Table 1 (p. 8): μ_s = 0.7, μ_d = 0.2,
  D_c = 0.8 m; λ = G = 32.04 GPa; ρ = 2.68 × 10³ kg/m³ in Table 1. The runs, however, have
  ρ = 2.67 × 10³ kg/m³ (fdfault's units: 2.67 MPa s²/(km m)). The evidence is the code: neither fabmogp
  v1.0 `create_problem` (`earthquake.py`, `utils.py`) nor `fdfault_problem.py` sets a material, so each
  block gets fdfault's default (`fdfault/python/fdfault/block.py:93` creates `material(mat)`, and
  `fdfault/python/fdfault/material.py:54` defaults to `rho = 2.67`). The reruns use 2.67 and reproduce
  the printed sizes, but no run at 2.68 was made, so they do not by themselves rule out 2.68. Barriers: μ_s = 10⁴ in boxcar perturbations centred
  at x = 2 and 30 km with `dx = 2` in the code (the paper: "strong barriers arrest rupture 2 km from the
  ends", p. 8). Nucleation: over |x − 16 km| < 2 km the
  shear load is raised to 0.1 MPa above the static strength.
- **Inputs** (m = 3), independent and uniform:

| name | symbol | units | range | source |
|---|---|---|---|---|
| `sigma_yy` | σ_yy | MPa | U[−120, −80] | Sec. 3(b)(i) and p. 8 |
| `shear_to_normal` | σ_xy/σ_yy (shear stress −ratio·σ_yy > 0) | – | U[0.1, 0.4] | Sec. 3(b)(ii) and p. 8 |
| `normal_ratio` | σ_xx/σ_yy | – | U[0.9, 1.1] | p. 8 ("±10 %") |

  The paper gives no nominal values; EMA's nominal point is the centre of the box, (−100, 0.25, 1.0),
  which is also the authors' example run for Fig. 1 (`fabmogp_paper/manuscript/run_earthquake.py`).
- **Quantity of interest**: the final slip integrated along the fault with Simpson's rule, in m km
  (Sec. 5(a), p. 10), proportional to the seismic moment because G is fixed. The text says it ranges
  "from 35 m km to around 300 m km"; the printed runs range from 29.24 (Table 2) to 296.45 m km.
- **Gradients**: none. Each run takes minutes here, and the QoI has steep transitions between rupture
  arrest and propagation (paper Fig. 3b, p. 12), so the notebook uses the global quadratic estimator.
- **Sampling design** (Sec. 5, pp. 10–11; `fabmogp_paper/manuscript/Makefile`): two independent Latin
  hypercubes from mogp_emulator 0.4.0, 20 training runs (Table 2, `seed=4742322`) and 10 validation runs
  (Table 3, `seed=63684`). In mogp's sampler each column is a random permutation of the 20 (or 10)
  strata with a uniform jitter inside each stratum.
- **Published results**: no active subspace. The GP emulator's correlation lengths are 43.755, 0.109
  and 0.599 (normal stress, shear/normal ratio, normal stress ratio) and its covariance scale 131.994
  (p. 10); 8 of 10 validation points lie within 3 standard errors (Fig. 3a); the output is most
  sensitive to the shear/normal ratio, then normal stress, and not highly sensitive to the normal stress
  ratio (Fig. 3 caption; p. 13); history matching against an "observed" 58 m km (10 000 query points,
  threshold 3) leaves a curve with ratio ≈ 0.16 at high compression and ≈ 0.2 at lower normal stress
  (p. 13, Fig. 4).

## Regenerating

```bash
python generate_data.py            # design_inputs.csv, under a second (numpy, scipy)
python generate_data.py --check    # redraw and compare: max |difference| 0.0e+00
FDFAULT_DIR=/path/to/fdfault python generate_data.py --rerun 1,11,13,21,29   # fdfault_reruns.csv
```

`design_inputs.csv` has no randomness of its own: it reproduces mogp_emulator 0.4.0's
`LatinHypercubeDesign.sample` after `np.random.seed(seed)` (legacy NumPy `RandomState`, so it is stable
across NumPy versions). Every input rounds to the printed value (2 decimals in MPa, 3 for the ratios),
each to exactly one row of the draw; the script stops otherwise.

The rerun needs fdfault. Steps used here (Ubuntu 24.04 container, 2 CPUs shared with other jobs,
2026-10-31): `apt-get install libopenmpi-dev openmpi-bin` (Open MPI 4.1.6, g++ 13.3.0);
`git clone https://github.com/edaub/fdfault` (tag v1.0.1 = commit ec066f0, the version the paper's
Dockerfile builds); `cd fdfault/src && make` (16 s); `pip install ./fdfault/python` (fdfault 1.0) into
a venv with numpy 2.5.4 and scipy 1.18.1 (Python 3.13.18). Then the command above, one MPI process per
run. The five reruns in the shipped file (rows 1, 11, 13, 21, 29: the smallest and largest sizes and two
validation runs) took 1188 s in all (19:23:58 to 19:43:46), 165–354 s each, on one process of a
container whose two CPUs other jobs were also using (load average about 5). The paper reports about 20 s per run on four cores of a MacBook (p. 9). One MPI
process and two give the same integrated slip to all 17 digits (row 1: 213.36664293573793 both ways).
The authors' own route is the Docker image of https://github.com/alan-turing-institute/fabmogp_paper
(`make` in the container, about 20 min on 4 cores); Docker Hub is not reachable from this container, so
it was not used.

## Accuracy

- **Reruns vs the paper**: the five reruns give 213.366643, 296.449618, 29.239023, 62.212413 and
  253.263079 m km against the printed 213.37, 296.45, 29.24, 62.21 and 253.26: largest difference
  0.0034 m km, inside the ±0.005 of the printed rounding. This confirms the inputs in
  `design_inputs.csv` and the set-up in `fdfault_problem.py`. It is consistent with the paper's runs
  using fdfault's default density 2.67 × 10³ kg/m³, but it does not discriminate 2.67 from Table 1's 2.68
  (no run at 2.68 was made); the evidence for 2.67 is the code default (see Friction above). Gradients: none are shipped, so there is no gradient check.
- **Input decks**: `fdfault_problem.py` was given keyword arguments for the held settings after the
  reruns; at the defaults it writes byte-identical input, surface and load files to the version that
  made the reruns (checked for three design points).
- **Rounding**: the printed sizes have two decimals, an error of at most 0.005 m km on values of
  29–296 m km; the notebook's results do not change at that level.
- **Discretization**: not checked. A grid refinement (`refine=2`, 8 times the cost) was out of the time
  budget; the paper does not report one either. The size is a property of this grid and this fault
  profile; the paper says other profile realizations change the values but not its conclusions (p. 8).

## How these files relate to the paper

- `paper_tables.csv` is Tables 2 and 3 as printed, row for row, in the paper's order (which is the
  FabSim3 result-directory order, not the order of the Latin hypercube draw; `design_inputs.csv` column
  3 gives the draw's row).
- The notebook analyses all 30 runs together; the paper used the 20 Table 2 runs to train its emulator
  and the 10 Table 3 runs to validate it. The notebook also reports the QPHD direction from the 20
  training runs alone and a clean validation on Table 3: the direction and the quadratic in y1 are
  both fitted to the 20 training runs only, then scored on the 10 validation runs (RMS error 13.6 m km,
  largest 20.6 m km).
- The paper's 50-point training designs (p. 13) and its 10 000 query points are not published.

## License and credit

Data: CC BY 4.0 (`LICENSE-DATA`); the printed values come from Daub et al. (2021), CC BY 4.0, and the
reruns from fdfault (MIT). Code: MIT (`LICENSE`); `fdfault_problem.py` is adapted from fabmogp v1.0
(the tag the paper's Dockerfile checks out; the set-up in its `earthquake.py` and `utils.py` was diffed
against `fdfault_problem.py` and is the same; v1.1 changes only the `mpiexec` launch line in
`run_simulation`) (MIT, Copyright (c) 2020 Alan Turing Institute, Eric Daub, Derek Groen, Hamid Arabnejad;
`LICENSE-fabmogp`). The paper's workflow repository (fabmogp_paper) is MIT as well. Cite Daub et al.
(2021), doi:10.1098/rsta.2020.0076, and fdfault.
