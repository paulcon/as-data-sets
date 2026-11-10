# Parcel_Droplet_Activation: data

Kind: **stored data (generated with pyrcel 2.0.0), with a regeneration script.** The model is the adiabatic
cloud-parcel model of Rothenberg and Wang (2016) as released by the first author in pyrcel, a JAX/diffrax code
outside this repository, so it is not a numpy/scipy "Python model" in the repository's sense (CLAUDE.md, "Two
kinds of data sets"), even though the driver is a Python module (`pyrcel_parcel_driver.py`). The data were
generated for this repository in a Linux container (x86-64, Python 3.13.18) on 2026-11-09/10. The shipped CSVs
are the record. **Regenerating the data, or rerunning the accuracy checks, needs pyrcel 2.0.0**
(`pip install pyrcel==2.0.0`, which brings JAX and diffrax): `generate_data.py` and `check_accuracy.py` stop with
an error naming what to install when it is missing. The notebook needs only numpy, matplotlib and
active_subspaces: it reads the shipped files and the driver's constants (the driver imports without pyrcel).

## Files

| file | contents | role |
|---|---|---|
| `Parcel_Droplet_Activation.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2016; outputs committed |
| `pyrcel_parcel_driver.py` | `run`, `run_with_grad`, `fd_gradient`; `NAMES`, `UNITS`, `LOWER`, `UPPER`, `NOMINAL`, settings (`NBINS`, `Z_CAP`, `S0`, `RTOL`) | drives pyrcel; its constants import without pyrcel |
| `generate_data.py` | writes the four CSVs (`--out DIR` required; `--rows a:b`, `--check`) | regenerates the data (needs pyrcel 2.0.0) |
| `check_accuracy.py` | recomputes the numbers of "Accuracy" below (`--quick`: checks 1–2 on three rows; `--fig3`: check 6, the paper's Fig. 3, only) | accuracy check (needs pyrcel 2.0.0) |
| `LICENSE-pyrcel.txt` | pyrcel's BSD 3-clause licence | credit for the outside code the driver calls |
| `inputs.csv` | 300 × 8: log10 N [log10 cm⁻³], log10 μ_g [log10 µm], σ_g, κ, log10 V [log10 m s⁻¹], T [K], P [Pa], α_c | Latin hypercube, seed 4482016 |
| `outputs.csv` | 300 × 8: log10 S_max, activated fraction, peaked (1/0), grad_ok (1/0), grad_method (0 none, 1 AD, 2 FD), height of S_max [m], S at 400 m, solver steps | model output and flags |
| `gradients.csv` | 300 × 8: ∂log10 S_max/∂x_j in the paper coordinates; nan where grad_ok = 0 | model output |
| `act_frac_gradients.csv` | 300 × 8: ∂(N_act/N)/∂x_j; nan where grad_ok = 0 | model output |

Analysis subset (the notebook's rule, from stored columns): rows with log10 μ_g ≥ −2.5 and κ ≥ 0.1, 240 rows,
all with gradients.

Every CSV has two `#` lines (a description, then the column names with units) and reads with
`np.loadtxt(path, delimiter=",")` (nan entries included). Inputs are rounded to the 16 written digits before the
model runs, so each output belongs exactly to its stored input row.

## Model

### Citation

D. Rothenberg and C. Wang, "Metamodeling of droplet activation for global climate models", *Journal of the
Atmospheric Sciences* 73(3) (2016) 1255–1272, doi:10.1175/JAS-D-15-0223.1 (Crossref-checked: volume 73,
issue 3, pp. 1255–1272, print March 2016). The PDF read is the publisher's version. No PDF is in this
directory.

### Physics and solver

- Lagrangian adiabatic, non-entraining parcel (Sec. 2a p. 1257; Appendix A pp. 1268–1270): Eqs. A1–A5 and A11
  for S, T, w_v, w_c, P and the wet radius of each bin, growth coefficient G with non-continuum corrections
  (Eqs. A6–A8; thermal accommodation 0.96), κ-Köhler equilibrium supersaturation (Eq. A9).
- One lognormal mode (Eq. 1) in 200 bins equally spaced in ln r (p. 1257). The paper prints the bin range as
  "(min[0.1 nm, μ_g/10σ_g], μ_g × 10σ_g)"; pyrcel 2.0.0, used here, spans [μ_g/(10σ_g), 10σ_g μ_g] with
  trapezoidal bin concentrations, rebuilt in JAX in the driver so the bins are differentiable (open question 1).
- Initial state: particles equilibrated at 100 % relative humidity, S0 = 0 (p. 1260).
- Solver: the paper used VODE (p. 1257, App. A); pyrcel 2.0.0 uses diffrax's Kvaerno5 (an L-stable ESDIRK
  method) with a PID step controller, rtol 1e-7 and pyrcel's per-component atol (radii 1e-12 m). The driver calls
  pyrcel's own right-hand side (`pyrcel.parcel_aux.parcel_ode_sys`), equilibration
  (`pyrcel.equilibrate.equilibrate_initial_state`) and tolerances; only the diffrax call is written out, to
  select the forward-mode adjoint for differentiation.
- **S_max (a choice of this data set).** The paper defines S_max as the maximum supersaturation of the
  ascending parcel (p. 1258) and does not say how long the parcel rises. Here S_max is the largest S in the first
  Z_CAP = 400 m of ascent (the default height cap of pyrcel's ensemble functions), located on 2001 saved heights
  (0 and 2000 geometric steps from 1 cm to 400 m) and refined by a cubic Hermite interpolant through the saved S
  and the exact dS/dt. For very large particles S settles into a slowly rising quasi-steady state and has no
  interior maximum below 400 m; then S at 400 m is used and `peaked` = 0 (74 of 300 rows, 67 of the 240 analysis rows, mostly μ_g > 1 µm); for
  these rows the quantity is S at 400 m, not an activation maximum. S is decimal.
- **Activated fraction.** Eq. 10 (p. 1260), N_act/N = ½{1 − erf[2 ln(S_m/S_max)/(3√2 ln σ_g)]}, with S_m the
  approximate κ-Köhler critical supersaturation of a dry particle of radius μ_g at the initial T (our choice of
  approximation and temperature; the paper does not specify them).

### Inputs (Table 1, p. 1259; each uniform between the bounds, p. 1259)

| input | symbol | units | bounds | Fig. 3 nominal (p. 1265) |
|---|---|---|---|---|
| mode number concentration | log10 N | log10 cm⁻³ | [1, 4] | 3 (N = 1000 cm⁻³) |
| mode geometric mean radius | log10 μ_g | log10 µm | [−3, 1] | log10 0.05 |
| mode geometric standard deviation | σ_g | — | [1.2, 3.0] | 2.0 |
| mode hygroscopicity | κ | — | [0.0, 1.2] | 0.54 |
| updraft velocity | log10 V | log10 m s⁻¹ | [−2, 1] | log10 0.5 |
| air temperature | T | K | [240, 310] | 283 |
| air pressure | P | Pa | [50 000, 105 000] | 85 000 (850 hPa) |
| accommodation coefficient | α_c | — | [0.1, 1.0] | 1.0 |

### Sampling design

A Latin hypercube of M = 300 points in the Table 1 box (numpy `default_rng(4482016)`: in each column a random
permutation of the 300 strata and a uniform position within each stratum). It is not the paper's design: the
paper computed Sobol' indices of the parcel model with the column swap-out Latin hypercube scheme of Weirs et
al. (2012), n = 1280, (2 + M) n runs (p. 1261), and validated its emulators on two maximin Latin hypercubes of
10 000 points each (p. 1261). Points near the edges of physical meaning, counted: the smallest sampled κ is
0.00209 (κ = 0 itself cannot be run: an insoluble particle has no equilibrium at 100 % RH), and in the 37 rows
with μ_g < 3.16 nm the smallest dry bin radius is 0.038–0.222 nm, below the size of a water molecule.

### Gradients

- Coordinates: the paper coordinates of `inputs.csv` (log10 N, log10 μ_g, σ_g, κ, log10 V, T, P, α_c), physical
  units per column header. The notebook scales them to [−1, 1] by each input's half-range (through `ac.Uniform`).
- Method: forward-mode automatic differentiation (`jax.jacfwd`) through the bin construction, the equilibration
  root-find (implicit differentiation in optimistix) and the diffrax solve with `ForwardMode` adjoint; the Hermite
  peak location is held fixed (envelope theorem). 260 rows.
- Rows with log10 μ_g < −2.5 (μ_g < 3.16 nm, 37 rows): value only (`MU_GRAD` in `generate_data.py`). There the
  smallest bins are below 0.3 nm in dry radius, the derivative solve returned non-finite values in 4 of 5 trial
  rows, and runs cost 10–100 times more.
- Rows above that where the derivative solve failed (3 rows: 19, 89 and 256): central differences of the value
  with step 1e-3 of each half-range (`fd_gradient`; one-sided second order at the box faces), `grad_method` = 2.

### Published results

- Fig. 5 (p. 1266): total Sobol' indices of S_max for the parcel model and emulators (a bar chart; the text says
  updraft speed and aerosol size most strongly contribute, followed by number, with α_c and κ weakest).
- Table 4 (p. 1267): leading main Sobol' indices of the OLS chaos expansions; order 5: log10 μ_g 0.359,
  log10 V 0.312, log10 N 0.129, σ_g 0.056, T 0.047, log10 N·log10 μ_g 0.021, κ 0.019, log10 V·log10 μ_g 0.017.
- Tables 2–3 (p. 1262): emulator error statistics (order-5 OLS: NRMSE 0.129, r² 0.976 for S_max).
- No active subspace is computed in the paper.

## Regenerating

External run, recorded here (`generate_data.py` regenerates the files where pyrcel 2.0.0 is installed):

```bash
python3.13 -m venv pvenv && pvenv/bin/pip install pyrcel==2.0.0 scipy   # jax 0.12.0.1, diffrax 0.7.2, numpy 2.5.4
pvenv/bin/python generate_data.py --out /tmp/parcel_gen                  # all 300 rows, resumable (rows.jsonl)
pvenv/bin/python generate_data.py --out /tmp/parcel_chk --rows 0:10 --check
```

Run time: the shipped files took 5084 s of CPU (1.41 CPU-hours; one process, one XLA thread) on a shared
2-CPU container, about 5 h of wall time in resumable chunks of 20 rows. Most rows cost 0.2–20 s; the three
finite-difference rows cost 350–850 s each, and the value-only rows with μ_g of 1–3 nm up to about 200 s. JAX
compiles the solve once per process (30–80 s CPU). `generate_data.py --out DIR --rows 0:10 --check` took 104 s
of CPU (243 s wall) and printed: inputs max difference 0, outputs max difference 2.89e-15, flags equal,
gradients and activated-fraction gradients max relative difference 0 (8 rows with gradients). `--check` tolerance: 1e-6 absolute on log10 S_max and the activated fraction, flags equal, 1e-6
relative 2-norm per row on the scaled gradients. JAX on another platform or version may differ at the solver-
tolerance level (rtol 1e-7), which is why the tolerance is not exact equality. The script writes only to
`--out`, which must be outside this directory (a subdirectory is refused too).

## Accuracy

All numbers below are printed by `check_accuracy.py` (pyrcel 2.0.0; about 13 min wall on the shared container;
`--fig3` runs check 6 alone in about 1 min),
except where a notebook cell is named.

- **Nominal point** (Fig. 3, p. 1265): log10 S_max = −2.737222 (S_max = 0.1831 %), peak 7.31 m above the start,
  activated fraction 0.5054. AD gradient against central differences (step 1e-3): relative 2-norm 4.1e-6
  (`check_accuracy.py`, check 1).
- **Gradients** (check 2): the shipped AD gradients against central differences of the model at ten fixed rows
  (eight with an interior peak, two without). Relative 2-norm error of the scaled gradient: with step 1e-3 of each
  half-range, median 5.7e-5 and max 2.0e-3; with step 1e-4, median 2.3e-6 and max 1.8e-4. The errors fall with
  the step, so they are the truncation error of the differences, and the AD gradients are accurate to better than
  about 1e-4. (The three FD rows use step 1e-3, so their gradients carry errors up to about 1e-3.)
- **Solver tolerance** (check 3): re-solving the same rows at rtol 1e-9 changes log10 S_max by at most 7.0e-7 and
  the gradient by at most 1.4e-5 (relative).
- **Locating the maximum** (check 4): pyrcel's own event solve (dS/dt = 0 root-find, `integrator.find_smax`) at
  the eight interior-peak rows agrees with the shipped log10 S_max to at most 3.8e-4 (median about 1e-9).
- **Ascent cap** (check 5): for three rows without an interior peak, the maximum over 800 m exceeds the shipped
  400 m value by 0.046, 0.022 and 0.004 in log10 S_max (still no interior peak by 800 m). These rows' values
  depend on the cap at that level, and a longer ascent is no remedy: for these coarse modes S is a slowly rising
  quasi-steady value, not an activation maximum, so for the 74 flagged rows (67 of the 240 analysis rows) the
  quantity is S at 400 m. The subspace does not depend on them: the interior-peak rows alone give |cos| = 0.999
  (λ1/λ2 = 21.62), and the rows with μ_g ≤ 1 µm (167, 23 still rising) |cos| = 0.989 (λ1/λ2 = 15.55; notebook,
  check cell after the summary plots).
- **Fig. 3 of the paper** (check 6; p. 1265, one input at a time, the others at the nominal point; paper values
  read from the black parcel-model curves). S_max (%) and activated fraction, ours (paper):

  | input changed | S_max (%) | activated fraction |
  |---|---|---|
  | N = 1e2, 1e4 cm⁻³ | 0.396 (≈ 0.40), 0.057 (≈ 0.06) | 0.775 (≈ 0.77), 0.135 (≈ 0.13) |
  | μ_g = 5 nm, 0.1 µm | 0.749 (≈ 0.75), 0.082 (≈ 0.09) | 0.025 (≈ 0.03), 0.596 (≈ 0.60) |
  | κ = 0.13, 1.2 | 0.255 (≈ 0.25), 0.142 (≈ 0.15) | 0.362 (≈ 0.37), 0.562 (≈ 0.56) |
  | V = 1, 10 m/s | 0.264 (≈ 0.27), 0.982 (≈ 0.98) | 0.643 (≈ 0.64), 0.948 (≈ 0.95) |

  Every value agrees with the paper's curve within reading accuracy, so near the nominal point pyrcel 2.0.0
  behaves as the paper's parcel model. Fig. 3c plots a point at κ = 0 with S_max ≈ 0.40 % and activated fraction
  ≈ 0.12; pyrcel gives 0.395 % and 0.122 at κ = 0.01, and 0.533 % (0.024) at κ = 0.001, so the paper's "κ = 0"
  point is consistent with a small positive κ, and S_max keeps growing as κ → 0.
- **Range against Fig. 1a** (notebook, cell after the analysis-box definition): the paper's Fig. 1a (p. 1261)
  plots parcel-model S_max on axes from 1e-2 % to 1e1 %. Here 86 of 300 rows (82 of the 240 analysis rows) lie
  below 1e-2 %, down to 1.5e-6 %; all have μ_g > 0.1 µm, and 60 of the 82 are still rising at 400 m.
  Var(log10 S_max) on the analysis box is 2.39 with them and 0.46 without (they hold 0.620 of the sum of squared
  deviations).
- **Hygroscopicity near 0** (notebook): κ ∂log10 S_max/∂κ stays between −0.04 and −0.32 at the smallest κ, so the
  derivative grows like 1/κ and E[(∂f/∂κ)²] is unbounded on κ ∈ [0, 1.2]; the notebook estimates the subspace for
  κ ≥ 0.1 and shows the same result for κ ≥ 0.05 and κ ≥ 0.2.

## How these files relate to the paper

The paper's parcel model was a 2015 code integrated with VODE; its runs were not archived (no data are linked
from the paper). These files are a new computation with the first author's current release, pyrcel 2.0.0, over
the paper's input box and distributions, with a different design (a 300-point Latin hypercube rather than the
Sobol' and validation designs) and with two choices the paper does not fix (the 400 m ascent cap and the
details of Eq. 10). The paper's numbers are compared in the notebook's last section.

## License and credit

Data: CC BY 4.0 (repository data licence); generated here, so no exception is needed. Code: MIT. pyrcel is
(c) Daniel Rothenberg, BSD 3-clause (`LICENSE-pyrcel.txt`); none of its code is copied here. Please cite
Rothenberg and Wang (2016) and pyrcel.

## Unknowns and open questions

1. The bin range printed on p. 1257, "(min[0.1 nm, μ_g/10σ_g], μ_g × 10σ_g)", reads literally as a lower edge of at
   most 0.1 nm; pyrcel uses μ_g/(10σ_g). Which did the paper's runs use?
2. How long the parcel rose, or how S_max was found when S had not peaked (large particles): not stated.
3. VODE tolerances of the paper's runs: not stated.
4. The κ = 0 point of Fig. 3c (S_max ≈ 0.40 %, activated fraction ≈ 0.12): was it run at a small positive κ? pyrcel
   gives these values at κ = 0.01 (κ = 0 has no equilibrium at 100 % RH; Accuracy, Fig. 3). Also how modes near
   μ_g = 1 nm were run.
5. The temperature and Köhler approximation behind S_m in Eq. 10: not stated.
6. Did the parcel runs for coarse modes (μ_g > 0.1 µm) reach S_max ≥ 1e-2 %, as the axes of Fig. 1a suggest, or
   are lower values clipped from the figure? How long was the ascent? Here 82 of the 240 analysis rows lie below
   1e-2 % and hold most of the variance of log10 S_max; they make μ_g dominate the main indices (0.630 from the
   surrogate; 0.72 binned from the stored values on the full Table 1 box, 0.66 on interior-peak rows) against
   Table 4's 0.359, with V 0.146 against 0.312.
7. Were the Sobol' samples of the parcel model (Fig. 5) filtered for runs whose S had not peaked?
