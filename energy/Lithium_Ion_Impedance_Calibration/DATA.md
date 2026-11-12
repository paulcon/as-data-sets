# Lithium_Ion_Impedance_Calibration: data

Kind: **stored data, generated for this repository with outside codes (PyBaMM 26.10.1.1 and PyBOP 26.3);
regeneration script included**. The data were generated with PyBaMM (its frequency-domain EIS method,
formerly PyBaMM-EIS) and PyBOP (the paper's grouped-parameter SPMe). The shipped files are the record of
those runs and are never altered. `spme_eis_driver.py` and `generate_data.py` regenerate them. **Regenerating
the data, or rerunning the model part of the accuracy checks (`check_accuracy.py`), needs PyBaMM 26.10.1.1,
pybammsolvers 0.11.1, PyBOP 26.3 and CasADi 3.8.1** (recipe under "Regenerating"); both scripts stop with an
error that says what to install when they are missing. The repo's `.venv` has none of them. The notebook needs
only numpy, scipy, matplotlib and active_subspaces; it imports `spme_eis_driver.py` for its constants, which
load without PyBaMM. In `.venv` (Python 3.11) it runs in about 9 s (`jupyter nbconvert --execute`, 2026-11-02; the notebook's own timer prints 4 s).

## Files

| file | contents | role |
|---|---|---|
| `Lithium_Ion_Impedance_Calibration.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom in `.venv` with seed 2025; outputs committed |
| `spme_eis_driver.py` | `PARAMETERS`, `NAMES`, `LOWER`, `UPPER`, `SOCS`, `FREQUENCIES`, `MESH`; `SPMeImpedance` (Eq. 9 with PyBaMM/PyBOP), `ImpedanceMisfit` (Eq. 7 with adjoint gradients) | the model set-up (needs PyBaMM and PyBOP to run) |
| `generate_data.py` | the design, the resumable runs, assembly, `--check`, `--gradcheck`, `--meshcheck`, `--reference`, `--jacobian`, `--checkpoints` | regenerates and checks the data (needs PyBaMM and PyBOP) |
| `check_accuracy.py` | recomputes the numbers of "Accuracy" below; `--data-only` runs the numpy checks alone, `--skip-slow` leaves out the finite differences | accuracy check (checks 5–10 need PyBaMM and PyBOP) |
| `inputs.csv` | 1000 × 18 grouped parameters, physical units, columns in the order of `NAMES` | design |
| `outputs.csv` | 1000 × 10: Φ, then Φ_m for SOC 10 %, …, 90 % [Ω²] | model output |
| `gradients.csv` | 1000 × 18: ∂Φ/∂θ_j with respect to the **physical** inputs [Ω² per input unit] | model output |
| `gradients_soc.csv` | 1000 × 162: column 18 m + j is ∂Φ_m/∂θ_j (SOC-major), physical | model output |
| `reference_impedance.csv` | 60 × 19: frequency [Hz], then Re Z and Im Z [Ω] for each of the 9 SOCs at the true θ | the synthetic data |
| `true_parameters.csv` | 1 × 18: the true θ (Table 3), full precision | reference point |
| `impedance_jacobian.csv` | 540 × 38: SOC [%], frequency [Hz], ∂Re Z/∂θ_j (18), ∂Im Z/∂θ_j (18) at the true θ, physical | local sensitivities |
| `check_parameters.csv` | 6 × 19: case, then θ (physical) at six extra points: 0 the Table 3 best fit θ̂ as printed; 1 θ̂ with R₀ = 0.01005 Ω; 2–5 θ_true + t·(half-range)·v_min(H), t = ±1e-3, ±1e-2 | extra model runs |
| `check_spectra.csv` | 3240 × 5: case, SOC [%], frequency [Hz], Re Z, Im Z [Ω] at the six points (case-major, then SOC-major) | extra model runs |

Each CSV has two `#` header lines (a description and the column names with units) and is read by
`np.loadtxt(path, delimiter=',')`. Values are written with `%.17g`, so float64 values round-trip exactly
(checked for `inputs.csv` when it is written). The data files total 5.3 MB (`gradients_soc.csv` is 3.6 MB of it).

## Model

**Paper.** N. Hallemans, N. E. Courtier, C. P. Please, B. Planden, R. Dhoot, R. Timms, S. J. Chapman,
D. Howey, S. R. Duncan, "Physics-Based Battery Model Parametrisation from Impedance Data", *J. Electrochem.
Soc.* 172(6) (2025) 060507, doi:10.1149/1945-7111/add41b (title, authors, volume 172, issue 6, article
060507 and CC BY 4.0 licence checked on Crossref, 2026-11-01). The PDF read for this data set is the arXiv
preprint, arXiv:2412.10896v4 (23 Apr 2025). Page, table and equation numbers below refer to it; the Table 3
values were not checked against the journal version (its pages could not be fetched here).

**Cell model** (Appendix C, Eqs. C.1–C.15; Table 2). This is the single particle model with electrolyte
(SPMe) with double-layer capacitance, written in grouped parameters:

- spherical diffusion in one particle per electrode, with time scale τ_d±;
- Butler–Volmer kinetics with the charge-transfer time scale τ_ct± in the exchange current;
- a double layer of capacitance C± on each electrode surface potential;
- electrolyte diffusion across the cell, with time scales τ_e±, τ_e^sep, relative porosities ζ±, reference
  electrolyte capacity Q_e and transference number t⁺;
- the terminal voltage v = v₊ − v₋ + η_e − R₀ i.

The electrode capacities are Q_th± = ∓Q_meas/(c±^100% − c±^0%) (Eq. 11), and the stoichiometry at an SOC
follows by linear interpolation (Eq. 13). The OCPs are those of the Chen2020 LG M50 parameter set.

The implementation is PyBOP's `GroupedSPMe(options={"surface form": "differential"})`, which PyBOP's source
attributes to this paper, with:

- PyBaMM finite volumes on the paper's mesh: 100 points in each particle radius, 100 across each electrode,
  20 across the separator (Section 2);
- 425 states, the paper's Nx = 424 (Table 1) plus PyBaMM's algebraic current variable for EIS.

**Impedance** (Eq. 9). Z_m(ω, θ) = [(jωM − J_θ,m)⁻¹B]_voltage, with J the Jacobian at the equilibrium of
SOC_m. It is computed as in PyBaMM's `EISSimulation`:

- the model is built once, with the 18 parameters and the SOC as PyBaMM input parameters;
- J and the output Jacobian are evaluated through CasADi functions of the inputs;
- one SuperLU factorization is done per frequency.

At two random θ and two SOCs, Z agrees with a separate `pybamm.EISSimulation` that has numeric parameter
values to 3.5e-15 relative (`check_accuracy.py` check 7; 3e-15 when the data were made). This checks the input ordering.

**QoI** (Eq. 7, Section 6.1). Φ(θ) = Σ_{m=1}^{9} Σ_{k=1}^{60} |Z_m(ω_k) − Z_m(ω_k, θ)|² [Ω²]:

- 9 SOCs: 10 %, 20 %, …, 90 %;
- 60 frequencies, `logspace(log10(2e-4), 3, 60)` Hz, i.e. 200 µHz to 1 kHz;
- the data Z_m(ω_k) are the model's spectra at the true θ: `reference_impedance.csv`, the paper's Fig. 7.
- the data are **noise-free**: the paper does not mention adding noise to its simulated spectra (Section 6.1),
  so none is added here, and Φ(θ_true) = 0 is the global minimum.

This is the paper's synthetic-data experiment (Section 6.1). The measured LG M50LT spectra of Section 7 are
not deposited. Φ spans 2.43 decades over the box (1.26e-2 to 3.36 Ω²), so the notebook's QoI is f = log₁₀ Φ, with
∇f = ∇Φ/(Φ ln 10). The conversion is done once, in the notebook's load cell.

**Inputs** (Eq. 16; Table 3, p. 12). There are 18 grouped parameters, uniform on the optimisation bounds of
Table 3, the box the paper's particle swarm searched. All are in physical units.

| name | symbol | units | true θ (Table 3) | range (Table 3) | source |
|---|---|---|---|---|---|
| `tau_d_p` | τ_d+ | s | 6812 | [5e2, 1e4] | Table 3, p. 12 |
| `tau_d_n` | τ_d− | s | 1041 | [5e2, 1e4] | Table 3 |
| `tau_e_p` | τ_e+ | s | 409.2 | [2e2, 1e3] | Table 3 |
| `tau_e_n` | τ_e− | s | 634.7 | [2e2, 1e3] | Table 3 |
| `tau_e_sep` | τ_e^sep | s | 246.2 | [2e2, 1e3] | Table 3 |
| `zeta_p` | ζ+ | – | 0.7128 | [0.5, 1.5] | Table 3 |
| `zeta_n` | ζ− | – | 0.5319 | [0.5, 1.5] | Table 3 |
| `Q_e` | Q_e | A s | 804.8 | [5e2, 1e3] | Table 3 |
| `tau_ct_p` | τ_ct+ | s | 4657 | [1e3, 5e4] | Table 3 |
| `tau_ct_n` | τ_ct− | s | 27592 | [1e3, 5e4] | Table 3 |
| `C_p` | C+ | F | 0.5935 | [0, 1] | Table 3 |
| `C_n` | C− | F | 0.6719 | [0, 1] | Table 3 |
| `c0_p` | c+^0% | – | 0.8540 | [0.8, 0.9] | Table 3 |
| `c0_n` | c−^0% | – | 0.02635 | [0, 0.1] | Table 3 |
| `c100_p` | c+^100% | – | 0.2638 | [0.2, 0.3] | Table 3 |
| `c100_n` | c−^100% | – | 0.9106 | [0.85, 0.95] | Table 3 |
| `t_plus` | t⁺ | – | 0.2594 | [0.2, 0.5] | Table 3 |
| `R0` | R₀ | Ω | 0.01 | [0, 0.05] | Table 3 |

The remaining inputs are held fixed:

- Q_meas and ℓ±, as in Table 3, which prints Q_meas = 18551 A s, ℓ₊ = 0.4375, ℓ₋ = 0.4930. The runs use
  PyBOP's Chen2020 grouping, Q_meas = 18551.5 A s (18551.514) and ℓ₋ = 0.49306, which the paper's printed
  values round or truncate.
- T = 298.15 K.

`true_parameters.csv` holds PyBOP's grouping of Chen2020 with R₀ = 0.01 Ω. Table 3 says the true R₀ is
10 mΩ, and its Fig. 12 caption says "Chen2020 parameters with contact resistance of 10 mΩ"; PyBOP's grouping
without a contact resistance gives R₀ = 5.41 mΩ, so the Table 3 value is used. The other 17 values agree with
Table 3's printed values to 3.9e-4 relative or better (notebook, "Comparison"). Only τ_e+ (409.06 vs 409.2)
and τ_e− (634.53 vs 634.7) differ by more than the 4-digit print rounding, by 3.3e-4 and 2.7e-4. This may come
from a difference in Chen2020's electrolyte diffusivity between PyBaMM 24.9 (the paper's) and 26.10; it is
not material.

**Sampling design.** N = 1000 independent Monte Carlo samples, uniform on the box, from numpy
`default_rng(20261101)` (`generate_data.design`). This is not a Latin hypercube. `inputs.csv` is the source of
truth: `generate_data.inputs()` checks that it equals the seeded design to 1e-15 relative.

**Uniform, not log-uniform.** τ_ct± span a factor of 50 and τ_d± a factor of 20, where a log-uniform density
is the usual alternative. The density is uniform because the paper's search space is the Table 3 box, and
because R₀ and C± have lower bound 0 and cannot be log-uniform. In PyBOP 26.3 with PINTS 0.6.1 (source read,
`pybop/optimisers/base_pints_optimiser.py` and `pints/_optimisers/_pso.py`), the particle swarm starts one
particle at the initial guess and draws the others uniformly between the bounds, in each parameter's
transformed coordinates (the physical ones unless a transformation, such as a log transform, is set). The paper
states neither its PyBOP version nor any transformation, so that its swarm searched uniformly in physical
coordinates is unverified. The choice affects the size of the gap, not the direction: under the uniform design
93 % of the samples have τ_ct+ above its true value and 9 % lie below the truth along w₁. Reweighting the
same samples to log-uniform τ_d±, τ_ct± (importance weights ∝ 1/∏τ, gradients chain-ruled to ln τ;
effective sample size 41 of 1000, so indicative only) gives λ₁/λ₂ = 5.2 (bootstrap 4.5–6.1) instead of
23.55, with w₁ = R₀ +0.67, τ_ct− +0.60, τ_ct+ +0.42; log coordinates with uniform weights give 9.1
(notebook, "Uniform or log-uniform time scales").

**Samples outside physical meaning.** None, by construction:

- every input lies in its Table 3 range;
- c₊^100% < c₊^0% and c₋^0% < c₋^100% hold throughout the box;
- the smallest capacitances are 1.66e-3 F (C₊) and 2.96e-4 F (C₋), and 16 samples have C₊ or C₋ < 0.01 F.

All are kept. C = 0 itself, the lower bound of Table 3, makes the double-layer equation singular, and it was
not drawn.

**Gradients.** ∂Φ/∂θ is exact up to round-off and is computed by an adjoint method (`spme_eis_driver.py`):

- at each SOC and frequency, the forward solve A x = b gives Z, and the adjoint solve Aᵀλ = g, with the same
  LU factors, gives dZ = λᵀ dJ x + (output-Jacobian terms), where A = jωM − J;
- the weights 2 conj(Z − Z_ref) are summed over frequencies into one bilinear form on the nonzeros of J;
- its gradient with respect to θ is one reverse-mode CasADi sweep through J(y₀(θ), θ) and the output
  Jacobian. The sweep includes the dependence of the operating point y₀ on the stoichiometry parameters.

The per-SOC gradients come from the same computation. `impedance_jacobian.csv` is ∂Z/∂θ at the true θ by
central differences, with a step of 1e-5 of each range (see "Accuracy").

**Published results.** The paper has no active-subspace or global sensitivity analysis. It reports:

- the particle-swarm fit to these synthetic spectra: best of ten runs, the relative standard deviation over
  the ten runs, and fitting errors FE_m (Eq. 17) of 0.84–0.97 % at the nine SOCs (Table 3);
- one-at-a-time sensitivities of the impedance over [0.5θ, 2θ] (Figs. 8 and 12);
- representative computation times (Table 1): SPMe 415 ms per spectrum (60 frequencies, one SOC), averaged
  over ten runs.

The notebook's "Comparison" section lists what can be compared.

## Regenerating

The data need a separate Python environment with PyBaMM and PyBOP. The recipe used for the shipped files
(written out here; the environment itself was an external run in a Linux container and is not part of the
repository):

```bash
python3.13 -m venv eis-venv          # Python 3.13.x; other Pythons with PyBaMM wheels should also work
eis-venv/bin/pip install "pybop==26.3" "pybamm==26.10.1.1" "pybammsolvers==0.11.1" "casadi==3.8.1" "numpy==2.3.5" "scipy==1.18.1"
eis-venv/bin/python -c "import pybamm, pybop; print(pybamm.__version__, pybop.__version__)"   # must print 26.10.1.1 26.3
export PYBAMM_DISABLE_TELEMETRY=true   # PyBaMM 26 otherwise asks about telemetry on first import, which can stall a script
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
eis-venv/bin/python check_accuracy.py --skip-slow   # checks 1-8, a few minutes; check 5 must print Phi = 0.294866279668775... Ohm^2 at the box centre
```

`generate_data.py` and `check_accuracy.py` stop if the installed PyBaMM, PyBOP or CasADi versions differ from
these (override with `--any-version`). `spme_eis_driver.py` uses PyBaMM internals (`sim._built_model`,
`jac_rhs_algebraic_eval`, `y_slices`, `pybamm.BaseSolver().set_up`), so other versions may not run. Then:

```bash
eis-venv/bin/python generate_data.py --reference --jacobian                    # reference_impedance.csv, true_parameters.csv, impedance_jacobian.csv
eis-venv/bin/python generate_data.py --rows 0:1000 --partial /tmp/eis_partial  # resumable: rerun until all rows exist
eis-venv/bin/python generate_data.py --assemble --partial /tmp/eis_partial     # writes inputs/outputs/gradients CSVs
eis-venv/bin/python generate_data.py --check 0,517,999                         # recompute rows and compare; writes nothing
eis-venv/bin/python generate_data.py --checkpoints                             # check_parameters.csv, check_spectra.csv
```

`--rows` and `--assemble` need `--partial`, a directory outside the data-set directory.

**Provenance and run time of the shipped files** (original run, Linux x86-64 container, 2026-11-01; Python
3.13.18, PyBaMM 26.10.1.1, pybammsolvers 0.11.1, PyBOP 26.3, CasADi 3.8.1, numpy 2.3.5, scipy 1.18.1):

- **Wall time 2941 s (49 min) for the 1000 samples**, a median of 2.58 s per sample for Φ, its gradient and
  the per-SOC gradients. The runs went in four resumable chunks of at most 19 min each (the first, rows 0–54,
  was stopped by hand and resumed), one process, with `nice -n 10`, on a 2-CPU container shared with other jobs.
- CPU time: the 945 runs with a CPU record took 1219 s, a median of 1.29 s per sample; rows 0–54 have no CPU
  record, and about 1290 s for all 1000 is the estimate.
- `--reference --jacobian` took 6.5 min of wall time (1.5 min CPU); `--check 0,517,999` 35 s;
  `--checkpoints` (six spectra sets, no gradients) 33 s.
- The paper's Table 1 timing is not comparable: it quotes 415 ms per SPMe spectrum (one SOC, 60 frequencies,
  averaged over ten runs); here one spectrum took about 0.1–0.2 s (9 spectra in about 1 s), on a different
  machine. The model set-up (build, discretise, reference spectra) takes about 3 s of CPU per process.

## Accuracy

`check_accuracy.py` recomputes every number in this section that the notebook does not print, except the
Table 3 fitting errors, which the notebook computes from `check_spectra.csv`. Checks 1–4 need only numpy and
were run in `.venv` (Python 3.11, numpy 2.4.6; under 1 s). Checks 5–10 need PyBaMM and PyBOP: they were run
with the versions above (PyBaMM 26.10.1.1, pybammsolvers 0.11.1, PyBOP 26.3, CasADi 3.8.1, numpy 2.3.5, scipy
1.18.1 from PyPI; Python 3.13, macOS x86-64, 2026-11-02) in 1370 s of wall time on one core of a heavily shared
machine (most of it checks 9–10; `--skip-slow` leaves them out), and
printed the values below. Where the same quantity was printed when the data were made (Linux container), it
agreed to the digits quoted unless stated.

- **Box centre** (check 5). Φ at the centre of the Table 3 box is 0.29486627966877 Ω², 3.2e-15 relative to the
  original run's 0.29486627966877590.
- **Reproduction** (check 6, `generate_data.check`). Rows 0, 517 and 999 recomputed from `inputs.csv`: Φ and
  the 9 Φ_m agree with the files to 2.6e-14, ∂Φ to 6.9e-14 and the 162 per-SOC derivatives to 5.9e-14
  (largest difference relative to the largest entry), and the reference spectra to 5.2e-14. In the original
  environment (Linux container) all agreed exactly; the differences here are rounding-level, from another
  platform.
- **Gradients vs. finite differences** (check 9, `generate_data.gradcheck`, rows 3, 400, 871). Central
  differences of Φ with step h·(upper − lower), compared in normalized coordinates as
  max |FD − adjoint| / max |adjoint|:

  | row | h = 1e-4 | h = 1e-6 |
  |---|---|---|
  | 3 | 2.2e-9 | 3.2e-9 |
  | 400 | 1.5e-6 | 1.2e-8 |
  | 871 | 1.8e-5 | 1.8e-9 |

  At h = 1e-6 all three agree to ≤ 1.2e-8 (the same values, to the digits shown, as in the original run).

  The larger differences at h = 1e-4 shrink with h, so they are truncation error of the differences, not of
  the adjoint. Φ is smooth; finite differences would also have worked, at 37 misfit evaluations per gradient.
- **Gradient coordinates** (check 2). A linear regression of log₁₀ Φ on the normalized inputs gives slopes
  0.650 (R₀), 0.397 (τ_ct−) and 0.243 (τ_ct+), against mean normalized gradients 0.567, 0.368 and 0.249;
  over all 18 inputs the two correlate at 0.995. They agree in sign and size, as expected for a curved but
  monotone trend, so the gradients are in physical units and are not scaled twice.
- **Impedance input ordering** (check 7). Z from the input-parameter build agrees with a separate
  `pybamm.EISSimulation` with numeric parameters to 3.5e-15 relative (two θ, two SOCs, coarse mesh).
- **Mesh** (check 8, `generate_data.meshcheck`). With every mesh doubled (845 states) the misfit, and its own
  reference spectra, change log₁₀ Φ by −1.95e-4, −4.50e-4 and +1.72e-4 at rows 3, 400 and 871. This is far below the 2.43-decade
  spread over the box.
- **Impedance Jacobian** (check 10). Central differences at steps 1e-5 and 1e-3 of each range differ by
  1.3e-5 of the largest entry (normalized coordinates); the step-1e-5 result agrees with
  `impedance_jacobian.csv` to 4.5e-9 of the largest entry.
- **Design** (check 1). Every column's kurtosis is 1.78–1.88 (uniform ≈ 1.8), the 1000 rows are distinct, and
  `inputs.csv` equals the seeded design exactly.
- **True parameters** (check 3). `true_parameters.csv` agrees with Table 3's printed θ to 3.9e-4 relative or
  better; only τ_e+ (409.06 vs 409.2, 3.3e-4) and τ_e− (634.53 vs 634.7, 2.7e-4) differ by more than the
  4-digit print rounding.
- **Table 3 fitting errors** (notebook, "The paper's best fit"; spectra from `generate_data.py --checkpoints`,
  case 0). At the printed θ̂, Φ = 1.12e-4 Ω² (log₁₀ −3.95, two decades below the smallest sampled value, −1.90),
  and FE_m = 0.95, 1.07, 1.07, 1.08, 1.05, 1.06, 1.01, 0.97, 0.95 % against the paper's 0.87, 0.97, 0.96, 0.97,
  0.94, 0.94, 0.90, 0.86, 0.84 % (1.09–1.13 times). R₀ shifts Z by exactly ΔR₀ (case 1 agrees with the shifted
  case 0 to 7e-18 Ω), and with R̂₀ = 0.01006 Ω, inside the rounding interval of the printed 0.0101, every FE_m
  matches Table 3 to 0.005, its print precision. Only R₀ was adjusted; the other printed θ̂ values are rounded too.
- **Quadratic regime at the truth** (notebook, cases 2–5). Along the weakest eigenvector of the Gauss–Newton
  matrix, Φ/t² is 2.1 λ_min(H) at t = 1e-3 and 112 λ_min(H) at t = 1e-2 (normalized coordinates), so Φ is
  quadratic only within about 1e-3 half-ranges of the truth. The Cramér–Rao bound would imply standard
  deviations of tens to hundreds of half-ranges for the poorly identified groups, far outside that region, so
  the notebook uses the local analysis **as a ranking only** and quotes no absolute standard deviations.

## How these files relate to the paper

- **Same as the paper:**
  - the model: grouped SPMe with double-layer capacitance, Chen2020 OCPs, and the paper's mesh;
  - the impedance method (Eq. 9);
  - the cost function (Eq. 7) with its 9 SOCs and 60 frequencies;
  - the 18 inputs and their bounds (Table 3);
  - the synthetic-data setting of Section 6.1.
- **Different from the paper:**
  - Software versions: PyBaMM 26.10.1.1 and PyBOP 26.3 here, against the paper's PyBaMM 24.9.0 and
    PyBaMM-EIS. PyBaMM-EIS is archived, and its method is now `pybamm.EISSimulation`.
  - The paper minimizes Φ by particle swarm; we sample Φ uniformly over the same box. The QoI is log₁₀ Φ.
  - The paper's measured LG M50LT spectra (Section 7) are not public and are not used.
- **Paper results compared** in the notebook's "Comparison" table:
  - the qualitative sensitivity statements of Section 5.4 and Fig. 12 agree with the global and per-SOC
    activity scores;
  - the global and local views of identifiability agree: the leading eigenvector of the Gauss–Newton matrix
    at the truth is w₁ (5.9° apart), the 6 weakest directions of C and the 5 weakest of H lie within 3.8° and
    0.4° of the coordinates of τ_e±, τ_e^sep, ζ±, Q_e and t⁺ (the electrolyte-diffusion groups, which the
    paper found hard to identify, and t⁺, which it estimated within 2 %), and the rankings agree
    (Spearman(local sd, 1/global share) = +0.63);
  - both rankings follow Table 3's best-fit errors, the local (Cramér–Rao) ranking more closely (Spearman
    +0.88) than the global shares (+0.59). The local analysis is used as a ranking only (see "Quadratic
    regime" under Accuracy); the noise-free data define no noise level. Caveats: the 7-against-11 group split alone gives +0.78; five of
    the seven poorly identified estimates sit on a box bound, so their errors reflect the box and the
    optimiser; the paper's practical non-identifiability is consistent with this structure, not proven by it;
  - Table 3's fitting errors are reproduced at the printed θ̂ (see "Accuracy").
- **Overlap with other data sets.** `energy/Lithium_Ion_Battery` (19 physical inputs, constant-current
  discharge, time domain) and `energy/Lithium_Ion_DFN_Drive_Cycle` (24 physical inputs, PyBaMM DFN, US06
  drive cycle, values only) are forward time-domain studies of the DFN. This one is new in four ways:
  - frequency-domain impedance from the linearized model at nine equilibria;
  - the SPMe in 18 grouped, identifiable parameters;
  - a calibration misfit as the QoI, analogous to `uq_benchmarks/PDE_Inverse_Problem_MCMC`;
  - exact adjoint gradients.
- **Open questions for the authors:** see Unknowns.

## Unknowns

- Questions for the authors:
  - Was the true R₀ = 10 mΩ added as a contact resistance to Chen2020's own series resistance?
  - Which PyBaMM version gave Table 3's τ_e± values?
  - Can the measured LG M50LT spectra be shared?
  - Were the particle-swarm runs bounded away from C = 0, and was any parameter searched on a log scale
    (τ_ct± span a factor of 50)?
- The Table 3 values were not checked against the journal version; the PDF read is the arXiv preprint (Model).
- The paper states neither its PyBOP version nor any parameter transformation, so that its swarm searched
  uniformly in physical coordinates is unverified (Model, "Uniform, not log-uniform").
- The cause of the τ_e± differences from Table 3 is not established; it may come from a difference in
  Chen2020's electrolyte diffusivity between PyBaMM 24.9 and 26.10 (Model).
- Rows 0–54 have no CPU-time record; the CPU time for all 1000 samples is an estimate (Regenerating).
- The licence of PyBaMM-EIS was not checked; it is not used directly (License and credit).

## License and credit

- **Data:** CC BY 4.0 (`LICENSE-DATA`). **Notebook and code** (`Lithium_Ion_Impedance_Calibration.ipynb`,
  `spme_eis_driver.py`, `generate_data.py`, `check_accuracy.py`): MIT (`LICENSE`), as for the repository.
  No PyBaMM or PyBOP source is copied; both are used as installed packages.
- **The paper:** published under CC BY 4.0 (version of record, Crossref). Cite it when using these data:
  Hallemans et al., J. Electrochem. Soc. 172 (2025) 060507, doi:10.1149/1945-7111/add41b.
- **PyBaMM** (BSD-3-Clause; Sulzer et al., J. Open Res. Softw. 9 (2021) 14, doi:10.5334/jors.309). Its
  frequency-domain EIS method comes from PyBaMM-EIS (github.com/pybamm-team/pybamm-eis, now archived; its
  licence was not checked here, and it is not used directly).
- **PyBOP** (BSD-3-Clause; Planden et al., J. Open Source Softw. 10 (2025) 7874, doi:10.21105/joss.07874):
  the `GroupedSPMe` model and its Chen2020 grouping.
- **Chen2020 parameter set:** Chen et al., J. Electrochem. Soc. 167 (2020) 080534,
  doi:10.1149/1945-7111/ab9050.
- **Funding acknowledged by the paper:** Faraday Institution (FIRG066, FIRG059); EU IntelLiGent (101069765)
  through the UKRI Horizon Europe Guarantee (10038031).
