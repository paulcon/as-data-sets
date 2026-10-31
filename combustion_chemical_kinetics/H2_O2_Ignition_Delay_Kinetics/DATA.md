# H2_O2_Ignition_Delay_Kinetics: data

Kind: **stored data (generated with Cantera 3.2.0), with a regeneration script.** The model runs in
Cantera, an outside code, so it is not a numpy/scipy "Python model" in the repository's sense (CLAUDE.md,
"Two kinds of data sets"), even though the Cantera driver is a Python module (`h2o2_ignition_model.py`). A port
to numpy/scipy (a stiff 9-species ODE with Troe falloff, reverse rates from NASA polynomials, and dense output
for the ignition time) was judged not worth the risk of a second, unchecked kinetics code. Following decision 1
of the build brief, Cantera was installed in the build container, the data were generated there, and the full
provenance (tool, version, mechanism file, settings, seeds, date, run time) is recorded below. The shipped CSVs
are the record. `h2o2_ignition_model.py` and `generate_data.py` need Cantera (tested with 3.2.0) and reproduce
the files bit for bit (`--check`). The notebook needs only numpy, because it reads the shipped files and the
module's constants.

## Files

| file | contents | role |
|---|---|---|
| `H2_O2_Ignition_Delay_Kinetics.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2019; outputs committed |
| `h2o2_ignition_model.py` | `IgnitionModel` (Cantera driver, not a numpy/scipy model: `ignition_delay`, `tau19`, `tau36`, `grad`); `REACTIONS`, `A_REF`, `EA_REF`, `EA_SET`, `NAMES19/36`, `NOMINAL19/36`, `LOWER19/36`, `UPPER19/36`, `STEPS19/36` | the model; its constants import without Cantera |
| `h2o2_li2004.yaml` | Cantera mechanism: 19 reactions (23 Arrhenius entries) of Li et al. (2004) Table I, 10 species, GRI-Mech 3.0 thermo | model input |
| `LICENSE-cantera.txt` | Cantera's BSD-3-Clause licence | covers the thermo data copied from Cantera's `h2o2.yaml` |
| `generate_data.py` | writes all CSVs (`--case`, `--rows`, `--out`, `--check`, `--workers`) | regenerates the data |
| `inputs_19.csv` | 1000 × 19: ln A_i, A in cm-mol-s units, order R1..R19 | Monte Carlo inputs, seed 2019019 |
| `outputs_19.csv` | 1000 × 1: ignition delay τ in s | model output |
| `gradients_19.csv` | 1000 × 19: ∂τ/∂ln A_i in s | model output |
| `inputs_36.csv` | 1000 × 36: ln A_1..ln A_19, Ea_i in kcal/mol for i in `EA_SET`, P0 in atm, T0 in K, φ | Monte Carlo inputs, seed 2019036 |
| `outputs_36.csv` | 1000 × 1: τ in s | model output |
| `gradients_36.csv` | 1000 × 36: ∂τ/∂x_j in s per unit of x_j (physical units) | model output |
| `mc_inputs_36.csv` | 10 000 × 36, as `inputs_36.csv` (13 significant digits) | Monte Carlo inputs, seed 2019100 |
| `mc_outputs_36.csv` | 10 000 × 1: τ in s | model output (no gradients) |

Every CSV has two `#` lines (a description, then the column names with units) and reads with
`np.loadtxt(path, delimiter=",")`. Inputs are rounded to the written digits before the model runs, so each output
belongs exactly to the stored input row.

## Model

### Citation

M. Vohra, A. Alexanderian, H. Guy, and S. Mahadevan, "Active subspace-based dimension reduction for chemical
kinetics applications with epistemic uncertainty", *Combustion and Flame* 204 (2019) 152–161,
doi:10.1016/j.combustflame.2019.03.006 (Crossref-checked; preprint arXiv:1810.00955). The PDF read for this
build is the publisher's version.

### Physics and solver

- Homogeneous, adiabatic, constant-pressure ignition of an H2/O2 mixture (paper p. 153), with
  k_i = A_i T^n_i exp(−Ea_i/RT) (Eq. 1) and the 19 reactions of the paper's Table 1 (p. 153). The temperature
  exponents n_i are fixed at their nominal values (p. 153).
- Initial state: P0 = 1 atm, T0 = 900 K, φ0 = 2.0 (pp. 153, 158). The mixture is H2 : O2 = 2φ : 1 by moles
  (paper Eq. 4) with **no diluent**: the paper names none (an assumption; listed under open questions).
- **Mechanism (departure from the paper).** The paper used the rate parameters of Yetter, Dryer and Rabitz,
  Combust. Sci. Technol. 79 (1991) 97–128 (its ref. [18]) in TChem. The paper does not print them, and they
  could not be obtained here (that paper is not open, and three web fetches were allowed). We use the
  **same 19 reactions in the same order** (R4 written in the opposite direction; see the next bullet) with the rate parameters of J. Li, Z. Zhao, A. Kazakov, F. L. Dryer,
  Int. J. Chem. Kinet. 36(10) (2004) 566–575, doi:10.1002/kin.20026, Table I. These were transcribed as printed
  (3 significant figures, cm-mol-s-kcal-K) from the copy on the Princeton combustion group's site. Li et al.
  update Yetter et al. through Mueller et al. (1999), and their Table I numbering is the paper's Table 1
  numbering. Differences that matter: R9 (H + O2 + M) and R15 (H2O2 + M) are Troe falloff reactions in Li et al.
  (N2-bath low-pressure limit used for R9, footnote c); R14 and R19 are sums of two expressions; the zero-Ea
  reactions are R6, R7, R8, R12 (and the first expression of R19), not R6–R9 and R13 as the paper reports for
  Yetter et al. (p. 158). The He-specific expressions are omitted (no He). Thermodynamic data: NASA7 from
  Cantera 3.2.0's `h2o2.yaml` (the H2/O2 subset of GRI-Mech 3.0), so reverse rates follow GRI-Mech
  thermochemistry, not Li et al.'s own. File: `h2o2_li2004.yaml`.
- **R4 is written the other way round.** The paper's Table 1 (p. 153) lists R4 as OH + OH → O + H2O. The
  mechanism follows Li et al. and writes it O + H2O ⇌ 2 OH, so A*_4 = 2.97e6 and Ea*_4 = 13.4 kcal/mol, and the
  perturbed ln A_4 and Ea_4, belong to that direction. Cantera computes the reverse rate from the equilibrium
  constant, so a multiplier on A_4 scales both directions alike; only the nominal values, and so the widths of
  the ranges on ln A_4 and Ea_4, depend on the written direction. This does not matter for the results: R4 barely
  affects τ. In normalized inputs, E[(∂τ/∂x_i)²]/λ1 is 2e-13 for ln A_4 (19 inputs), and 9e-14 for ln A_4 and
  1e-13 for Ea_4 (36 inputs), as printed in the notebook; a range several times wider would change nothing.
- Solver: Cantera 3.2.0 `IdealGasConstPressureReactor` / `ReactorNet` (CVODES BDF), rtol = 1e-9,
  atol = 1e-18. One ignition takes about 15 ms (the step count is set by the ignition front).
- **Quantity of interest.** The ignition delay τ (s): the first time T reaches T0 + 400 K. The crossing is
  located by cubic Hermite interpolation of T between the bracketing solver steps, with the exact dT/dt from
  the energy equation. The paper says only "the time required for the rate of temperature increase to exceed
  a given threshold" (p. 153), with no threshold. Its companion paper (Vohra et al., J. Sci. Comput. 79 (2019)
  1335–1359) uses the inflection point of T(t). On the slow branch these definitions agree closely: τ for
  rises of 200, 800 and 1200 K differs from τ(400 K) by −4.6e-5, +2.3e-5 and +4.3e-5 relative at the nominal
  point, and by −1.8e-5, +7.6e-6 and +1.5e-5 at a slow sample (τ = 0.48 s). On the fast branch the definition
  matters at the per-cent level: at a sample with τ = 0.149 ms they give −1.2 %, +1.4 % and +5.0 %.

### Inputs

All inputs are independent and uniform; they are stored in physical units (the notebook normalizes once,
through `ac.Uniform(lower, upper)`, which maps every input by its own half-range to [−1, 1]).

| case | inputs | units | nominal | range | source |
|---|---|---|---|---|---|
| 19 | ln A_i, i = 1..19 | ln of A in cm-mol-s units | ln A*_i (Li et al. Table I, first expression) | [0.97, 1.03] · ln A*_i | paper Sec. 4, p. 156 |
| 36 | ln A_i, i = 1..19 | as above | as above | [0.98, 1.02] · ln A*_i | paper Sec. 5, p. 158 |
| 36 | Ea_i, i in R1–R5, R9, R10, R11, R13–R18 (14) | kcal/mol | Li et al. Table I | [0.98, 1.02] · Ea*_i (R13: [−0.51, −0.49]) | paper Sec. 5, p. 158 (rule: nonzero nominal Ea) |
| 36 | P0 | atm | 1.0 | [0.98, 1.02] | p. 158 |
| 36 | T0 | K | 900 | [882, 918] | p. 158 |
| 36 | φ0 | – | 2.0 | [1.96, 2.04] | p. 158 |

- **Units of A.** The interval [0.97, 1.03]·ln A* scales k_i by a factor of up to (A*)^0.03. That factor does
  not depend on the base of the logarithm, but it does depend on the units of A: here 1.38 (R2) to 4.76
  (R8). We use cm-mol-s, the units of the Yetter and Li mechanisms and of Chemkin and TChem input. In SI
  (m³/kmol) the factors would be smaller.
- **Reactions with two expressions.** A*_i and Ea*_i are those of the first expression in Li et al. Table I
  (the low-pressure limit for R9 and R15). A perturbation of reaction i multiplies **every** expression's A by
  A_i/A*_i (Cantera's rate multiplier, which scales a falloff rate exactly) and every expression's Ea by
  Ea_i/Ea*_i. For R9 this convention widens the range: A*_9 is the low-pressure-limit A (6.37e20), so ±3 % on
  ln A_9 multiplies k_9 by up to 4.21 (±1.437 in ln k_9); the high-pressure A (1.48e12) would give 2.32 (±0.841).
  The notebook's threshold rule for the branch switch (below) predicts a fast-branch fraction of 0.214 as built
  (observed 0.226) and 0.149 with the high-pressure reading, so the wider R9 range adds to the fast branch, but
  most of it comes from where the Li et al. nominal point sits.
- **The 14 activation energies.** The paper perturbs Ea of all reactions "except R6–R9, R13 (due to zero
  nominal values for Ea)". Applying the same rule (nonzero first-expression Ea) to Li et al. gives R1–R5, R9,
  R10, R11, R13–R18. The count is still 14, but R9 and R13 replace the paper's R12 and R19, whose Ea is 0 in
  Li et al.
- Checks in the notebook: normalized inputs fill [−1, 1], with kurtosis 1.72–1.88 (uniform 1.8); no duplicate
  rows; no value outside physical meaning (all A > 0, every Ea keeps its sign, P0, T0, φ0 > 0).

### Gradients

Central differences in the physical inputs, with step h_j = 1e-3 × (half-range of input j), the same step in
normalized coordinates for every input (`IgnitionModel.grad`, `STEPS19`, `STEPS36`). That is 38 extra runs per
sample (19 inputs) and 72 (36 inputs). `gradients_19.csv` holds ∂τ/∂ln A_i (s). `gradients_36.csv` holds ∂τ/∂x_j
in each input's physical unit (s per unit ln A, s per kcal/mol, s/atm, s/K, s). The gradient of ln τ is ∇τ/τ.

### Sampling design

Plain Monte Carlo (independent uniform draws with numpy's `default_rng`), as in the paper. Seeds: 2019019
(19 inputs, 1000 samples), 2019036 (36 inputs, 1000 samples), 2019100 (36 inputs, 10 000 samples, τ only, the
analogue of the paper's cross-validation set). The paper's own reference set had N = 1000 for the 19 inputs
(20 000 runs); its 36-input subspace came from its iterative Algorithm 1 (925 finite-difference runs, or 1850
regression runs), plus 10^4 cross-validation samples. The paper reports no seeds.

### Published results (paper)

- Fig. 1 (p. 156): the normalized 19-input spectrum for N = 1000 and n = 20, 40, 80, 120; "λ1 is roughly an
  order of magnitude larger than λ2". Eq. (12) error O(1e-2); 0.0657 (finite differences, 45 samples) and
  0.1050 (regression, 900 runs) (p. 157).
- Figs. 2–3 (p. 158; text p. 157): the 19-input dominant eigenvector and activity scores (A9 largest; A1, A15, A17
  significant), and the summary plot.
- Fig. 4 (p. 159): the 36-input dominant eigenvector; 4 iterations (925 runs) and 9 iterations (1850 runs) to
  converge. Fig. 5: summary plots with quadratic and cubic fits. Table 2: mean and sd of τ, 0.15 and 0.14
  (model), 0.15 and 0.13 (both surrogates). Fig. 6: PDFs of τ, "from about 2 ms to 400 ms". Fig. 7 (p. 160): activity
  scores and total Sobol' indices (A1, A9, T0 dominant; A15, A17, Ea15 significant; P0 and φ0 unimportant).

### What the notebook finds (from these files)

- 19 inputs, τ: λ1/λ2 = 6.7, λ2/λ3 = 56.2. Activity scores: A9 0.448, A1 0.237, A15 0.158, A17 0.129. Leave-one-out
  R² is 0.9006 (quadratic in y1) and 0.9766 (cubic in two active variables). ln τ: λ1/λ2 = 76.9.
- 36 inputs, τ: λ1/λ2 = 9.6, λ2/λ3 = 107.2. Activity scores: A9 0.368, T0 0.204, A1 0.189, A15 0.087, A17 0.081,
  Ea15 0.037. P0 and φ0 score below 5e-5. On the 10 000 Monte Carlo samples, the mean and sd of τ are 0.1067 s and
  0.1109 s; the quadratic surrogate gives 0.1043 s and 0.1054 s.
- τ is bimodal: 226 of the 1000 19-input samples (1239 of the 10 000 36-input samples) ignite in under 1 ms, on
  the fast branch beyond the second explosion limit. The switch is governed by ln(k1/k1*) − ln(k9/k9*): no
  sample with a difference below 0.5 is fast, every sample above 1.0 is, and 20.6 % of those in between are.
  Applied to the input ranges, this rule predicts a fast fraction of 0.214 (0.149 if A*_9 were R9's
  high-pressure A).

## Regenerating

```bash
# Cantera environment (Python 3.13 was used)
python3.13 -m venv cantvenv && cantvenv/bin/pip install cantera==3.2.0 numpy scipy
cantvenv/bin/python generate_data.py                                        # all files
cantvenv/bin/python generate_data.py --case 36 --rows 0:20 --out /tmp/h2 --check   # subset vs shipped files
```

Provenance of the shipped files: generated 2026-10-30 in the build container (Linux x86-64, 2 cores shared
with other jobs; load average about 5), Python 3.13.18, Cantera 3.2.0, numpy 2.5.4, scipy 1.18.1, with
`generate_data.py` as shipped and two worker processes. Wall time: 428 s (19 inputs), 504 s (36 inputs), 110 s
(10 000 τ values); 1041 s in all. The measured cost of one ignition, about 15 ms, implies about 0.6 s of CPU per
19-input sample (39 runs), 1.1 s per 36-input sample (73 runs) and 150 s for the 10 000 single runs: about 1850 s of
CPU in all. There is no randomness after
the seeded draws, and the runs are deterministic: `--rows 0:20 --check` reproduces all eight files bit for bit
(maximum difference 0 for every file and case).

## Accuracy

Scratch checks (in the build's `_work` folder, not shipped):

- **τ.** Re-solving at rtol = 1e-12, atol = 1e-22 changes τ by 3.7e-9 to 2.5e-8 relative (six shipped rows).
- **Gradients** at six random shipped rows (3 per case), against an independent central difference at
  rtol = 1e-12 with the same step: relative errors 2.1e-7, 2.9e-6 and 4.8e-7 (19 inputs) and 9.0e-5, 2.4e-6 and
  2.8e-5 (36 inputs). Against a tight central difference with a step ten times smaller: 1.8e-6 to 9.0e-5.
- **Cantera forward sensitivities** (19 inputs only; ∂τ/∂ln A_i = −T(τ) S_i(τ)/Ṫ(τ), S_i = ∂ln T/∂ln k_i from
  CVODES at fixed t, solver rtol 1e-9, sensitivity rtol 1e-7, atol 1e-9) agree with the shipped gradients to
  1.3e-5, 3.7e-6 and 1.0e-4 relative at the same three rows. The largest difference is at a sample near the
  branch switch (τ = 0.90 ms). These sensitivities are themselves erratic: depending on CVODES's sensitivity
  tolerances, the agreement on three test points ranged from 3e-6 to 2.7e-2. So the finite differences, which
  are stable under changes of step and tolerance, are the primary gradients.
- In the notebook, the least-squares slope of τ on the normalized inputs and the mean normalized gradient
  agree in direction (cosine 0.993 and 0.992), which confirms the gradient coordinates and the single
  normalization.

## How these files relate to the paper

- **Not the paper's computation.** The paper's data were not published, and its mechanism parameters were not
  available. These files come from an independent Cantera computation with the same reactions, input ranges
  and design type, but different rate constants and thermo, and a stated ignition criterion. Expect the same
  structure (a dominant direction led by A9, A1, A15, A17, and T0 in the 36-input case), not the paper's
  numbers. The comparison table in the notebook lists what agrees and what does not. The main differences are
  a mean τ about 30 % shorter (0.107 s against 0.15 s), smaller first eigenvalue gaps, and a fast-ignition mode
  (τ < 1 ms) that the paper's Fig. 6 does not show.

## Open questions (for the authors)

1. The paper's ignition threshold on dT/dt.
2. Is the mixture pure H2/O2 at φ0 = 2, or diluted (N2, Ar)?
3. The Yetter et al. (1991) parameter file as used in TChem (A units, falloff treatment of R9 and R15): does
   it place the ±3 % (±2 %) ranges entirely on the slow-ignition branch, as the paper's Fig. 6 (2–400 ms)
   suggests?
4. The units of Table 2 (seconds assumed).

## License and credit

Code (`*.py`): MIT. Data (the CSVs): CC BY 4.0. The mechanism file combines rate parameters published by
Li et al. (2004), for which no licence is stated (credit: J. Li, Z. Zhao, A. Kazakov, F. L. Dryer), with
thermodynamic data copied from Cantera's `h2o2.yaml` (GRI-Mech 3.0, G. P. Smith et al.), distributed under
Cantera's BSD-3-Clause licence (`LICENSE-cantera.txt`, kept beside the file). Study design: M. Vohra,
A. Alexanderian, H. Guy, and S. Mahadevan; cite their paper (above). Cantera: D. G. Goodwin, H. K. Moffat,
I. Schoegl, R. L. Speth, B. W. Weber, https://www.cantera.org.

