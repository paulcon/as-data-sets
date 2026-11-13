# EMT_Switch_ODE: data

Kind: **Python model**. The model is `emt_model.py` (numpy and scipy), written from the paper's equations. `generate_data.py` draws the inputs with saved seeds and writes all six CSVs; a rerun reproduces them to the tolerances under "Regenerating".

## Files

| file | contents | role |
|---|---|---|
| `EMT_Switch_ODE.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 575; outputs committed |
| `emt_model.py` | `qoi(x, C, T, grad=False, return_info=False)`, `qois`, `trajectory`, `equilibria`, `count_equilibria`, `unique_equilibrium`, `forward_sensitivities`, `rhs`, `jac_y`, `jac_p`; `NAMES`, `UNITS`, `DESCRIPTIONS`, `NOMINAL`, `HILL`, `LOWER`, `UPPER`, `INITIAL`, `T_ON`, `CONTACTS`, `T_END`, `RTOL`, `ATOL`, `QOI_NAMES` | the model |
| `generate_data.py` | draws the samples and writes the six CSVs; `--out DIR --check` compares a rerun with the shipped files; checks gradients by finite differences | regenerates the data |
| `check_accuracy.py` | recomputes the numbers below that the notebook and `generate_data.py` do not print | check |
| `inputs_pm10.csv`, `inputs_pm25.csv` | 2000 × 11 physical inputs (α1, α2, β1, β2, k0, k1, k2, IC_S, IC_E, IC_T, IC_C) on the ±10 % and ±25 % boxes | model input |
| `outputs_pm10.csv`, `outputs_pm25.csv` | 2000 × 3: E (ng/mL), S (ng/mL) at t = 10,000 min, and `n_equilibria` (number of equilibria of the treatment group at the draw; 1 at every row) | model output |
| `gradients_pm10.csv`, `gradients_pm25.csv` | 2000 × 22: ∂E/∂x in columns 0–10, ∂S/∂x in columns 11–21, with respect to the **physical** inputs | model output |

The CSVs have two `#` lines (a description and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits.

## Model

**Source.** K. I. Gasior, "Determining parameter impact in systems biology models via sensitivity analysis: a comparative approach", *npj Systems Biology and Applications* 12:143, 2026, doi:[10.1038/s41540-026-00757-5](https://doi.org/10.1038/s41540-026-00757-5) (publisher PDF read). The model is that of K. Gasior, M. Hauck, S. Bhattacharya, *J. Theor. Biol.* 546:111160, 2022 (the paper's ref. 15; not read). Supplementary Information (publisher PDF `41540_2026_757_MOESM1_ESM.pdf`, https://static-content.springer.com/esm/art%3A10.1038%2Fs41540-026-00757-5/MediaObjects/41540_2026_757_MOESM1_ESM.pdf, linked from the article page; read, not shipped): Table S1 (parameter ranges) and Table S2 (initial conditions), both on its p. 1. The two tables are short and are transcribed into `emt_model.py` (`LOWER`, `UPPER`, `INITIAL`) with their source; no other content of the supplement is used.

**Equations** (Eqs. (1)–(2), p. 10): dE/dt = α1/(1 + (S/IC_S)^n1) + k0 (C/IC_C)^n2/(1 + (C/IC_C)^n2) − β1 E; dS/dt = α2 − k1 (E/IC_E)^n3/(1 + (E/IC_E)^n3) + k2 (T/IC_T)^n4/(1 + (T/IC_T)^n4) − β2 S, with E membrane-bound E-cadherin and S Slug (ng/mL), C cell–cell contact (cells) and T exogenous TGF-β (ng/mL). The PDF's text layer drops the minus signs; the page image has them. Hill coefficients n1..n4 = 3, 4, 2, 3 (Table 1, p. 9), held fixed.

**Inputs** (m = 11), independent uniform (p. 11: "This study assumes a uniform distribution for each parameter"):

| name | symbol | units | Table 1 value (p. 9) | ±10 % box (Table S1) | ±25 % box (Table S1) |
|---|---|---|---|---|---|
| `alpha1` | α1, E-cadherin production | ng/(mL min) | 0.19355 | [0.1742, 0.2129] | [0.1452, 0.2419] |
| `alpha2` | α2, Slug production | ng/(mL min) | 0.080 | [0.0772, 0.0932] (shifted) | [0.0772, 0.1172] (shifted) |
| `beta1` | β1, E-cadherin degradation | 1/min | 6.035 | [5.4315, 6.6385] | [4.5263, 7.5438] |
| `beta2` | β2, Slug degradation | 1/min | 0.830 | [0.7470, 0.9130] | [0.6225, 1.0375] |
| `k0` | E-cadherin to the membrane by contact | ng/(mL min) | 0.1245 | [0.1120, 0.1370] | [0.0938, 0.1556] |
| `k1` | Slug suppression by membrane E-cadherin | ng/(mL min) | 0.080 | [0.0669, 0.0829] (shifted) | [0.0429, 0.0829] (shifted) |
| `k2` | Slug upregulation by TGF-β | ng/(mL min) | 0.040 | [0.0360, 0.0440] | [0.0300, 0.0500] |
| `ICS` | half-maximal Slug | ng/mL | 0.01920 | [0.0173, 0.0211] | [0.0144, 0.024] |
| `ICE` | half-maximal E-cadherin | ng/mL | 0.010 | [0.0090, 0.0110] | [0.0075, 0.0125] |
| `ICT` | half-maximal TGF-β | ng/mL | 3.64 × 10⁻⁶ | [3.276, 4.004] × 10⁻⁶ | [2.73, 4.55] × 10⁻⁶ |
| `ICC` | half-maximal number of cells | cells | 2.17 | [1.9530, 2.3870] | [1.6275, 2.7125] |

Table S1 says the α2 and k1 ranges "have been shifted but the sizes have not changed" because of monotonicity and negativity constraints (also p. 10). Two entries of Table S1 look like typos; the values are used as printed. (1) The β2 ±25 % bracket has a stray "]". (2) The k0 ±25 % lower bound is printed as 0.0938, but 0.75 × 0.1245 = 0.0934; every other unshifted entry equals the Table 1 value × (1 ± 0.10) or (1 ± 0.25) to the printed rounding, so the k0 range used here is 0.0618 wide instead of 0.06225.

**Treatment group (this data set's choice).** The paper studies eight groups, C ∈ {0, 1, 2, 6} cells and T ∈ {0, 3.64 × 10⁻⁶} ng/mL (pp. 10–11). This data set uses one: **C = 2 cells, T = 3.64 × 10⁻⁶ ng/mL**. The PDF's text layer prints the TGF-β level as "3.64 × 106"; the page images and the supplement's figure labels show 3.64 × 10⁻⁶, equal to IC_T.

**Initial conditions.** Supplementary Table S2 gives (E, S)(0) = (0.0003, 0.0963), (0.0280, 0.0109), (0.0399, 0.0057), (0.0522, 0.0034) ng/mL for C = 0, 1, 2, 6, the same for both TGF-β levels: epithelial for C ≥ 1, mesenchymal for C = 0 (also the Fig. 7 caption, p. 10). They are the equilibria at the Table 1 values with T = 0 (notebook, model cell). The data use the C = 2 row.

**Bistability and how it is handled.** For C = 1, 2 without TGF-β the model is bistable (Fig. 7B, p. 10), so E(10⁴) jumps between branches across the input box. With TGF-β on, the C = 2 group is monostable at the Table 1 values (one equilibrium). `emt_model.equilibria` reduces the system to one scalar equation in S (from dE/dt = 0), scans every S an equilibrium can have and refines each sign change; `generate_data.py` records the count at every draw. **All 4000 stored draws have exactly one equilibrium**, so no stored draw can switch branch and E, S are smooth along the sample; no rows are dropped. The boxes are not monostable everywhere: of their 2048 corners, 3 (±10 %) and 25 (±25 %) are bistable (notebook, cell after the data load; `emt_model.count_equilibria`), and at the first bistable ±25 % corner the trajectory from Table S2 stays on the epithelial branch, E = 0.0310 ng/mL, above the largest stored value (0.02592), so E is discontinuous at the edge of this small region. Among the 106,496 uniform evaluations per box of the notebook's all-input Sobol' study, 0 (±10 %) and 1 (±25 %, a fraction of 9.4 × 10⁻⁶) are bistable, and none of the subset-design evaluations; the notebook evaluates such rows by the trajectory. `check_accuracy.py` confirms the count on a 5× finer scan for rows 0–499 of each box and shows that trajectories started from the mesenchymal initial condition end at the same state (rows 0–99, max relative difference at rounding level, below 10⁻¹⁵ on both boxes; check 3). For context, check 5 tabulates the other groups (500 draws per box, seed 577):

| group (C, T) | ±10 %: bistable (epithelial / mesenchymal end), S < 0, E(10⁴) < E(0)/2 | ±25 %: same |
|---|---|---|
| C = 0, T = 0 | 27 (0 / 27), 0, 89 | 7 (0 / 7), 0, 273 |
| C = 1, T = 0 | 34 (34 / 0), 0, 466 | 7 (7 / 0), 0, 493 |
| C = 2, T = 0 | 71 (71 / 0), 0, 382 | 6 (6 / 0), 0, 468 |
| C = 6, T = 0 | 0 (0 / 0), 9, 90 | 0 (0 / 0), 1, 344 |
| C = 0, T = 3.64 × 10⁻⁶ | 0 (0 / 0), 0, 422 | 0 (0 / 0), 0, 407 |
| C = 1, T = 3.64 × 10⁻⁶ | 0 (0 / 0), 0, 500 | 0 (0 / 0), 0, 500 |
| C = 2, T = 3.64 × 10⁻⁶ (this data set) | 0 (0 / 0), 0, 500 | 0 (0 / 0), 0, 495 |
| C = 6, T = 3.64 × 10⁻⁶ | 0 (0 / 0), 0, 477 | 0 (0 / 0), 0, 414 |

Counts out of 500. "Bistable" means three equilibria at the draw; the epithelial / mesenchymal split says on which stable branch the trajectory from Table S2 ended. The last column counts draws whose E(10⁴) is below half of E(0), a rule of this data set: it says only that E fell by more than half from its initial value (for C = 0 from 0.0003 ng/mL, and with TGF-β on it includes the expected TGF-β response), not that the phenotype switched. For C = 1 and C = 2 without TGF-β, where the start is the epithelial equilibrium at the Table 1 values, 466 and 382 of 500 draws on the ±10 % box end with E below half its start although only 34 and 71 draws are bistable (and those all stay epithelial): in most draws only one equilibrium, a low-E one, is left, so E(10⁴) jumps between the epithelial level and a low level across the box and is discontinuous. With TGF-β on every group is monostable at every draw. S < 0 occurs only for C = 6 without TGF-β (9 and 1 draws), although the paper shifted the α2 and k1 ranges to avoid negative values (p. 10; Table S1).

**Quantities of interest.** E and S at t = 10,000 min (p. 10). The trajectory is integrated from the Table S2 initial condition with `scipy.integrate.solve_ivp`, LSODA, rtol 1e-10, atol 1e-14, analytic Jacobian, and the end state is polished to the equilibrium by Newton's method. The iteration stops when the correction is below 10⁻¹⁵ (relative); at some inputs the corrections cycle at the rounding level without meeting that, and the state after 20 iterations is then accepted if its last correction is below 10⁻¹². `qoi` raises if the iteration does not get that far or if the end state is not within 10⁻⁶ (relative) of an equilibrium. The largest polish correction over the 4000 draws is at rounding level, about 10⁻¹⁵ relative on both boxes, and the slowest Jacobian eigenvalue is −0.494 and −0.356 per minute (`generate_data.py`), so at t = 10⁴ min the transient is far below rounding. At the Table 1 values: E = 0.009144, S = 0.076590 ng/mL. The paper rounds E and S to 10⁻⁴ (p. 10); the stored values are not rounded. No draw has S < 0.

**Gradients.** Exact up to rounding: the implicit function theorem at the equilibrium, ∂y/∂p = −J_y⁻¹ J_p, with analytic J_y and J_p, with respect to the 11 physical inputs. `forward_sensitivities` integrates the sensitivity equations instead (check 2).

**Sampling design.** For each box, M = 2000 independent uniform (Monte Carlo) draws, `numpy.random.default_rng(seed).random((2000, 11))` scaled to the box, seed 575 (±10 %) and 576 (±25 %). The paper used Latin hypercube samples (N = 10,000) for PRCC and Saltelli samples for Sobol' (pp. 10–11); the draws here are new.

**Published results** (no active subspace in the paper). Sobol' S1 and ST of reduced subsets (MMS or PRCC; the paper does not say what values the other inputs take, see "Agreement"), as bar charts in Figs. 2–5 (pp. 3–6) with no printed values; for this group panels 2F, 3F (E) and 4F, 5F (S). Morris results in Supplementary Figs. S1–S4, PRCC in Tables S3–S6 (not used). p. 4: for C = 1, 2 with TGF-β, "Sobol analysis for all ranges shows ICC controlling E-cadherin".

**What the notebook finds.** On the ±10 % box n = 1 for both outputs: λ1/λ2 = 227 (E) and 174.5 (S), next ratios 7.0 and 4.2; w1 of E is dominated by IC_C (−0.838), β1 (−0.389), k0 (0.367); w1 of S by α2 (0.513), IC_C (0.51), β2 (−0.46). A quadratic in y1 has leave-one-out R² = 0.99753 (E); a line 0.99417 (S). On the ±25 % box λ1/λ2 = 13.7 (E, with λ2/λ3 = 12.5 comparable) and 33.3 (S); LOO R² 0.97759 and 0.97224 for n = 1, 0.98269 and 0.99137 for n = 2. Normalized activity scores agree with all-input Sobol' total indices within 0.008 (E) and 0.004 (S) on ±10 %, 0.050 and 0.024 on ±25 %.

## Regenerating

```bash
python generate_data.py                                 # writes the six CSVs here
python generate_data.py --check                         # writes to a temporary directory and compares
python generate_data.py --out DIR --check               # writes to DIR (not this directory) and compares
python check_accuracy.py                                # the numbers marked below
```

One process. `--check` refuses an `--out` that is this directory or a subdirectory of it. The shipped files were written on 2026-11-09 in a Linux container (Python 3.11.18, numpy 2.4.6, scipy 1.17.1). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-11-12, on a loaded machine), `python generate_data.py --check` took about 186 s (about 80 s for the 2000 draws of each box, the rest the finite-difference check); the inputs were identical to the shipped ones, the outputs differed by a relative amount below 10⁻¹⁵ and the gradients by 10⁻¹⁵ to 10⁻¹⁴ (elementwise). `--check` prints these differences and does not judge them; it exits with status 1 only for a structural mismatch: a different shape or NaN pattern, a different number of equilibria at a draw, or regenerated inputs that differ from the shipped ones by more than a relative 1e-8.

## Accuracy

`check_accuracy.py` recomputes the numbers marked with its name; in `.venv` it took about 275 s (2026-11-10).

- **Outputs** (check 1). Rows 0–19 of each box against integrations at rtol 1e-12, atol 1e-16: largest relative difference of order 10⁻¹⁶ on both boxes; the unpolished LSODA state at t = 10⁴ min against the files: also of order 10⁻¹⁶, i.e. rounding level. The notebook re-solves 5 rows per box and matches the files to below 10⁻¹⁵.
- **Gradients.** Implicit-function gradients against the forward sensitivity equations at t = 10⁴ min (rtol 1e-11), rows 0–19: about 10⁻¹⁵ relative on both boxes (row 2-norm; check 2). Against central differences in the normalized inputs (step 10⁻⁶ of the half-width), first 50 rows: 1.6 × 10⁻⁹ (E), 1.9 × 10⁻⁹ (S) on ±10 % and 4.7 × 10⁻¹⁰, 8.6 × 10⁻¹⁰ on ±25 % (`generate_data.py`), the size of the differences' truncation and rounding error.
- **Vectorized solver.** `unique_equilibrium`, used by the notebook for the Sobol' indices, matches the files at all 2000 rows of each box to about 10⁻¹⁵ relative (check 4). It is valid only at monostable rows (at a bistable row it returns one of the three equilibria), so the notebook counts the equilibria at every Sobol' row with `count_equilibria` and evaluates bistable rows by the trajectory.
- **The largest E on the ±25 % box** (check 6) is row 1039, E = 0.02592 ng/mL (next largest 0.02389), the isolated point above the trend in the notebook's ±25 % summary plot. Its inputs move k2 by −24 %, IC_E by −20 %, IC_S by +23 %, β2 by +18 % and k0 by +15 % from Table 1; it is monostable (Jacobian eigenvalues −7.064 and −0.442 per min). Along the segment from the Table 1 point to it, E is 0.01001, 0.01124, 0.01320, 0.01750, 0.02180, 0.02592 at fractions 0.5, 0.7, 0.8, 0.85, 0.9, 1.0 of the way, and 0.02841, 0.03143 at 1.1 and 1.3 (beyond the box), with one equilibrium throughout: a steep but continuous rise of E toward its epithelial level, not a branch switch.

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from the author's MATLAB code, available on request (p. 12; not requested). The data here are an independent Python computation of Eqs. (1)–(2) with Table 1, Table S1 and Table S2, for one treatment group chosen here.
- **Agreement.** With the subset varied and the other inputs held at their Table 1 values (this data set's reading of the paper's design: the paper does not state the values of the inputs outside the subset, and the agreement below supports the reading), the notebook's Sobol' indices for C = 2 with TGF-β reproduce the bars of Figs. 2F–5F, read by eye, within 0.01 for every total index and every first-order index except α2 and IC_C in the three-input subset of Fig. 5F (0.315 against about 0.30, and 0.339 against about 0.32). This checks the transcription of the equations, ranges and initial condition, as far as bar-reading allows.
- **Open questions** (only the author can settle; not contacted). (1) The exact sample sets and Sobol' sample sizes behind Figs. 2–5 (Tables S7–S13 give the counts; not used). (2) Whether the k0 ±25 % lower bound of Table S1 (0.0938) is a typo for 0.0934, and whether the stray bracket of the β2 ±25 % entry hides another typo. (3) The MATLAB solver and tolerances; the paper rounds outputs to 10⁻⁴, which matters for E ≈ 0.01 ng/mL. (4) What values the inputs outside each Sobol' subset take in Figs. 2–5 (Table 1 values are implied but not stated).

## License and credit

Code (`emt_model.py`, `generate_data.py`, `check_accuracy.py`) and the notebook: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the published equations and parameter tables; no data or code were copied. The article and its supplement are CC BY-NC-ND 4.0; neither is shipped, and only the parameter values of Table 1 and Tables S1–S2 (facts needed to run the model) are used, with citation. Cite K. I. Gasior, npj Syst. Biol. Appl. 12:143, 2026, doi:10.1038/s41540-026-00757-5, and K. Gasior, M. Hauck, S. Bhattacharya, J. Theor. Biol. 546:111160, 2022. The paper states no funding.

## Unknowns

- Initial conditions are taken from Supplementary Table S2 (not a choice here); the treatment group (C = 2, T on), the Monte Carlo design and M = 2000 are this data set's choices.
- Paper figure values are read by eye from bar charts (about ±0.01).
- Whether the paper's MATLAB integration to t = 10⁴ min ends on the same equilibrium in every draw cannot be checked; here every stored draw of this group is monostable, so it must for these draws; a draw in the small bistable region near some corners of the boxes could end on the epithelial branch.
- What values the paper gives the inputs outside each Sobol' subset (Figs. 2–5); Table 1 values are read here, not stated in the paper.
