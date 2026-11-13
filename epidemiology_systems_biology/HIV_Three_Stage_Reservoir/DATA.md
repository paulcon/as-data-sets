# HIV_Three_Stage_Reservoir: data

Kind: **Python model**. The model is `hiv3_model.py` (numpy and scipy), written from the paper's equations. `generate_data.py` draws the inputs with a saved seed and writes all three CSVs; a rerun reproduces them to the tolerances under "Regenerating".

## Files

| file | contents | role |
|---|---|---|
| `HIV_Three_Stage_Reservoir.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2024; outputs committed |
| `hiv3_model.py` | `qoi(p, grad=False)`, `qois(P, grad=False)`, `trajectory(p, times)`, `rhs`, `jac`, `reproduction_numbers(p)`; `NAMES`, `LATEX`, `UNITS`, `DESCRIPTIONS`, `NOMINAL`, `LOWER`, `UPPER`, `Y0`, `DAYS_PER_YEAR`, `T_AIDS`, `HORIZON_YEARS`, `QOI_NAMES`, `QOI_UNITS`, `RTOL`, `ATOL` | the model |
| `generate_data.py` | draws the samples and writes the three CSVs to `--out DIR`; resumable in blocks of 50 rows; `--check` compares with the shipped files; checks gradients by finite differences | regenerates the data |
| `check_accuracy.py` | recomputes the numbers below that the notebook and `generate_data.py` do not print (checks 1–7) | checks |
| `inputs.csv` | 1000 × 15 physical parameters, columns in the order of `NAMES` (Table 1 of the paper) | model input |
| `outputs.csv` | 1000 × 3: T(5 years), T(10 years) in cells/mm³, F (time to AIDS) in years | model output |
| `gradients.csv` | 1000 × 45: column 15 q + j is ∂(output q)/∂p_j, q = T5, T10, F, with respect to the **physical** parameters | model output |

The CSVs have two `#` lines (a description and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits. The notebook passes the physical inputs and gradients to `ac.Uniform(LOWER, UPPER)`, which scales each input by its range to [−1, 1].

## Model

**Source.** C. Clarke and S. Pankavich, "Three-stage modeling of HIV infection and implications for antiretroviral therapy", *J. Math. Biol.* 88(3):34, 2024, doi:[10.1007/s00285-024-02056-1](https://doi.org/10.1007/s00285-024-02056-1) (publisher PDF read). The equations are those of E. A. Hernandez-Vargas and R. H. Middleton, *J. Theor. Biol.* 320:33–40, 2013, doi:[10.1016/j.jtbi.2012.11.028](https://doi.org/10.1016/j.jtbi.2012.11.028) (not read; cited by the paper as the source of the model, the parameter values and the initial state).

**Equations** (Eq. (1), p. 5): five ODEs for uninfected and infected CD4+ T cells T, T*, uninfected and infected macrophages M, M* (cells/mm³) and free virus V:
T' = s_T + ρ_T V T/(c_T + V) − k_T T V − δ_T T; T*' = k_T T V − δ_T* T*; M' = s_M + ρ_M V M/(c_M + V) − k_M M V − δ_M M; M*' = k_M M V − δ_M* M*; V' = p_T T* + p_M M* − δ_V V. Time in days.

**Inputs** (m = 15), independent uniform on the printed ±10 % ranges (Table 1, p. 6; uniform distribution stated in Appendix B, p. 37):

| # | name | symbol | units | nominal | range | source |
|---|---|---|---|---|---|---|
| 1 | `sT` | s_T | cells/mm³/day | 10 | 9–11 | Table 1, p. 6 |
| 2 | `sM` | s_M | cells/mm³/day | 0.15 | 0.135–0.165 | Table 1, p. 6 |
| 3 | `kT` | k_T | mm³/day/copy | 4.57e-5 | 4.11e-5–5.02e-5 | Table 1, p. 6 |
| 4 | `kM` | k_M | mm³/day/copy | 4.33e-8 | 3.90e-8–4.76e-8 | Table 1, p. 6 |
| 5 | `pT` | p_T | copies/cell/day | 38 | 34.2–41.8 | Table 1, p. 6 |
| 6 | `pM` | p_M | copies/cell/day | 35 | 31.5–38.5 | Table 1, p. 6 |
| 7 | `deltaT` | δ_T | 1/day | 0.01 | 0.009–0.011 | Table 1, p. 6 |
| 8 | `deltaTs` | δ_T* | 1/day | 0.4 | 0.36–0.44 | Table 1, p. 6 |
| 9 | `deltaM` | δ_M | 1/day | 1e-3 | 9e-4–1.1e-3 | Table 1, p. 6 |
| 10 | `deltaMs` | δ_M* | 1/day | 1e-3 | 9e-4–1.1e-3 | Table 1, p. 6 |
| 11 | `deltaV` | δ_V | 1/day | 2.4 | 2.16–2.64 | Table 1, p. 6 |
| 12 | `rhoT` | ρ_T | 1/day | 0.01 | 0.009–0.011 | Table 1, p. 6 |
| 13 | `rhoM` | ρ_M | 1/day | 0.003 | 0.0027–0.0033 | Table 1, p. 6 |
| 14 | `cT` | c_T | copies/mm³ | 300 | 270–330 | Table 1, p. 6 |
| 15 | `cM` | c_M | copies/mm³ | 220 | 198–242 | Table 1, p. 6 |

The printed ranges are rounded (k_T: 0.9 × 4.57e-5 = 4.113e-5 is printed 4.11e-5); they are used as printed.

**Initial state** (p. 6): T(0) = 1000, M(0) = 150 cells/mm³, T*(0) = M*(0) = 0, V(0) = 10 copies/ml. **Chosen (this data set's reading): V(0) = 0.01 copies/mm³**, the printed value converted to the copies/mm³ of Table 1. `check_accuracy.py`, check 5, compares the nominal run with V(0) = 0.01 and with V(0) = 10 (the number taken in Table 1's units): the acute T minimum at 0.198 against 0.124 years, the time to AIDS 7.105 against 7.031 years, and V first exceeds 5000 after year 1 at 8.615 against 8.541 years. Located by pixel position on 300-dpi renderings of the paper's figures (this data set's measurement, about ±0.02 year), the AIDS crossing in Fig. 1 (p. 3) is at about 7.1 years, the acute minimum at about 0.18 years, and the total viral load in Fig. 4 (p. 8) leaves the top of the axis (5000) at about 8.6 years; all three favour V(0) = 0.01. The reading barely changes the gradient directions: the nominal normalized gradients for the two readings have cosines 0.9998 (T5), 0.9998 (T10) and 0.9999 (F) (check 5).

**Quantities of interest** (p. 12): T5 = T(5 years) and T10 = T(10 years), taken at 1825 and 3650 days (365 days per year, this data set's choice; with 365.25 the nominal values change from 382.0336, 1.7536, 7.1052 to 381.8166, 1.7284, 7.1004; check 6), and the time to AIDS F(p) = max{t ≥ 0 : T(t; p) > 200}, in years. F is located as the last downward crossing of T = 200 (a `solve_ivp` event) in a 40-year window; `qoi` raises `RuntimeError` if T never reaches 200 or is above 200 at the end of the window. None of the 1000 samples raises. 28 samples dip below 200 in the acute phase (lowest T 166.2) and recover before the final decline; for them F is not the first crossing (check 7).

**Solver.** `scipy.integrate.solve_ivp`, LSODA, rtol = atol = 1e-10, analytic Jacobian of the states (block-diagonal Jacobian for the sensitivity system, used only by the Newton iteration). The paper used MATLAB `ode23s` (p. 6).

**Gradients.** Exact up to the ODE tolerance: the relative sensitivities ∂y/∂ln p_j (75 ODEs) are integrated with the states and divided by p_j. At the crossing, ∂F/∂p_j = −(∂T/∂p_j)/Ṫ, converted from days to years. `gradients.csv` holds gradients with respect to the **physical** parameters.

**Sampling design.** M = 1000 independent Monte Carlo draws, p = LOWER + (UPPER − LOWER) ⊙ U, `numpy.random.default_rng(2024).random((1000, 15))`. Every sample is inside the box and within the model's meaning (all parameters positive). Ranges: T5 186.70–658.47 and T10 0.578–359.31 cells/mm³, F 4.863–11.505 years (`generate_data.py`).

**Published results.** Fig. 6 (p. 13): the first eigenvector ("parameter weights") and sufficient summary plots for T(5; p), T(10; p) and F(p) (the F inset is capped at 10 years), from N ≈ 1000 Monte Carlo samples with forward-difference gradients in the normalized inputs (Appendix B, p. 39; step, seed and tolerances not given). Text pp. 12–14: p_T, δ_T* and δ_V dominate T(5); ρ_M also drives T(10) and F; the weight vectors of T(10) and F are "nearly identical". No eigenvalues are reported. The notebook's comparison uses the weights read from Fig. 6 by locating each marker on a 400-dpi rendering of p. 13 (reading error about 0.005; each read vector has unit norm to within 0.001).

**What the notebook finds.** n = 1 for all three outputs: λ1/λ2 = 103 (T5), 43.5 (T10) and 127.7 (F); for T10 also λ2/λ3 = 31.8, so a second direction is real there. w1 has cosines 0.9990, 0.9996 and 0.9921 with Fig. 6; the largest gap is the ρ_M weight of F (−0.434 against −0.519); capping F at 10 years, as the paper's Fig. 6c inset does, gives −0.439 (cosine with Fig. 6c 0.9926 against 0.9921), so the cap does not explain it. A cubic in y1 has leave-one-out R² = 0.9898, 0.9594 and 0.9929.

## Regenerating

```bash
python generate_data.py --out DIR                          # all 1000 samples, written to DIR
python generate_data.py --check                            # rows 0:50 (the first block), compared with the same rows here
python generate_data.py --check --full                     # all 1000 samples, compared with the files here
python generate_data.py --out DIR --rows 0:20 --check      # another subset
python generate_data.py --out . --here                     # rewrite the shipped files (deliberate)
python check_accuracy.py                                   # the numbers marked below
```

One process. The script never writes into its own directory, or a subdirectory of it, without `--here`, and `--check` and `--rows` refuse any `--out` inside it. Finished blocks of 50 rows are kept in `DIR/_blocks/` and reused, so an interrupted run resumes. The shipped files were written on 2026-11-09 in a Linux container (Python 3.11.18, numpy 2.4.6, scipy 1.17.1). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-11-10), `python generate_data.py --out DIR --rows 0:20 --check` took about 140 s: 48 s for the 20 samples with sensitivities (about 2.4 s each, so about 40 min for all 1000 at that rate) and 94 s for the finite-difference check on 10 samples. The inputs were identical to the shipped rows, and the outputs and gradients differed from them by a relative 1.5e-9 and 5.7e-9 (row 2-norm), the size of the solver tolerance (1e-10; checks 1–2 below show differences of order 1e-9 to 1e-8 between solves at different tolerances). On 2026-11-12, `python generate_data.py --check` (rows 0:50) took 239 s and `--check --full` 2339 s (39 min); the full check gave largest relative differences of 3.5e-9 (outputs, elementwise) and 9.5e-7 (gradients, row 2-norm), on macOS (Python 3.11, numpy 2.4.6, scipy 1.17.1) against files written on Linux. `--check` prints these differences and does not judge them; it exits with status 1 only for a structural mismatch: a different shape or NaN pattern, regenerated inputs that differ from the shipped rows by more than a relative 1e-8, or a run stopped by `--max-blocks` before it finished the rows it was asked to check.

## Accuracy

`check_accuracy.py` recomputes the numbers marked with its name; in `.venv` it took about 260 s (2026-11-10).

- **Outputs** (`check_accuracy.py`, check 1). At rows 0–19 the shipped outputs agree with state-only solves at rtol = atol = 1e-12 to a relative 6.5e-9 (LSODA) and 7.0e-9 (Radau). The notebook re-solves 3 samples without sensitivities and agrees with the files to 9.2e-9.
- **Gradients.** Against sensitivities solved at rtol = atol = 1e-12, rows 0–4: relative 1.0e-9 (T5), 9.4e-9 (T10), 1.3e-9 (F) (row 2-norm; check 2). Against central differences in ln p with tight solves: step 1e-5, first 10 samples, at most 1.2e-6, 1.1e-6, 1.2e-6 (`generate_data.py`); step 1e-4, rows 0–2, 1.9e-8, 4.8e-6, 3.2e-8, and step 1e-6, 2.5e-6, 2.6e-6, 2.3e-6 (check 3), so the remaining differences are the differences' own truncation and round-off error.
- **Window.** For rows 0–9 and the 10 rows with the largest F (up to 11.505 years), T stays below 200 from one day after F to 100 years (largest value 199.76, one day after the crossing; check 4), so the 40-year window finds the last crossing.
- **Paper checks** (notebook): R_T = 1.8090 and R_M = 0.0947 at the nominal values (paper p. 10: 1.809 and 0.09); with p_M → 0 the solution tends to T = 552.80, V = 283 and with p_T → 0 to T = 1000, as in Fig. 5 (p. 11).

## How these files relate to the paper

- **Not the paper's computation.** The authors' MATLAB code and samples were not published (no code or data statement in the paper). The data here are an independent Python computation of Eq. (1) with the Table 1 ranges, plus the choices above (V(0) in copies/mm³, 365-day years, a 40-year window, exact gradients instead of forward differences, new random samples).
- **Agreement.** The first eigenvectors agree with Fig. 6 to cosines 0.9990 (T5), 0.9996 (T10) and 0.9921 (F). For T10 all 15 read weights agree within the combined sampling and reading error; for T5 five differ by 2.9–4.6 times that error (at most 0.023 in absolute terms); for F seven differ by 3.2–13.9 times it, the largest being ρ_M (−0.519 in the figure, −0.434 here), k_T (−0.125, −0.178) and p_T (−0.329, −0.373). The notebook prints these ratios.
- **The paper's F (Fig. 6c inset) is not the F computed here** (notebook, comparison section; read from a 600-dpi rendering of p. 13, about ±0.15 year). (1) The paper's F is capped at 10 years: markers sit at F = 10 for wᵀx above about 0.9, matching the 10-year simulation of Fig. 1 (p. 3). This data set's F is uncapped; 15 samples have F ≥ 10 (up to 11.505 years). (2) Along the paper's Fig. 6c direction the paper's band centre lies 0.80–1.02 years above this data set's median F (about 6.0, 7.2, 7.9, 9.1 years against 4.981, 6.334, 7.102, 8.078 at wᵀx = −1.5, −0.5, 0, 0.5). At wᵀx = 0 the paper's band is at about 7.9 years, while its own Fig. 1 nominal run crosses 200 at about 7.1 years, as does this data set's (7.105); the offset is internal to the paper. The reading of V(0) does not explain it (nominal F 7.105 years with V(0) = 0.01 against 7.031 with V(0) = 10 copies/mm³; `check_accuracy.py`, check 5). Capping F at 10 years here (gradient zero for the 15 capped samples) moves the ρ_M weight only from −0.434 to −0.439 (Fig. 6c −0.519). A different computation of F for Fig. 6c is the most concrete candidate cause of the F weight differences; it cannot be reproduced from the paper.
- **A discrepancy in the paper.** The text on p. 14 says δ_V, δ_T* and p_T are positively correlated with the outputs; Fig. 6 shows a negative p_T weight in all three panels, and so does this data set (−0.457, −0.374, −0.373).
- **Open questions (only the authors can settle).** (1) The units of V(0) (copies/ml as printed against the copies/mm³ of Table 1). (2) How F was computed for Fig. 6c: its values are capped at 10 years and lie about 0.8–1.0 year above this data set's (and above the paper's own Fig. 1 nominal of about 7.1 years at wᵀx = 0). Does that computation account for the ρ_M, k_T and p_T weight differences? (The cap alone does not.) Also the finite-difference step and solver tolerances, which are not given. (3) The length of a year used for T(5) and T(10).

## License and credit

Code (`hiv3_model.py`, `generate_data.py`, `check_accuracy.py`) and the notebook: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the published equations and parameter table; no data or code were copied. The paper is not open access (exclusive licence to Springer); it is linked by DOI, not included. Cite C. Clarke and S. Pankavich, *J. Math. Biol.* 88(3):34, 2024, doi:10.1007/s00285-024-02056-1, and E. A. Hernandez-Vargas and R. H. Middleton, *J. Theor. Biol.* 320:33–40, 2013, doi:10.1016/j.jtbi.2012.11.028. The paper's second author was supported by the US National Science Foundation (DMS-1911145, DMS-2107938).

## Unknowns

- V(0) units (see above); chosen 0.01 copies/mm³.
- Year length; chosen 365 days.
- The paper's finite-difference step, sample seed, solver tolerances and method of locating F; not given.
- The paper's F in Fig. 6c is capped at 10 years and about 0.8–1.0 year later than this data set's at the same wᵀx (and than its own Fig. 1 nominal); how it was computed is unknown. Capping here does not reproduce the ρ_M weight (−0.439 against −0.519).
- Fig. 6 values are available only as a plot; read by pixel position (about ±0.005).

## Relation to `HIV_Long_Term_Dynamics`

That data set is the seven-equation model of Loudon and Pankavich (2017, doi:10.3934/mbe.2017040), with latently infected T cells and cytotoxic T lymphocytes, 27 parameters on ±2.5 % ranges, and T at 19 times as outputs, from stored 2016 inputs with central-difference gradients. This data set is a different, five-equation model (Hernandez-Vargas and Middleton 2013, analysed by Clarke and Pankavich 2024) with no latent compartment and no immune response, 15 parameters on ±10 % ranges, the time to AIDS as an output, exact sensitivity gradients and a new seeded design.
