# Zika_SEIR_SEI_Brazil: data

Kind: **Python model**. The model is `zika_model.py` (numpy and scipy only), and `generate_data.py` writes `inputs.csv`, `outputs.csv` and `gradients.csv` from it with a saved seed. `data_cases.csv` is stored data (the weekly case counts the paper fitted), copied unaltered.

## Files

| file | contents | role |
|---|---|---|
| `Zika_SEIR_SEI_Brazil.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 2018`; outputs committed; about 8 s |
| `zika_model.py` | `trajectory(p)`, `cumulative_with_sens(p)`, `qoi_and_grad(p)`, `qoi(p)`, `load_cases()`; `NAMES`, `UNITS`, `NOMINAL`, `LOWER`, `UPPER`, held constants | the model |
| `generate_data.py` | draws the inputs (seed 2018) and writes the three CSVs; `--check`, `--fd-check K` | regenerates the data |
| `inputs.csv` | 1000 × 10 physical inputs, columns in the order of `NAMES` | model input |
| `outputs.csv` | 1000 × 2: C₅₂ (people), J (people²) | model output |
| `gradients.csv` | 1000 × 20: columns 0–9 ∂C₅₂/∂p, columns 10–19 ∂J/∂p, with respect to the **physical** inputs p | model output |
| `data_cases.csv` | 52 × 2: epidemiological week (EW) 1–52 of 2016, probable Zika cases that week | stored data (Brazilian Ministry of Health) |

Each CSV has two `#` lines (a description, then the column names with units) and is read by `np.loadtxt(path, delimiter=",")`.

## Model

**Citation.** E. Dantas, M. Tosin, A. Cunha Jr, "Calibration of a SEIR–SEI epidemic model to describe the Zika virus outbreak in Brazil", *Applied Mathematics and Computation* 338:249–259 (2018), doi:10.1016/j.amc.2018.06.024. The paper card was written from the arXiv preprint (arXiv:2105.06036v1, May 2021); every number and page reference below was checked against the published version (AMC pp. 249–259), which the authors distribute in their ARBO repository (`publications/JournalPaper_2018_AMC_v338_pp249-259.pdf`). The two versions agree on Tables 1–2 and on the quoted fit statistics.

**Equations.** Eq. (1), p. 251: humans S_h, E_h, I_h, R_h (people, N_h = 206 × 10⁶), vectors S_v, E_v, I_v (proportions of N_v = 1), and the cumulative number C of humans who have become infectious:

- dS_h/dt = −β_h S_h I_v/N_v, dE_h/dt = β_h S_h I_v/N_v − α_h E_h, dI_h/dt = α_h E_h − γ I_h, dR_h/dt = γ I_h, dC/dt = α_h E_h;
- dS_v/dt = δ N_v − β_v S_v I_h/N_h − δ S_v, dE_v/dt = β_v S_v I_h/N_h − (α_v + δ) E_v, dI_v/dt = α_v E_v − δ I_v.

Time is in days. The run starts at t₀ = 7 (the end of EW 1 of 2016) with C(7) = C_i = 8201 and ends at t = 364 (EW 52); EW w ends at t = 7w. New cases per week are N₁ = C(7), N_w = C(7w) − C(7(w−1)) (Eq. 3, p. 251). This time convention is not spelled out in the paper beyond "one to fifty-two weeks (7 to 365 days)" (p. 253); it was taken from the authors' MATLAB code (`ObjFun_SEIR_SEI.m`: `tspan = 7:1:364`, C sampled every 7 days), which reproduces the paper's fit statistics (below).

**Inputs (m = 10).** Independent uniform on the box below. Rates in 1/day, human initial conditions in people, vector initial conditions as proportions.

| name | symbol | units | nominal (TRR input) | range (uniform) | source |
|---|---|---|---|---|---|
| `alpha_h` | α_h, human incubation rate | 1/day | 1/5.9 | [1/12, 1/3] | Table 1, p. 255; Table 2, p. 256 |
| `alpha_v` | α_v, vector incubation rate | 1/day | 1/9.1 | [1/10, 1/5] | Tables 1–2 |
| `gamma` | γ, human recovery rate | 1/day | 1/7.9 | [1/8.8, 1/3] | Tables 1–2 |
| `delta` | δ, vector death rate | 1/day | 1/11 | [1/21, 1/11] | Tables 1–2 |
| `beta_h` | β_h, mosquito-to-human transmission | 1/day | 1/11.3 | [1/16.3, 1/8] | Tables 1–2 |
| `beta_v` | β_v, human-to-mosquito transmission | 1/day | 1/8.6 | [1/11.6, 1/6.2] | Tables 1–2 |
| `E_hi` | initial exposed humans | people | 8201 | [0, 10,000] | Table 2, p. 256 (Table 1: [0, N_h]) |
| `I_hi` | initial infectious humans | people | 8201 | [0, 10,000] | Table 2 (Table 1: [0, N_h]) |
| `E_vi` | initial exposed vectors | – | 2.2 × 10⁻⁴ | [0, 10⁻³] | **chosen here** (Tables 1–2: [0, 1]) |
| `I_vi` | initial infectious vectors | – | 2.2 × 10⁻⁴ | [0, 10⁻³] | **chosen here** (Tables 1–2: [0, 1]) |

Held fixed, as in the paper's calibration (p. 253): N_h = 206 × 10⁶, N_v = 1, R_hi = 29,639 (the suspected cases of 2015, p. 252), C_i = 8201 (the cases of EW 1, p. 252).

**The three flags on the paper card, and how they were resolved.**

1. *Weekly case data file.* The paper's data are in its supplementary material A (not in the preprint). The same 52 weekly counts are in the authors' MIT-licensed repository github.com/americocunhajr/ARBO, as `ARBO-1.0/CalibrationProblem/2016zika_Prob250117.dat` (identical to `ARBO-1.0/Datasets/2016zika_Prob250117.dat`; commit 21508721, 2025-04-04), which the authors' calibration script loads. `data_cases.csv` holds them unaltered (only transposed to two columns). Its first value, 8201, is the paper's C_i, and its peak, 21,280 in EW 7, matches the paper's Fig. 2. Its total, 263,099, is larger than the 215,319 probable cases the paper quotes from the Ministry's EW 52 bulletin (p. 249); the paper does not discuss the difference (the weekly file was retrieved through a separate access-to-information request, ref. [38]).
2. *Initial-condition constraints.* The paper requires S_hi = N_h − R_hi − E_hi − I_hi and S_vi = N_v − E_vi − I_vi (p. 252). Its calibration treated S_hi and S_vi as two more free unknowns, added the two sums as extra fitting points, and corrected S_hi and S_vi afterwards (p. 255). Here S_hi and S_vi are **derived** from the sums, so the constraints hold exactly and the study has 10 inputs, not the paper's 12. (Varying S_hi over [0.9 N_h, N_h] independently would break the human-population constraint by up to 2 × 10⁷ people.)
3. *Nominal S_vi outside its bound.* Table 1 bounds S_vi to [0.99, 0.999], but the nominal value 0.99956 and both calibrated values (1 in Table 1, 0.999586 in Table 2) are above 0.999: the post-hoc correction of item 2 moved them out of the box. Since S_vi is derived here, the bound is not used. Instead E_vi and I_vi each range over [0, 10⁻³] (so S_vi ∈ [0.998, 1]). This range contains the nominal value (2.2 × 10⁻⁴) and the Table 2 value (4.14 × 10⁻⁴), and its width is the "order 10⁻³" that the paper says already changes the response significantly (p. 255). The paper's [0, 1] only says these are proportions. Over [0, 5 × 10⁻³] (a test, not shipped) the median C₅₂ was 2.3 × 10⁶, nine times the data's total, so the wider range was not used.

**Quantities of interest.**
- C₅₂ = C(364), the cumulative number of cases at the end of EW 52 (people): the end point of the paper's Figs. 5a and 6a. Not defined as a QoI in the paper.
- J = Σ_{w=1}^{52} (y_w − N_w)², the paper's misfit (Eq. 7, p. 254), people². N₁ = C_i = y₁, so the first term is always zero, as in the authors' code.
- Both span orders of magnitude over the box (C₅₂ from 3.5 × 10⁴ to 1.5 × 10⁸), because the reproduction number R₀ = [β_h β_v α_v/(γ δ (α_v + δ))]^{1/2} ranges from 0.37 to 1.47. The notebook analyzes ln C₅₂ and ln J (gradients ∇C/C, ∇J/J). The files hold C₅₂ and J and their gradients untransformed.

**Gradients.** Forward sensitivities: the 7 states (R_h follows from conservation) and the 7 × 10 sensitivity matrix are solved together (77 ODEs), with the analytic Jacobian and parameter derivatives. Human compartments are scaled by N_h internally, so all unknowns are O(1) or smaller. Gradients of J use ∂N_w/∂p from the sensitivities of C at the 52 weekly times.

**Solver.** `scipy.integrate.solve_ivp`, DOP853, rtol = 10⁻¹¹, atol = 10⁻¹⁵ (scaled variables). About 0.035 s per solve with sensitivities, 0.01 s without. The authors used MATLAB `ode45` with rtol = 10⁻⁶, atol = 10⁻⁹ (their `ObjFun_SEIR_SEI.m`).

**Sampling design.** M = 1000 points, independent uniform on the box, `numpy.random.default_rng(2018)`. The paper has no sampling design: it is a deterministic trust-region-reflective calibration (MATLAB `lsqcurvefit`, tolerance 10⁻⁷, p. 254).

**Published results to check.** The paper has no active-subspace results (it names the analysis as future work, p. 258). It reports two calibrated parameter sets and their fit statistics. With `zika_model.py` at those parameter values (S_hi derived; the table values are rounded):

| quantity | paper | this model |
|---|---|---|
| Table 1 fit: C₅₂ over the data's total | +6.00 % (p. 255) | +6.73 % |
| Table 1 fit: peak N_w vs. data maximum | 7.87 %, one week off (p. 255) | −7.21 %, EW 6 vs. EW 7 |
| Table 2 fit: C₅₂ over the data's total | +5.74 % (p. 256) | +5.61 % |
| Table 2 fit: peak N_w vs. data maximum | 10.57 %, two weeks off (p. 256) | −10.66 %, EW 5 vs. EW 7 |

The notebook's results: ln C₅₂ has λ₁/λ₂ = 7.25 and λ₂/λ₃ = 5.90 (two directions hold 97.2 % of the trace); w₁ is within 20° of the gradient of ln R₀² and w₂ is mostly the initial vector infection (I_vi, E_vi) and γ; a cubic in two active variables has LOO R² = 0.972. ln J has the same two directions (λ₁/λ₂ = 7.87, λ₂/λ₃ = 8.48; LOO R² = 0.955).

## Regenerating

```bash
python generate_data.py                                   # 1000 points, seed 2018, writes here
python generate_data.py --out /tmp/zika --check           # regenerate elsewhere and compare
python generate_data.py --fd-check 20                     # gradients vs. independent central differences
```

The run that wrote the shipped files (2026-10-30, Python 3.13.18, numpy 2.5.4, scipy 1.18.1, one core of the build container) took 38 s. A rerun with `--check` reproduced the files to a relative 5.2 × 10⁻¹⁶ (outputs) and 4.2 × 10⁻¹⁶ (gradients), with identical inputs. In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-10-31) `--check` gives 6.4 × 10⁻¹³ (outputs) and 2.7 × 10⁻¹² (gradients), with identical inputs. `--fd-check 20` took 211 s.

## Accuracy

- **Outputs.** At seven rows (0–4 and the rows with the largest and smallest C₅₂), DOP853 at the shipped tolerances agrees with Radau at rtol = 10⁻¹³, atol = 10⁻¹⁸ to a relative 1.7 × 10⁻¹⁴ in C₅₂ and 5.9 × 10⁻¹³ in J (scratch script, not committed).
- **Gradients.** `generate_data.py --fd-check 20`: the sensitivity gradients at the first 20 rows agree with central differences of an independent solver (Radau, rtol = 10⁻¹², atol = 10⁻¹⁶, step 10⁻⁴ of each input's half-range; a second-order one-sided difference within one step of a zero lower bound) to a relative 3.8 × 10⁻⁹ for C₅₂ (median 8.8 × 10⁻¹¹) and 2.1 × 10⁻⁸ for J (median 1.6 × 10⁻¹⁰), in the 2-norm of the 10-vector.
- The notebook re-solves five rows and prints the largest relative difference from `outputs.csv` (1.5 × 10⁻¹², from the solve without sensitivities taking different steps).

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from MATLAB (ode45 and `lsqcurvefit`) in the authors' ARBO code. `zika_model.py` is an independent Python implementation of Eq. (1); the MATLAB files `rhs_SEIr_SEI.m`, `ObjFun_SEIR_SEI.m` and `main_SEIR_SEI_TRR_example{1,2}.m` (MIT licence) were read for the time convention and the fixed constants only, and no code was copied.
- **Differences from the paper's setup.** S_hi and S_vi are derived rather than free (flag 2); the S_vi bound is replaced by the E_vi, I_vi range above (flag 3); the parameter study is new (the paper only calibrates). The fit statistics above agree with the paper's to within about 0.7 percentage points, which we attribute to the rounded values printed in Tables 1–2 (e.g. 1/16.86, 1/10.40, I_hi = 253,360) and the looser ode45 tolerance; this was not verified with the authors' unrounded values, which are not published.
- The ARBO script `main_SEIR_SEI_TRR_example1.m` (which starts from the Table 2 initial conditions and refits the six rates) bounds δ to [1/8.8, 1/3], whereas the paper's Tables 1–2 and the script `main_SEIR_SEI_TRR_example2.m` use [1/21, 1/11]; this data set follows the paper.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs written by `generate_data.py`): CC BY 4.0 (`LICENSE-DATA`). `data_cases.csv`: probable Zika cases per EW, Brazil 2016, from the Brazilian Ministry of Health (SINAN), as distributed by E. Dantas, M. Tosin and A. Cunha Jr in github.com/americocunhajr/ARBO under the MIT licence; credit both when reusing it. Model: E. Dantas, M. Tosin and A. Cunha Jr. Cite the paper: Appl. Math. Comput. 338:249–259 (2018), doi:10.1016/j.amc.2018.06.024. Python implementation, data and notebook: Claude (Anthropic) for P. Constantine, 2026.
