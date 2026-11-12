# Tumour_Competition_ODE: data

Kind: **Python model**. The model is `tumour_model.py` (numpy and scipy), written from the paper's equations. `generate_data.py` draws the inputs with a saved seed and writes the four files `lv_samples_*.npz`; a rerun reproduces them to the tolerances under "Regenerating".

## Files

| file | contents | role |
|---|---|---|
| `Tumour_Competition_ODE.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 246; outputs committed |
| `tumour_model.py` | `qoi(x, grad=False)`, `qois(X, grad=False)`, `trajectory(x, t=None)`, `equilibrium(x)`; `NAMES`, `UNITS`, `DESCRIPTIONS`, `LOWER`, `UPPER`, `S0`, `R0`, `T_END`, `T_GRID`, `RTOL`, `ATOL`, `QOI_NAMES`, `QOI_UNITS` | the model |
| `generate_data.py` | draws both designs and writes `lv_samples_*.npz`; `--check` compares a rerun with the shipped files; checks gradients by central differences | regenerates the data |
| `check_accuracy.py` | recomputes the numbers below that the notebook and `generate_data.py` do not print | check |
| `reduce_gradients.py` | rounds the gradients of a float64 single-file `lv_samples.npz` to float32 and copies everything else (`--in FILE --out DIR`) | storage reduction |
| `lv_samples_grid4.npz`, `lv_samples_sub8.npz` | each design's cell indices, inputs and values (6.1 MB each, compressed; float64); the labels are in the grid4 file | model inputs and outputs |
| `lv_samples_grid4_grad.npz`, `lv_samples_sub8_grad.npz` | each design's gradients (4.5 MB each; float32) | model outputs |

The arrays are split over the four files so that each is under 10 MB; `tumour_model.load_samples()` reads them back as one dict, and `tumour_model.SAMPLE_FILES` lists which file holds which array (load with `np.load`, no pickle; the notebook converts the gradients to float64 on loading). The full set is also archived as one file, `lv_samples.npz` (21.2 MB, SHA-256 45fca08835b404b3b9caa98c3e9decb57e3e7e9d205a5492b78368a2fe851b41), on Zenodo: doi:[10.5281/zenodo.24174099](https://doi.org/10.5281/zenodo.24174099). Its arrays are byte-identical to those of the four files.

| array | shape | contents |
|---|---|---|
| `grid4_X`, `sub8_X` | (102400, 6) | inputs r_S (1/day), r_R (1/day), K_S (mm^3), K_R (mm^3), gamma_S (–), gamma_R (–) |
| `grid4_f`, `sub8_f` | (102400, 2) | q_pri (mm^3 day), q_sec (–) |
| `grid4_grad`, `sub8_grad` | (102400, 2, 6) float32 | `grad[i, k, j]` = ∂q_k/∂x_j with respect to the physical inputs (exact forward sensitivities, computed in float64, stored as float32) |
| `grid4_region`, `sub8_region` | (102400,) int | C-order flat index of the point's cell in the 4^6 or 8^6 grid (r_S slowest); 25 consecutive rows per cell |
| `names`, `units`, `qoi_names`, `qoi_units`, `description` | strings | labels and provenance |

## Model

**Source.** H. Zou, A. L. Lewis, "Using active subspaces to explore discrepancies between global and local parameter sensitivities in a Lotka-Volterra system", *Math. Biosci. Eng.* 23(7):2178–2207, 2026, doi:[10.3934/mbe.2026079](https://doi.org/10.3934/mbe.2026079) (publisher PDF, open access, read). The model is from Cho et al., *J. Theor. Biol.* 559:111377, 2023, doi:[10.1016/j.jtbi.2022.111377](https://doi.org/10.1016/j.jtbi.2022.111377) (the paper's ref. [29]; not read).

**Equations** (Eqs. (3.1)–(3.2), p. 2186): dS/dt = r_S S (1 − S/K_S − γ_R R/K_S), dR/dt = r_R R (1 − R/K_R − γ_S S/K_R); S, R are the volumes (mm^3) of Type S and Type R cells. Initial state: total volume 0.02 mm^3 in the ratio 9:1 (p. 2187), so S(0) = 0.018, R(0) = 0.002 mm^3.

**Inputs** (m = 6), independent uniform (Sec. 3.2.1, p. 2187, text: "parameter ranges of [0, 1] for all parameters"; Latin hypercube designs, p. 2188):

| name | symbol | units | range | source |
|---|---|---|---|---|
| `r_S` | r_S | 1/day | [0, 1] | Sec. 3.2.1, p. 2187 |
| `r_R` | r_R | 1/day | [0, 1] | Sec. 3.2.1, p. 2187 |
| `K_S` | K_S | mm^3 | [0, 1] | Sec. 3.2.1, p. 2187 |
| `K_R` | K_R | mm^3 | [0, 1] | Sec. 3.2.1, p. 2187 |
| `gamma_S` | γ_S | – | [0, 1] | Sec. 3.2.1, p. 2187 |
| `gamma_R` | γ_R | – | [0, 1] | Sec. 3.2.1, p. 2187 |

The paper prints no units for the inputs; the units above follow from the equations with S, R in mm^3 and t in days (this data set's reading).

**Quantities of interest.** q_pri = ∫₀⁵⁶ (S + R) dt (Eq. (3.3), p. 2187), computed, as the paper does, by the trapezoidal rule on days 0, 1, …, 56; this discretised value is the QoI here (it differs from the exact integral by a median relative 2.4 × 10⁻⁵ and at most 1.2 × 10⁻² over 200 points, `check_accuracy.py`, check 5; the largest differences are at fast early transients). q_sec = S(56)/(S(56) + R(56)) (Eq. (3.5), p. 2198).

**Solver.** `scipy.integrate.solve_ivp`, LSODA, in u = ln S, v = ln R (u' = r_S (1 − (S + γ_R R)/K_S), similarly for v), rtol = atol = 1e-10, output at the 57 day nodes. The log states keep both volumes positive and resolve a dying population to a relative accuracy, which q_sec needs.

**K = 0 (singular).** The right-hand side divides by K_S and K_R. `tumour_model.qoi` raises ValueError for K ≤ 0 and accepts any K > 0. As K → 0 the population collapses to about K within a day (a stiff transient that LSODA's stiff method handles) and the QoIs have finite limits: with the other inputs at 0.5, q_pri = 22.509145, 22.487165, 22.487453, 22.487456 mm^3 day at K_S = 1e-2, 1e-4, 1e-6, 1e-8 (`check_accuracy.py`, check 6). The designs are continuous, so K = 0 has probability zero; the smallest values drawn are K_S = 5.4 × 10⁻⁵, K_R = 9.4 × 10⁻⁶ (grid4) and 1.1 × 10⁻⁵, 4.5 × 10⁻⁶ (sub8) (`generate_data.py` prints them). `generate_data.py` raises if a K of exactly 0 is ever drawn. No point was dropped or redrawn.

**Gradients.** Exact derivatives of the discretised QoIs up to the solver tolerance: the forward sensitivities of (u, v) with respect to the six inputs are integrated with the states (14 equations); ∂S/∂x = S ∂u/∂x, and the trapezoidal rule is applied on the same nodes; ∂q_sec/∂x = q_sec (1 − q_sec)(∂u/∂x − ∂v/∂x) at day 56. Coordinates: the physical inputs (equal to the paper's [0, 1] coordinates). **Storage precision:** the gradients are stored as float32. The rounding is at most 5.9 × 10⁻⁸ (q_pri) and 9.5 × 10⁻⁸ (q_sec) relative per row (2-norm) over every row whose norm is inside float32's normal range (≥ 1.2 × 10⁻³⁸); 2725 grid4 and 2568 sub8 q_sec rows have a smaller nonzero norm (points where q_sec is 0 or 1 to far below rounding) and become float32 subnormals or 0 (2279 and 2100 become 0), and 9739 and 9513 q_sec rows were already 0 (`reduce_gradients.py` prints these counts). Every number the notebook prints is the same to the printed digits with float64 and float32 gradients, except the notebook's five-point central-difference check, which now sees the storage rounding (4.3 × 10⁻⁸ and 4.5 × 10⁻⁸). Storing the gradients as float32 cuts the data from 30.6 MB to 21.2 MB (as one file); double-precision gradients barely compress. The notebook passes them to `ac.Uniform(0, 1)`, which scales every gradient by 1/2. The paper used forward differences with step 1e-5 (p. 2188) on MATLAB `ode45` solves (rtol 1e-6, atol 1e-8, maximum step 1 day, in its sample code).

**Sampling design.** Two designs, each 4096 cells × 25 points, seed 246:

- **grid4**: every cell of a 4^6 grid of the unit cube (cells of width 0.25). This is a coarser grid than the paper's 8^6 = 262,144 cells of width 0.125 (p. 2187), chosen by this data set because the paper's full sweep is 6.5 million gradient evaluations (about 20 CPU hours at the cost measured here).
- **sub8**: 4096 of the paper's 8^6 cells: the two cells named on p. 2201 (r_S, γ_S ∈ [0.125, 0.25], r_R ∈ [0.25, 0.375], K_S ∈ [0.375, 0.5], K_R, γ_R ∈ [0.5, 0.625]; and [0.875, 1]^6) and 4094 drawn uniformly without replacement (`numpy.random.default_rng([246, 8])`). This is this data set's choice, so that the paper's cell size can be compared on a 1/64 random subset.

In each cell the 25 points are a Latin hypercube (one point in each of 25 equal slices of every coordinate, uniform within the slice; generator `default_rng([246, G, cell])`), as in the paper (p. 2188). Pooled, grid4 is a uniform stratified sample of the cube; the paper's sample code likewise pools its local samples into the global sample (the text calls the global design "25 × 8^6 sample points from a Latin hypercube design", p. 2188). Every point is inside the model's meaning. q_sec is within 1e-6 of 0 or 1 at 26,673 grid4 and 25,894 sub8 points (one population has died out); there its gradient is tiny or underflows to 0.

**Published results** (p. 2188–2198): Table 1, normalized global activity scores α_i/α_max = 1.0000, 0.6715, 0.9364, 0.6942, 0.0746, 0.0567 (with Morris and Sobol' results); Table 2, the ten most frequent local rankings for each method (activity scores: first K_R, r_R, K_S, γ_S, γ_R, r_S in 12,311 cells); 304 distinct local activity-score rankings; the global ranking 53rd (1525 cells); Fig. 7 and p. 2192, cells per distance bin 63, 4362, 44,939, 133,127, 79,653 and the g-scores; p. 2197, mean distance 0.7334 (0.6665 bifurcation-matched); Table 3, 25 %, 25 %, 50 % stable outcomes; p. 2198, mean distance 0.6369 for q_sec; p. 2201, distances 0.1355 and 0.9996 for the two named cells.

**What the notebook finds.** Global subspace of q_pri from the 102,400 grid4 gradients: eigenvalue ratios 2.14, 1.43, 1.26, 8.97, 2.19; n = 4 (the only gap; first four eigenvalues 0.9799 of the trace; quadratic surrogate LOO R² 0.8264 for n = 3, 0.9170 for n = 4, 0.9291 for n = 5); the inactive directions are the two interaction strengths. Activity scores 1.0000, 0.6690, 0.9347, 0.6934, 0.0745, 0.0568 (the paper's values inside every 95 % bootstrap interval). distance-bin shares on the 4094 randomly drawn sub8 cells 0.024, 1.759, 17.196, 50.757, 30.264 % (paper 0.024, 1.664, 17.143, 50.784, 30.385 %; largest difference 0.12 percentage points, the lowest bin from 1 cell), mean distance 0.73048 (0.73040 on all 4096 sub8 cells, which include the two named cells added by hand, one of them in the lowest bin); named cells 0.1332 and 0.9997; q_sec mean distance 0.6316.

## Regenerating

```bash
python generate_data.py                                   # writes lv_samples_*.npz here (22 to 60 min, see below)
python generate_data.py --out DIR --check --blocks 1      # first 256 cells of each design, compared (about 1.5 min)
python generate_data.py --out DIR --cache CDIR --max-minutes 9   # resumable chunks (rerun until done)
python check_accuracy.py                                  # the numbers marked below
python reduce_gradients.py --in OLD.npz --out DIR         # float64 gradients -> float32 (not needed for a new run)
```

One process. `--check`, `--blocks`, `--cache` and `reduce_gradients.py` refuse a directory that is this one or a subdirectory of it. The data were computed on 2026-11-09 in a Linux container (Python 3.11.18, numpy 2.4.6, scipy 1.17.1), in resumable chunks of at most about 10 minutes on a shared machine: 3589 s (about 60 min) of model runs for the 204,800 points (9–12 ms per point with sensitivities). That run stored the gradients in float64 (30.6 MB, sha256 6086605def7119f56222bac86a2cfed9ef009053cd6f58031920d883e1369460; not kept in the repository); the shipped arrays are its output passed through `reduce_gradients.py` (no rerun of the model), and `generate_data.py` now writes the float32 gradients directly, so a full rerun gives the shipped arrays. In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-11-10), `python generate_data.py --out DIR --check --blocks 1` took about 85 s (82 s of model runs for the 12,800 points it recomputes, 6.4 ms per point with sensitivities; at that rate a full run takes about 22 min) and passed: the cells and inputs are identical to the shipped files', q_pri differs by at most a relative 3.0 × 10⁻¹¹, q_sec by 1.0 × 10⁻¹¹ (absolute), and the gradients, after rounding the new ones to float32, by at most a relative 9.5 × 10⁻¹⁰ (row 2-norm). The tolerance is 1e-15 absolute for the inputs (they are computed from numpy's generator), 1e-8 relative for q_pri, 1e-8 absolute for q_sec, and 1e-6 relative (row 2-norm, floor 1e-8) for the gradients after rounding to float32 (about 8 float32 units in the last place), which allows for LSODA's step choices changing with the platform's floating point. The generator also prints a central-difference check of the first 20 grid4 points against its float64 gradients: largest relative error 5.6 × 10⁻⁷ (q_pri) and 1.3 × 10⁻⁷ (q_sec, the 19 points with q_sec (1 − q_sec) > 1e-6).

## Accuracy

`check_accuracy.py` recomputes the numbers marked with its name; in `.venv` it took about 40 s on the shipped (float32-gradient) file (2026-11-10).

- **Values** (`check_accuracy.py`, check 1). At 20 points of each design the stored q_pri agrees with the model at rtol = atol = 1e-12 to a relative 8.9 × 10⁻¹² (grid4) and 7.3 × 10⁻¹² (sub8), and with an independent solve of Eqs. (3.1)–(3.2) in the original states S, R (Radau, rtol 1e-12, atol 1e-15) to 8.8 × 10⁻¹² and 7.4 × 10⁻¹²; q_sec to 1.3 × 10⁻¹² and 4.6 × 10⁻¹² (absolute). The notebook's check (solve without sensitivities, 5 points) agrees to 1.4 × 10⁻⁹.
- **Gradients.** The stored float32 gradients against sensitivities solved at rtol = atol = 1e-12, 20 points per design: q_pri 3.9 × 10⁻⁸ and 3.6 × 10⁻⁸, q_sec 4.1 × 10⁻⁸ and 5.1 × 10⁻⁸ (rows with gradient norm above 1e-8) (`check_accuracy.py`, check 2): the size of the float32 storage rounding (at most 5.9 × 10⁻⁸ and 9.5 × 10⁻⁸ per row, `reduce_gradients.py`). Against central differences (step 1e-6, tight solves), first 100 grid4 points: q_pri largest 1.4 × 10⁻⁶, median 3.3 × 10⁻⁸; q_sec (94 points with q_sec (1 − q_sec) > 1e-6) largest 3.9 × 10⁻⁷, median 3.0 × 10⁻⁸ (check 3). Closer to 0 or 1 the differences of q_sec are round-off, so q_sec is not compared there.
- **The paper's gradient recipe.** Forward differences with step 1e-5 on solves at rtol 1e-6, atol 1e-8, maximum step 1 day (scipy RK45 standing in for MATLAB `ode45`), against the exact gradients of q_pri at the first 100 grid4 points: median relative error 3.2 × 10⁻⁵, largest 8.6 × 10⁻⁴ (`check_accuracy.py`, check 4).
- **Quadrature.** Trapezoidal rule against the exact integral (check 5): see "Quantities of interest".

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from MATLAB (p. 2184) on all 8^6 cells with forward-difference gradients. The authors' sample code (github.com/lewisallison10/Global-vs.-Local-Sensitivity-Using-the-Active-Subspace, `LotkaVolterraEx_SampleCode.m`; no licence) was read to confirm the settings (pooled global sample, activity scores over all m eigenpairs, local dimension at more than 95 % of the trace, the solver options) and nothing was copied. The data here are an independent Python computation of the paper's equations with exact gradients, on fewer cells.
- **Agreement.** Table 1's activity scores to within 0.0025 (inside the 95 % bootstrap intervals of our 102,400 points). On the sub8 cells (the paper's cell size), the distance-bin shares of the 4094 randomly drawn cells to within 0.12 percentage points of the paper's (the lowest bin, 0.024 % in both, rests on 1 of our cells), the mean distance 0.73048 against 0.7334, the two named cells 0.1332 and 0.9997 against 0.1355 and 0.9996, the most frequent local ranking (4.37 % against 4.70 % of cells) and the next three rankings in a slightly different order, and the g-score 0.344 [0.330, 0.359] for k = 2 at distances 0.8–1.0 against the paper's 36 %. The grid4 cells, four times wider per input, give a larger mean distance (0.8218): the paper's diagnostics depend on the cell size.
- **Not reproduced.** Morris and Sobol' results (Tables 1–2, Fig. 4), the bootstrap studies (Figs. 8–9), the bifurcation-matched distances (Figs. 10–11), the calibration (Sec. 3.3.1) and surrogate (Sec. 3.3.2) experiments, and the g-scores of the paper's lowest distance bin (2 sub8 cells, 1 of them random; the paper has 63 of 262,144).

## Open questions (only the authors can settle)

1. How the bifurcation-specific global subspace is matched to a cell whose 25 points have different long-term outcomes (Figs. 10–11).
2. How cells where q_sec is flat to rounding (both populations' fractions saturated at every point) were treated in Fig. 12; this notebook leaves them out (491 of 4096 sub8 cells).
3. Whether the "25 × 8^6 sample points from a Latin hypercube design" of the global analysis (p. 2188) are the pooled local samples, as in the sample code, or a separate design.
4. Units of the inputs (not printed; read here from the equations).

## License and credit

Code (`tumour_model.py`, `generate_data.py`, `check_accuracy.py`, `reduce_gradients.py`) and the notebook: MIT (`LICENSE`). Data (`lv_samples_*.npz`, and `lv_samples.npz` on Zenodo): CC BY 4.0 (`LICENSE-DATA`). The data were generated here from the published equations; no data or code were copied. The paper is open access; cite H. Zou, A. L. Lewis, *Math. Biosci. Eng.* 23(7):2178–2207, 2026, doi:10.3934/mbe.2026079. The paper prints no funding statement.
