# HIV_Long_Term_Dynamics: data

Kind: **Python model**. The model is `HIV_model.py`, and `generate_data.py` writes `outputs.csv` and `gradients.csv` from it. The 1000 input samples in `inputs.csv` were drawn without a seed in 2016 and cannot be redrawn, so they are treated as stored data.

## Files

| file | contents | role |
|---|---|---|
| `HIV_Long_Term_Dynamics.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2017; outputs committed |
| `HIV_model.py` | `Tcells(p, times)`, `trajectory(q, times)`; `NAMES`, `NOMINAL`, `LOWER`, `UPPER`, `Y0`, `T_FLOOR` | the model |
| `generate_data.py` | writes `outputs.csv` and `gradients.csv` (and rewrites `inputs.csv`) from `inputs.csv` | regenerates the data |
| `inputs.csv` | 1000 × 27 physical parameters, columns in the order of `NAMES` | stored inputs |
| `outputs.csv` | 1000 × 19: T(t) in cells mm⁻³ at the 19 times | model output |
| `gradients.csv` | 1000 × 513: column 27 i + j is ∂T(t_i)/∂x_j, with x the **normalized** inputs | model output |

The CSVs are plain comma-separated numbers after two `#` lines (a description and the column names), so `np.loadtxt(path, delimiter=',')` reads them. The 19 times, in days, are 24, 38, 40, 45, 50, 55, 65, 90, 140, 500, 1000, 1600, 1700, 1800, 2000, 2200, 2400, 2800, 3400: the 18 of the paper's Appendix B (Figs. 15–17) plus 1700, the paper's worked example (Figs. 2–6).

## Model

The seven ODEs of Loudon and Pankavich (2017), Eq. (1), with the 27 parameters of their Table 1.

- **Inputs**: independent uniform on nominal ± 2.5 % (paper p. 718). The normalized inputs are x = (2p − (upper + lower)) / (upper − lower) ∈ [−1, 1]²⁷ (paper Eq. (8)), and `gradients.csv` is ∂T/∂x. Divide by (upper − lower)/2 for gradients with respect to the physical parameters.
- **Initial conditions**: the same for every sample, by design (author, 2026-10-30): (T, T_I, T_L, M, M_I, CTL, V)(0) = (1000, 0, 0, 0.15/0.005, 0, 5/0.015, 1), the uninfected equilibrium at the nominal parameters plus one virion. The paper does not state its initial conditions.
- **Floor**: when T falls to 0.1 the state is frozen and T stays 0.1. This is in the 2016 code, not in the paper. `HIV_model.py` implements it as a terminal event of `solve_ivp`; the 2016 code returned a zero right-hand side below 0.1. At a time after a sample reaches the floor, its output is 0.1 and its gradient is 0.
- **Solver**: `scipy.integrate.solve_ivp`, LSODA, rtol = atol = 1e-10, analytic Jacobian; 0.04–0.08 s per 3400-day trajectory.
- **Gradients**: central differences in x with step h = 3e-4, so 55 trajectories per sample, each giving all 19 times; set to 0 at times after the floor.

## Regenerating

```bash
python generate_data.py                                  # all 1000 samples, 3 processes
python generate_data.py --rows 0:20 --out /tmp/hiv --check   # a subset, compared with the files here
```

There is no randomness, so a run reproduces the files. Samples are independent; `--workers` changes only the run time. The run that wrote the shipped files (2026-10-30, scipy 1.17.1, numpy 2.4.6, Python 3.11) took 2000 s (33 min) on three processes of an 8-core Mac that other jobs were also using; that is about 4 s of CPU per sample (55 trajectories of 0.04–0.08 s), so about 70 min on one core. A 30-sample subset run took 54 s on three processes. Tolerance: a rerun reproduces the files to the digits written; `--rows 0:6 --check` prints outputs 1.8e-13 and gradients 2.8e-11, with the same zero rows.

## Accuracy

Checked against tighter solves (scratch scripts, not committed):

- **Outputs**: LSODA at 1e-10 agrees with Radau at rtol = atol = 1e-13 to a relative 4e-9 for typical samples and 1.5e-6 for one with T(3400) ≈ 0.6.
- **Gradients**: central differences with h = 3e-4 at 1e-10 agree with the same differences at rtol = atol = 1e-12 to a relative 1e-5 (row norm, worst time, three samples) and to ≤ 6e-6 at the three rows checked below. h = 1e-4 gives 3–5e-5 (solver noise), and h = 1e-3 gives up to 4e-4 at late times (truncation), so 3e-4 is the step used.

The notebook's check cell re-solves five samples with `HIV_model.py` and prints the largest relative difference from `outputs.csv` (1.8e-13: the 13 significant digits written).

## How these files relate to the paper and to the 2016 files

- **Not the paper's computation.** The paper's results come from MATLAB `ode23s` and forward differences with step 1e-6 in x (pp. 717–718), on its own random samples; those scripts were posted at http://inside.mines.edu/~pankavic/activeHIV (paper p. 710; current availability not checked) and are not in this repository. The data here are an independent Python computation of the same model, inputs and ranges.
- **The 2016 files.** The 2016 notebook shipped CSVs written by Ryan Howard's 2016 `generate_data.py`: the same 1000 inputs, scipy `ode` (vode, BDF, default tolerances, `max_step = 1`), forward differences with step 1e-7 in x, at 21 times (also 5, 15 and 750 days; not 1700), as pandas CSVs with an index column. They are at the tag `legacy-master` (`HIV/`). Compared with the new files on the 18 times both have (1000 samples):
  - The same samples are at the floor at every time (6, 24, 61, 130, 283, 474 at t = 1800 … 3400; 2 at the new t = 1700).
  - **Outputs** above the floor: median relative difference 1.0e-7, 99th percentile 1.2e-5, maximum 5.5e-4. The ten differences above 1e-4 are all at samples with T < 3 near the floor, where relative error is magnified; at the three largest (rows 714, 51, 175), a Radau solve at 1e-13 agrees with the new value to ≤ 1.3e-5 and differs from the 2016 one by up to 5.5e-4. At the floor the 2016 values are 0.09999512–0.09999999 (vode stopped one step below 0.1), the new ones exactly 0.1.
  - **Gradients** above the floor (relative difference of the 27-vector): median 2.9e-5, 99th percentile 2.6e-3, maximum 0.16. The largest differences are at t = 38–90 (rows 197, 469, 482, 714, 952 exceed 1e-2); at rows 482, 197 and 469 the new gradient agrees with a solve at 1e-12 to ≤ 6e-6, and the 2016 one differs from it by 0.16, 0.05 and 0.04. So the differences are errors in the 2016 gradients (vode at its default rtol = 1e-6 under a forward step of 1e-7), not in the new ones.
  - At the floor the 2016 gradients have norms of about 0.1–66 (finite-difference noise); the exact value under the model is 0, which is what the new files hold.
  - The 2016 code cannot be rerun to reproduce its own files: under scipy 1.17 its `Tcells` differs from the 2016 outputs by up to 1.6e-2 at one sample.
  - The analysis is unchanged to the printed digits: on the 2016 files, v1 gives λ1 = 1.0461e4 and λ1/λ2 = 3957 at t = 24 (new files: 1.046e4, 3954), and λ1 = 4.5356e5 and λ1/λ2 = 853 at t = 3400 with the floor samples included (new: 4.535e5, 874; the ratio differs because the 2016 floor rows are noise rather than 0). The first eigenvectors (all samples, unoriented) differ by at most 9e-5 in any component at any of the 18 common times.

  The tolerance of 1e-5 that `docs/port-2026/HIV.md` §7 expected for reproducing the 2016 files holds for 99 % of the outputs but not for the gradients, for the reason above.
- **What changed in the files.** `inputs.csv` holds the same values (checked bit for bit by `generate_data.py` when it rewrote the file). `outputs.csv` and `gradients.csv` are new: written by `generate_data.py` at the 19 times, without the pandas index column, with a `#` header.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). Model and analysis: T. Loudon and S. Pankavich; 2016 notebook, model code and data: Ryan Howard. Cite T. Loudon and S. Pankavich, "Mathematical analysis and dynamic active subspaces for a long term model of HIV", Math. Biosci. Eng. 14(3):709–733, 2017, doi:10.3934/mbe.2017040.
