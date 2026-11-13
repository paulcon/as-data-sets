# HIV_Long_Term_Dynamics: data

Kind: **Python model**. The model is `HIV_model.py`, and `generate_data.py` writes `outputs.csv` and `gradients.csv` from it. The 1000 input samples in `inputs.csv` were drawn without a seed in 2016 and cannot be redrawn, so they are treated as stored data.

## Files

| file | contents | role |
|---|---|---|
| `HIV_Long_Term_Dynamics.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2017; outputs committed |
| `HIV_model.py` | `Tcells(p, times)`, `trajectory(q, times)`; `NAMES`, `NOMINAL`, `LOWER`, `UPPER`, `Y0`, `T_FLOOR` | the model |
| `generate_data.py` | writes `outputs.csv` and `gradients.csv` from `inputs.csv` (never rewrites `inputs.csv` here; `--check` and `--rows` need `--out` elsewhere) | regenerates the data |
| `check_accuracy.py` | recomputes the accuracy checks below (checks 1–2) | checks; 0.5–3.5 min on 4 processes in `.venv`, depending on the machine's load |
| `inputs.csv` | 1000 × 27 physical parameters, columns in the order of `NAMES` | stored inputs |
| `outputs.csv` | 1000 × 19: T(t) in cells mm⁻³ at the 19 times | model output |
| `gradients.csv` | 1000 × 513: column 27 i + j is ∂T(t_i)/∂x_j, with x the **normalized** inputs | model output |

The CSVs are plain comma-separated numbers after two `#` lines (a description and the column names), so `np.loadtxt(path, delimiter=',')` reads them. The 19 times, in days, are 24, 38, 40, 45, 50, 55, 65, 90, 140, 500, 1000, 1600, 1700, 1800, 2000, 2200, 2400, 2800, 3400: the 18 of the paper's Appendix B (Figs. 15–17) plus 1700, the paper's worked example (Figs. 2–6).

## Model

The seven ODEs of Loudon and Pankavich (2017), Eq. (1), with the 27 parameters of their Table 1.

- **Inputs**: independent uniform on nominal ± 2.5 % (paper p. 718). The normalized inputs are x = (2p − (upper + lower)) / (upper − lower) ∈ [−1, 1]²⁷ (paper Eq. (8)), and `gradients.csv` is ∂T/∂x. Divide by (upper − lower)/2 for gradients with respect to the physical parameters.
- **Initial conditions**: the same for every sample, by design: (T, T_I, T_L, M, M_I, CTL, V)(0) = (1000, 0, 0, 0.15/0.005, 0, 5/0.015, 1), the uninfected equilibrium at the nominal parameters plus one virion. The paper does not state its initial conditions.
- **Floor**: when T falls to 0.1 the state is frozen and T stays 0.1. The floor comes from the model code first written for this notebook (Ryan Howard, 2016), not from the paper. `HIV_model.py` implements it as a terminal event of `solve_ivp`. At a time after a sample reaches the floor, its output is 0.1 and its gradient is 0.
- **Solver**: `scipy.integrate.solve_ivp`, LSODA, rtol = atol = 1e-10, analytic Jacobian; 0.04–0.08 s per 3400-day trajectory.
- **Gradients**: central differences in x with step h = 3e-4, so 55 trajectories per sample, each giving all 19 times; set to 0 at times after the floor.

## Regenerating

```bash
python generate_data.py                                  # all 1000 samples, 3 processes
python generate_data.py --check                          # samples 0-49, compared with the files here
python generate_data.py --check --full                   # all 1000 samples, compared with the files here
python generate_data.py --rows 0:20 --check              # another subset
```

There is no randomness, so a run reproduces the files. Samples are independent; `--workers` changes only the run time. The run that wrote the shipped files (2026-10-30, scipy 1.17.1, numpy 2.4.6, Python 3.11) took 2000 s (33 min) on three processes of an 8-core Mac that other jobs were also using; that is about 4 s of CPU per sample (55 trajectories of 0.04–0.08 s), so about 70 min on one core. A 30-sample subset run took 54 s on three processes. A rerun reproduces the files to the digits written; `--rows 0:6 --check` prints outputs 1.8e-13 and gradients 2.8e-11, with the same zero rows. On 2026-11-12 (`--workers 2`), `python generate_data.py --check` (samples 0–49) took 105 s and `--check --full` 1458 s, with largest relative differences of 4.1e-13 (outputs, elementwise) and 3.6e-11 (gradients, 2-norm of each row) over all 1000 samples, on macOS (Python 3.11, numpy 2.4.6, scipy 1.17.1). `--check` prints these differences and does not judge them; it exits with status 1 only for a structural mismatch: a different shape or NaN pattern, or a different set of zero-gradient rows.

## Accuracy

Checked against tighter solves:

- **Outputs** (`check_accuracy.py`, check 1): LSODA at 1e-10 agrees with Radau at rtol = atol = 1e-13 to a relative 3e-9 to 3.5e-7 at rows 0–4 (4e-9 at rows 0, 1 and 4), and to at most 1.1e-5 at rows 714, 51 and 175, whose T comes close to the floor.
- **Gradients** (`check_accuracy.py`, check 2): central differences with h = 3e-4 at 1e-10 agree with the same differences at rtol = atol = 1e-12 to a relative 1e-5 (9.0e-6 to 1.6e-5; row norm, worst time) at rows 0, 1, 2 and 714 and at rows 482, 197 and 469. At rows 0, 1 and 2, h = 1e-4 gives 2–3e-5 (solver noise) and h = 1e-3 gives 3e-6. At row 714, whose T(3400) = 0.23 is just above the floor, h = 1e-4 and h = 1e-3 give 2e-2 and 4e-2 at t = 3400, against 1.6e-5 for h = 3e-4. So 3e-4 is the step used.

The notebook's check cell re-solves five samples with `HIV_model.py` and prints the largest relative difference from `outputs.csv` (1.8e-13: the 13 significant digits written).

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from MATLAB `ode23s` and forward differences with step 1e-6 in x (pp. 717–718), on its own random samples; those scripts were posted at http://inside.mines.edu/~pankavic/activeHIV (paper p. 710; current availability not checked) and are not in this repository. The data here are an independent Python computation of the same model, inputs and ranges.

## Unknowns

- The paper does not state its initial conditions; the ones used here are this data set's choice (Model).
- The 1000 input samples were drawn without a seed in 2016 and cannot be redrawn.
- The current availability of the paper's MATLAB scripts (http://inside.mines.edu/~pankavic/activeHIV) was not checked.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). Model and analysis: T. Loudon and S. Pankavich; 2016 notebook, model code and data: Ryan Howard. Cite T. Loudon and S. Pankavich, "Mathematical analysis and dynamic active subspaces for a long term model of HIV", Math. Biosci. Eng. 14(3):709–733, 2017, doi:10.3934/mbe.2017040.
