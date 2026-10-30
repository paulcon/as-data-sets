# PDE_Inverse_Problem_MCMC: data

Kind: **Python model**. The model is `pde_model.py` (class `PDEMisfit`). `generate_data.py` writes the shipped chain file `pde_mcmc_chains.npz` from saved seeds, and reproduces it. The KL basis `kl-short.npz` is an input that cannot be regenerated, so it is treated as stored data.

## Files

| file | contents | role |
|---|---|---|
| `PDE_Inverse_Problem_MCMC.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2016; outputs committed; loads the chains, does not rerun them |
| `pde_model.py` | `PDEMisfit`: `observations`, `set_data`, `misfit`, `misfit_grad` (and `PDESolver` for the Elliptic_PDE QoI) | the model; byte-identical to `../Elliptic_PDE_Random_Coefficients/pde_model.py` |
| `as_mcmc.py` | `random_walk_metropolis`, `ActiveSubspaceMCMC` (Algorithm 1, reconstruction (4.2)), `batch_means_se` | the samplers; numpy only |
| `generate_data.py` | the script that writes `pde_mcmc_chains.npz` | regeneration |
| `pde_mcmc_chains.npz` | inputs, subspace and chains, listed below (7.6 MB) | generated |
| `kl-short.npz` | `K` (10000, 100) float64, `lam` (100,) float64 | KL basis (stored); see `../Elliptic_PDE_Random_Coefficients/DATA.md` |

## Model

`-∇·(a ∇u) = 1` on [0, 1]², u = 0 on the left, top and bottom sides, zero flux on the right side, `log a = B x` with `B = K * sqrt(lam)` and x ~ N(0, I_100). The KL kernel is exp(−‖s − t‖₁/0.02), which is the paper's eq. (5.5) with β = 0.02. The discretization (five-point vertex-centred finite volumes on 100 × 100 cells, sparse LU, discrete adjoint for the gradient) is described in `../Elliptic_PDE_Random_Coefficients/DATA.md`.

- **Observations**: u at the unknown rows `indz = [19, 29, …, 79]` of the last unknown column, i.e. s2 = 0.2, 0.3, …, 0.8 at s1 = 0.99. The paper puts the sensors on the boundary, s1 = 1. The 2016 code used rows 20, …, 80 (s2 = 0.21, …, 0.81).
- **Misfit**: f(x) = ‖d − m(x)‖²/(2σ²), with gradient from one adjoint solve.
- **Check**: `python uq_benchmarks/check_pde_model.py`.

## `pde_mcmc_chains.npz`

Load with `np.load` (no pickle). Indices are 0-based.

| key | shape, dtype | contents |
|---|---|---|
| `seed`, `streams` | (), (5,) str | master seed 2779; `np.random.SeedSequence(seed).spawn(5)` gives one generator per stream, in this order: `data`, `grad_points`, `bootstrap`, `vanilla`, `active` |
| `x_true`, `d`, `sig2` | (100,), (7,), () | synthetic data: x_true ~ N(0, I), d = m(x_true) + e, e ~ N(0, σ² I), σ² = 1e-4 ‖m(x_true)‖² |
| `indz`, `sensor_s1`, `sensor_s2` | (7,) | sensor rows and coordinates |
| `kl_md5` | () str | md5 of `kl-short.npz` (079370cfa2411de13b20ab62c7b22879) |
| `X_grad`, `G` | (1000, 100) float64 | prior samples and the misfit gradients at them |
| `W`, `eigenvalues` | (100, 100), (100,) | the active subspace of `G` (v1 `active_subspace`, n = 2, `orient` so that the first component of w1 and w2 is positive). The active-variable chain used `W[:, :2]` and `W[:, 2:]` |
| `n_active`, `prop_std`, `M`, `burn_in`, `thin`, `trace_index` | () | 2; 0.1 (proposal standard deviation, both chains); 10 (misfit evaluations per ĝ); 0.2 (the burn-in fraction the notebook discards); 10; (9, 89) |
| `van_x0` | (100,) | start of the vanilla chain, a prior draw |
| `van_trace` | (100000, 2) float64 | x_10 and x_90 at every vanilla step |
| `van_proj` | (100000, 4) float32 | `X @ W[:, :4]`, the first four components of Wᵀx at every vanilla step |
| `van_thin` | (10000, 100) float32 | every 10th vanilla state |
| `van_misfit` | (100000,) float64 | f at every vanilla state |
| `van_accept` | (100000,) bool | whether the proposal made at step k was accepted |
| `van_last` | (100,) | the last vanilla state |
| `av_y0`, `av_y` | (2,), (10000, 2) | start (a prior draw) and states of the active-variable chain |
| `av_ghat`, `av_accept` | (10000,), (10000,) bool | ĝ at each state; acceptance flags |
| `wall_*`, `versions` | | wall times (s) of the run and the package versions |

The full 100-dimensional vanilla chain (80 MB as float64) is not stored. The notebook needs only the reductions above, and the full chain is reproducible.

## Regenerating

```bash
cd uq_benchmarks/PDE_Inverse_Problem_MCMC
python generate_data.py                       # writes pde_mcmc_chains.npz
python generate_data.py --n-van 2000 --n-av 200 --out /tmp/short.npz   # a short check
```

- **Steps**: synthetic data; 1000 misfit gradients and the subspace (20 to 50 s); then the vanilla chain (100,000 steps of random-walk Metropolis–Hastings on x) and the active-variable chain (10,000 steps of Algorithm 1 with ĝ cached at the current state, M = 10). Each chain costs 10⁵ forward solves. The two chains run in two processes (`--workers 1` runs them one after the other; the result is the same).
- **Runtime**: the shipped file took 47 minutes (2817 s, stored as `wall_total`) on a 4-core Intel laptop (i7-1068NG7) shared with other jobs, 16 to 50 ms per solve depending on the load. With the two chains in parallel on an idle machine, expect about 40 minutes.
- **Seeds**: every random draw comes from the streams above, drawn one step at a time in a fixed order, so a shorter run reproduces the first steps of a longer one.
- **Reproduction check**: `python generate_data.py --n-van 2000 --n-av 200` reproduced every array of the shipped file bitwise, taken to the same length (the data, `X_grad`, `G`, `W`, the first 2000 vanilla and 200 active-variable steps of every chain array, and the first 200 rows of `van_thin`), except the wall times and `van_last` (2026-10-30, on the machine that wrote the file; `--workers 1` and `--workers 2` also agree bitwise). The full 47-minute run was not repeated. The notebook reruns the first 1000 vanilla and 100 active-variable steps and prints the differences, which are 0. On another machine, BLAS and SuperLU rounding may differ in the last bits; an accept/reject decision then flips only if a log acceptance ratio falls within rounding of log t, so short reruns should agree to rounding, while long chains may eventually part.

## Relation to the paper's runs

Constantine, Kent & Bui-Thanh (2016) §5.2–5.6 ran 500,000 vanilla steps and 50,000 active-variable steps (two variants) with the numpad model on the Mio cluster; their data, seeds and chains are not in this repository. The 2016 `.mat` files that this file replaces (`Vanilla_Chain.mat`, `Vanilla_Acceptance.mat`, `AV_Chain.mat`, `AV_Acceptance.mat`, `Gradient_Samples.mat`; 2016-07-12/13) were a smaller rerun at the present scale, without seeds or synthetic data saved, so they could not be checked; they are removed from `main` but remain in the git history.

Differences from the paper and the 2016 files:

- different synthetic data (x_true, d) and sensors at s1 = 0.99, s2 = 0.2, …, 0.8;
- a different discretization (finite volumes, not the paper's finite differences);
- ĝ is cached at the current state, as in the paper's Algorithm 1. The 2016 code recomputed it at the current state every step with fresh draws. That variant costs twice as many solves and gives a higher acceptance rate. On the shipped data and W, a 300-step test of the recompute variant started at the posterior mean accepted 0.58, against 0.289 for the shipped (cached) chain. With M = 10, ĝ at the posterior mean has a standard deviation of about 1.5 (printed in the notebook), and a state whose ĝ came out low holds the chain. The paper's 62% and the 2016 file's 0.615 are in line with the recompute variant;
- 20% burn-in, as in the paper; the 2016 notebook used none.

## License and credit

Code: MIT (`LICENSE`). Data (`pde_mcmc_chains.npz`, `kl-short.npz`): CC BY 4.0 (`LICENSE-DATA`). Notebook by Ryan Howard. Cite P. G. Constantine, C. Kent and T. Bui-Thanh, "Accelerating Markov chain Monte Carlo with active subspaces", SIAM J. Sci. Comput. 38(5):A2779–A2805, 2016, doi:10.1137/15M1042127.
