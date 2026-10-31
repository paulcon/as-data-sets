# PDE_Inverse_Problem_MCMC: data

Kind: **Python model**. The model is `pde_model.py` (class `PDEMisfit`). `generate_data.py` writes the shipped file `pde_mcmc_chains.npz` from saved seeds, and reproduces it. The KL basis `kl-short.npz` is an input that cannot be regenerated, so it is treated as stored data.

## Files

| file | contents | role |
|---|---|---|
| `PDE_Inverse_Problem_MCMC.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2016; outputs committed; loads the chains, does not rerun them |
| `pde_model.py` | `PDEMisfit`: `observations`, `set_data`, `misfit`, `misfit_grad` (and `PDESolver` for the Elliptic_PDE QoI) | the model; byte-identical to `../Elliptic_PDE_Random_Coefficients/pde_model.py` |
| `as_mcmc.py` | `random_walk_metropolis`; `ActiveSubspaceMCMC` (Metropolis–Hastings on the active variables with ĝ recomputed at every step or kept with the state; reconstruction (4.2)); `chain_diagnostics` (τ, ESS, split R̂); `batch_means_se` | the samplers and diagnostics; numpy only |
| `generate_data.py` | writes `pde_mcmc_chains.npz` | regeneration |
| `tune_proposal.py` | the pilot runs that chose the active-variable proposal | tuning |
| `pde_mcmc_chains.npz` | inputs, subspace, chains and grid, listed below (24.0 MB) | generated |
| `kl-short.npz` | `K` (10000, 100) float64, `lam` (100,) float64 | KL basis (stored); see `../Elliptic_PDE_Random_Coefficients/DATA.md` |

## Model

`-∇·(a ∇u) = 1` on [0, 1]², u = 0 on the left, top and bottom sides, zero flux on the right side, `log a = B x` with `B = K * sqrt(lam)` and x ~ N(0, I_100). The KL kernel is exp(−‖s − t‖₁/0.02), which is the paper's eq. (5.5) with β = 0.02. The discretization (five-point vertex-centred finite volumes on 100 × 100 cells, sparse LU, discrete adjoint for the gradient) is described in `../Elliptic_PDE_Random_Coefficients/DATA.md`.

- **Observations**: u at the unknown rows `indz = [19, 29, …, 79]` of the last unknown column, i.e. s2 = 0.2, 0.3, …, 0.8 at s1 = 0.99. The paper puts the sensors on the boundary, s1 = 1. The 2016 code used rows 20, …, 80 (s2 = 0.21, …, 0.81).
- **Misfit**: f(x) = ‖d − m(x)‖²/(2σ²), with gradient from one adjoint solve.
- **Check**: `python uq_benchmarks/check_pde_model.py`.

## What was run (author's decisions, 2026-10-30)

| sampler | chains × steps | proposal | ĝ | solves |
|---|---|---|---|---|
| vanilla | 20 × 100,000 | x' = x + 0.1 e (paper) | — | 2 × 10⁶ |
| cached | 4 × 16,000 | y' = y + 1.2 L e | M = 10, kept with the state (Algorithm 1 as printed; pseudo-marginal, exact for π_M) | 6.4 × 10⁵ |
| recompute | 4 × 5,000 | y' = y + 1.7 L e | M = 10, recomputed at the current state and the proposal every step (the 2016 code behind the paper's results; noisy Metropolis–Hastings) | 4 × 10⁵ |

- **Starts and burn-in.** Every chain starts at its own prior draw, and the notebook discards the first 20%.
- **The factor L** is the Cholesky factor of the posterior covariance of (y1, y2) from vanilla chain 1 after burn-in, rounded to 4 decimals: [[0.2438, 0], [−0.1044, 0.3561]].
- **The scales s** come from the pilot below and are fixed during the stored runs.
- **Chain lengths** come from the pilot autocorrelation times τ: 4 × steps × 0.8 / τ ≥ 1000 × 1.5 for y1 and y2 in each active-variable sampler.
- **Vanilla chains.** One chain of 100,000 steps gives an ESS of about 350 for y1, y2 (τ ≈ 230 steps) after burn-in. The first run had 4 vanilla chains: ESS about 1100–1400 for y1, y2, but split R̂ = 1.011 for y2, and the spread of the 4 chain means suggested a standard error twice the computed 0.011. For the 100 x-moments the smallest ESS was 290 and the largest R̂ 1.027. The author added 16 chains (2026-10-31; the 10 MB cap on this file was lifted for them). With 20 chains, the ESS of y1, y2 is about 7000, split R̂ ≤ 1.004, the spread of the 20 chain means agrees with the computed standard errors, and the smallest ESS of the x-moments is 1761 (largest R̂ 1.016).

**Pilot (`tune_proposal.py`).** Each run is one chain of 2,500 steps started at the vanilla posterior mean, with τ measured after 300 steps. The proposal used L from the unrounded factor. Seeds were 100–107, from a scratch version of the script; the shipped script uses `SeedSequence([2779, 1])` and the rounded L, so a rerun differs statistically.

| sampler | s | acceptance | τ(y1) | τ(y2) |
|---|---|---|---|---|
| recompute | 0.50 | 0.590 | 18.4 | 35.8 |
| recompute | 0.85 | 0.490 | 15.3 | 20.7 |
| recompute | 1.20 | 0.419 | 8.5 | 13.7 |
| recompute | **1.70** | 0.315 | 8.1 | 9.9 |
| cached | 0.50 | 0.286 | 42.6 | 81.4 |
| cached | 0.85 | 0.244 | 32.1 | 34.8 |
| cached | **1.20** | 0.159 | 14.5 | 33.0 |
| cached | 1.70 | 0.107 | 17.4 | 71.8 |

For comparison, the first version of this file ran one chain of each sampler with the paper's isotropic proposal, standard deviation 0.1, 10,000 steps. τ for (y1, y2) was:
- recompute: (87, 217) steps, ESS (91, 37) after burn-in;
- cached: (57, 126) steps, ESS (141, 63).

## `pde_mcmc_chains.npz`

Load with `np.load` (no pickle). Indices are 0-based; `k` is the chain: 1, …, 20 for vanilla, 1, …, 4 for cached and recompute.

| key | shape, dtype | contents |
|---|---|---|
| `seed`, `streams` | (), (33,) str | master seed 2779. `np.random.SeedSequence(seed).spawn(33)` gives one generator per stream, in the order of `streams`: `data`, `grad_points`, `bootstrap`, `vanilla_1`, `unused` (the first versions' active-variable chain), `grid`, `vanilla_2..4`, `cached_1..4`, `recompute_1..4`, `vanilla_5..20`. Child i of `spawn` has the same seed whatever the number of children, so appending `vanilla_5..20` left every earlier stream unchanged |
| `x_true`, `d`, `sig2` | (100,), (7,), () | synthetic data: x_true ~ N(0, I), d = m(x_true) + e, e ~ N(0, σ² I), σ² = 1e-4 ‖m(x_true)‖² |
| `indz`, `sensor_s1`, `sensor_s2` | (7,) | sensor rows and coordinates |
| `kl_md5` | () str | md5 of `kl-short.npz` (079370cfa2411de13b20ab62c7b22879) |
| `X_grad`, `G` | (1000, 100) float64 | prior samples and the misfit gradients at them |
| `W`, `eigenvalues` | (100, 100), (100,) | the active subspace of `G` (v1 `active_subspace`, n = 2, `orient` so that the first component of w1 and w2 is positive). The active-variable chains used `W[:, :2]` and `W[:, 2:]` |
| `n_active`, `M`, `burn_in`, `van_prop_std` | () | 2; 10; 0.2 (the burn-in fraction the notebook discards); 0.1 |
| `av_chol`, `av_scale_cached`, `av_scale_recompute` | (2, 2), (), () | the active-variable proposal: y' = y + s L e |
| `thin_proj`, `thin_x`, `trace_index` | () | 10, 50, (9, 89) |
| `vanilla_k_x0`, `vanilla_k_last` | (100,) | first and last state |
| `vanilla_k_proj` | (10000, 4) float32 | `(X @ W[:, :4])`, the first four components of Wᵀx, at every 10th step |
| `vanilla_k_thin` | (2000, 100) float32 | every 50th state |
| `vanilla_k_misfit` | (10000,) | f at every 10th step |
| `vanilla_k_accept` | (100000,) bool | whether the proposal of step j was accepted |
| `vanilla_1_trace` | (100000, 2) | x_10 and x_90 at every step of chain 1 |
| `cached_k_y0`, `cached_k_y`, `cached_k_ghat`, `cached_k_accept` | (2,), (16000, 2), (16000,), (16000,) bool | start, states, ĝ at the state used in the test, flags |
| `recompute_k_*` | the same with 5000 steps | ĝ at the current state, recomputed at every step |
| `grid_y1`, `grid_y2` | (21,) | grid axes: center (0.64, −0.62), half-widths (1.0, 1.7), about 5 cached-chain standard deviations |
| `grid_f_fine` | (21, 21, 160) float32 | f(W1 y + W2 z) at every grid point for the first 160 common draws z |
| `grid_f_coarse` | (11, 11, 2560) float32 | the same at every other grid point for all 2560 draws |
| `wall`, `wall_total`, `versions` | | wall times of the jobs and of the run, package versions |

The full 100-dimensional vanilla chains (80 MB each as float64) are not stored. The notebook needs only the reductions above, and the chains are reproducible.

## Regenerating

```bash
cd uq_benchmarks/PDE_Inverse_Problem_MCMC
OPENBLAS_NUM_THREADS=1 python generate_data.py --workers 16   # writes pde_mcmc_chains.npz (one BLAS thread per worker)
python generate_data.py --n-van 2500 --n-cached 120 --n-recompute 60 --grid-draws 20 \
    --workers 2 --out /tmp/short.npz            # a short check
python tune_proposal.py --workers 6             # the pilot (needs pde_mcmc_chains.npz)
```

- **Jobs.** Every chain and every block of grid points is a separate job with its own seed stream. `--workers` sets how many run at once, and the output does not depend on it. `--reuse FILE` copies chains of the same seed, data and length from FILE.
- **BLAS.** The BLAS, and its number of threads, change the last bits of the matrix products (the gradients, W and every ĝ). The shipped file was written on Linux with one OpenBLAS thread per worker (`OPENBLAS_NUM_THREADS=1`); `versions` in the file lists the platform and the package versions. Elsewhere a rerun agrees to rounding (below). Set one BLAS thread per worker whenever several workers run: with OpenBLAS's default of one thread per CPU in every worker, 16 workers on 16 vCPUs ran about 7 times slower.
- **Runtime.** About 3.4 × 10⁶ PDE solves (2 × 10⁶ vanilla, 6.4 × 10⁵ cached, 4 × 10⁵ recompute, 3.6 × 10⁵ grid). On an AWS c7a.4xlarge (16 vCPUs, AMD EPYC 9R14) with 16 workers, one vanilla chain or recompute chain takes about 29 min and one cached chain about 43 min; the 12 original chains and the grid took 43.8 min, and vanilla chains 5–20 another 30.6 min. On a 4-core Intel Mac with 6 workers, the 12 original chains and the grid took 179 min.
- **How the shipped file was made** (2026-10-31, on the AWS instance above: Amazon Linux 2023, Python 3.11.18, numpy 2.4.6, scipy 1.17.1). One run computed the 12 original chains and the grid from scratch; a second run added vanilla chains 5–20 with `--reuse` of the first run's output, which copies the first 12 chains and the grid unchanged.
- **Seeds.** Every random draw comes from the streams above, drawn one step at a time in a fixed order, so a shorter run reproduces the first steps of a longer one. The grid draws are one block, so the first K draws of a shorter run equal the first K of a longer one.
- **Reproduction checks** (2026-10-30 and 2026-10-31):
  - The 12 original chains and the grid were also run in full on a Mac (macOS 14, Python 3.11.16, the same numpy and scipy, 6 workers). All 87 fields agree with the shipped file to rounding (at most 4 × 10⁻¹³ relative, 1.9 × 10⁻⁸ for the float32 grid), and 54 are bitwise equal, including every accept/reject array: the same seeds give the same chains, so this run is a reproduction, not an independent sample. Its vanilla chain 1 was copied from the first version of this file, so that version is reproduced as well.
  - A short run on the Mac (`--n-van 1000 --n-cached 100 --n-recompute 50 --grid-draws 20`) matches the start of all 28 chains and the grid: 121 of 160 fields bitwise, the rest to at most 5.5 × 10⁻¹³ relative, every accept array bitwise.
  - Short runs with `--workers 1` and `--workers 6` agree bitwise, and a short run of the 20-chain generator reproduces every field of the 12-chain generator bitwise.
  - The notebook reruns the first steps of chain 1 of each sampler and prints the differences, which are at the rounding level (at most 5 × 10⁻¹² in the misfit, 0 in the accept flags).
  - An accept/reject decision flips only if a log acceptance ratio falls within rounding of log t. No decision flipped in the runs above, but very long chains on another machine could eventually part.

## Results (printed by the notebook)

Mean ± standard error of (y1, y2) = W1ᵀx; standard errors from the autocorrelation (`chain_diagnostics`), and for the grid densities from a bootstrap over the draws.

| | τ (steps) y1, y2 | ESS y1, y2 | split R̂ | acceptance | y1 | y2 |
|---|---|---|---|---|---|---|
| vanilla (posterior), 20 chains | 229, 235 | 6990, 6802 | ≤ 1.004 | 0.547 (paper 60%) | 0.739 ± 0.003 | −0.829 ± 0.005 |
| cached, 4 chains | 26.8, 24.8 | 1910, 2068 | 1.002 | 0.167 | 0.657 ± 0.004 | −0.664 ± 0.007 |
| recompute, 4 chains | 8.9, 10.2 | 1804, 1569 | 1.002 | 0.313 (paper 62%, other proposal) | 0.643 ± 0.006 | −0.636 ± 0.011 |
| π_ε (grid) | | | | | 0.629 ± 0.003 | −0.601 ± 0.005 |
| π_M, M = 10 (grid) | | | | | 0.649 ± 0.005 | −0.651 ± 0.006 |

- **Cached vs π_M** (the test of spec §9: |offset| ≤ 3 combined SE): +1.3 and −1.4 SE, pass. Recompute vs π_M: −0.7 and +1.2 SE.
- **Active-variable chains vs the posterior:** the cached chains are off by −0.082 and +0.165 (−16 and +20 SE), the recompute chains by −0.095 and +0.193 (−15 and +16 SE), 0.35–0.5 posterior standard deviations. This is the approximation error of the two-dimensional subspace (λ3 + … + λ100 = 124, against λ1 + λ2 = 1934). The data also inform y3 (posterior mean −1.28, standard deviation 0.53), which the active-variable samplers draw from the prior.
- **Moments of x** (the paper's Fig. 5.9): 44% (cached) and 45% (recompute) of the 100 posterior means, and 84% and 82% of the variances, fall inside the vanilla 99% intervals (median half-widths 0.054 and 0.053). Of the squared difference of the means, 65% lies along w3 and 99% along w3, …, w100.
- **Grid convergence:** halving the spacing and shrinking the extent change the means by less than 10⁻⁴; doubling the draws changes them by 0.003, consistent with the grid's bootstrap standard errors (0.003–0.006).

## Relation to the paper's runs

Constantine, Kent & Bui-Thanh (2016) §5.2–5.6 ran one vanilla chain of 500,000 steps and two active-variable chains of 50,000 steps with the numpad model on the Mio cluster; their data, seeds and chains are not in this repository. The 2016 `.mat` files that this file replaces (`Vanilla_Chain.mat`, `Vanilla_Acceptance.mat`, `AV_Chain.mat`, `AV_Acceptance.mat`, `Gradient_Samples.mat`; 2016-07-12/13) were a smaller rerun at 10⁵ / 10⁴ steps, without seeds or synthetic data saved; they are removed from `main` but remain in the git history. Departures from the paper (all listed in the notebook):

1. different synthetic data and sensors at s1 = 0.99, s2 = 0.2, …, 0.8;
2. a finite-volume solver, not the paper's finite differences;
3. two active-variable samplers: Algorithm 1 as printed (cached ĝ), and the 2016 code's variant (recomputed ĝ) that produced the paper's numbers;
4. a tuned proposal on y (full covariance, scale from a pilot), not the paper's isotropic standard deviation 0.1 (printed as "variance") or 0.3;
5. 20 vanilla chains and 4 chains per active-variable sampler at the lengths above, with split R̂ and autocorrelation-based standard errors (the paper: one chain each, batch means);
6. reference densities π_ε and π_M on a grid (not in the paper).

The vanilla proposal (standard deviation 0.1), M = 10, P = 10 and the 20% burn-in are as in the paper.

## License and credit

Code: MIT (`LICENSE`). Data (`pde_mcmc_chains.npz`, `kl-short.npz`): CC BY 4.0 (`LICENSE-DATA`). Notebook by Ryan Howard (2016). Cite P. G. Constantine, C. Kent and T. Bui-Thanh, "Accelerating Markov chain Monte Carlo with active subspaces", SIAM J. Sci. Comput. 38(5):A2779–A2805, 2016, doi:10.1137/15M1042127.
