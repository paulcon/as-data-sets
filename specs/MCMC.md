# MCMC: update spec

Kind: **Python model**: the shared `pde_model.py`; `generate_data.py` writes the chains with saved seeds (see [README.md](README.md#two-kinds-of-data-sets)).

Status line: `MCMC/MCMC.ipynb` (39 cells, kernel `python2`; also on origin/dev, where master is newer). Paper: Constantine, Kent & Bui-Thanh, "Accelerating Markov chain Monte Carlo with active subspaces", SIAM J. Sci. Comput. 38(5):A2779–A2805, 2016, doi:10.1137/15M1042127 (arXiv:1510.00024).
- **Two examples:**
  - quadratic, m = 2;
  - PDE, m = 100 KL coefficients.
- **QoI:** data misfit f(x) = ‖d − m(x)‖²/(2σ²) with a N(0, I) prior.
- **Legacy API:** `Subspaces.compute/partition`, `asub.Subspaces` (exists only on legacy-dev), `utils.simrunners.SimulationGradientRunner`, `utils.plotters.{eigenvalues,subspace_errors,eigenvectors,sufficient_summary}`.
- **py2 issues:** yes. `print` statements in cells 8, 11, 20, 28 and 33, and in `generate_data.py` and `pde_mcmc_study.py`.
- **Hard dependency:** numpad.
- **Data:** an 80 MB `Vanilla_Chain.mat`.

## 1. Summary

The notebook demonstrates Algorithm 1 of the paper (Metropolis–Hastings on the active variables, prior samples on the inactive ones) on the paper's two examples.

What I checked:
- **Data loading and subspace.** The stored data load with scipy only. v1 `active_subspace` on `Gradient_Samples.mat` reproduces the legacy eigenvalue figure exactly: 1093, 222, 24.6, 7.8.
- **Live model.** The numpad-free solver reproduces the 2-D structure with fresh synthetic data.

Main problems:
1. **Orphaned stored chains.** They cannot be reproduced or validated. The synthetic data (x_true, d, σ²), the points where the gradients were taken and the seeds were never saved.
2. **Wrong reconstruction of `X_AS` (cell 35).** It maps the stored `AV_Chain` through a subspace from *different* data (cell 23). Under v1 the second eigenvector also flips sign relative to the legacy W used to generate `AV_Chain` (confirmed below).
3. **Code differs from the paper's Algorithm 1.**
   - The sampler recomputes ĝ at the current state every step (fresh z), which doubles cost and changes the algorithm.
   - The quadratic example uses Monte Carlo where the paper uses 10-point Gauss–Hermite.
   - Proposal "0.1" is a standard deviation in the code. The paper says "variance", but acceptance rates show the paper also used std 0.1, so the paper's wording is wrong, not the code.
   - No burn-in.
4. **Math transcription errors.** c_pos⁻¹ is written for c_pos; L replaces C instead of multiplying it. The gradient sign is wrong (copied from published eq. (3.5)). Published paper errata, confirmed against the paper's own figures and tables: σ² = 0.1 should be σ = 0.1 (quadratic example), and the sign in eq. (3.5).
5. **`Vanilla_Chain.mat` is 80 MB** (over GitHub's 50 MB warning). The notebook needs < 4 MB of it.

Solver: see `specs/numpad_replacement.md` (class `PDEMisfit`). This spec covers the notebook-level plan only.

## 2. Inventory

| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| `MCMC.ipynb` | 1,187,941 B | nbformat 4, python2 | case study | — | Execution counts out of order: cells 1:6, 3:2, 19:2, 33:3, 37:7 vs 20:13 … 31:20. Saved outputs come from ≥ 2 kernel sessions. |
| `pde_model.py` | 2,488 B | py source | `PDESolver` with `observations`, `set_data`, `misfit`, `misfit_grad` | `import pde_model as pdem` | `from numpad import *` (shadows numpy names), `import pdb`. Solver code identical to `Elliptic_PDE/pde_model.py`; different QoI. |
| `pde_mcmc_study.py` | 5,925 B | py2 | `ActiveSubspaceAcceleratedMCMC`, `compute_misfit_active_subspace`, `__main__` demo | `from pde_mcmc_study import *` | See §4. |
| `generate_data.py` | 2,705 B | py2 script | produced the 5 `.mat` files (2016-07-12/13) | not imported | No seed. Saves neither x_true nor d nor σ² nor gradient points. |
| `kl-short.npz` | 8,001,158 B | npz `K` (10000,100), `lam` (100,) | KL basis, exp(−‖s−t‖₁/0.02) | `np.load` (CWD-relative) | Byte-identical to `Elliptic_PDE/kl-short.npz`. |
| `Vanilla_Chain.mat` | 80,000,192 B | MAT v5, `Chain` (100000,100) f8 | vanilla MH chain, prop. std 0.1 | `scipy.io.loadmat` | 54,561 unique rows. Starts at a random x (‖x‖ = 9.4); no burn-in. |
| `Vanilla_Acceptance.mat` | 800,192 B | MAT v5, `Accept` (100000,1) f8 of 0/1 | acceptance flags | loadmat | Redundant: equals `any(Chain[1:] != Chain[:-1])`, checked. Mean 0.5456. |
| `AV_Chain.mat` | 160,192 B | MAT v5, `Chain` (10000,2) f8 | active-variable chain y (legacy W sign convention) | loadmat | Mean (0.138, 1.358), std (0.25, 0.49). |
| `AV_Acceptance.mat` | 80,192 B | MAT v5, `Accept` (10000,1) f8 | flags | loadmat | Mean 0.6152. |
| `Gradient_Samples.mat` | 800,192 B | MAT v5, `Samples` (1000,100) f8 | misfit gradients used to build the AV chain's subspace | loadmat | The x at which they were taken are not stored. |

## 3. Text and math review (vs. literature)

Source: the published version, SISC 38(5):A2779–A2805 (PDF), read §2–5 and App. A. Equation, page, figure and table numbers below refer to it. The published numbering is one section higher than arXiv's (published §3 = arXiv §2), and the content agrees with the arXiv source in every item checked. The Constantine 2015 book discusses posterior approximation and MH on the active variables generically (§4.5 "Inversion", around p. 67), but does not contain this example.

| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| 0 | "$c_{\text{pos}}^{-1} = \int\rho_{\text{lik}}(\mathbf d,\ \mathbf x)\ \rho_{\text{pri}}(\mathbf x)\ d\mathbf x$" | Published eq. (3.3), p. A2784: ρ_pos = c_pos⁻¹ ρ_lik ρ_pri with c_pos = ∫ρ_lik ρ_pri. | Drop the ⁻¹ on the left. | confirmed |
| 0 | "$c_\pi^{-1} = \int\exp(-g(\mathbf W_1^T\mathbf x))\rho_{\text{pri}}(\mathbf x)\ d\mathbf x$" | Same error. Published eqs. (3.7)–(3.8), p. A2785: c_π = ∫exp(−g)ρ_pri. | Same fix. | confirmed |
| 0 | "has the same formula as the error bounds … replacing $C$ with $L$" and "$H(\rho_{\text{pos}},\hat\pi_\epsilon) \leq L(1+M^{-1/2})(…)$" | Published Theorem 3.1, eqs. (3.10)–(3.13), p. A2786: H ≤ **L C**(…). L is defined in (3.9), p. A2785, and the notebook's L² matches it. Numerically harmless, because C = 1 for a Gaussian prior. | Say "multiplying by L"; write L C in the example. | confirmed |
| 0 | "we choose … $\rho$ to be the prior likelihood of $\mathbf x$"; "Then we an construct" | Should be "prior density"; typo "we an". | Fix. | confirmed |
| 0 | Algorithm steps 1–7 (ĝ(y₁) computed once; only ĝ(y′) per step) | The text matches published Algorithm 1, p. A2788: "Pick an initial value ŷ1, and compute ĝε(ŷ1)". Step 2 computes only ĝε(ŷ′), and the ratio (4.1) reuses ĝε(ŷk). The text below the algorithm says "each step … uses M forward model evaluations". The **code** (cell 11 `g(y_MCMCAS[:,i-1])`; `pde_mcmc_study.sample_active_variables` line 61) recomputes ĝ at the current state each step with fresh z. That is a Monte-Carlo-within-Metropolis variant, not Algorithm 1, and it costs 2M solves/step instead of M. | Cache ĝ at the current state (§4). | confirmed |
| 0 | References: titles, volumes, pages, DOIs | Paper 1 matches the published version: title, 38(5):A2779–A2805, 2016, DOI 10.1137/15M1042127 (p. A2779). Paper 2's title lacks "methods" (published: "Active subspace **methods** in theory and practice: applications to kriging surfaces", 36(4):A1500–A1524). | Fix paper 2's title. | confirmed |
| 0 | Authors: "Carson Kent, Stanford", "Tan Bui-Thanh, UT Austin", Constantine at Mines, "<pconstan@mines.edu>" | Match the published 2016 affiliations (p. A2779 footnotes): crkent@stanford.edu, tanbui@ices.utexas.edu. The paper prints Constantine's address as **paul.constantine@mines.edu**, not pconstan@mines.edu. Ryan Howard is a notebook author, not a paper author. Current affiliations were not checked. | Use the paper's author block, with "notebook by Ryan Howard". Ask about current addresses (§11). | confirmed (vs paper); unverified (current) |
| 0 | "$\mathbf Q = \frac 1 2[\sqrt2,\sqrt2;-\sqrt2,\sqrt2]$", ε = .01, d = .9 | Matches published eqs. (5.1)–(5.2), p. A2789. | — | confirmed OK |
| 0 | "noise parameter in the likelihood of $\sigma = .1$" | Published p. A2789 says "noise parameter σ² = 0.1", but the paper's own results were computed with σ = 0.1 (σ² = 0.01): <br>(1) Fig. 5.1(a), p. A2790, shows eigenvalues of C ≈ 2e4 and ≈ 0.7 for ε = 0.01. A 50-point Gauss–Hermite computation gives 1.87e4 and 0.648 for σ² = 0.01, vs 187 and 0.0065 for σ² = 0.1 (`prototypes/g2/quad_C.py`). <br>(2) The reported 12% vanilla acceptance (p. A2791) is reproduced only with σ = 0.1: 0.118 vs 0.406 (`quad_acc.py`); the notebook printed 0.1193. | Keep σ = 0.1. Add a footnote saying the paper prints σ² = 0.1 but its Fig. 5.1 corresponds to σ = 0.1. | confirmed (paper typo) |
| 3 | `return 1./sigma**2*dm(x)*(d - m(x).T)` | The gradient of f = (d−m)²/(2σ²) is −(1/σ²)(d−m)∇m, so the sign is wrong. It copies published eq. (3.5), p. A2784, which has the same sign error. The paper's own linear case, eq. (3.15), p. A2786, has the correct sign (Mᵀ(Mx − d)). Harmless for C = E[∇f∇fᵀ]. | Negate. List (3.5) as a paper erratum (§11). | confirmed |
| 7 | "normal proposal … covariance $.5\mathbf I$. For $10^6$ samples" | Matches published p. A2791: variance σ² = 0.5, 10⁶ steps, 12% acceptance. | — | confirmed OK |
| 10 | "we will sample 10 $\mathbf z$'s according to $\rho(\mathbf z\|\mathbf y)$ and compute the conditional average of $f$ given $\mathbf y$ with Monte Carlo" | Published p. A2791 evaluates ĝε "with a 10-point Gauss–Hermite quadrature rule", which is deterministic: proposal variance 0.5, 10⁵ steps, 13% acceptance, P = 10. | Use `np.polynomial.hermite_e.hermegauss(10)`, weights/√(2π). This also removes the recompute issue for this example. | confirmed |
| 10 | "so the total number of forward model evaluations is the same as with vanilla MCMC" | True as coded: both loops recompute the current state, giving 2·10⁶ each. With caching it is 10⁶ vs 10⁵·10, still equal, as the paper states (p. A2791). | Keep, after caching both. | confirmed OK |
| 15 | "The KDE only uses $\sim 10^5$ samples" | The paper used all 10⁶ (p. A2791). Fine as a runtime choice. | Keep, or use a histogram-based density. | — |
| 18 | "observations of u on a few points of the nontrivial boundary" | Published p. A2792: "seven points on the right boundary, s2 ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}", with s1 = 1. Code: `indz=[20,…,80]` index unknown rows, giving s2 = 0.21, …, 0.81 at s1 = 0.99 (numpad_replacement.md §2). | State the 7 locations as coded (or fix `indz` to 19…79, author's choice). Also state the field (published eqs. (5.4)–(5.5): √σ_i φ_i x_i and exp(−‖s1−s2‖₁/β) with β = 0.02, identical to `kl-short`), m = 100, prior N(0, I), and σ² = 10⁻⁴‖m(x_true)‖² ("roughly 1%", p. A2793). | confirmed |
| 18 | "Qiqi Wang's Numpad … about 10 minutes … per 1000 forward model evaluations; this is why we only use 1000 replicates" | The replacement takes 20–30 s per 1000 evaluations. | Rewrite: describe the scipy.sparse solver with adjoint gradient. Live chains can then be about 5× longer. | confirmed |
| 18 | "solves the PDE with second-order finite differences—discretized using 100 points in each spatial dimension" (published p. A2792; not in the notebook) | The code is a 5-point vertex-centred FV with 99² unknowns and a first-order Neumann ghost. | Describe the code in the notebook. | confirmed |
| 20 | `prop_sig = .1 #Proposal standard deviation` and comment "# proposal step in the active variables" (in the vanilla loop) | Published p. A2796 and Table 5.1 (p. A2797) say "proposal density with variance 0.1" (vanilla; AS2 0.1 and 0.3). The code uses std 0.1, i.e. variance 0.01. A test on seeded synthetic data with the new solver (`prototypes/g2/van_prop.py`, 400 steps) gives acceptance 0.54–0.56 for std 0.1 and 0.09–0.11 for std √0.1. The paper reports 60% and the legacy chain has 0.546, so the paper's "variance" almost certainly means standard deviation. Table 5.1 also swaps the column labels ("AS2, 0.1" lists variance 0.3). The vanilla comment is a copy-paste. | State "standard deviation 0.1" and fix the comment. Footnote the paper's wording. | likely (paper wording; code is right) |
| 25 | "significant gaps between the 1st and 2nd and between the 2nd and 3rd eigenvalues" | Holds: 1093, 222, 24.6, 7.8 (ratios 4.9, 9.0, 3.1), consistent with published Fig. 5.5. v1 EVG (linear gap) picks n = 1, so the notebook must pass n = 2. | Keep; set `n=2` explicitly. | confirmed |
| 25–27 | "we can examine sensitivity by looking at the coefficients, $a(\mathbf s, \mathbf x)$, with $\mathbf x$ as one of the eigenvectors"; code plots `np.exp(model.B.dot(W1[:,j]))` | Published Fig. 5.6 (p. A2795) shows "the corresponding **log-coefficient** fields when these eigenvectors are plugged into x". exp() distorts it and depends on the eigenvector sign. | Plot B w_j with a diverging colormap and colorbar; label the axes s1, s2. | confirmed |
| 30 | "$\hat{\mathbf y}_k = \hat{\mathbf W}^T\mathbf x_k$ … first 2 or so components … diverge from standard Gaussian" | Matches published Fig. 5.10 (p. A2801). | — | confirmed OK |
| 32 | "generate 100000 replicates in the vanilla case and 10000 replicates for the active variable" | Matches the files. Published p. A2796: 500k vanilla steps with 100k burn-in; 50k AS2 steps with 10k burn-in, M = 10, P = 10; proposal "variances" 0.1 and 0.3. The notebook uses no burn-in. | Say the stored chains are a smaller rerun; apply burn-in (20%, as in the paper). | confirmed |
| 35 | `X_AS = asamcmc.sample_original_variables(Y, P)` | `asamcmc` holds W1/W2 from cell 23 (live data), but `Y` comes from `generate_data.py` with different data. Cell 37 recomputes `ss` only afterwards, and `asamcmc` keeps stale copies anyway. Result: X_AS is wrong. | Reconstruct with the W stored next to the chain (§5). | confirmed |
| 37 | "#We need to recompute the subspace since the simulated data used to generate the 100000-replicate chains is different" | The comment admits the mismatch. d, σ² and x_true were not saved, so the stored chains cannot be checked against the posterior. | Regenerate with saved inputs (§8). | confirmed |
## 4. Code review

Algorithmic deviations and bugs:
- **`pde_mcmc_study.ActiveSubspaceAcceleratedMCMC.sample_active_variables`.**
  - Recomputes `mf = self.misfit(y0, M)[0]` each iteration (not paper Algorithm 1; 2× cost).
  - `self.misfit` returns a tuple `(mean, var)`. `(mf-self.misfit(yc, M))[0]` works only because `np.float64 - tuple` broadcasts to an array. Fragile; return the mean only.
  - `prop_sig` is a std.
  - Uses the global `np.random`.
- **`compute_misfit_active_subspace`.** Uses `asub.Subspaces()`. That exists only on legacy-dev (`from base import *`); on legacy master it is an AttributeError.
- **Vanilla loops** (cell 20, `generate_data.py`): recompute `model.misfit(y0)` every step (2× cost). Use log acceptance with a cached current misfit.
- **Quadratic cells.**
  - Cells 8 and 11 call `np.random.multivariate_normal` per step for 10⁶ steps (minutes).
  - The ratio of exponentials risks overflow; use log-ratios.
  - `f_evals` in `g` is unused.
  - Names are reused: `M` = MC samples, then chain length; `x` = samples, then linspace, then meshgrid.
  - Cell 14: `np.arange(M/10 - 10000, M/10)` yields floats in py3; use `//`.
- **cell 26.** `exp` of the eigenvector field (see §3); `meshgrid(1..100)` axes are unlabeled.
- **cell 31/37 KDE.** `gaussian_kde` on 10⁵ points evaluated on 10⁴ grid points, 6 times: several minutes. Thin the chain or use `np.histogram2d` with Gaussian smoothing.

Randomness and reproducibility:
- **No seeds anywhere** (notebook, `generate_data.py`, `pde_mcmc_study.py`).
- **Notebook and script disagree on σ².** The `__main__` of `pde_mcmc_study.py` uses absolute `sig2 = 0.0001`; the notebook and `generate_data.py` use the relative 10⁻⁴‖obs‖².

Paths and imports:
- **Hard-coded paths.** None, but `kl-short.npz` is loaded CWD-relative.
- **Leftover imports.** `import pdb` in two files.

## 5. Data management review

Fixes that do not depend on regeneration:
- **Redundant flag files.** `Vanilla_Acceptance` is derivable from the chain. Both acceptance files store 0/1 as float64; bool or uint8 is enough.
- **Size.** `Vanilla_Chain.mat` is 80 MB. It is under GitHub's 100 MB hard limit but over the 50 MB warning, and the repo pack is already 210 MB.
- **What the notebook actually uses from it** (cells 34 and 37):
  - (a) trace of x₁₀ and x₉₀ over the last 2,000 steps;
  - (b) the projections y = Wᵀx, first 4 components, for KDEs.
- **Measured sizes for alternatives:**

  | storage option | size |
  |---|---|
  | `savez_compressed`, float64, full chain | 43.6 MB |
  | unique states + flags | 41.9 MB |
  | thin ×10, float64 | 7.7 MB |
  | thin ×10, float32 | 3.7 MB |
  | the 2 trace columns (full length) + the 4 projected columns (full length), float32 | ≈ 2.4 MB |

**Recommendation: regenerate.** The current chains cannot be validated (§3, cells 35 and 37), so regenerate with the new solver and fixed seeds rather than convert. Store one `pde_mcmc_chains.npz` (≤ 10 MB) with:
- `seed`, `x_true` (100,), `d` (7,), `sig2`, `indz` / sensor coordinates, and `kl_md5`;
- `X_grad` (N_est,100) and `G` (N_est,100), so the subspace is recomputable;
- `W` (100,100), the eigenvectors actually used, with the sign convention recorded;
- vanilla:
  - `van_trace` (N,2): x₁₀, x₉₀;
  - `van_proj` (N,4): Wᵀx[:4], float32;
  - `van_thin` (N/10,100) float32, optional, for moments;
  - `van_accept` bool;
- AV:
  - `av_y` (N_av,2);
  - `av_accept` bool;
  - `av_ghat` (N_av,);
- metadata: chain lengths, proposal std, M, burn-in, wall time, and package versions.

Remove the 5 `.mat` files from HEAD.

**History.** Deleting the file does not shrink clones. Purging it from history (`git filter-repo --path MCMC/Vanilla_Chain.mat --invert-paths`) rewrites every SHA and breaks forks. This is the author's call (§11). External hosting (Zenodo DOI) is not recommended: the notebook must run headless and offline, and the reduced file fits easily in git.

**`kl-short.npz` duplicate.** Keep it, as numpad_replacement.md §7.3 recommends. The stored K defines the x coordinates, so do not regenerate the basis. If the author prefers, load it from `../Elliptic_PDE` instead.

**Provenance and license.** Chains generated 2016-07-12/13 by `generate_data.py` on the legacy stack. No license file exists in the repo.

**Pandas-free loading.** Legacy files, tested in .venv:
```python
import scipy.io as sio
Xv  = sio.loadmat('Vanilla_Chain.mat')['Chain']          # (100000, 100)
acc = sio.loadmat('Vanilla_Acceptance.mat')['Accept'].ravel()
Y   = sio.loadmat('AV_Chain.mat')['Chain']               # (10000, 2), legacy W sign convention
G   = sio.loadmat('Gradient_Samples.mat')['Samples']     # (1000, 100)
```
New file: `z = np.load('pde_mcmc_chains.npz'); Y, W = z['av_y'], z['W']`.

## 6. Migration to active_subspaces v1

Spaces:
- quadratic: `ac.Gaussian.reference(2)`;
- PDE: `ac.Gaussian.reference(100)`. The prior N(0, I) means physical = normalized coordinates, with no gradient scaling.

| legacy call | v1 replacement | notes |
|---|---|---|
| `ss = ac.subspaces.Subspaces(); ss.compute(df=grad, nboot=500)` | `sub = ac.active_subspace(space=ac.Gaussian.reference(2), grads=grad, nboot=500, rng=rng)` | `grad` (M,2). Eigenvalues equal to rounding. |
| `ac.utils.plotters.eigenvalues(ss.eigenvals, ss.e_br)` | `ac.plotting.plot_eigenvalues(sub, interval="minmax")` | The default band is a 95% percentile band, narrower than 2016's. Use `"minmax"` to match the legacy figure, or keep the default and say so. |
| `ac.utils.plotters.subspace_errors(ss.sub_br)` | `ac.plotting.plot_subspace_errors(sub)` | |
| `ac.utils.plotters.sufficient_summary(x.T.dot(ss.W1), output)` | `ac.plotting.plot_sufficient_summary(sub, X=x.T, f=output)` | Takes X, not y. `f` is (M,). |
| `ss.W1`, `ss.W2` (quadratic `g`, cells 11 and 13) | `sub.W1`, `sub.W2` (after `with_dimension(sub, 1)`) | **Sign rule changed:** the largest-\|·\| component is positive (2016: the first component). The y chain may be mirrored relative to legacy; the KDE of x is unaffected. |
| `SimulationGradientRunner(pde_model.misfit_grad).run(X)` | `ac.evaluate_grads(ac.Model(space=space, value_and_grad=batch_misfit_and_grad), X=X)` or a comprehension | Optional `executor=` for parallel runs; the result is independent of the worker count. |
| `asub.Subspaces(); ss.compute(df=df, nboot=100); ss.partition(2)` | `sub = ac.active_subspace(space=space, grads=G, nboot=100, rng=rng, n=2)` | EVG alone gives n = 1 here (largest linear gap λ1−λ2), so pass `n=2`. Verified on `Gradient_Samples.mat`: eigenvalues match legacy (rel. diff 3e-15). |
| `ss.eigenvals[:10,0], e_br=ss.e_br[:10,:]` | `ac.plotting.plot_eigenvalues(sub, k_max=10, interval="minmax")` | |
| `ac.utils.plotters.eigenvectors(ss.eigenvecs[:,:2])` | `ac.plotting.plot_eigenvectors(sub, indices=[0, 1])` | 0-based indices. |
| `ss.eigenvecs.T.dot(X_van.T)` | `X_van @ sub.W` | |
| `ActiveSubspaceAcceleratedMCMC(model, ss)` → `.W1`, `.W2` | pass `sub.W1`, `sub.W2` (or W from the chain file) | **Critical with legacy chains.** v1 flips w2 relative to the legacy W used to make `AV_Chain.mat`. Measured: vanilla projected mean of y2 is −1.46 with the v1 w2 but +1.46 with the legacy sign, against an AV chain mean of +1.36. If any legacy chain is reused, multiply column j of W by sign(W[0,j]) first. With regenerated chains, store W in the file and use it, never a recomputed one. |

## 7. Associated .py files

- **`pde_model.py`.** Replace with the shared numpad-free module (numpad_replacement.md, class `PDEMisfit`, byte-identical copy of `Elliptic_PDE/pde_model.py`).
  - Add `misfit_and_grad`.
  - Set data via `set_data(d, sig2)` as before.
  - Delete `pdb`.
- **`pde_mcmc_study.py`.** Port to py3 and rename to `as_mcmc.py` (optional).
  - `ActiveSubspaceMCMC(misfit, W1, W2, rng)`.
  - `sample_active_variables(y0, N, M, prop_std)`: caches ĝ(y_k), works in log space, returns `(Y, ghat, accept)`.
  - `sample_original_variables(Y, P)`: vectorized, `X = Y@W1.T` repeated plus `Z@W2.T`.
  - `misfit_subspace(model, N, rng)`: v1-based.
  - Delete the `__main__` demo (it uses an inconsistent σ²).
- **`generate_data.py`.** Rewrite with these steps:
  1. Seeded `default_rng`.
  2. Synthetic data: x_true, d, σ² = 10⁻⁴‖m(x_true)‖².
  3. N_est = 1000 gradients; v1 subspace with n = 2.
  4. Vanilla chain: 10⁵ steps, std 0.1, cached misfit.
  5. AV chain: 10⁴ steps, M = 10, std 0.1, cached ĝ.
  6. Write `pde_mcmc_chains.npz` (§5) and print the wall time.
  - Estimated cost with about 27 ms per solve:
    - 1000 gradients: 30 s;
    - vanilla 10⁵ solves: about 45 min;
    - AV 10⁵ solves: about 45 min (parallelizable over the M = 10 inner evaluations with an executor).
  - Paper scale (500k / 50k) is about 4 h per chain; that is an author's choice (§11).

## 8. Update plan

1. **Shared solver.** Add `pde_model.py` (`PDEMisfit`) to MCMC, identical to the Elliptic_PDE copy (`cmp` check).
2. **Sampler module.** Port `pde_mcmc_study.py` (→ `as_mcmc.py`): cached ĝ, log acceptance, rng, no tuple arithmetic.
3. **Regeneration script.** Rewrite `generate_data.py` (seeded; saves inputs, W and reduced chains).
4. **Regenerate.** Run it (about 1.5 h). Add `pde_mcmc_chains.npz` and delete the 5 `.mat` files. Decide separately whether to purge the 80 MB blob from history.
5. **Notebook, quadratic part.**
   - py3 kernel and imports, `rng`.
   - Fix the sign of `df`.
   - v1 subspace and plots.
   - Vanilla MH with pre-drawn proposals and log acceptance.
   - AS-MH with 10-point Gauss–Hermite ĝ.
   - Integer arithmetic.
   - Faster KDE.
6. **Notebook, PDE part.**
   - Live demo with the new solver: 1000 gradients plus short chains (about 1000 vanilla steps and 200 AV steps, about 2 min). Alternatively load `X_grad` and `G` from the npz.
   - Then load the stored chains.
   - Reconstruct X_AS with the stored W.
   - Burn-in.
   - Log-coefficient sensitivity plots.
   - Remove the cell 35/37 mismatch.
7. **Text fixes (§3).** c_pos, L·C, typos, σ footnote, the Gauss–Hermite statement, sensor locations and field definition, the proposal std, the chain-length vs paper statement, numpad removal, references.
8. **Execute headless and commit outputs.**
9. **Docs.** README bullet (data now `.npz`) and CLAUDE.md (shared solver, no `.mat`).

## 9. Acceptance criteria

- **Headless run.** `cd MCMC && jupyter nbconvert --to notebook --execute MCMC.ipynb` succeeds in .venv in < 10 min, using the stored chain file for the long chains. Imports only numpy, scipy, matplotlib, active_subspaces, and the local modules. No numpad, pandas or py2 prints.
- **Seeding.** All randomness comes from seeded `default_rng`. A rerun reproduces the outputs exactly.
- **Quadratic example.**
  - vanilla acceptance 0.12 ± 0.01 (paper 12%; legacy 0.1193);
  - AS acceptance 0.13 ± 0.02 (paper 13%; legacy 0.1346);
  - eigenvalue gap λ1/λ2 > 10;
  - the AS-chain KDE shows the two posterior lobes.
- **PDE subspace.** Eigenvalues show gaps after λ1 and λ2, with λ2/λ3 > 5. Legacy data give 1093/222/24.6/7.8; prototype with fresh data (seed 1510) gives 2081/262/20.5/15.8.
- **Stored chain file.**
  - ≤ 10 MB;
  - loaded with `np.load` only;
  - contains x_true, d, σ², seed, X_grad, G, W;
  - `W[:, :2].T @ W[:, :2] = I` to 1e-12;
  - recomputing the subspace from `G` reproduces `W[:, :2]` up to sign (\|cos\| > 1 − 1e-10).
- **AV chain vs the approximate posterior** (rewritten 2026-10-30 with the author's sampler decision, PR #52). The AV chain does not sample the posterior, so it is tested against what it approximates:
  - **Reference.** π_M ∝ E[exp(−ĝ_M(y))] ρ(y) with M = 10: the average over the Monte Carlo noise of the likelihood estimate. Algorithm 1 as printed (ĝ kept with the state) is a pseudo-marginal method and samples π_M exactly. The 2016 variant (ĝ recomputed at every step, the shipped chain) is the noisy approximation of it called Monte Carlo within Metropolis (Andrieu & Roberts 2009), with no closed-form target.
  - **Grid.** Compute the mean of y under π_M on a 15 × 15 grid covering ±4 AV standard deviations around the AV mean. Use the same 40 prior draws of z at every grid point, and estimate E[exp(−ĝ_10)] by averaging over 400 random 10-draw subsets of the 40.
  - **SE_grid** is the standard deviation over 200 bootstrap resamples of the 40 draws. It is about 0.03 for y2, consistent with the 0.04–0.06 spread between two independent 40-draw grids. A two-halves estimate (one degree of freedom) gave 0.001 and is too small to use.
  - **SE_chain** is the consistent batch-means standard error of the AV mean after burn-in (Flegal, Haran & Jones 2008, b = ⌊N^{2/3}⌋).
  - **Pass** if |mean_AV − mean_π_M| ≤ 3 (SE_chain² + SE_grid²)^{1/2} for each of y1, y2. Three standard errors give a two-sided false-alarm rate of about 0.3% per component if both errors are roughly normal.
  - **Expected deviation.** The 2016 variant's chain is shifted from π_M toward the posterior and is wider, because the noise in the acceptance test lets it visit states that π_M weights less. A failure in that direction is the variant's noise bias, not a bug. The notebook reports whichever happens, together with the π_ε grid mean (exp(−g_ε) with the exact conditional mean).
  - **Vanilla.** The vanilla chain's projected mean is reported, not tested. π_ε and π_M differ from the posterior by the bias that Theorem 3.1 bounds (λ3 + … + λ100 = 124 for the shipped data), and the gap is several standard errors.
  - **Original criterion.** AV within 3 SEs of the vanilla projected mean. It fails for both shipped variants for this reason (5 combined SEs with ĝ cached, 3.1 and 2.3 with ĝ recomputed). The legacy files failed it without the sign fix (y2: +1.36 vs −1.46).
- **Acceptance rates are reported.** Vanilla is expected near 0.55 (paper 60%). AV is expected near 0.6 at std 0.1 (paper 62%) with ĝ recomputed at every step (author, 2026-10-30), but near 0.29 with ĝ kept with the state (Algorithm 1 as printed; `generate_data.py --cached-ghat`).
- **Figures.**
  - quadratic: eigenvalues, subspace error, sufficient summary, 2 trace plots, 2 KDE contour plots, 1-D y density;
  - PDE: eigenvalues, subspace error, eigenvectors, 2 log-coefficient sensitivity maps, trace plots (vanilla x₁₀/x₉₀, AV y, AV-reconstructed x), 3×3 bivariate KDE of Wᵀx (vanilla).
- **Text.** Every confirmed row in §3 is resolved.

## 10. Prototype results

Scripts in `prototypes/g2/`.

- **`proto_mcmc_data.py`** (legacy files, .venv, scipy.io):
  - All 5 `.mat` files load (MAT v5).
  - Acceptance: vanilla 0.5456, AV 0.6152 (match the cell 33 printout).
  - Vanilla row changes == the acceptance flags (54,561 unique rows).
  - v1 `active_subspace(space=Gaussian.reference(100), grads=G, nboot=100, rng=0)`:
    - λ[:7] = 1093.4, 222.4, 24.63, 7.84, 7.46, 6.70, 5.68, matching the legacy cell 37 figure;
    - max rel. diff vs a plain `eigh` is 3e-15;
    - EVG n = 1.
  - Sign rule: v1 w1 has first component +; v1 w2 has first component − (largest \|·\| at index 14), so it is flipped vs legacy; w3 is unchanged.
  - Projected vanilla chain (second half): y2 mean −1.464 with the v1 w2. AV chain y2 mean +1.358, so the legacy sign convention is confirmed for `AV_Chain.mat`.
  - Storage sizes as listed in §5.
- **`proto_mcmc.py`** (new solver, `fvsolver.py`, seed 1510):
  - value-only solve 29 ms;
  - σ² = 6.3e-6;
  - 1000 misfit gradients in 29 s;
  - λ[:5] = 2081, 262, 20.5, 15.8, 14.4;
  - 100 AV steps with cached ĝ (M = 10, std 0.1): 27 s, acceptance 0.44.
  - Adjoint misfit gradient vs central FD (directional): agree to 9 digits (`check_solver.py`).
- **`quad_acc.py`** (quadratic vanilla MH, 2·10⁵ steps): acceptance 0.118 with σ = 0.1 vs 0.406 with σ² = 0.1. The paper reports 12%.
- **`quad_C.py`** (eigenvalues of C for the quadratic, 50×50 Gauss–Hermite, the paper's own method):
  - ε = 0.01: 1.866e4 and 0.648 with σ² = 0.01; 186.6 and 0.0065 with σ² = 0.1.
  - ε = 0.95: 3.1e4 and 2.6e4 with σ² = 0.01.
  - Published Fig. 5.1(a),(c) (p. A2790) shows ≈ 2e4 / 0.7 and ≈ 3e4 / 3e4, so σ = 0.1 is what the paper used.
- **`van_prop.py`** (PDE vanilla MH, new solver, seed 1510 synthetic data, 400 steps): acceptance 0.56 (std 0.1) vs 0.11 (std √0.1). The paper reports 60% for "variance 0.1", so the paper's "variance" means standard deviation.
- **Not run.** Full regeneration of the chains (about 1.5 h). The notebook end-to-end.

## 11. Open questions for the author

1. **Purge the 80 MB blob?** Should `Vanilla_Chain.mat` be purged from git history (rewrites SHAs), or only removed from HEAD?
2. **Chain scale.** Keep the notebook's 10⁵ / 10⁴ steps, or match the paper (500k / 50k, AS2 proposal variances 0.1 and 0.3, burn-in 100k / 10k), at about 4 h per chain?
3. **Proposal "0.1".** The paper (p. A2796, Table 5.1 p. A2797) says "variance 0.1". Acceptance rates show it was std 0.1: 0.55 vs 0.10 for std √0.1, against the paper's 60%. Add to an errata list, together with the swapped Table 5.1 column labels ("AS2, 0.1" lists 0.3)?
4. **Paper errata.** Record these on an errata list?
   - p. A2789 prints σ² = 0.1 for the quadratic example. Fig. 5.1(a) and the 12% acceptance both correspond to σ = 0.1, so this is confirmed.
   - eq. (3.5) has the wrong sign of ∇f (contradicted by (3.15)).
   - The proposal "variance" wording (item 3).
5. **Sensor rows.** Keep `indz = [20,…,80]` (s2 = 0.21…0.81) for continuity, or shift to s2 = 0.2…0.8 as the paper states?
6. **Quadratic ĝ.** Switch the quadratic AS chain to Gauss–Hermite ĝ as in the paper (recommended), or keep Monte Carlo and say so?
7. **Authors and license.** Author list, affiliations and emails for the header. Data license.
