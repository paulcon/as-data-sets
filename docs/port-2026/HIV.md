# HIV: update spec

Kind: **Python model**: `HIV_model.py`; `generate_data.py` writes the shipped CSVs and must reproduce them (see [README.md](../README.md#two-kinds-of-data-sets)).

Status line: `HIV/HIV.ipynb` | paper: T. Loudon and S. Pankavich, *Mathematical analysis and dynamic active subspaces for a long term model of HIV*, Math. Biosci. Eng. 14(3) (2017) 709–733, doi:10.3934/mbe.2017040 (checked against the published PDF; arXiv:1604.04588 v2 read first. Page, equation, table and figure refs below are to the published version). It is **Mathematical Biosciences and Engineering** (AIMS), not *Mathematical Biosciences* | m = 27 (README's "21-parameter" is wrong; 21 is the number of output times) | QoI: T-cell count T(t) at 21 times (5 … 3400 days), one active subspace per time | legacy API: `ac.gradients.finite_difference_gradients`, `ac.subspaces.Subspaces().compute(df=)/.partition(2)/.W1/.eigenvals`, `ac.utils.misc.BoundedNormalizer`, `ac.utils.plotters.*` | py2 issues: yes. `generate_data.py` has a `print` statement; the notebook uses pandas `DataFrame.from_csv(...).as_matrix()`; kernelspec python2.

## 1. Summary

The model equations (Eq. 1, p.711), all 27 names, nominal values and units (Table 1, p.713) and the ±2.5% uniform ranges (p.718) match the published paper. The notebook's claims about dominant parameters (t = 24: K1, K9, δ7, ψ; late times: K4, δ7, δ4, K5) are confirmed on the shipped 1000-sample data with v1. The late-time set agrees with the paper's T(1700) result (K4, δ4, δ7 > 0.3; p.719, Fig. 3). The published paper itself links to this notebook (p.710: `nbviewer…/as-data-sets/blob/master/HIV/HIV.ipynb`), so the path must not change. Main findings:

1. **The shipped gradients are accurate, but the notebook's own gradient computation is not reproducible in Python 3.**
   - A tight-tolerance `solve_ivp` (LSODA, rtol = atol = 1e-10) with central FD agrees with the shipped `gradients.csv` rows to a relative 1e-5 at the few samples first tested (24 to 3400 days; cos = 1.000). **Corrected after the port (PR #51):** over all 1000 samples the 2016 files are less accurate than that. Gradients above the floor agree with the regeneration to a median of 2.9e-5, a 99th percentile of 2.6e-3 and a maximum of 0.16 (at t = 38–90). Outputs agree to a median of 1e-7 and a maximum of 5.5e-4 (at T < 3). At the worst rows, tighter solves agree with the new values to ≤ 1.3e-5, so the errors are in the 2016 vode run (default rtol 1e-6, 1e-7 forward step). The eigenvectors are unaffected (≤ 9e-5). The CSVs were replaced by the regeneration (author, 2026-10-30).
   - Running the notebook's own path (`HIV_model.Tcells`, scipy `ode`/vode, `nsteps=1500`, forward FD with step 2.5e-9 relative) under the venv scipy gives a t = 3400 gradient that is **130×** wrong in relative norm. vode also emits "Excess work done" warnings, and `Tcells` never checks `solver.successful()`.
   - So the small-sample section (cells 3–13) would produce garbage at late times. It would also take an estimated ~30 min (the notebook says 10).
2. **The 0.1 "freeze" in `HIV_model.py` is a code artifact, not in the paper.** `if T < .1: return np.zeros(7)` freezes the state. In the shipped data, 6, 24, 61, 130, 283 and 474 of the 1000 samples are frozen at t = 1800, 2000, 2200, 2400, 2800 and 3400.
   - For frozen samples the exact gradient is 0, because perturbations do not change the frozen value. Event-based recomputation confirms this. The shipped gradients there have norm about 15, which is FD noise (non-frozen median 325).
   - The effect on eigenvalues is small: λ1/λ2 at 3400 is 853 with them and 874 without. But nearly half of the late-time summary-plot points sit at f ≈ 0.1, and the text does not mention it.
3. **The shipped data are not the paper's computation, and they are cleaner than it.** The paper used MATLAB ode23s, forward FD with δ = 1e-6 in normalized coordinates (pp.716–718), and 86 time subintervals (p.721); Appendix B shows 18 times.
   - The paper's eigenvalue tails sit 1–2 orders above the notebook's. Fig. 2 (T(1700), p.717) has λ1 ≈ 1.5e4, λ2 ≈ 2e3 and a tail of about 20–500. The shipped data give λ1 = 1.2e4 / 2.6e4 (t = 1600 / 1800), λ2 ≈ 400–600 and λ3 ≈ 10.
   - At t = 2000 the paper's Fig. 7 (p.721) has λ2 ≈ λ1 (≈ 4e4 vs 5e4) and the paper picks n = 2. The shipped data give λ1/λ2 = 75 (n = 1).
   - Because the regenerated gradients are verified to about 1e-5, the paper's tails and its t = 2000 λ2 are most likely FD/solver noise (ode23s with a 1e-6 forward step). This is likely, not proven.
   - The notebook must not import the paper's t = 2000 "2-D" statement. The notebook data come from `generate_data.py` (scipy vode, FD step 1e-7·half-range). The text implies the two are the same analysis.
4. **Citations and README.** The README says 21 parameters (it is 27), the journal name is wrong, the CDW 2014 title is wrong, and there are typos.
5. **Migration.** Shipped gradients are w.r.t. normalized inputs, so divide by (xu − xl)/2 and pass them with `ac.Uniform(xl, xu)`, or pass them with `Uniform.reference(27)`. Both were tested and give identical results. The hand sign flip has to act on a copy, because v1 `W1` is read-only. **Superseded (V1-05, [#51](https://github.com/paulcon/as-data-sets/pull/51)):** `ac.orient` returns a new result, so no hand flip is needed.

## 2. Inventory

| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| HIV.ipynb | 1.6 MB | nbformat 4, python2 | case study | — | 23 PNGs, no printed numbers |
| HIV_model.py | 2.9 KB | py (py3-clean syntax) | `Tcells(p, times)`: 7-ODE model via `scipy.integrate.ode('vode', bdf, nsteps=1500, max_step=1)` with an analytic Jacobian; returns T at `times` | `from HIV_model import *` | Jacobian verified correct (built transposed, passed as `.T`); T<0.1 freeze; fixed ICs |
| generate_data.py | 1.2 KB | py2 (print stmt), pandas, legacy `ac.gradients` | regenerates the three CSVs (1000 samples, "about 8 hours") | script | unseeded `np.random.uniform`; re-integrates from t = 0 for each of the 21 times (21× redundant) |
| inputs.csv | 444 KB | pandas CSV: header `,0..26`, index column 0..999 | 1000×27 **physical** parameters (12 significant digits) | `pn.DataFrame.from_csv` | all inside [xl, xu] (normalized range ±0.99997) |
| outputs.csv | 298 KB | same, header `,0..20` | 1000×21 T(t_i) | same | min 0.0999951 (frozen samples), max 1001.6 |
| gradients.csv | 8.7 MB | same, header `,0..566` | 1000×567. Column `27*i + j` = ∂T(t_i)/∂x_j w.r.t. the **normalized** input x_j ∈ [−1, 1] | same | no NaN; noise rows where frozen (§1.2) |

## 3. Text and math review (vs. literature)

| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| 0 | seven ODEs; parameter table (27 rows) | match published Eq. (1) (p.711) and Table 1 (p.713) term by term and value by value. Table 1's "Range" column holds literature ranges, not the ±2.5% analysis ranges | add units (Table 1: s_i mm⁻³d⁻¹, K mm³d⁻¹ except K9, K10 d⁻¹, rates d⁻¹, ψ unitless) | confirmed (published) |
| 0 | "The model's 27 parameters are summarized in the table below." | correct (published p.710: "twenty-seven distinct parameters"); **README says "a 21-parameter dynamical system model"** | README → 27 | confirmed |
| 0 | "The limits on the uniform distributions are 2.5% above and below the nominal values." | matches published p.718; the affine map is Eq. (8) | none | confirmed (published) |
| 0 | "The active sbspace method" | typo | subspace | confirmed |
| 0 | "[[1]] T. Loudon and S. Pankavich. … arXiv:1604.04588, 2016" | published version exists | add Math. Biosci. Eng. 14(3):709–733 (June 2017), doi:10.3934/mbe.2017040 (verified on the published title page) | confirmed (published) |
| 0 | "[[2]] … _Active subspaces in theory and practice: applications to kriging surfaces_" | title is "Active subspace methods in theory and practice: applications to kriging surfaces" | fix | confirmed |
| 0 | "__Tyson Loudon__, University of Minnesota"; "__Stephen Pankavich__, Colorado School of Mines" | matches the published p.709 (School of Mathematics, University of Minnesota-Twin Cities; Applied Mathematics and Statistics, Colorado School of Mines) | none | confirmed (published) |
| 0 | (absent) initial conditions | the code uses T0 = 1000, M0 = .15/.005, CTL0 = 5/.015, V0 = 1 and zeros elsewhere. This is the uninfected equilibrium at the **nominal** parameters plus one virion, and is not sample dependent. The published version also never states ICs. It gives E_NI = (s1/δ1, 0, 0, s2/δ4, 0, s3/δ6, 0) (Theorem 2.1, p.713), and Fig. 1 (p.712) starts at T ≈ 1000, both consistent with the code | state the ICs in the text as the notebook's choice | unverified (paper silent; consistent with Fig. 1) |
| 2, 3 | "uses SciPy to solve the ODE system" | the solver silently fails ("Excess work done") in current SciPy and the T<0.1 freeze is undocumented | describe the solver, tolerances and the floor; see §4 | confirmed |
| 4 | "Gradients are approximated with first order finite differences, which takes about 10 minutes to compute with 20 samples" | py3 estimate is ~30 min (560 FD evaluations re-integrating to each of 21 times, ~0.15 ms per simulated day), and late-time results are noise (§1.1) | replace (§8) | confirmed |
| 8 | "the 6th, 14th, 25th, and 27th variables (K1, K9, δ7, and ψ …) are the most important" | index-to-name mapping correct. Confirmed on 1000 samples: w1 = K1 .51, K9 .49, δ7 −.47, ψ .47. The published paper gives no t = 24 list. Its acute-stage reduction (first 40 days, threshold 0.032, pp.726–727) keeps s1, p1, C1, K1, K3, K9, δ1, δ2, δ7, ψ (then sets ψ = 1), which contains all four | none (but the text describes an unseeded random draw; seed it) | confirmed (data); consistent with published p.726 |
| 8 | "sufficeint summary plots" | typo | sufficient | confirmed |
| 10 | "tight univariate trends in each case, though deviation from the trend increases as time increases" | \|corr(y1, f)\| is ≥ 0.98 up to t = 1000, then 0.94 (1600), 0.90 (2000), 0.88 (3400). Not monotone after 2000. Late-time scatter includes the f ≈ 0.1 floor cluster. The published paper (p.723) describes the late-time trend as "an arctangent function multiplied by a heaviside step function" for t ≥ 1800, with T(2600; y) = 0 for y ≥ 0.5 | "…deviation grows in the chronic/AIDS stage; for t ≥ 1800 a growing fraction of trajectories reach T ≈ 0 (here, the 0.1 floor), giving the arctangent × step shape described in the paper" | confirmed (data, published p.723) |
| 12 | "The gap between the first and second eigenvalues is not quite as large as the early time" | λ1/λ2 = 3957 (t = 24) vs 853 (t = 3400); the smallest is 30 at t = 1600 | fine; optionally mention 1600 | confirmed |
| 14 | "K4 is unimportant in the acute stage but very important in the chronic and AIDS stages while K1 is important in the acute and chronic stages but unimportant in the AIDS stage" | w1(K4): 0.00 → 0.36 (t = 1000) → 0.65; w1(K1): 0.5–0.6 until t = 1000 → ≈0 after 2000. Published: K4 is removed in the acute stage (p.726) and is the top weight at t = 1700 (p.719) | none | confirmed (data, published) |
| 14 | "few variables have weight near zero for the entire course of the disease, so one can't significantly reduce the dimension … globally" | agrees with the published conclusion (p.726: "a true global-in-time parameter reduction is highly unlikely"). The paper says at most K11, K12, K13 could be dropped. The shipped data give 7 parameters with max \|w1\| ≤ 0.0014 over all 21 times: K2, K8, K11, K12, K13, δ3, α1 | optionally name these 7 | confirmed (data); differs from published count |
| 14 | "code similar to above (contained in the 'generate_data.py' file) was used to generate 1000 samples … This took about 8 hours" | the paper's 1000 samples used MATLAB ode23s and forward FD with δ = 1e-6 on normalized parameters (published pp.716–718), so the notebook data are an independent regeneration. Runtime not in paper. The paper's MATLAB scripts were posted at http://inside.mines.edu/~pankavic/activeHIV (p.710; current availability not checked) | say that the data were regenerated in Python for this notebook, with solver and step; drop or update the runtime | confirmed (methods differ) |
| 5, 14 | times list includes 5, 15 and 750 days | the published Appendix B (Figs. 15–17, pp.732–733) shows 18 times (24, 38, 40, 45, 50, 55, 65, 90, 140, 500, 1000, 1600, 1800, 2000, 2200, 2400, 2800, 3400), not 5, 15, 750. The text also uses 1700 and 2600 | fine, but say so | confirmed (published) |
| 20 | "summary plots, eigenvalues/vectors, and the time-dependence … look extremely similar between the sample sizes" | true for the first 20 shipped rows at t = 24 (top weights identical to 2 decimals) | none | confirmed |

### 3b. Errata / issues in the published paper (Math. Biosci. Eng. 14(3) 2017), for the coordinator's errata table

| page | published text | issue | evidence | confidence |
|---|---|---|---|---|
| 721, Fig. 7 | t = 2000: "the largest spectral gap exists between the second and third eigenvalues" (λ2 ≈ λ1) | not reproduced: verified Python gradients (1000 samples, same ±2.5% box) give λ1/λ2 = 75 and λ2/λ3 = 22.5 | regenerated `gradients.csv` (PR #51), checked against rtol 1e-12 `solve_ivp` to about 1e-5 | likely an artifact of noisy MATLAB FD gradients |
| 717, Fig. 2 | eigenvalue tail at T(1700) of about 20–500 | 1–2 orders above the verified data (λ3 ≈ 10 at t = 1600–1800) | same | likely (FD/solver noise) |
| 726 | "at most 3 of the 27 parameters - namely, K11, K12, and K13 - could be eliminated" | verified data also give K2, K8, δ3, α1 with max \|w1\| ≤ 0.0014 over 21 times | same | likely (different time mesh; noise inflates small weights) |
| 726–727 | acute stage: K3 "maximum weight during the first 40 days is 0.1256"; s3, K7, δ6 below 0.032 | verified data: K3 max 0.203; s3 0.056, K7 0.125, δ6 0.064 | same (includes t = 5, 15, not in the paper's mesh) | unverified (mesh-dependent) |
| 717, 723 | "the patient's T-cell count could vanish"; "T(2600; y) = 0" | under Eq. (1), dT/dt = s1 > 0 at T = 0, so T stays positive. "Vanish" should read "becomes negligibly small" | Eq. (1), p.711 | confirmed (wording) |
| 721 vs Fig. 7 right | "[0, 3400] is divided into 86 non-uniform subintervals" | the Fig. 7 right panel (dimension vs time step) has an x-axis ending near 55 | figure read at 110 dpi | unverified (axis may be truncated) |

## 4. Code review

- **Solver silently fails.** `Tcells` uses `ode(...).set_integrator('vode', nsteps=1500, method='bdf', max_step=1)` and never checks `solver.successful()`. In the venv it warns "Excess work done". A forward FD with step 1e-7·half-range (2.5e-9 relative) on top of that gives a t = 3400 gradient for sample 0 that differs from the shipped one by a relative error of 130.
- **Discontinuous RHS.** `if T < .1: return np.zeros(7)` and the zero Jacobian below the threshold. With LSODA at tight tolerances a frozen trajectory takes minutes, because the solver chatters at the switch. An event at T = 0.1 (`terminal=True`, then hold T = 0.1) gives exactly the same frozen value semantics in 0.07 s and an exact zero gradient.
- **Wasted computation.**
  - Cell 5 and `generate_data.py` call `Tcells(x, np.linspace(1, t_i, t_i))` once per time, re-integrating from 0 each time: 21× redundant. One trajectory per FD perturbation gives all 21 outputs.
  - Daily output (`linspace(1, 3400, 3400)`) with `max_step=1` forces at least 3400 steps.
  - `solve_ivp(LSODA, rtol=atol=1e-10, t_eval=times)` takes 0.04–0.07 s per 3400-day solve, against 0.5 s for legacy.
- **Fixed ICs.** `M0 = .15/.005` and `CTL0 = 5/.015` are hard-coded at nominal values, not `s2/d4`, `s3/d6` of the sample. Keep this, since it reproduces the shipped data, but document it.
- **Randomness.** `np.random.uniform` is unseeded in cell 3 and in `generate_data.py`. The shipped data cannot be regenerated bit-for-bit. Use `np.random.default_rng(seed)`.
- **State across cells.** `ss` is reused across cells 7–19, as are `weights` (from 13 to 19) and `times`. Cell 9 does `ss.W1 *= -1.0`, which mutates shared state; v1 `W1` is read-only, so use `w = sub.W1[:, 0].copy()`.
- **Sign rule (cells 9, 13, 17, 19).** "enforce downward trend" via `f[argmin(y)] < f[argmax(y)]` compares two single points and is fragile. Use `s = -np.sign(np.corrcoef(y, f)[0, 1])`. This is needed for the weight-vs-time plot, because v1's largest-component rule can flip between times. **Superseded (V1-05, [#51](https://github.com/paulcon/as-data-sets/pull/51)):** t = 24 is oriented with `ac.orient(by="covariance")` against −f (downward trend), and each later time `by="reference"` against the previous one.
- **py2.** `print '...'.format` in `generate_data.py`. `np.linspace(1, times[i], times[i])` works in numpy 2, since an `np.int64` count is accepted.
- **pandas.** `pn.DataFrame.from_csv` and `.as_matrix()` were removed in pandas 1.0.
- **Hard-coded paths.** Relative CSV names only.

## 5. Data management review

- **Format.** Three pandas CSVs with an index column (0..999) and integer headers. Names, units and coordinate conventions are not in the files.
  - `gradients.csv` is w.r.t. **normalized** inputs, while `inputs.csv` is physical.
  - The normalization is not stored but is reconstructible: nominal ± 2.5%.
- **Size.** 9.5 MB total, under the 50 MB warning. Recommendation: keep the CSV format.
  - Optionally rewrite with `np.savetxt(..., header='s1,s2,…', comments='# ')`, drop the index column, and add a provenance comment line (solver, FD step, freeze threshold, ICs, coordinates).
  - Do not convert to physical gradients, because that would change the shipped numbers.
- **Quality.**
  - At frozen samples (f < 0.1001) the gradient rows are FD noise; the true value is 0 under the model as coded. Options: zero them in the notebook with a documented mask, or leave them and add a note. Prefer the mask plus note; the effect on λ1/λ2 is under 3%.
  - Non-frozen rows verified against an independent solver (§10).
- **Provenance and license.** Generated by `generate_data.py` (Ryan Howard, 2016), not by the paper's MATLAB code. Say so.
- **Tested pandas-free loading (venv, 0.08 s):**

```python
import numpy as np
X  = np.loadtxt('inputs.csv',    delimiter=',', skiprows=1)[:, 1:]   # (1000, 27) physical
f  = np.loadtxt('outputs.csv',   delimiter=',', skiprows=1)[:, 1:]   # (1000, 21) T(t_i)
df = np.loadtxt('gradients.csv', delimiter=',', skiprows=1)[:, 1:]   # (1000, 567) d T(t_i)/d x_j, normalized x
G = df.reshape(1000, 21, 27)                                        # G[:, i, :] = gradients at times[i]
```

## 6. Migration to active_subspaces v1

| legacy call | v1 replacement | notes |
|---|---|---|
| `xl = .975*nominal; xu = 1.025*nominal`; hand affine map; `BoundedNormalizer(xl, xu).normalize(X)` | `space = ac.Uniform(lower=xl, upper=xu, names=labels)`; `space.sample(N, rng=rng)`; `space.to_normalized(X)` only where y is needed by hand | |
| `ac.gradients.finite_difference_gradients(p, fun, h)` (physical step `h = 1e-7*(xu-xl)/2`, then `*= (xu-xl)/2`) | `ac.finite_difference_gradients(model, X=P, space=space, h=1e-6)` → `GradientResult`, with `model` a batch callable `(k, 27) → (k,)` | `h` is relative and in normalized coordinates; do not rescale by hand. Tested at t = 24, N = 20: 3.0 s, λ1 = 9.7e3, w1 matches the shipped-data result |
| `ss.compute(df=df[:, 27*i:27*(i+1)])` on shipped data | `ac.active_subspace(space=space, grads=G[:, i, :] / ((xu-xl)/2))`, or `ac.active_subspace(space=ac.Uniform.reference(27), grads=G[:, i, :])` | the shipped gradients are already normalized: divide by the half-range or use the reference space, **never both** and never neither. Both variants were tested and give identical eigenvalues |
| `ss.partition(2)` | `ac.with_dimension(sub, 2)` | EVG picks n = 1 at all 21 times |
| `ss.eigenvals`, `ss.W1` | `sub.eigenvalues` (27,), `sub.W1` (read-only) | |
| `ss.W1 *= -1.0` | `w1 = s * sub.W[:, 0]` (copy), with `s = -sign(corr(y, f))` | v1 sign rule: largest-magnitude component positive. **Superseded (V1-05, [#51](https://github.com/paulcon/as-data-sets/pull/51)):** `ac.orient`, as in §4 "Sign rule". |
| `ac.utils.plotters.eigenvalues/eigenvectors/sufficient_summary` | `ac.plotting.plot_eigenvalues(sub)`, `plot_eigenvectors(sub, indices=[0] or [0,1], labels=in_labels)`, `plot_sufficient_summary(sub, X=X_phys, f=f[:, i], dims=1 or 2)` | the summary plot takes **physical** X. For the 7×3 grid and the weight-vs-time plot, keep hand matplotlib with the oriented `w1` |

## 7. Associated .py files

- **`HIV_model.py`: rewrite (keep the name and the `Tcells(p, times)` signature).**
  - Per sample, call `solve_ivp(rhs, (0, times[-1]), y0, method='LSODA', t_eval=times, rtol=1e-10, atol=1e-10, events=floor)`. Use `floor(t, y) = y[0] - 0.1`, `terminal=True`, `direction=-1`, and fill the remaining outputs with 0.1.
  - Keep the same ICs. Optionally pass the analytic Jacobian, which is verified correct.
  - Measured cost: 0.04–0.07 s per 3400-day trajectory.
  - Agrees with the 2016 `outputs.csv` to a relative 1.4e-5 at the samples first tested, and to 5.5e-4 over all samples (at T < 3). Agrees with the 2016 non-frozen `gradients.csv` rows to 1e-5 at those samples, but only to 0.16 over all samples, because of errors in the 2016 files (see §1.1).
- **`generate_data.py`: port to py3 or mark as legacy.**
  - Port: numpy I/O, seeded rng, one trajectory per FD perturbation for all 21 times, central FD with h = 3e-4 (normalized), and zero gradients after the floor event. With h = 1e-4 the error against an rtol 1e-12 solve is 3–5e-5, from solver noise; with h = 3e-4 it is about 1e-5. Step 3e-4 confirmed (author, 2026-10-30).
  - Estimated cost: 1000 × 54 solves × ~0.05 s ≈ 45 min.
  - ~~Do not regenerate the shipped CSVs by default.~~ Superseded (author, 2026-10-30). The CSVs were regenerated at the 19 times and replace the 2016 values, which stay at `legacy-master`. The measured cost was 33 min on 3 processes.
  - If the author prefers, delete the script and document provenance in the notebook.

## 8. Update plan

1. Fix the text and README:
   - README: 27 parameters.
   - Cite MBE 14(3) and fix the CDW title.
   - Fix typos.
   - State the ICs and the T = 0.1 floor.
   - Say that the data were regenerated in Python (solver, step) and are not the paper's MATLAB runs.
   - Note the extra times 5, 15 and 750.
   - Note that the data are a cleaner regeneration than the paper's MATLAB runs. Do not repeat the paper's t = 2000 "2-D" claim.
   - **Keep the notebook path `HIV/HIV.ipynb` on master unchanged**, because the published paper links to it (p.710).
2. Rewrite `HIV_model.py` with `solve_ivp` + event (§7). Add a check cell: max relative difference against `outputs.csv` < 1e-4 for 5 rows.
3. Replace the small-sample section (cells 3–13). Keep the 10 representative trajectories (20 solves, < 2 s). Compute v1 FD gradients **only at t = 24** for N = 20 seeded samples (3 s), with eigenvalue, eigenvector and summary plots. Drop the in-notebook 21-time FD loop.
4. Large-sample section: numpy loading, v1 subspaces per time, oriented `w1`, mask frozen samples' gradients (documented), and the summary grid and weight-vs-time plot.
5. Port or retire `generate_data.py` (§7).
6. Convert the kernel to py3, execute headless, and commit the outputs.

## 9. Acceptance criteria

- `jupyter nbconvert --execute` from `HIV/` in `.venv` finishes in < 2 min, with no pandas and no warnings from the ODE solver.
- Shipped data: t = 24 has λ1 = 1.046e4 (±0.5%), λ1/λ2 > 3000, and w1 top-4 (oriented) = K1 .51, K9 .49, ψ .47, δ7 −.47 (±0.01).
- t = 3400 has λ1 = 4.54e5 (±1% without masking), λ1/λ2 ≈ 850, and w1 top = K4 .66, δ7 −.38, δ4 −.35, K5 −.28 (signs per orientation).
- Weight-vs-time plot: K1 ≈ 0.5–0.6 for t ≤ 500, ≈ 0.38 at t = 1000 (also in the 2016 data), and ≈ 0 for t ≥ 2000; K4 ≈ 0 for t ≤ 140 and ≥ 0.57 for t ≥ 1600.
- Small-sample t = 24 (N = 20, v1 FD): λ1 within 20% of the shipped-data λ1, and the same four dominant parameters.
- Optional, if t = 1700 is added: w1 has \|w\| > 0.3 exactly for K4, δ4 and δ7, as in published p.719 / Fig. 3 (true at both 1600 and 1800 in the shipped data).
- Figures: trajectories (1), t = 24 eigen/eigvec/summary (3), 7×3 summary grid (1), t = 3400 eigen/eigvec/1-D/2-D summary (4), weight-vs-time (1), for both the small and large sample where applicable.

## 10. Prototype results

Scripts are in `prototypes/g1/`: `hiv_load.py`, `hiv_fd*.py`, `hiv_v1fd.py`, and `hiv_model_copy.py` (a copy of `HIV_model.py`).

- **Loading:** numpy loads all three CSVs in 0.08 s with the shapes above. The index columns are 0..999 and there are no NaNs.
- **v1 subspaces:** v1 on the shipped data reproduces the legacy figures: t = 24 λ1 ≈ 1e4, t = 3400 λ1 ≈ 4.5e5 with λ2 ≈ 5e2, and the same eigenvector shapes.
- **Gradients, non-frozen samples:** an independent tight `solve_ivp` with central FD (h = 1e-3 and 1e-4) matches the shipped gradients. For sample 0 the relative error is ≤ 7e-5 at t = 24, 500, 1600, 2200 and 3400. For samples 1, 2 and 4 at t = 2800 and 3400, cos = 1.000 and the norms are equal to 3 digits.
- **Gradients, frozen samples:** for samples 3, 7, 9 and 12, the floor crossings are at t = 2434–2978. The event-based gradient is exactly 0; the shipped norms are 12–17 (noise).
- **Legacy code path under py3:** the notebook's `Tcells` plus forward FD with h = 1e-7, for sample 0 at t = 3400, gives a relative error of 130 against the shipped gradient (12 s for 28 solves).
- **Timing:** legacy `Tcells` takes about 0.5 s per 3400-day trajectory; `solve_ivp` with the event takes about 0.07 s.
- **Comparison with the published figures** (PDF pp.717, 718 and 721, read at 110 dpi):
  - Fig. 3 (w1 at 1700) has K4 ≈ .58, δ4 ≈ −.33, δ7 ≈ −.46, δ6 ≈ .29 and K9 ≈ .26. The shipped data at 1600 give K4 .57, δ4 −.33, δ7 −.47, δ6 .29, K9 .26: they match.
  - Fig. 2 eigenvalue tail and Fig. 7 (t = 2000, λ2 ≈ λ1) do not match: see §1.3.
  - For the acute window (t ≤ 40) at threshold 0.032, the shipped data drop 14 of the paper's 17 parameters. s3 (0.056), K7 (0.125) and δ6 (0.064) stay above the threshold, and K3's max is 0.203 vs the paper's 0.1256 (p.727). The likely cause is different times and the noisier MATLAB gradients.

## 11. Open questions for the author

1. The T < 0.1 freeze is not in the paper. Keep it as a floor (it matches the shipped data), or drop it and let T decay, which would require regenerating the late-time data?
2. Frozen-sample gradients are noise: zero them in the notebook, or leave them as shipped and add a note?
3. Initial conditions are fixed at the nominal uninfected equilibrium. Is that intended, or should they be sample-dependent?
4. Should the notebook reproduce the paper's 18 times only, or keep 5, 15 and 750? Add t = 1700 (the paper's worked example, Figs. 2–6)?
6. The paper's t = 2000 result (λ2 ≈ λ1, n = 2; Fig. 7) and its eigenvalue tails are not reproduced by the verified Python data (λ1/λ2 = 75). Were the MATLAB gradients noisy (ode23s, 1e-6 forward step)? Should the notebook comment on it?
7. The published p.726 says at most K11, K12, K13 could be eliminated globally. The shipped data also give K2, K8, δ3 and α1 with max |w1| ≤ 0.0014. Which statement should the notebook make?
5. Keep `generate_data.py` (ported, ~45 min) or retire it?
