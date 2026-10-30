# active_subspaces v1: library changes needed to port as-data-sets

2026-10-29

**Purpose.** This report lists what must change in `active_subspaces` v1 so that the as-data-sets notebooks can be ported without hand-written workarounds. It is written for the maintainer agent of `ActiveSubspacesRevision`. Every item was reproduced against the installed v1 (`upstream-v1` at `b4e82fb`, `active_subspaces 1.0.0.dev0`, Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1, macOS, 34 GB) using the real data files. Each item names the rulings it touches (D-numbers in `design-specs/decisions.md`) and the `design-specs/target/` spec it would amend. Where a ruling excludes a change, the item says so and gives either the argument for revisiting it or a docs-only resolution. Under D-03b and CLAUDE.md, the author rules before any change to the mathematics. **How to use it:** take the summary table as the work list. Each item's "Design constraints" says whether an author ruling is needed first. The acceptance tests are written so they can go into `tests/` or `docs/`. The repro scripts (`r01`–`r15`, with their outputs) are in `/Users/paulconstantine/PycharmProjects/as-data-sets/specs/prototypes/v1report/`. The data files, notebook specs (`specs/<Notebook>.md`) and published papers (`papers/`, local only) are in the same as-data-sets checkout. The code and output needed for each item are quoted below, so the report does not depend on the scripts.

Notation: "normalized" and "physical" as in target/03. File references are to `upstream-v1/src/active_subspaces/`.

## Summary

Priority: **blocks** = the default v1 call gives a wrong or misleading published number, or the notebook must hand-code a published method's numerics. **workaround** = v1 can do it, but only through an undocumented or mislabelling idiom. **nice** = convenience, speed or clarity.

| ID | Change | Kind | Priority | Notebooks | Rulings / target spec |
|---|---|---|---|---|---|
| V1-01 | RBF length-scale rule: cite CDW 2014 §4.2–4.3; flag the departures; add the published bounds on the hyperparameter | bug + D-03b process | blocks | Elliptic_PDE (also ONERA-M6, Atacamac RBF use) | D-36, D-100, D-101, D-172, D-03b; target/09 §3.3, §5.2; references.md |
| V1-02 | Bootstrap replicates of the leading eigenvectors, sign-aligned (published Algorithm 2) | new feature (schema) | blocks | HyShotII, Lithium_Battery, SingleDiodePV | D-30, D-131(b), D-178, D-63; target/07 §3.7, §4.4; target/13 §3.3 |
| V1-03 | Prediction and confidence intervals for polynomial surrogates | new feature | blocks | Hydrology, HyShotII | D-97, D-99, D-104; target/09 §3.2, §4.2, §4.4 |
| V1-04 | OLS reports the published signed direction ŵ = b/‖b‖ | behavior change | workaround | Hydrology, Stomatal_Resistance, Subsurface_Permeability, Lithium_Battery, HyShotII | D-81, D-188, D-03b (P7); target/07 §3.2, §3.4 (OLS) |
| V1-05 | `orient` helper: data- or reference-defined eigenvector signs that plots and `physical_directions` keep | new feature | workaround | HIV, MCMC, Lithium_Battery (QPHD), Elliptic_PDE, NACA0012, MHD, Stomatal_Resistance | D-81, D-11 (amended), D-18; target/07 §3.2, §4.5; target/03 §3.3 |
| V1-06 | Product space of independent `Uniform` and `Gaussian` blocks | new feature | workaround | Hydrology (Reentry, borehole E4: evidence) | D-12, D-13, D-02b, D-21, D-39, D-40; target/03, 07 §3.5, 04, 08 |
| V1-07 | `activity_scores(subspace, n=)` | new feature | nice | Ebola | none; target/07 §4.5 |
| V1-08 | `subspace_from_matrix(C, space=)`, and lower peak memory in the AS estimator | new feature + perf | nice | Ebola | D-102 (amendment), D-29, D-63; target/07 §3.3, §4.5 |
| V1-09 | Finite differences for a model with several outputs (one run per stencil point) | new feature | nice | HIV | D-08, D-69, D-23, D-83; target/07 §3.8.2, target/12 |
| V1-10 | `input.out_of_space`: count rows, name the input and bound, print a plain float | bug (message) | nice | NREL_Wind, Hydrology | D-16; target/02, 06 |
| V1-11 | QPHD least squares: QR instead of `gelsd` when full rank (about 3× faster) | perf | nice | Subsurface_Permeability | D-40; target/07 §3.4, §5.1 |
| V1-12 | `Uniform.reference(m, *, names=, units=)`, same for `Gaussian` | new feature | nice | Lithium_Battery, Hydrology, Atacamac, Reentry | D-58; target/03 §4, §5.4 |
| V1-13 | Docs corrections (OLS citation, OLS under hand normalization, RBF "R²" wording, discrete inputs, eigenvalue scaling, stored normalized gradients) | docs | nice | all | D-41, D-97, D-166; docs only |

**Candidate gaps that need no library change** (§"No library change needed"): many QoIs at once (#10), the 8^8 Ebola rule (#5, as stated), data outside the declared box and log of zero in plots (#8), eigenvalue scaling conventions (#9), dimension-rule defaults and in-sample R² (#11), and `ValidityWarning` for OLS (#12). **Stays in notebooks:** AS-accelerated MCMC (#13) and the models themselves.

---

## V1-01 RBF length-scale rule: attribution, departures and published bounds

**Need.** Elliptic_PDE reproduces Constantine, Dow and Wang 2014 (CDW), Fig. 5/Table: kriging on the active variables from a 5-point (n = 1) or 5×5 (n = 2) design, β = 1 (`kl-long`), mean relative error on 500 test points. The paper reports 0.178 (n = 1) and 0.149 (n = 2).

**Current v1 behavior.** Data as in `specs/prototypes/g2/proto_long2.py` (seed 2014, M = 300 adjoint gradients, `Gaussian.reference(100)`):

```python
rs = ac.fit_surrogate(X=Xd, f=fd, subspace=ac.with_dimension(sub, n), kind="rbf", degree=2)
```
```
sum|lam|=1.1844e-02 lam1=1.1812e-02 sigma_hat^2 (300 samples)=6.9901e-03; CDW (4.13): 0.590 <= alpha <= C1 = 1
n=1 v1 default: log10_eta=-1.87 ell=[0.0137] nugget=3.7e-05 mean rel err 1.013
   v1 J(log10_eta) on -10..1 (step 1): [-13.41 -13.41 -13.41 -13.41 -13.41 -13.41 -13.41 -13.41 -13.41 -13.41 -13.2 -10.94]
n=1 v1 with log10_eta_bounds from CDW (4.13) (eta = 2 alpha) [0.072 0.301]: mean rel err 0.095
n=1 CDW §4.3 implemented by hand (alpha in [0.590, 1]):                   mean rel err 0.095
n=2 v1 default: log10_eta=-1.34 ell=[0.0456 41.9] mean rel err 0.435
n=2 v1 with (4.13) bounds: 0.300;  CDW §4.3 by hand: 0.319;  v1 log10_eta fixed so ell1 = 1: 0.154
kl-short (beta = .01), n = 1, 2: default 0.008; with alpha = C1: 0.008. Here 2*sig2/sum|lam| = 2.22 > 2*C1, so the (4.13) interval is empty (sampling error).
```

- With 5 points and 3 tail functions, the concentrated objective J is flat to 4 digits from log10 η = −10 to −1. The grid search (surrogates.py:535–560, D-101) then returns an arbitrary point in the flat region, and the surrogate is a near-interpolant with ℓ1 = 0.014.
- The docstring at surrogates.py:1548 says the length-scale rule is "a heuristic of the 2016 package with no published reference". So do target/09 §3.3, `design-specs/references.md:36`, `docs/theory/surrogates_and_designs.md:50` and `:87`, and `docs/user_guide/surrogates.md:127`. D-100 says the same: "the 2016 heuristic has no published reference, so no published method is departed from".
- **The rule is published:** CDW 2014 §4.2, eqs. (4.8)–(4.14), pp. A1513–A1514, and Algorithm §4.3 step 5, eq. (4.16), p. A1515. CDW sets ℓ_i² = α Σλ/λ_i, σ² = α Σλ (4.10) and noise η² = α Σ_{q>n} λ_q (4.14). The single hyperparameter α is chosen by maximum likelihood **within σ̂²/Σλ ≤ α ≤ C1** (4.13), where σ̂² is the biased sample variance of the M gradient-sample values (4.11) and C1 ≤ 1 for a standard Gaussian. (CDW §3.3 is the generic response-surface section; the rule is in §4.2–4.3.) v1's kernel is exp(−Σ_j Δ_j²/ℓ_j) with ℓ_j = η Σ|λ|/|λ_j|, so η ≡ 2α.

**Design constraints.** D-36 (concentrated likelihood), D-100 (nugget δ = η·Σ_{q>n}|λ|/Σ|λ|, amplitude profiled), D-101 (grid on [−10, 1]) and D-172 (accepted unit-scale accuracy) all rest on the premise that there is no published method. That premise is false, so under D-03b and P7 the departures must be flagged and ruled. Relative to CDW §4.2–4.3, v1 departs in three ways:
1. The amplitude is profiled (σ̂²_η), where CDW ties it to α (σ² = αΣλ).
2. The relative nugget is η·Σ_{q>n}λ/Σλ; in CDW it is Σ_{q>n}λ/Σλ, independent of α.
3. The search interval is [−10, 1] with no lower bound, where CDW bounds α by (4.13).

Departure 3 causes the failure. Restoring only the bounds inside v1 gives the same error as a hand implementation of CDW (0.095 and 0.300 against 0.095 and 0.319).

**Proposed API.**
```python
fit_surrogate(*, X, f, space=None, subspace, kind="rbf", degree=2, ...,
              log10_eta=None, log10_eta_bounds=None,          # None: rule below
              f_variance: float | None = None)                 # CDW (4.11): variance of the gradient-sample f
```
- When `log10_eta_bounds is None`, `subspace` has eigenvalues and `f_variance` is given, the bounds become [log10(2·f_variance/Σ|λ|), log10(2·C1)]. C1 = 1 for a Gaussian space; for a uniform space the author must choose between 2√m/π and its square (open Elliptic_PDE Q4). An empty interval collapses to its upper end and issues `AccuracyWarning` `accuracy.empty_bounds`. Otherwise the bounds stay [−10, 1].
- New warning `AccuracyWarning` `accuracy.flat_objective` when J over the grid varies by less than 1e-8·|J| across the selected point's neighbourhood of at least 4 decades.
- The author also rules departures 1 and 2: keep them, flagged, or adopt CDW's tied amplitude and α-free nugget.
- `RBFSurrogate` records `log10_eta_bounds` and `f_variance` in provenance.

**Acceptance tests.**
- Elliptic β = 1, data of `proto_long2.py`, `f_variance=f0.var()`: n = 1 mean relative error ≤ 0.12 (measured 0.095; paper 0.178); n = 2 ≤ 0.33 (0.300).
- β = 0.01: unchanged, ≤ 0.01 (0.008), with `accuracy.empty_bounds` issued.
- The n = 1 default call without `f_variance` issues `accuracy.flat_objective`.
- The `surrogates/rbf_scale` baseline (D-172) keeps its equivariance check.

**Amend.** target/09 §3.3, §5.2, §7, §9; target/16 (traceability of D-100); `design-specs/references.md:36` (remove "RBF length-scale heuristic" from the unpublished list; cite ConstantineDowWang2014 §4.2–4.3); new D-entry amending D-100 and D-101; surrogates.py docstring at :1548; `docs/theory/surrogates_and_designs.md:50`, `:87`; `docs/user_guide/surrogates.md:127`; `docs/migration.md` "Method choices ruled by the author".

## V1-02 Bootstrap replicates of the leading eigenvectors

**Need.** Three published analyses report component-wise bootstrap variability of w:
- HyShotII (Constantine et al., JCP 2015, §4.2 **Algorithm 2**, Fig. 8): histograms of the bootstrap replicates w_k from 10 000 resamples of the OLS fit.
- Lithium_Battery (Constantine and Doostan 2017, §2, Fig. 9): "the bootstrap standard error for each component of w". It covers the OLS w and the first QPHD eigenvector, at 50 capacity and 48 voltage points × 3 rates.
- SingleDiodePV (Constantine, Zaharatos, Campanelli 2015, §2.2): bootstrap replicates Ŵ₁ʲ used for bands and in summary plots.

All three notebooks hand-code the resampling and the sign alignment.

**Current v1 behavior.**
```python
s = ac.active_subspace(X=X, f=C[:, 2], space=space, estimator="ols", n=1, nboot=100, rng=0)   # Lithium 0.25C
```
```
SubspaceResult fields: [... 'bootstrap_eigenvalues', 'bootstrap_distances', 'bootstrap_dets']
bootstrap arrays: (100, 19) (100, 18) (100, 18)
rank-one OLS, bootstrap_distances[:,0] (sin angle w, w^(b)): median 0.0064 max 0.0107
hand bootstrap SE of sign-aligned w (idx 2): max 0.0019
```
Each replicate's eigenvectors `Wb` are computed and discarded at subspaces.py:2038 (`_replicate`). The fields are at subspaces.py:1794–1796. `docs/migration.md:554` lists `W_br` as removed.

**Design constraints.** D-131 declined (b) for three reasons:
1. Storing (B, m, m) costs 800 MB at B = 100, m = 10³.
2. Signs and rotations make component bands ill-posed.
3. Alignment is "not part of the published method".

Argument for revisiting, limited to the leading k ≤ n columns:
- Cost is 8·B·m·k bytes: 0.8 MB at B = 100, m = 10³, k = 1. It sits within D-63.
- For k = 1, and for any column whose eigenvalue is separated, sign alignment (w_kᵀw_k^(b) > 0) is canonical.
- The alignment *is* in the published methods. JCP 2015 eq. (22) defines w = b/‖b‖ (signed). Constantine and Doostan 2017 apply standard bootstrap SEs to regression coefficients (Efron and Tibshirani ch. 7). The PV paper uses the replicate eigenvectors directly.
- Rotation within near-repeated eigenvalues remains ill-posed. The library can warn when |w_kᵀw_k^(b)| is small instead of refusing.

This needs a new ruling amending D-131(b) and D-30.

**Proposed API.**
```python
active_subspace(..., nboot=0, bootstrap_vectors: int = 0, rng=None)
# SubspaceResult.bootstrap_eigenvectors: (B, m, k) float64 | None, k = bootstrap_vectors <= n,
# column j of replicate b multiplied by sign(W[:, j] @ Wb[:, j]) (positive inner product with the full-sample vector)
bootstrap_interval(result.bootstrap_eigenvectors[..., 0], kind="percentile" | "minmax", level=0.95)  # existing helper
ac.plotting.plot_eigenvectors(sub, indices=[0], interval="percentile" | "minmax" | None, level=0.95)
```
- `AccuracyWarning` `accuracy.unaligned_replicates` when min_b |w_jᵀw_j^(b)| < 0.5 for a stored column.
- Random streams are unchanged (D-15), and `SCHEMA_VERSION` of `SubspaceResult` goes to 2 with a migration (results.py `_register_migration`).

**Acceptance tests.**
- For OLS, n = 1: `bootstrap_eigenvectors[b, :, 0] @ W[:, 0] > 0` for all b, and `bootstrap_distances[b, 0] == sqrt(1 - (W[:,0] @ Wb)**2)` to 1e-12, on Lithium 0.25C idx 2 and HyShotII φ = 0.30 (50 rows).
- Lithium 0.25C capacity idx 2, B = 100: max component SE in [0.0012, 0.0028] (hand: 0.0019).
- HyShotII φ = 0.30, B = 10 000: the component ranges of the replicates equal a hand Algorithm 2 run on the same index draws to 1e-12.
- SingleDiodePV (`grads=G/((xu-xl)/2)`, physical space, B = 500): the w₁ band contains w₁ = [0.767, …] and has width < 0.05 per component.
- With the default `bootstrap_vectors=0`, results are bitwise equal to the current ones.

**Amend.** New D-entry amending D-131(b) and D-30; target/07 §3.7, §4.4, §5.3 (stream layout unchanged); target/13 §3.3, §4.2 (band on `plot_eigenvectors`); target/04 (schema version); target/18 and `docs/migration.md:554` (`W_br` now has a replacement).

## V1-03 Prediction and confidence intervals for polynomial surrogates

**Need.**
- Hydrology (Jefferson et al. 2015) shows quadratic fits with confidence and prediction bands; the legacy notebook used a straight-line leverage formula (bug, TODO §3).
- HyShotII needs the quadratic prediction limit to bound the safe box (JCP 2015 §5.3; spec §9: φ = 0.35 AoA ≤ 3.193 with the exact limit).

**Current v1 behavior.** `PolynomialSurrogate` stores `Y`, `f`, `sample_weights`, `indices`, `coefficients`, `residuals`, `loo_residuals` (surrogates.py:628). No interval function exists: `ac.surrogates.__all__` has none. The notebooks hand-code ŷ ± t_{M−p} s √(1 + b(y)ᵀ(BᵀSB)⁻¹b(y)):
```
HyShotII phi=0.30 (M=50) 95% PI at y=-2.0127,0,+2.0127: lower [2.0348 2.3672 2.7402] upper [2.0961 2.4065 2.7982]
   95% CI at y=0: [2.3836, 2.3901]
Hydrology grass LH 95% PI half-width at y=0: 1.3919e+05; CI half-width 1.0119e+04
```

**Design constraints.** No ruling covers intervals. The D-97 leverages h_ii = s_i b_iᵀ(BᵀSB)⁻¹b_i are already computed for LOO, so the interval reuses that factorization. Under D-104, active-coordinate methods take `Y`, functions take physical `X`. Gaussian-error OLS theory is the standard method (cite e.g. Montgomery, Peck and Vining, *Introduction to Linear Regression Analysis*, §3.4–3.5). Restrict to `kind="poly"`. RBF predictive variance is a separate question; under D-36 it would be the GP variance, so leave it out.

**Proposed API.**
```python
ac.surrogates.predict_interval(surrogate: PolynomialSurrogate, X: ArrayLike, *, level: float = 0.95,
                               kind: Literal["prediction", "confidence"] = "prediction",
                               side: Literal["two", "lower", "upper"] = "two") -> IntervalResult
PolynomialSurrogate.interval(Y, *, level=0.95, kind="prediction", side="two") -> (lower (K,), upper (K,))
# IntervalResult (frozen): estimate, lower, upper, level, kind, side, dof = (#positive weights) - p, residual_scale
```
- `InputError` `input.parameter` for an RBF surrogate, and for dof ≤ 0 with `details["reason"]="no_dof"`.
- With `sample_weights`, s² = Σ s_i r_i²/(M − p) and a new point has weight 1 (documented).

**Acceptance tests.**
- HyShotII φ = 0.30, 50 rows, OLS n = 1, degree 2: the bounds above to 1e-4.
- Hydrology grass LH: PI half-width at y = 0 is 1.3919e5 and CI half-width 1.0119e4 (rtol 1e-4). Both equal the hand formula to 1e-10 for all six cases.
- `side="lower"`, `level=0.99` equals the one-sided t quantile.
- Root-finding the HyShotII φ = 0.35 limit with `xmin = -sign(w)` reproduces AoA ≤ 3.193 (spec §9).

**Amend.** target/09 §3.2, §4.2, §4.4, §6 (codes), §8; target/13 optionally adds `plot_surrogate_fit(..., interval=)`; `docs/user_guide/surrogates.md`.

## V1-04 OLS: report the published signed direction

**Need.** Every OLS notebook follows its paper in orienting w as the normalized slope ŵ = b/‖b‖. Then f increases along w, and directions stay continuous across voltages, months and hours:
- Constantine 2015 book, §1.1 Algorithm 1.3, eq. (1.6), p. 5;
- JCP 2015 eq. (22);
- Jefferson et al. 2015 eq. 10;
- Jefferson et al. 2017 Fig. 9;
- Constantine and Doostan 2017 eq. 10.

**Current v1 behavior.** OLS returns w₁ = û/‖û‖ re-signed by the max-magnitude rule (`_from_rows` → `_orient_columns`, subspaces.py:159, spaces.py:265–277; D-81/D-188):
```python
s = ac.active_subspace(X=X, f=F[:, i], space=space, estimator="ols", n=1)   # one call per column
```
```
(v1 sign != â sign, v1 reversals between neighbouring columns, â reversals, N)
Lithium 0.25C capacity: (44, 1, 0, 50)   voltage: (12, 2, 0, 48)
Lithium 1C capacity:    (12, 1, 0, 50)   4C voltage: (5, 1, 0, 48)
Stomatal mo_avg_rssha (12, 0, 0, 12)  mo_avg_rssun (12, 0, 0, 12)  JJA_hr_avg_rssha (14, 0, 0, 14)  JJA_hr_avg_rssun (12, 1, 0, 14)
Hydrology: v1 disagrees with â for bare GH and grass LH, SH, GH (spec §10);  Subsurface Hortonian: disagrees (paper Fig. 7a w1 ≈ −0.65)
```
The notebooks multiply by `np.sign(w @ a_hat)` by hand. The plot functions then disagree with the hand-oriented w (see V1-05).

**Design constraints.** D-81/D-188 calls the sign "a presentation convention". For OLS the published method returns a signed vector, so the max-magnitude rule is a departure from Algorithm 1.3 (P7, D-03b). The OLS docstring (subspaces.py:632–680) cites Stein 1981 and Li 1991 but not the book's Algorithm 1.3 or JCP 2015. The change needs a ruling amending D-81 with an OLS exception. Bootstrap metrics are sign-invariant (D-81), so no other output changes.

**Proposed API.** No signature change. For `estimator="ols"`, `W[:, 0] = û/‖û‖`. The other columns are an orthonormal complement oriented by D-188. For û = 0, `accuracy.degenerate_matrix` and W = I, as now. The OLS docstring and theory pages cite Constantine 2015 Alg. 1.3 and JCP 2015 Alg. 1–2.

**Acceptance tests.**
- `W[:, 0] @ û > 0` for all 50 Lithium 0.25C capacity fits; consecutive reversals 0/49.
- Stomatal: 0 reversals in all six series; rssha/rssun weights for bp negative (spec §10 ranges).
- Hydrology grass lai weight > 0 for LH and < 0 for GH (paper p. 18); Subsurface Hortonian w₁ = −0.627 ± 1e-3.
- HyShotII w(0.30) = [0.6506, 0.5565, −0.0002, 0.3685, −0.3566, −0.0196, 0.0607] ± 1e-4 (unchanged).

**Amend.** New D-entry amending D-81; target/07 §3.2 (exception), §3.4 OLS, §3.9, §7; target/18 §7 and `docs/migration.md` "What changes in your numbers"; `docs/theory/estimators.md` (OLS references).

## V1-05 `orient`: data- or reference-defined signs that the whole result keeps

**Need.** Several notebooks compare a sequence of gradient-based subspaces, or must match a stored W:
- HIV: w₁ at 21 output times.
- Lithium_Battery: the first QPHD eigenvector across voltages.
- MCMC: the stored `AV_Chain.mat` was made with the 2016 W; v1 flips w₂, so the projected vanilla chain mean is −1.46 against the AV chain's +1.36 (spec §6).
- Elliptic_PDE: β = 1 w₂ mirrored against the legacy figure.
- NACA0012: drag w₂ mirrored.
- MHD: Hartmann-u w₁ flipped.

**Current v1 behavior.**
```python
subs = [ac.active_subspace(grads=Gh[:, 27*i:27*(i+1)], space=ac.Uniform.reference(27)) for i in range(21)]  # HIV shipped gradients
```
```
HIV v1 w1 reversals between consecutive times: 3 /20
dataclasses.replace(sub, eigenvectors=-W) accepted; provenance kept: True       # undocumented; provenance now wrong
plot_sufficient_summary x == X @ v1 w: True                                        # plotting.py:773 uses sub.eigenvectors
W1 [0.7071 0.7071] -> flipped [-0.7071 -0.7071]; physical_directions of flipped [0.995 0.0995] (sign restored by max-abs rule)
```
The flip is re-applied at spaces.py:561 (`directions_to_physical` → `_orient_columns`). So even a hand-flipped result shows inconsistent signs between `W1` and `physical_directions()`. `with_dimension` keeps the given eigenvectors (subspaces.py:2543–2561), so an oriented result would propagate correctly through it.

**Design constraints.** This keeps D-81 as the default and adds a pure function returning a new frozen result (D-02, D-18). `physical_directions` documents its output as "sign-oriented like W" (subspaces.py:1866), and target/07 §3.2 says `directions_to_physical` "applies the same sign rule" (D-81). With V1-04 and V1-05 this must change: the sign of each physical direction follows its normalized column, not an independent max-magnitude rule. Otherwise V1-04 and V1-05 are undone by `physical_directions`.

**Proposed API.**
```python
ac.orient(subspace: SubspaceResult, *, by: Literal["reference", "covariance", "mean_gradient"],
          reference: ArrayLike | SubspaceResult | None = None,   # (m, k) normalized, or a result
          X: ArrayLike | None = None, f: ArrayLike | None = None, grads: ArrayLike | None = None,
          columns: int | None = None) -> SubspaceResult
```
- Column j is multiplied by s_j ∈ {±1}:
  - `"reference"`: s_j = sign(w_jᵀr_j);
  - `"covariance"`: s_j = sign(Σ_i ω_i (y_ij − ȳ_j)(f_i − f̄)), with y = normalized X @ W;
  - `"mean_gradient"`: s_j = sign(ḡᵀw_j) from normalized grads.
- Ties (|criterion| ≤ √ε·scale) keep the D-188 sign.
- Records `orientation` and its inputs' digests in provenance. Bootstrap fields are unchanged (sign-invariant); V1-02 replicates are re-aligned with s_j.
- `directions_to_physical`/`physical_directions` sign each physical column so that its inner product with A^{-ᵀ}w_j is positive. The max-magnitude rule applies only to the default orientation.

**Acceptance tests.**
- HIV: chaining `orient(sub_t, by="reference", reference=sub_{t-1})` gives 0/20 reversals.
- MCMC: `orient(sub, by="reference", reference=W_legacy)` makes the vanilla y₂ mean +1.46 (sign of the AV chain, spec §10).
- After `orient`, `plot_sufficient_summary` x-data equal `space.to_normalized(X) @ W[:, 0]` with the new sign, and `physical_directions()[:, 0] @ (A^{-T} W[:, 0]) > 0`.
- `orient(sub, by="covariance", X=X, f=f)` on OLS input equals V1-04's sign.

**Amend.** target/07 §3.2, §4.5 (new helper), §8; target/03 §3.3 (sign of the physical view; D-11 note); target/13 §3.4 (plots follow the result's signs); `docs/user_guide/discovering.md`.

## V1-06 Product space of independent uniform and Gaussian blocks

**Need.** Hydrology (Jefferson et al. 2015, eq. 14): 18 uniform inputs plus log z0m ~ N(−5.11, 0.52²). Supporting evidence:
- Reentry_freestream_conditions (excluded from the port): 3 uniform + 4 Gaussian. With the correct QPHD Σ = diag(1/3, 1/3, 1/3, 1, 1, 1, 1), p_st λ₂ goes 4.97e4 → 1.66e4.
- The 2016 borehole tutorial: migration.md E4.

Related cases need no product space: discrete Stomatal inputs wj1 ∈ {1, 4, 4.5}, wj2 ∈ {2, 8, 10.5}, and log-transformed inputs (SingleDiodePV log I_S, MHD); see "No library change needed".

**Current v1 behavior.**
```
ac.Uniform(lower=[..., -inf, ...], upper=[..., inf, ...]) -> InputError input.parameter | lower: must be finite   (spaces.py:317, :853)
active_subspace(X=Z, f=F[:,0], space=ac.Uniform.reference(19), estimator="ols", n=1)
   -> InputError input.out_of_space | X: row 2 lies outside the support ... (coordinate 16 = np.float64(-1.3445...))   (_validation.py:214-226)
Workaround: ac.Gaussian(np.zeros(19), np.ones(19), names=ORDER) on hand-normalized Z: lambda1 = 6.68147e+10 (= legacy), no warnings;
   but space.kind = 'gaussian', reference_covariance diag = 1 for the 18 uniform columns, sample() draws N(0,1).
Uniform on the 18 uniform columns only: ValidityWarning validity.assumption (drops z0m).
```
The OLS result is exact under the workaround (OLS uses only the normalized X). QPHD, SIR, pHd and SAVE would use the wrong Σ (D-39, D-40). `docs/user_guide/input_spaces.md:145–159` says not to use value-based estimators under hand normalization; that is too strict for OLS (V1-13).

**Design constraints.** D-12 rules homogeneous spaces only. D-13 guards geometry with `unsupported.space`, which is already implemented (geometry.py:505, integration.py:621, optimization.py:499, designs.py:612). D-21 handles the borehole by hand. Argument for revisiting:
- Three as-data-sets cases and the 2016 tutorial need it.
- Every component the estimators use factorizes over independent blocks: the affine maps (diagonal per block), the support check (finite bounds on uniform coordinates only), `reference_covariance` = blockdiag(I/3, I), and sampling.
- D-13's guard already rejects the geometry, so no new geometry is needed.

Scope: subspaces, gradients, surrogates on all inputs or on a subspace, plots. Geometry, integration and optimization raise `unsupported.space`. Validity table (target/07 §3.5): a new `"product"` column equal to the stricter of the two existing columns.

**Proposed API.**
```python
ac.Product(blocks: Sequence[Uniform | Gaussian], *, order: Sequence[int] | None = None)  # kind = "product"
# .lower/.upper (NaN on Gaussian coords), .scale, .shift, .reference_covariance (block diagonal),
# to_normalized/from_normalized/grads_to_*/directions_to_physical/contains/sample(rng=)/to_dict
# names/units concatenated; `order` places block columns into data-column order (Hydrology z0m is column 16)
```

**Acceptance tests.**
- Hydrology grass with `Product([Uniform(YL[:16], YU[:16]), Gaussian([-5.11], [0.52**2]), Uniform(YL[16:], YU[16:])])` and physical X (log z0m in column 16):
  - OLS λ₁ = 6.68147e10 (LH), 7.68786e10 (SH), 6.77473e9 (GH) to rtol 1e-10;
  - |w·w_legacy| = 1 to 1e-10.
- `reference_covariance` diagonal = 1/3 on 18 columns and 1 on column 16.
- `active_domain(sub)` raises `UnsupportedError` `unsupported.space`.
- Round trip through `to_dict` and `save`/`load`.
- Reentry QPHD (n = 2) p_st λ₂ = 1.66e4 ± 1 % (spec §10 option B).

**Amend.** New D-entry amending D-12 (and D-02b, D-21 note); target/03 §3.1, §4, §5.3; target/04 (`_space_from_dict`); target/07 §3.5; target/08, 10 (guards listed); target/18 E4 and `docs/migration.md` E4; `docs/user_guide/input_spaces.md` "Spaces v1 does not have".

## V1-07 Activity scores

**Need.** Ebola reproduces Diaz et al. 2018, Fig. 9: activity scores α_i(n) = Σ_{j≤n} λ_j w_ij², from Constantine and Diaz 2017 (*Reliab. Eng. Syst. Saf.* 162:1–13). The notebook hand-codes them.

**Current v1 behavior.** No such function. `[n for n in dir(ac) if "activ"/"sobol"/"score" in n]` lists only active-domain names. The hand formula on the 6^8 rule gives:
```
Liberia:      lambda1=1.0194, alpha(2)/max: b1 .228 b2 .006 b3 .291 rho1 .003 g1 .099 g2 .290 om .002 psi 1.0
Sierra Leone: lambda1=1.1014, alpha(2)/max: b1 .184 b2 .005 b3 .082 rho1 .002 g1 .046 g2 .133 om .002 psi 1.0
```

**Design constraints.** No ruling. It is a published method with a one-line formula, so it fits target/07 §4.5 helpers. It must refuse `"basis"` results (NaN eigenvalues) and, for indefinite estimators, decide between |λ| and signed λ; recommend |λ| per D-31.

**Proposed API.** `ac.activity_scores(subspace, n: int | None = None) -> NDArray (m,)`, normalized coordinates, default `n = subspace.n`. `InputError` `input.parameter` with `details["reason"]="no_eigenvalues"` for a basis result.

**Acceptance tests.** It equals `(lam[:n] * W[:, :n]**2).sum(1)` to 1e-14. On the Ebola 6^8 rule, ψ has the largest score in both countries and the ratios above hold to 1e-3.

**Amend.** target/07 §4.5, §8; `design-specs/references.md` (ConstantineDiaz2017); `docs/user_guide/discovering.md`.

## V1-08 Subspace from a precomputed matrix; peak memory of the AS estimator

**Need.** Ebola's published C uses an 8^8 tensor Gauss–Legendre rule (16.8 M nodes). The spec planned 6^8 because the 8^8 rule was assumed infeasible.

**Current v1 behavior.**
```
k=6: N=1679616   active_subspace: 2.5s,  numpy peak inside call 0.34 GB (3.1 x G), lambda1=1.0194
k=7: N=5764801   active_subspace: 8.1s,  numpy peak inside call 1.15 GB (3.1 x G), lambda1=1.0213
k=8: N=16777216  active_subspace: 26.2s, numpy peak inside call 3.36 GB (3.1 x G), lambda1=1.0217, process max RSS 7.57 GB
subspace_from_basis(V[:, :2] ...): eigenvalues [nan nan nan]; plot_eigenvalues -> InputError input.parameter "a 'basis' result has no eigenvalues to plot" (plotting.py:344)
```
- The paper's 8^8 rule therefore works directly with `quad_weights` on a 34 GB machine, matching the hand C (λ₁ = 1.0217).
- The 3.1× overhead comes from the normalized-gradient copy, the √ω-scaled rows (subspaces.py:460) and the thin SVD's U factor (subspaces.py:152).
- A hand-accumulated C cannot become a result with eigenvalues (subspaces.py:2817, D-102 amendment).

**Design constraints.** P10 and the closed estimator set require a registered estimator name. `SubspaceResult._check` (subspaces.py:1890) accepts only `ESTIMATORS` or `"basis"`. D-29 and D-63 document the envelope and do not enforce it.

**Proposed API.**
- `ac.subspace_from_matrix(C, *, space, n=None, dimension_rule=None, qoi=None) -> SubspaceResult`, with estimator `"matrix"`, C symmetric (m, m) in normalized coordinates and checked finite, `eigh` with D-31 ordering and D-81 signs, `n_samples=0`, no bootstrap.
- Memory: compute the AS eigenpairs from `R = sqrt(w) * G` by QR (`mode="r"`) followed by the SVD of the (m, m) factor. This avoids the (M, m) U, gives the same accuracy as the thin SVD, and costs ≤ 2× G.

**Acceptance tests.**
- `subspace_from_matrix(C_hand_8^8, space=)` gives λ₁ = 1.0217 (L) and w₁ equal to the `quad_weights` result to 1e-10.
- `plot_eigenvalues` accepts it.
- The 7^8 peak inside `active_subspace` is ≤ 2.1× G (now 3.1×), with eigenvalues unchanged to 1e-13 relative.

**Amend.** target/07 §3.3, §4.5, §5.1; target/04 (estimator tag); target/00 §2.3 envelope text.

## V1-09 Finite differences for a model with several outputs

**Need.** HIV: one ODE trajectory per stencil point yields T cells at all 21 output times. The legacy code differenced the whole trajectory once.

**Current v1 behavior.**
```
vector-valued model (k,21) -> ModelError model.shape | ... returned values of shape (16, 21); expected (16,) or (16, 1)   (models.py:674)
21 scalar FD calls: 21 model calls, 336 trajectory rows (= 21 x M(m+1) = 336)
```
`finite_difference_gradients` (gradients.py:462) evaluates one stacked batch per call, so 21 outputs cost 21× the solves. `GradientResult` already stores the signed `steps`, `central_mask` and `values`.

**Design constraints.** D-08 rules scalar QoI only in v1. D-69 says vector QoI "returns a container of per-output results" later; this is the first such container, and it touches no estimator. The alternative is a documented caching wrapper in the notebook (no change).

**Proposed API.** `finite_difference_gradients(model, *, X, space=None, h=1e-6, scheme="forward", outputs: int | None = None, qoi_names=None, ...) -> tuple[GradientResult, ...]` when `outputs` is given. The model returns (k, outputs), one `GradientResult` per column shares `steps` and `central_mask`, and there is one model evaluation per stencil point.

**Acceptance tests.** On a toy (k, 21) model, `n_evaluations` = M(m+1) and each output's gradients equal the scalar call bitwise. On HIV at t = 24 with N = 20, the output matches the shipped gradients' w₁ (spec §6).

**Amend.** New D-entry under D-08/D-69; target/07 §3.8.2, §4.6; target/12 (model value shape); `docs/user_guide/models.md`.

## V1-10 Clearer `input.out_of_space` message

**Need.** NREL_Wind has 724 of 50 000 rows outside [−1, 1]⁵: 441 in WaveDir, 284 in PlatformDir. Hydrology's z-scored column shows the same error.

**Current v1 behavior** (_validation.py:214–226):
```
local_linear_gradients(space=Uniform.reference(5)) -> InputError input.out_of_space | X: row 7 lies outside the support of the input space (coordinate 4 = np.float64(1.0389279698213882))
```
The message reports only the first row, prints a NumPy-2 repr, and gives neither the input name nor the bound.

**Design constraints.** D-16 (stable code; `details` free); D-58.

**Proposed API.** Message `X: 724 of 50000 rows lie outside the support; first: row 7, input 'PlatformDir' (column 4) = 1.0389 > upper 1.0`. `details` gain `n_outside`, `name`, `bound`, `side`. The code is unchanged.

**Acceptance tests.** On the NREL data, `e.details["n_outside"] == 724`, `"np.float64" not in str(e)`, and `e.details["name"] == "PlatformDir"` when names are given.

**Amend.** target/02 (validation), target/06 §codes table (details), `docs/user_guide/errors.md`.

## V1-11 QPHD least squares speed

**Need.** Subsurface_Permeability QPHD: m = 100, M = 10 000, design 10 000 × 5 151 (0.41 GB).

**Current v1 behavior.**
```
v1 QPHD (gelsd) m=100 M=10000: 40.8s     (subspaces.py:196-206 _lstsq gelsd; QPHD._matrix subspaces.py:770)
scipy lstsq gelsy: 27.5s;  numpy QR + solve_triangular: 13.3s;  QR eigenvalues match gelsd: True
```

**Design constraints.** D-40 (QPHD form) is unchanged. The rank-deficiency warning `accuracy.rank_deficient` must survive: detect it from |diag R| ≤ tol·max|diag R| and fall back to `gelsd`.

**Proposed API.** None; internal change to `_lstsq` for M ≥ p.

**Acceptance tests.** Eigenvalues on the Subsurface data equal the current ones to 1e-8 relative ([12252.5, 630.6, 424.3, 363.6] for the first four); time ≤ 0.5× current on the reference runner; the rank-deficient HyShotII φ = 0.35 bootstrap still warns.

**Amend.** target/07 §5.1; benchmarks.

## V1-12 Names and units on reference spaces

**Need.** Notebooks whose data are already normalized pass a reference space and want labelled plots: Lithium_Battery (19 inputs, header names), Hydrology workaround, Atacamac, Reentry.

**Current v1 behavior.** `Uniform.reference(m: int)` and `Gaussian.reference(m: int)` take no names (spaces.py:874, :1098). `Uniform.reference(3, names=...)` raises `TypeError`. Notebooks write `ac.Uniform(-np.ones(19), np.ones(19), names=hdr)`. LaTeX header names such as `$\epsilon_a$` are accepted.

**Design constraints.** D-58 (reference spaces for normalized data). No conflict.

**Proposed API.** `Uniform.reference(m, *, names=None, units=None)`, same for `Gaussian`. The space equals `Uniform(-1, 1, names=…)`.

**Acceptance tests.** `Uniform.reference(2, names=["a", "b"]) == Uniform([-1, -1], [1, 1], names=("a", "b"))`; `plot_eigenvectors` labels come from it.

**Amend.** target/03 §4, §5.4.

## V1-13 Documentation corrections

| Doc | Change | Evidence |
|---|---|---|
| `docs/theory/estimators.md`, OLS docstring (subspaces.py:632) | Cite Constantine 2015 Alg. 1.3, eqs. (1.5)–(1.6), and JCP 2015 Alg. 1–2 (with V1-04) | book §1.1, p. 5 |
| `docs/user_guide/input_spaces.md:150–157` | Under hand normalization with a reference space, OLS is exact (it uses only normalized X); the Σ-based estimators (QPHD, SIR, pHd, SAVE), sampling and geometry are not | Hydrology: λ₁ equal to legacy to rounding |
| same page | Discrete inputs (Stomatal wj1, wj2) pass the support check; fine for OLS, AS and local linear; wrong for Σ-based estimators, `sample` and geometry | `Uniform(bounds).contains(all rows): True` |
| `docs/migration.md` "What changes in your numbers" | The 2016 RBF "R²" was 1 − rᵀK⁻¹r/fᵀK⁻¹f [N-05], not an in-sample R²; it is not reproducible (D-97, D-166). The polynomial in-sample R² is `1 - sum(residuals**2)/SS_tot` | ONERA-M6 drag: LOO 0.703/0.884/0.932; RBF residual-based value 0.710/1.000/1.000 |
| `docs/user_guide/choosing_n.md` | Eigenvalue scaling: scaled outputs via `grads=c*G` (LM104X min–max: λ₁ lift 0.191, drag 0.0844, EA 1.42e-3 = dissertation fig. 4.14); physical-gradient eigenvalues via `space=Uniform.reference(m)` with physical grads (NACA0012 rows 1–122: 877.2 = book Fig. 5.11); inputs on [0, 1] multiply λ by 4 (Ebola Fig. 7) | r09 output |
| `docs/migration.md` "Coordinates" | Name the 2016 data sets whose stored gradients are normalized (HIV, SingleDiodePV, Atacamac) and the double-scaling result (PV w₁ becomes [0.013, −0.330, 0.039, −0.004, 0.943]) | SingleDiodePV spec §10 |

---

## No library change needed

| Need (candidate) | v1 idiom | Verified |
|---|---|---|
| Many QoIs at once (#10): HIV 21 times, Lithium 150, Stomatal 78, NREL 7 | Loop over columns; `f` with k > 1 columns raises `input.vector_qoi` by design (D-08, _validation.py:261). Cost is negligible. Sign consistency across the loop is V1-04/V1-05 | 50 OLS calls (M = 3600, m = 19) 0.20 s; 21 AS calls 0.06 s |
| Ebola 8^8 rule (#5) | `ac.active_subspace(space=, grads=, quad_weights=W)` with the tensor rule (weights sum to 1 to 1e-15) | 26 s, λ₁ = 1.0217 = hand C; 3.4 GB peak in call, 7.6 GB RSS. Use 6^8 (2.5 s, λ₁ 0.3 % low) on small machines |
| Log of zero outputs in plots (#8, Subsurface) | `m = f > 0; plot_sufficient_summary(sub, X=X[m], f=np.log10(f[m]))`; report the 3 137 masked rows. Raising is D-17 | `input.nonfinite` at plotting.py:739; masked plot draws 6 863 points |
| Data outside the declared box (#8, NREL) | Declare the box explicitly, e.g. `ac.Uniform(np.minimum(X.min(0), -1), np.maximum(X.max(0), 1), names=)`. Inferring a space from data is excluded by P2/[A-01]; V1-10 improves the message | 2 000 centers, p = 60: 4.4 s; EVG n = 1 |
| Scaled outputs or physical inputs (#9: LM104X, NACA0012, ONERA-M6, Ebola Fig. 7) | Scale gradients by the output scale (`grads=c*G`); pass physical gradients with `Uniform.reference(m)` to get eig(GᵀG/M); D-11 keeps eigenvalues normalized | LM104X λ₁ 0.191/0.0844/1.42e-3; NACA0012 877.2 |
| Dimension rule disagrees with published n (#11) | Pass `n=` or `with_dimension`; show `plot_dimension_criterion`. Rules behave as ruled (D-38, D-79) | ONERA-M6 drag: EVG linear n = 1, log n = 49, RS n = 49, LI n = 1 (`accuracy.zero_dimension`); tail |λ| to 1e-13 |
| In-sample R² (#11) | Polynomial: `1 - sum(s.residuals**2)/SS_tot` (Hydrology reproduces 0.952/0.917/0.992/0.819/0.826/0.964). RBF: report `loo_r_squared`; 2016 RBF "R²" not reproducible (V1-13) | ONERA-M6 above |
| `ValidityWarning` for OLS on uniform (#12) | By design (D-41). Under default filters a 78-fit loop prints it twice (attributed to the user's line). Silence locally with `warnings.filterwarnings("ignore", message=r".*\[validity\.assumption\]$")`, as in `docs/user_guide/errors.md:86–96`, and explain once in markdown | r12: 2 printed for 78 fits |
| Log-transformed inputs (SingleDiodePV log I_S, MHD logs) | Treat the log as the physical coordinate: `Uniform(log lo, log hi)`, gradients ∂f/∂log x = x ∂f/∂x (`docs/user_guide/input_spaces.md:149`) | specs §10 |
| Discrete inputs (Stomatal) | Uniform box over the levels with OLS (see V1-13) | contains all rows |
| Stored normalized gradients (HIV, SingleDiodePV, Atacamac, MHD) | Pass with `Uniform.reference(m)`/`Gaussian.reference(m)` or divide by the half-range, never both (`docs/migration.md` Coordinates) | specs §10 |
| 2016 bootstrap band look; first k eigenvalues | `plot_eigenvalues(sub, interval="minmax", k_max=k)` | API |
| Overlays and grids | `ax=` on every plot (D-130); several calls on one `ax` overlay | API |

## Stays in the notebooks (out of library scope)

- **AS-accelerated MCMC (#13; Constantine, Kent, Bui-Thanh 2016, Algorithm 1).**
  - Not in target/00 §2.1. It was never library code: 2016 had it as the notebook class `MCMC/pde_mcmc_study.py:ActiveSubspaceAcceleratedMCMC`, not in master or dev. D-03 sets the v1 scope to master plus the chosen dev features.
  - It depends on model-specific misfits, proposal tuning and chains of hours.
  - v1 already supplies the pieces: `forward_map`, an exact Gaussian `inverse_map`, and `evaluate_grads` with `executor=`.
  - A library sampler would need a new ruling and a new spec. Leave it in the notebook as about 40 lines (cached ĝ, log acceptance, `rng`); reconsider for v1.x.
- **Models.** The numpad-free PDE solver, the HIV `solve_ivp` model, the single-diode model and the Ebola R₀ are data-set models, not library code (D-122).
- **Problem-specific post-processing:**
  - the HyShotII safe-box optimization (it uses V1-03);
  - Hydrology's grass-plus-bare weight overlay (different m);
  - Stomatal multi-slice summary panels and the Lithium 2×3 grids;
  - masking HIV's 474 frozen samples;
  - data conversion to `.npz`;
  - errata notes.
