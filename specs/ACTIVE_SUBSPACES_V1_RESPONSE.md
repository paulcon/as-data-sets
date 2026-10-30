# active_subspaces v1: response to the as-data-sets change report

2026-10-30

**Purpose.** This answers `specs/ACTIVE_SUBSPACES_V1_CHANGES.md` (2026-10-29, items V1-01 to V1-13). It says what landed in `active_subspaces`, where the shipped behavior differs from the proposal, which later rulings change what the notebooks see, and what the as-data-sets port should do now. It is written for the agent porting the notebooks. Ruling numbers (D-xx) refer to `ActiveSubspacesRevision/design-specs/decisions.md`. The `target/` specs there are amended to match.

## Where the code is

- Every item is on GitHub `main` of `paulcon/active_subspaces`, head **`ca41b3d`** (PRs #58 to #61, merged 2026-10-30). CI is green on all 36 jobs, including Windows and the oldest supported dependencies.
- The version is still `1.0.0.dev0`. The PyPI release of 1.0.0 waits on an account recovery, so install from the checkout.
- `as-data-sets/.venv` has an editable install of `ActiveSubspacesRevision/upstream-v1/src`. That worktree is now detached at `ca41b3d`, so the venv already imports the merged code. Check with `python -c "import active_subspaces, pathlib; print(pathlib.Path(active_subspaces.__file__).parent)"`.
- Acceptance scripts on `main`, run with that venv:

  | Script | Result |
  |---|---|
  | `prototypes/v1report/accept_subspaces.py` | 20 passed, 1 skipped (MCMC: no loader) |
  | `prototypes/v1report/accept_surrogates.py` | 17 passed |
  | `prototypes/v1report/accept_spaces.py` | 21 passed |

## Summary

All thirteen items landed before v1.0, as four themed PRs [D-203]. "As proposed" means the proposal's API and acceptance tests hold. Differences are listed under each item below.

| ID | Status | Rulings | What shipped |
|---|---|---|---|
| V1-01 | Landed, with differences | D-192, D-193, D-194, D-204, D-206, D-207, D-211, D-212 | `fit_surrogate(..., f_variance=)` sets the CDW (4.13) bounds; departures 1 and 2 kept and documented; C₁ = 4/π² (uniform), 1 (Gaussian, product); field renamed `training_variance` |
| V1-02 | As proposed | D-196, D-213 | `active_subspace(..., bootstrap_vectors=k)`, `SubspaceResult.bootstrap_eigenvectors` (B, m, k), `plot_eigenvectors(..., interval=)` |
| V1-03 | As proposed (Student t) | D-200, D-211 | `ac.predict_interval`, `PolynomialSurrogate.interval`, `IntervalResult` |
| V1-04 | As proposed | D-195 | OLS `W[:, 0] = û/‖û‖` |
| V1-05 | As proposed | D-197 | `ac.orient(...)`; `physical_directions()` keeps the signs of `W` |
| V1-06 | Landed, with differences | D-198, D-205, D-208, D-210, D-214 | `ac.Product(blocks)` with **no `order=`**; SIR, pHd and SAVE back-transform by Σ^{-1/2} |
| V1-07 | As proposed | D-201, D-208 | `ac.activity_scores(subspace, n=None)` with \|λ\| |
| V1-08 | As proposed | D-202, D-213 | `ac.subspace_from_matrix(C, space=)`; AS peak memory 1 + 2/m times G without a bootstrap |
| V1-09 | Landed as a separate function | D-199 | `ac.finite_difference_gradients_multi(model, X=, space=, outputs=)` |
| V1-10 | As proposed | D-203 (V1-10) | Message counts rows and names the input, value, side and bound; `details` gain `n_outside`, `column`, `name`, `bound`, `side` |
| V1-11 | As proposed, thresholds adjusted | D-203 (V1-11) | QR solve in `_lstsq`, gelsd fallback with an explicit rank cutoff |
| V1-12 | As proposed | D-203 (V1-12) | `Uniform.reference(m, *, names=, units=)`, same for `Gaussian` |
| V1-13 | All six corrections made | D-194, D-195, D-203 | See V1-13 below |

## Per item: what differs from the proposal and what to do

### V1-01 RBF hyperparameter bounds

- **API.** `fit_surrogate(X=, f=, subspace=, kind="rbf", f_variance=σ̂²)`. When `f_variance` is given, `log10_eta_bounds` is None and the subspace has eigenvalues, the search interval is [log₁₀(2σ̂²/Σ|λ|), log₁₀(2C₁)]. Explicit `log10_eta_bounds` or `log10_eta` take precedence. An empty interval collapses to its upper end and warns `accuracy.empty_bounds`. Provenance records `f_variance` and the bounds' source.
- **C₁** [D-193, D-204]: 4/π² on a uniform space (the sharp constant; CDW's printed 2√m/π is valid but loose), 1 on a Gaussian space, and 1 on a `Product`. Open Elliptic_PDE Q4 is answered by this.
- **Departures 1 and 2** (profiled amplitude; nugget η·Σ_{q>n}|λ|/Σ|λ|) are **kept and flagged** [D-194]. The docs now cite CDW 2014 §4.2–4.3 and (4.16) for the length-scale rule.
- **Field rename** [D-206]: the RBF field formerly called `f_variance` is `RBFSurrogate.training_variance`. `RBFSurrogate` is schema 2; version-1 files load through a registered migration. The keyword on `fit_surrogate` is still `f_variance`.
- **Model path** [D-207]: `fit_surrogate_from_model` has no `f_variance`. Pass `log10_eta_bounds` computed from σ̂², the eigenvalues and C₁, as the docs show.
- **`accuracy.flat_objective` changed after the report** [D-212]. It no longer fires when the flat run of J reaches the lower end of the search interval. There the kernel part vanishes and the RBF reduces to its polynomial tail, so η does not change the fit. Interior or upper-end plateaus of ≥ 4 decades still warn. **Consequence:** your acceptance test "the n = 1 default call without `f_variance` issues `accuracy.flat_objective`" is reversed. The Elliptic n = 1 default plateau is [−10, −1], which reaches the lower end, so it no longer warns. `accept_surrogates.py` was edited to expect no warning; that edit is **uncommitted** in this repo.
- Accuracy targets hold: Elliptic β = 1 n = 1 mean relative error 0.095, n = 2 0.300; β = 0.01 0.008 with `accuracy.empty_bounds`.

### V1-02 Bootstrap eigenvector replicates

- As proposed: `active_subspace(..., nboot=B, bootstrap_vectors=k, rng=)` with 0 ≤ k ≤ n. Replicates are sign-aligned to the full-sample vectors, and `accuracy.unaligned_replicates` warns when min_b |w_jᵀw_j^(b)| < 0.5.
- `plot_eigenvectors(sub, indices=[0], interval="percentile" | "minmax", level=0.95)` draws the band.
- `SubspaceResult` is schema 2; version-1 files load with `bootstrap_eigenvectors=None`. With the default k = 0, results are bitwise unchanged.
- **Memory** [D-213]: with a bootstrap, AS holds two (M, m) arrays plus (M,) index and weight vectors at once, about 2.57 G at m = 8. Without one it is 1 + 2/m times G. Bootstrapping a quadrature rule warns `validity.bootstrap_weights` anyway.

### V1-03 Intervals for polynomial surrogates

- As proposed: `ac.predict_interval(surrogate, X, level=, kind="prediction" | "confidence", side="two" | "lower" | "upper")` on physical X, and `PolynomialSurrogate.interval(Y, ...)` on active coordinates. `IntervalResult` has `estimate, lower, upper, level, kind, side, dof, residual_scale`. An RBF surrogate or ν ≤ 0 raises `input.parameter`.
- **HyShotII safe box** [D-211]: v1 keeps Student's t. The paper's φ = 0.35 limit uses the normal quantile, ĝ(y) + 2.33 s √(1 + h(y)). With t (t_{ν,0.99} ≈ 2.40), the angle-of-attack (AoA) limit is 3.179. To reproduce the paper's 3.193, scale the one-sided half-width by z₀.₉₉/t_{ν,0.99}:

  ```python
  import scipy.stats as st
  iv = ac.predict_interval(poly, X, level=0.99, side="upper")
  z_limit = iv.estimate + (iv.upper - iv.estimate) * st.norm.ppf(0.99) / st.t.ppf(0.99, iv.dof)
  ```

  `accept_surrogates.py` already does this. The notebook should do the same and say why in markdown.

### V1-04 OLS sign

- As proposed [D-195]. Drop the notebooks' `np.sign(w @ a_hat)` multiplications. The OLS docstring and theory page cite Algorithm 1.3 and JCP 2015 Algorithms 1–2.

### V1-05 `orient`

- As proposed [D-197]: `ac.orient(sub, by="reference" | "covariance" | "mean_gradient", reference=, X=, f=, grads=, columns=)` returns a new result. Stored replicates are re-signed with it.
- `physical_directions()` and `space.directions_to_physical` now keep each column's sign. The max-magnitude rule is no longer re-applied to physical directions.
- Replace the hand flips in HIV, MCMC, Lithium (QPHD), Elliptic_PDE, NACA0012 and MHD with `orient`. Chaining `orient(sub_t, by="reference", reference=sub_prev)` gives continuous signs.

### V1-06 Product space

- **No `order=` argument** [D-198]. Blocks are concatenated in the order given, so build the product in data-column order. For Hydrology, log z0m is column 16:

  ```python
  space = ac.Product([ac.Uniform(YL[:16], YU[:16]),
                      ac.Gaussian([-5.11], [0.52**2]),
                      ac.Uniform(YL[16:], YU[16:])])
  ```

- Subspaces, gradients, surrogates (on all inputs or on a subspace) and plots accept it. Geometry (`active_domain`, `inverse_map`), integration, optimization, designs and `build_reduced_model` raise `UnsupportedError` `unsupported.space`.
- **SIR, pHd and SAVE** [D-205]: on a mixed product, their directions are back-transformed to Li's Σ^{-1/2}η by a QR. span(w₁..w_k) = Σ^{-1/2} span(η₁..η_k) for every k, and the eigenvalues are those of the whitened matrix. The columns of W are then not eigenvectors of that matrix. QPHD uses C = ggᵀ + HΣH with the declared Σ directly. Results on `Uniform` and `Gaussian` are bitwise unchanged.
- **Activity scores on a mixed product** [D-208]: for SIR, pHd and SAVE they pair |λ_j| with the normalized directions b_j = Σ^{-1/2}η_j/‖Σ^{-1/2}η_j‖, not with the columns of W.
- **Validity warnings** [D-210]: a mixed product takes the stricter of the uniform and Gaussian cells. A product whose blocks are all of one kind takes that kind's cells.
- **Diagonal Gaussian covariances** [D-214]: an (m, m) covariance that is exactly diagonal is stored as its (m,) variances. `Gaussian(mu, d)` and `Gaussian(mu, np.diag(d))` are then equal spaces with identical maps, and `cov.shape` is (m,) for such input.
- The Reentry QPHD check (p_st λ₂ ≈ 1.66e4) and the Hydrology OLS λ₁ values hold (`accept_spaces.py`).

### V1-07 Activity scores

- As proposed: `ac.activity_scores(sub, n=None)` returns Σ_{j≤n} |λ_j| w_ij² in normalized coordinates. A `"basis"` result raises `input.parameter` with `reason="no_eigenvalues"`.
- On a mixed `Product`, SIR, pHd and SAVE use the D-208 directions described under V1-06.

### V1-08 Matrix input and memory

- As proposed: `ac.subspace_from_matrix(C, space=, n=None, dimension_rule=None, qoi=None)` returns a result with estimator `"matrix"`, which `plot_eigenvalues` accepts.
- AS uses a blocked QR then the SVD of the (m, m) factor. The 7^8 Ebola peak is 1.25 G (was 3.1 G), and eigenvalues agree to 1e-13 relative.
- For the Ebola 8^8 rule, `active_subspace(grads=, quad_weights=)` and `subspace_from_matrix` give the same λ₁ = 1.0217.

### V1-09 Several outputs

- **A separate function, not an `outputs=` argument** on `finite_difference_gradients` [D-199]:

  ```python
  results = ac.finite_difference_gradients_multi(model, X=X, space=space, outputs=21,
                                                 h=1e-6, scheme="forward", qoi_names=None)
  # tuple of 21 GradientResult; default qoi labels "f0", "f1", ...
  ```

- The model returns (K, outputs). Each stencil point is evaluated once, and each output equals the scalar call on that column bitwise. `finite_difference_gradients` is unchanged.

### V1-10 `input.out_of_space` message

- As proposed. One change after the report: the value and bound print with full precision (`repr`). A point just past a bound now reads `= 1.000000001 > upper 1.0`, not `= 1 > upper 1`.
- On the NREL data, `details["n_outside"] == 724` and `details["name"] == "PlatformDir"`.

### V1-11 QPHD least squares

- As proposed, with two numerical adjustments that came out of Windows CI:
  - The QR route is taken when every |R_jj| > √ε·max|R_ll|, so every design with κ₂ < ε^{-1/2} uses it.
  - The gelsd fallback uses rank cutoff max(M, p)·ε·σ₁, as NumPy does, so an exactly dependent column always warns `accuracy.rank_deficient`.
- Subsurface QPHD: eigenvalues [12252.52, 630.56, 424.28, 363.64], about 18 s against 40.8 s with gelsd.

### V1-12 Names on reference spaces

- As proposed. `Uniform.reference(m, names=hdr)` replaces `ac.Uniform(-np.ones(m), np.ones(m), names=hdr)`.

### V1-13 Documentation

All six corrections are in the docs on `main`:

1. OLS citations: `docs/theory/estimators.md`, OLS docstring.
2. OLS under hand normalization: `docs/user_guide/input_spaces.md`.
3. Discrete inputs: `docs/user_guide/input_spaces.md`.
4. The 2016 RBF "R²" and the in-sample polynomial R²: `docs/migration.md`.
5. Eigenvalue scaling (`grads=c*G`, physical gradients with a reference space, the factor 4 for [0, 1]): `docs/user_guide/choosing_n.md`.
6. The data sets with stored normalized gradients: `docs/migration.md`.

## Other changes after the report that the notebooks may notice

- **Schema versions.** `SubspaceResult` and `RBFSurrogate` are schema 2. Results saved with the b4e82fb build load through registered migrations.
- **Sketched subspace** (`active_subspaces.experimental`, not used by the notebooks) applies its boundary rule to a product's uniform coordinates [D-209].
- **Warning filters.** Default RBF fits on noise-free data no longer raise `accuracy.flat_objective` (D-212). Remove any filter a notebook added for it.

## Actions for the as-data-sets port

1. Commit the D-212 edit to `specs/prototypes/v1report/accept_surrogates.py`; it is uncommitted.
2. Replace the hand-coded workarounds:

   | Notebooks | Replace | With |
   |---|---|---|
   | Lithium, Stomatal, Hydrology, Subsurface | OLS sign flips | Nothing; v1 returns the signed direction (V1-04) |
   | HIV, MCMC, Elliptic_PDE, NACA0012, MHD, Lithium QPHD | Hand flips, `dataclasses.replace(..., eigenvectors=-W)` | `ac.orient` (V1-05) |
   | HyShotII, Lithium, SingleDiodePV | Bootstrap and alignment | `bootstrap_vectors=`, `plot_eigenvectors(..., interval=)` (V1-02) |
   | Hydrology, HyShotII | Interval formulas | `ac.predict_interval` (V1-03) |
   | Hydrology | Gaussian-reference workaround | `ac.Product` in column order (V1-06) |
   | Ebola | Activity-score formula, hand C | `ac.activity_scores`, `ac.subspace_from_matrix` (V1-07, V1-08) |
   | HIV | 21 scalar finite-difference calls | `ac.finite_difference_gradients_multi` (V1-09) |
   | Lithium, Hydrology, Atacamac, Reentry | `Uniform(-1, 1, names=)` | `Uniform.reference(m, names=)` (V1-12) |
   | Elliptic_PDE | Hand CDW bounds | `fit_surrogate(..., f_variance=f0.var())` (V1-01) |

3. Rename any use of the RBF field `f_variance` to `training_variance`.
4. In HyShotII, explain the 2.33 versus t difference and use the rescaling above.
5. Update spec expectations that relied on the old behavior:
   - the Elliptic `flat_objective` warning (D-212);
   - the out-of-space message text (V1-10);
   - signs of physical directions, which now follow `W` (V1-05);
   - `Gaussian.cov` shape for a diagonal matrix (D-214).
6. The release of 1.0.0 is still to come. Keep installing from `upstream-v1/src` until it is on PyPI.
