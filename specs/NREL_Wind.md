# NREL_Wind: update spec

Kind: **stored data**: FAST runs by Graf (NREL) (see [README.md](README.md#two-kinds-of-data-sets)).

Status line: `NREL_Wind/NREL_Wind.ipynb` | paper (source of the model and data, not an active-subspace paper): P. A. Graf, G. Stewart, M. Lackner, K. Dykes, P. Veers, "High-throughput computation and the applicability of Monte Carlo integration in fatigue load estimation of floating offshore wind turbines", *Wind Energy* 19(5):861–872 (2016; online 2015), doi:10.1002/we.1870, NREL/JA-2C00-62929. The citation and DOI are correct (verified via OSTI 1248074 and the BL catalogue). **Checked on 2026-10-29 against the accepted manuscript** in the user's folder (`papers/Graf - High throughput computation … - 2016.pdf`). Its header reads "Wind Energ. 0000; 00:1–11", "DOI: 10.1002/we", "Prepared using weauth.cls", so it is not the typeset journal version. Page numbers p. 1–11 below refer to this manuscript, not to journal pages 861–872. The paper does not mention active subspaces, Constantine or dimension reduction: the AS analysis in this notebook is unpublished. | m = 5 (Vhub, Hs, WaveDir, Tp, PlatformDir) | QoIs: 7 fatigue-load outputs `rf4_RootMxc1, rf4_RootMyc1, rf4_TwrBsMxt, rf4_TwrBsMyt, rf4_Anch1Ten, rf4_Anch2Ten, rf4_Anch3Ten` (no units given) | legacy API: `ac.gradients.local_linear_gradients(X, f, 60)`, `ac.subspaces.Subspaces().compute(df=, nboot=100)`, `.partition`, `.W1`, `.eigenvals`, `.e_br`, `ac.utils.plotters.eigenvalues/eigenvectors/sufficient_summary`, `ac.utils.response_surfaces.PolynomialApproximation(3|4|5).train/.predict/.Rsqr` | py2 issues: yes (`print` statements in cell 7; pandas `from_csv`/`as_matrix`)

## 1. Summary
This is a small, cheap notebook: 50,000 samples with no gradients. It estimates gradients by local linear fits, computes a subspace for each of 7 outputs, and fits polynomial surrogates of degree 3–5. The port to v1 is straightforward and the prototype reproduces the legacy results: eigenvalues within sampling error, and R² within 0.03 for all 21 fits. Most important findings:
1. **Wrong sample count.** The text says "The dataset has nearly 500,000 pairs". The shipped file has **50,000** rows. The 500k set was `windData500k.mat` (64.5 MB), deleted in 554642d but still in git history. The paper explains both numbers (§4.3, p. 7): the 4D study (no PlatformDir) used 50,000 Monte Carlo samples and the **5D study used 500,000**. The shipped file has all 5 inputs, so it is most likely a 50,000-row subset of the 5D 500k set, not the 4D 50k set. The notebook's "nearly 500,000" describes the deleted .mat file. Confidence: likely.
2. **The inputs are not what the table says, and two encodings are mixed.** The table matches the paper's distributions word for word (p. 2: Weibull; von Mises conditioned on wind speed; gamma conditioned on wind speed and wave direction; gamma conditioned on wind speed and wave height; von Mises conditioned on wind speed). The stored columns are not draws from those distributions, and the paper says nothing about how they were normalized. Data evidence (prototype) points to a mixed encoding:
   - **Vhub, Hs, Tp are probability coordinates** (likely 2u − 1 with u a marginal or conditional CDF value). They are exactly uniform on [−1, 1], and Hs and Tp show no dependence on Vhub (mean and std flat across Vhub quintiles), although the paper makes both conditional on Vhub.
   - **WaveDir and PlatformDir are angles in units of π rad, not wrapped to [−1, 1]** (likely). PlatformDir spans exactly 2.000 (from −0.953 to 1.046), i.e. θ/π ∈ [μ/π − 1, μ/π + 1] with μ ≈ 0.146 rad: a von Mises sample on [μ − π, μ + π]. WaveDir keeps the conditional structure the paper describes. Its mean rises from −0.068 to +0.024 and its std falls from 0.539 to 0.434 across Vhub quintiles. All 435 values below −1 occur at low wind (implied V ≤ 10.9 m/s), and all 6 values above 1 at high wind (≥ 12.3 m/s). This is consistent with a location that increases with V, as in the paper's East Coast input file (p. 3: `WaveDir = VM(0.03 + 0.12*Vhub, -0.620 + 0.031*Vhub)`). The exact coefficients there do not fit (1119 rows violate them), so the shipped data were probably made with a different parameter set (the paper's 5D distribution is not given).
   - **So the 724 rows outside [−1, 1] come from unwrapped von Mises samples, not from bad normalization.** Linear correlations look small (≤ 0.07) only because the dependence is in the spread.
   - **Consequence.** Eigenvector entries for Vhub, Hs and Tp are sensitivities to quantiles, not to m/s, m or s. The angle entries are per π rad. This must be stated in the notebook (Q1).
3. **Very few gradients.** The legacy `local_linear_gradients` estimated gradients at only `ceil(10·m·ln m)` = **81 random centers** out of 50,000 points, unseeded. The legacy neighbor selection also has the indexing defect fixed in v1 as [C-02]. The bootstrap then resamples these 81 gradients. Cell 3 ("fit linear models at many points") hides this.
4. **v1 needs a space that contains the data.** `ac.Uniform.reference(5)` raises `InputError input.out_of_space` because of the 724 rows outside [−1, 1]. The prototype uses a box from the data bounds (§6). The real fix depends on Q1.
5. **Dimension choice is not supported by the plots.** The chosen dimensions "1, 1, 3, 3, 2, 2, 2" are not all justified by eigenvalue gaps. For TwrBsMyt the legacy ratios are λ₁/λ₂ ≈ 6, λ₂/λ₃ ≈ 2 and λ₃/λ₄ ≈ 2.5, so there is no gap after 3. v1 EVG picks 1, 1, 2, 1, 1, 1, 1. The anchor-tension subspaces are essentially the PlatformDir coordinate axis (w₁ ≈ e₅), so "active subspace" is no more than variable screening there.

## 2. Inventory
| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| NREL_Wind.ipynb | 3,648,636 B | nbformat 4, kernel python2 | analysis | — | 29 PNGs (28 in cell 5 = 7 outputs × [eigenvalues, eigenvectors, 1-D summary, 2-D summary]; 1 surrogate grid in cell 7) + printed R² table. Master is newer than dev (refs only). |
| NRELwind_testing.txt | 15,074,183 B | CSV, 1 header row, 50,000 × 13, `%.16e` | samples | `pn.DataFrame.from_csv` (`Run` → index) | cols `Run, Vhub, Hs, WaveDir, Tp, PlatformDir, rf4_RootMxc1, rf4_RootMyc1, rf4_TwrBsMxt, rf4_TwrBsMyt, rf4_Anch1Ten, rf4_Anch2Ten, rf4_Anch3Ten`. No NaN, no duplicate input rows, `Run` = 1..50000 sorted. The name "testing" and the 5 inputs suggest a subset of the paper's 5D 500,000-sample MC set (§4.3, p. 7) (Q2). |
| *(history)* windData500k.mat | 64,519,673 B | MATLAB | first draft's data | — | added 79a3271 (2016-07-12), deleted 554642d. It still accounts for 64.5 MB of the 225 MB `.git`. |

Output ranges: RootMxc1 4,097–7,395; RootMyc1 1,093–7,782; TwrBsMxt 1,501–60,765; TwrBsMyt 11,668–93,407; Anch{1,2,3}Ten 9.2–231.8. The units are presumably kN·m and kN (FAST conventions). The paper gives no units for its DEL plots, so this is not verified. The paper (§4.3, p. 7) computes m = 4 and m = 10 DELs for 14 FAST sensors (RootMxc1, RootMyc1, LSSGagMya, LSSGagMza, YawBrMxp, YawBrMyp, TwrBsMxt, TwrBsMyt, Fair1–3Ten, Anch1–3Ten). The notebook's 7 outputs are a subset, and the prefix `rf4_` very likely means the rainflow-based short-term DEL with material exponent m = 4 (p. 3, eq. (1)). The paper uses m = 10 for composite blades (p. 3), so the blade-root outputs here are the m = 4 variants, which the paper would not use for blade fatigue. Say so in the text.

## 3. Text and math review (vs. literature)
| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| 0 | "The dataset has nearly 500,000 pairs of inputs and corresponding outputs." | The file has 50,000 rows. The paper's 5D Monte Carlo study used 500,000 samples (§4.3, p. 7), which is what the deleted `windData500k.mat` held. | "The data set has 50,000 input/output pairs, a subset of the 500,000-sample 5D Monte Carlo study in [3]" (confirm in Q2) | confirmed (count); likely (subset) |
| 0 | "The model and data are found in [3]." | [3] describes the model (OC3 Hywind spar-buoy turbine, FAST v7 + TurbSim + HydroDyn/JONSWAP, 10-min simulations, IEC DLC 1.2, class A NTM, shear exponent 0.14; pp. 2, 4) and the sampling pipeline (WISDEM/openMDAO, p. 3). It publishes no data, gives no data link, and does not mention active subspaces. | "The model and the Monte Carlo sampling are described in [3]; the 50,000 samples used here were provided by NREL" (provider to be confirmed, Q3) | confirmed (paper content); unverified (provider) |
| 0 | table "Wind Speed \| $V_{hub}$ \| Weibull … Wave Direction \| $\theta_w$ \| Von Mises conditioned on $V_{hub}$ …" | The table matches the paper's list exactly (§2, p. 2). The paper adds context the notebook lacks: the metocean data are from NOAA NDBC buoys, the representative **East Coast** site is used, WaveDir is measured w.r.t. the wind direction, PlatformDir is the wind direction w.r.t. a fixed platform orientation (no yaw), and angles are in radians (grid −2.792…π, p. 7). The 4D distribution parameters are on p. 3 (e.g. Vhub ~ W(2.12, 9.77)); no PlatformDir parameters are given. The stored columns are encoded differently from the table (§1 item 2). | Keep the table. Add units, the East Coast site, the angle conventions, and one paragraph on the encoding of x (Q1). | confirmed (table = paper p. 2); likely (encoding) |
| 0 | table row order Vhub, θ_w, H_s, T_p, θ_p | The data order is Vhub, Hs, WaveDir, Tp, PlatformDir, and the plots use the data labels. | Reorder the table to match, and add the column names | confirmed |
| 0 | "we examine 7 for demonstration" | The 7 outputs are never defined. `RootMxc1/RootMyc1` = blade-1 root in-plane/out-of-plane bending moment; `TwrBsMxt/TwrBsMyt` = tower-base side-to-side/fore-aft moment; `Anch{1,2,3}Ten` = anchor (mooring-line) tensions. These are standard FAST channel names. The paper confirms three: RootMxc1 is "the 'edgewise' load at the base of the blade" (p. 7), TwrBsMxt is "the side to side tower base bending moment" (Fig. 5, p. 8), and Anch1Ten is "the tension … of the first anchor cable" (Fig. 6, p. 8). The statistic is the short-term damage equivalent load: the amplitude of a 1 Hz sinusoid with the same fatigue damage, from rainflow counting (MLife) of one 10-min simulation with zero fixed mean (pp. 3–4). `rf4` = m = 4 (§2 note). | Add a table (name, meaning, unit, statistic = DEL with m = 4) | confirmed (3 names, DEL); likely (RootMyc1, TwrBsMyt, rf4 = m 4); unverified (units) |
| 0 | "Offshore wind turbines …"; title "Wind Turbine Fatigue" | The cited study is about *floating* offshore turbines: the OC3 Hywind spar buoy (p. 2). Platform direction and anchor tensions only make sense for a floating platform. | Say "floating offshore (OC3 Hywind spar buoy)" | confirmed (p. 2) |
| 0 | "What we know about land turbines can't be applied directly to offshore turbines because of the differences in operating conditions." | Imprecise. The paper's point (p. 1) is that IEC 61400-3 covers *fixed-bottom* offshore turbines, but there is no standard yet for *floating* ones, which are more sensitive to wind/wave misalignment because of "more compliance and low side-to-side damping". | Rephrase along those lines | confirmed |
| 0 | "We can simulate loads on wind turbines with physics simulations such as TurbSim … and FAST" | Agrees with the paper (§3.1.2, p. 4: TurbSim, FAST v7, rainflow counting; waves via HydroDyn with a JONSWAP spectrum) | Add HydroDyn and the 10-min simulation length | confirmed |
| 0 | header "<pconstan@mines.edu>", "<ryhoward@mines.edu>" | Possibly stale | Q6 | unverified |
| 3 | "One such approach is to fit linear models at many points using several of its nearest neighbors" | The legacy code used 81 random centers, 60 neighbors each. | In v1 state the number of centers and p explicitly, e.g. "at 2,000 randomly chosen points, each using its 60 nearest neighbors" | confirmed (legacy source) |
| 6 | "the tight trend of rf4_RootMxc1 and the first eigenvector placing almost all its weight on $V_{hub}$" | Agrees with the legacy figure and v1 (w₁ = [1.00, 0.01, 0.01, 0, 0]) | keep | confirmed |
| 6 | "Large gaps in the eigenvalues indicate accuracy in estimating the active subspace." | Loose. The subspace error bound scales like 1/(λ_n − λ_{n+1}) (subspace-error bound, CDW 2014 §3; Constantine 2015 §3.3), so a large gap means the estimated subspace is accurate. | "A large gap between λ_n and λ_{n+1} means the n-dimensional subspace can be estimated accurately" | confirmed |
| 6 | "we choose subspace dimensions of 1, 1, 3, 3, 2, 2, and 2" | Not all supported by gaps (§1 item 5). In the legacy plots TwrBsMxt has λ₁ ≈ λ₂ (ratio 1.3) and TwrBsMyt has no gap after 3. For anchors 2–3 the gap after 2 is real (ratio ≈ 20), but w₁, w₂ ≈ e₅, e₁, so the subspace is just {PlatformDir, Vhub}. The eigenvector figure shows only 2 vectors even when n = 3. | Justify each n by a gap, or say it was chosen for surrogate accuracy. Note the coordinate-aligned cases. | confirmed (legacy plots, v1) |
| 8 | "rf4_Anch1Ten's $R^2$ maxes out at under 0.80 while the rest have values above 0.80 (for the quintic model), and 4 outputs have values above 0.90" | Agrees with the legacy table. v1 LOO R² gives the same picture (Anch1Ten quintic 0.775, the other six > 0.86, four > 0.92). | Keep, computed from the run | confirmed |
| 8 | "Overall, most of outputs we examine have useful active subspaces." | Defensible, but for 3 of 7 outputs the active direction is a single input (PlatformDir); for 2, it is Vhub. The paper's own grid-coarsening study agrees. For Anch1Ten, "wind direction w.r.t. platform (PlatformDir) … should not be" coarsened (Fig. 6, p. 8). For RootMxc1, "the discretization of wave height Hs could be significantly coarsened" (Fig. 7, p. 9). The paper also motivates the WaveDir sensitivity of the side-to-side tower moment (low side-to-side damping, p. 1). | Add that sentence and cite these paper observations as an independent check | confirmed (consistent with paper) |
| 0 | "[[3]] … Wind Energy, 19(5), 861-872, 2016" with doi 10.1002/we.1870 | Correct | Add "floating" to the title check (the title is already right). Optionally add NREL/JA-2C00-62929. | confirmed |

**Errata in Graf et al. (accepted manuscript; these do not affect the notebook):**
- p. 2: "our turbulence simulation model TurbSim (reference)": an unresolved citation placeholder.
- p. 5, after eq. (2): "σ² is the variance of α". α is the fixed true mean; σ² should be the variance of the integrand samples f(x)^m.
- Placeholder report numbers "NREL/TP-5000-xxxxx" in refs. 9, 12 and 14; refs. 8 and 15 are "under review"/"forthcoming"; the DOI line reads "10.1002/we". These are manuscript-stage gaps, presumably fixed in the journal version (not checked).
- Convergence claims vary across the paper: "10,000 MC samples give accurate estimates" (p. 7), "within 5% after 5000 samples" (Fig. 6, p. 8) and "often within 5% … within 1,000 or less iterations" (p. 10). They describe different cases, but read as inconsistent; minor.
- p. 3: in the East Coast input file, `VM(a, b)` does not say which argument is the concentration and which the location. The first argument stays positive for every V, so it is presumably κ. Ambiguous.

The general AS exposition in cell 0 (C, eigendecomposition, partition, "each eigenvalue quantifies how much f changes, on average, along…") is correct (CDW 2014 Lemma 2.1; Constantine 2015 §3.1).

## 4. Code review
- **py3.** `print` statements in cell 7; `pn.DataFrame.from_csv` and `.as_matrix()` were removed.
- **Gradient estimates (cell 4).** 2016 `local_linear_gradients(XX, f, 60)` uses `MM = min(ceil(10 m ln m), M−1)` = 81 random centers per output, unseeded. Its neighbor list is `argsort(D2)[D2 != 0]`: a boolean mask in the original order applied to the sorted index array. This is wrong in general, fixed in v1 [C-02]. v1 computes one gradient per row of `X` by default, which is about 25× too slow here. Use `centers=` with a seeded subset (§6).
- **Randomness.** Unseeded: the gradient centers, `nboot=100`, the plot masks (`np.random.choice`, cell 5), and **a new mask for every (output, degree)** in cell 7 (21 draws, so the panels show different points). Use one `rng` and one plotting mask.
- **State mutation.** `ss.partition(2)` in cell 5, then `ss.partition(partitions[i])` in cell 7, on the same objects. Three `PolynomialApproximation` objects are retrained in place for each output. In v1, results are immutable, so use `ac.with_dimension`.
- **R² is in-sample** (legacy `.Rsqr`). With 50,000 points and ≤ 56 coefficients this barely matters: v1 LOO and in-sample R² differ by < 4e-4.
- **Wasted work.** `range(XX.shape[0])` is passed to `np.random.choice`; use `rng.choice(M, 10000, replace=False)`.
- **Labels.** `in_labels = keys[:5]` are the raw column names. Readable labels with units belong in the space `names`.
- No hard-coded absolute paths.

## 5. Data management review
- **Format.** CSV with 16-digit mantissas; `Run` is a redundant index. Inputs are stored only in the transformed/normalized form (Q1), and outputs have no units.
- **Normalization.** Vhub, Hs, Tp are uniform on [−1,1] (excess kurtosis −1.20) and look like probability coordinates. WaveDir and PlatformDir are bell-shaped (kurtosis −0.86/−0.82, std 0.489/0.452), with 441/284 values outside [−1,1]. They look like angles/π from unwrapped von Mises samples (§1 item 2). Neither encoding is documented here or in the paper.
- **Size.** 15.1 MB, below the 50 MB warning. Measured alternatives: `gzip -9` 6.21 MB; `npz` float64 (X, f) 4.55 MB; `npz` float32 2.21 MB. History: the deleted `windData500k.mat` (64.5 MB) and this file (15.1 MB) both stay in `.git` unless the history is rewritten.
- **Recommendation.** Ship `nrel_wind.npz` (X, F, `input_names`, `output_names`, units once known) at 4.55 MB, or `.txt.gz` (6.2 MB) to keep text. If the 500k set is meant to be public, deposit it on Zenodo (it would be about 45 MB as npz) rather than in git.
- **Provenance and license.** NREL: Graf et al., OC3 Hywind, East Coast NDBC site (paper pp. 2–4). The paper has no data-availability statement, and the license for redistribution is unknown (Q3).
- **pandas-free loading (tested):**
```python
import numpy as np
fname = "NRELwind_testing.txt"
with open(fname) as fh:
    names = fh.readline().strip().split(",")
d = np.loadtxt(fname, delimiter=",", skiprows=1)      # (50000, 13), 0.2 s
X, F = d[:, 1:6], d[:, 6:]
in_labels, out_labels = names[1:6], names[6:]
```

## 6. Migration to active_subspaces v1
| legacy call | v1 replacement | notes |
|---|---|---|
| (implicit: `XX` treated as normalized) | `space = ac.Uniform(lower=np.minimum(X.min(0), -1), upper=np.maximum(X.max(0), 1), names=in_labels)` | `ac.Uniform.reference(5)` **fails** (`input.out_of_space`, 724 rows). The data-bounds box rescales WaveDir and PlatformDir by ≤ 5 % relative to 2016, so eigenvalues and vectors change slightly. This is a stopgap until Q1 gives the true transformation. If the angle hypothesis holds, the angle support is [μ/π − 1, μ/π + 1], with μ depending on Vhub for WaveDir, so it is not a product box. Do not wrap the angles to [−1, 1]: that would cut the von Mises mode. Do not use `ac.active_domain`, `inverse_map` or integration with this space: the density is not uniform. |
| `ac.gradients.local_linear_gradients(XX, f[:,i][:,None], 60)` | `g = ac.local_linear_gradients(X=X, f=F[:, i], space=space, p=60, centers=rng.choice(M, 2000, replace=False))` | returns a `GradientResult`. v1 `p` counts the center [C-02]. 2000 centers take 5–15 s per output (prototype); all 50,000 would be about 25× slower. |
| `ss.compute(df=df, nboot=100)` | `sub = ac.active_subspace(grads=g, nboot=100, rng=rng)` | the space comes from `g`. Bootstrap bands are 95 % percentile by default; use `interval="minmax"` in the plot for the 2016 look. |
| `ss.partition(k)` | `ac.with_dimension(sub, k)` or `ac.active_subspace(..., n=k)` | v1 EVG default picks 1, 1, 2, 1, 1, 1, 1. Pass `n` explicitly. |
| `ac.utils.plotters.eigenvalues(ss.eigenvals, ss.e_br, label)` | `ac.plotting.plot_eigenvalues(sub, interval="minmax")` | |
| `ac.utils.plotters.eigenvectors(ss.W1, in_labels=…, out_label=…)` | `ac.plotting.plot_eigenvectors(ac.with_dimension(sub, 2))` | **sign rule:** v1 makes the largest entry positive (2016: the first). For the anchors, w₁ ≈ −e₅ in 2016 becomes +e₅, which mirrors the 1-D summary plots. This is expected; say so in the text. |
| `ac.utils.plotters.sufficient_summary(XX.dot(ss.W1)[mask], f[mask,i], label)` | `ac.plotting.plot_sufficient_summary(sub2, X=X[mask], f=F[mask, i], qoi_label=…)` | takes X, not y |
| `PolynomialApproximation(N).train(y, f)`, `.predict(y)[0]`, `.Rsqr` | `s = ac.fit_surrogate(X=X, f=F[:, i], subspace=ac.with_dimension(sub, n_i), kind="poly", degree=N)`; `ac.predict(s, X)`; `s.loo_r_squared`; `ac.plotting.plot_surrogate_fit(s, X=X[mask], f=F[mask, i])` | X is in the space's coordinates. LOO R² ≈ in-sample R² here. |
| `print '{:<15}…'.format(...)` | f-strings | |

## 7. Associated .py files
None.

## 8. Update plan (one commit each)
1. Data: convert to `nrel_wind.npz` (or `.txt.gz`) after Q1–Q3, and add a `README` with columns, units, transformation and provenance.
2. Notebook to Python 3: kernelspec, numpy loader, one `rng = np.random.default_rng(2016)`, clear outputs.
3. v1 calls per §6: space, local-linear gradients at 2000 seeded centers, `active_subspace(nboot=100)`, v1 plots, `fit_surrogate` with explicit `n`, one fixed plotting mask.
4. Text fixes from §3: 50,000 rows (subset of the paper's 500k 5D MC set), the input encoding, output definitions (DEL, m = 4) and units, "floating (OC3 Hywind)", fixed-bottom vs floating standards, the center count, the gap/dimension justification, and the coordinate-aligned remark with the paper's Fig. 6/7 observations.
5. Top-level README: "fatige" → "fatigue". Label the link as the model/data source ("[model]"), since it is not an active-subspace paper.
6. Execute headless and commit with outputs (keep the PNG count small: about 28 figures at 3.6 MB of notebook today; consider grouping the 4 plots per output into one figure).

## 9. Acceptance criteria
- `cd NREL_Wind && jupyter nbconvert --to notebook --execute NREL_Wind.ipynb` succeeds in `.venv` in < 5 min (prototype: about 40 s for gradients, subspaces and 21 surrogate fits, without the bootstrap).
- No pandas, no Python 2 syntax, a single seeded `rng`. No warnings in the output.
- Leading eigenvalues (2000 centers, p = 60) are within ±30 % of the legacy plots: RootMxc1 λ₁ ≈ 0.9–1.0e6 with λ₁/λ₂ ≳ 100; RootMyc1 λ₁ ≈ 4.2e6 with λ₁/λ₂ ≳ 50; anchor tensions λ₁/λ₂ ≈ 3–4 and λ₂/λ₃ ≈ 15–20. |w₁| matches: RootMxc1 and RootMyc1 ≈ e₁ (Vhub) with |w₁₁| > 0.99; Anch1–3 ≈ e₅ (PlatformDir) with |w₁₅| > 0.99.
- Surrogate LOO R² with dimensions (1,1,3,3,2,2,2), degrees 3/4/5: within 0.035 of the legacy table. Prototype: RootMxc1 0.975/0.976/0.983; RootMyc1 0.970/0.972/0.976; TwrBsMxt 0.733/0.869/0.942; TwrBsMyt 0.797/0.907/0.925; Anch1 0.211/0.756/0.775; Anch2 0.767/0.869/0.878; Anch3 0.776/0.856/0.866.
- Figures: for each output, eigenvalues, eigenvectors (n ≤ 2 shown), and 1-D + 2-D summary plots; one surrogate-fit grid (7 × 3); one R² table printed. Axes carry units once Q4 is answered.
- The text states 50,000 samples (and their relation to the paper's 500k set), the number of gradient centers, the input encoding (probability coordinates vs angles/π), and that the outputs are m = 4 DELs.

## 10. Prototype results
Run in `.venv` (scripts in `prototypes/g5/wind1.py`, `wind2.py`):
- Loading: `np.loadtxt` 0.2 s, (50000, 13). Input statistics as in §5. 724 rows lie outside [−1,1]⁵.
- `ac.local_linear_gradients(..., space=ac.Uniform.reference(5))` → `InputError input.out_of_space`. Since V1-10 the message reads `X: 724 of 50000 rows lie outside the support; first: row 7, input 'PlatformDir' (column 4) = 1.0389279698213882 > upper 1.0 [input.out_of_space]`, with `details["n_outside"] == 724` and `details["name"] == "PlatformDir"` ([#43](https://github.com/paulcon/as-data-sets/pull/43)).
- With the data-bounds box, 2000 centers, p = 60, `nboot=100`: eigenvalues per output (v1 vs a re-implementation of the legacy 81-center version):
  - RootMxc1: [8.94e5, 4.63e3, 4.50e3, …] vs [7.86e5, 7.92e3, 5.63e3, …]; w₁ = [1.00, .01, .01, 0, 0].
  - TwrBsMxt: [2.53e8, 2.00e8, 3.10e7, …] vs [3.66e8, 1.28e8, …]; w₁ = [.57, .20, .80, −.02, 0] vs [.73, .21, .65, …]. w₁ is not stable because λ₁ ≈ λ₂, which matches the legacy figure (WaveDir-dominated).
  - TwrBsMyt: w₁ = [1.00, .09, 0, −.01, −.01].
  - Anch1/2/3: w₁ ≈ e₅; λ₁/λ₂ = 3.3 / 3.5 / 3.1.

  These match the legacy PNGs (cell 5) within sampling noise.
- `ac.fit_surrogate(kind="poly")`: the R² table in §9 reproduces the legacy in-sample table to ≤ 0.03 (largest difference: Anch1 quintic 0.775 vs 0.746). In-sample and LOO R² agree to < 4e-4. No warnings were raised.
- Input-encoding check (after reading Graf et al.):
  - By Vhub quintile, WaveDir mean/std go from −0.068/0.539 to +0.024/0.434. PlatformDir, Hs and Tp are flat (PlatformDir mean ≈ 0.046, std ≈ 0.45; Hs and Tp std 0.576–0.580).
  - PlatformDir range width is 1.9997.
  - Mapping x_Vhub through the paper's Weibull W(2.12, 9.77) gives V median 8.2 m/s and 99th percentile 20 m/s. WaveDir < −1 occurs only for V ≤ 10.9 m/s (435 rows), and > 1 only for V ≥ 12.3 m/s (6 rows).
  - The paper's 4D coefficients (κ = 0.03 + 0.12V, μ = −0.62 + 0.031V) are violated by 1119 rows, so the shipped data used other parameters.
- Sizes: §5.

## 11. Open questions for the author
1. How do the stored inputs relate to the physical values (Vhub [m/s], Hs [m], θ_w [rad], Tp [s], θ_p [rad])? The data suggest that Vhub, Hs and Tp are 2u − 1 with u a (conditional) CDF value, and that WaveDir and PlatformDir are θ/π from unwrapped von Mises samples (§1 item 2). Is that right? Which distribution parameters were used for the 5D set (the paper, p. 3, lists only the 4D East Coast file)? Can physical values be supplied?
2. Is `NRELwind_testing.txt` a random 50,000-row subset of the paper's 5D 500,000-sample MC set (§4.3, p. 7)? Should the 500k set be published (e.g. Zenodo) and the text say so?
3. Provenance and license: who generated the data (P. Graf, NREL?), and may it be redistributed under the repo's terms? Should the notebook credit NREL/Graf in the header?
4. Units of the outputs. Confirm that `rf4_` = short-term DEL with m = 4, 1 Hz equivalent, from a 10-min simulation (paper pp. 3–4). Should the notebook switch the blade-root outputs to m = 10, as the paper does for composites, or keep m = 4 for all?
8. Is it worth noting that the AS analysis of these data is unpublished (the cited paper is about Monte Carlo integration and does not use active subspaces)?
5. Which subspace dimensions should be used? Keep (1,1,3,3,2,2,2) for the surrogates, or justify by gaps?
6. Current affiliations and emails for the header.
7. Should the history be rewritten to purge `windData500k.mat` (64.5 MB of the 225 MB `.git`)? This is a repo-wide decision.
