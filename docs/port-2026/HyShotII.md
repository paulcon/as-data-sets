# HyShotII: update spec

Kind: **stored data**: HyShot II scramjet simulations (JCP 2015; book §5.1) (see [README.md](../README.md#two-kinds-of-data-sets)).

Status line: `HyShotII/HyShotII.ipynb` | paper: Constantine, Emory, Larsson, Iaccarino, "Exploiting active subspaces to quantify uncertainty in the numerical simulation of the HyShot II scramjet", J. Comput. Phys. 302 (2015) 1–20, doi:10.1016/j.jcp.2015.09.001 (published version checked: Table 2 p. 8, eq. (15) p. 10, Table 3 p. 14, Table 4 p. 17); **and** Constantine, *Active Subspaces*, SIAM 2015, §5.1 pp. 71–78 (Tables 5.1–5.2, Figs. 5.2–5.6), doi:10.1137/1.9781611973860. The notebook's φ=0.35 analysis and the 0.30 column of `HyShotII35.txt` follow the **book**, not the paper (see §3) | m = 7 | QoI: exit pressure [bar], the pressure averaged over the last 10 mm of the combustor, at fuel plenum pressure P0,H2 = 4.8 bar (φ ≈ 0.30) and 5.6 bar (φ ≈ 0.35) | legacy API: `Subspaces.compute(sstype='OLS', nboot=500)`, `utils.misc.BoundedNormalizer`, `utils.misc.atleast_2d_col`, `utils.plotters.sufficient_summary/eigenvectors`, `utils.response_surfaces.PolynomialApproximation(2)` | py2 issues: yes

## 1. Summary
This is a small, cheap notebook: 52 + 16 runs, global OLS direction, bootstrap, a quadratic in the active variable, a safe box from SLSQP, and a CDF. The port is straightforward. The OLS direction reproduces to 5e-5 in v1. Problems found:
1. **The notebook mixes two published studies.** φ=0.30 (`HyShotII30.txt`, 50+2 runs) is the JCP paper's 4.8 bar study: w matches paper Table 3. φ=0.35 (`HyShotII35.txt`, 14+2 runs) is the **book's** §5.1 study: w matches book Table 5.2 exactly, but not paper Table 3. The notebook cites only the paper. The safe ranges in cell 21 match neither paper Table 4 nor book §5.1.3, though they are close to the book.
2. The bootstrap (cell 13) uses all 52 rows, including the 2 validation runs that cell 6 says are excluded.
3. Under Python 2, `1/M30` in cell 18 is integer division (= 0). Python 3 changes the printed safe ranges in the 3rd digit.
4. The stored "extreme" corners (last 2 rows of both files) do not equal `±sign(w)` for the notebook's `w`: the cowl-transition sign is flipped.
5. The data files mix physical and normalized rows. The "Exit press 0.30" column of `HyShotII35.txt` (unused by the notebook) is the **book's φ=0.30 data**: OLS on its 14 rows reproduces book Table 5.2 φ=0.30 (0.6933, 0.5033, 0.0067, 0.3468, −0.3732, −0.0524, 0.0605) to 4 digits. Its 2 corner runs are missing (`0` placeholders).
6. There are no units anywhere. The turbulence-intensity and length-scale units differ from the paper's (%, mm vs. fraction, m).
7. Errors in the published sources: the turbulence length-scale lower bound 0.1325 m is a typo (it should be 0.1225) in both paper Table 2 and book Table 5.1. In book §5.1.3 (p. 77), the stagnation enthalpy bound "greater than 2.23 MJ/kg" lies outside the input range [3.0551, 3.4280], so it is a typo (3.23?).

## 2. Inventory
| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| HyShotII.ipynb | 315 KB | nbformat 4, python2 kernel | analysis | — | 24 cells, 12 PNG outputs |
| HyShotII30.txt | 9.5 KB | CSV, **no header**, 52×8 | φ=0.30: 7 inputs **already normalized** to [-1,1] + exit pressure | `pn.DataFrame.from_csv(..., header=None, index_col=None)` | rows 0–49 design, rows 50–51 = validation runs at min/max corners |
| HyShotII35.txt | 3.5 KB | CSV, header, 16×10 | φ=0.35: Run, 7 inputs, Exit press 0.30, Exit press 0.35 | `pn.DataFrame.from_csv` (Run becomes the index) | rows 0–13 **physical** inputs. Rows 14–15 (Run 15, 16) **normalized** corner inputs, with `0` in the 0.30 column |

## 3. Text and math review (vs. literature)
| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| 0 | table "Turbulence intensity \| 0.1 \| 1.9" | Paper Table 2 gives 0.001–0.019 (a fraction). The notebook and data use percent. No units in the table. | Add a units column: AoA [deg], TI [%], L [mm], P0 [Pa], H0 [J/kg], x_c, x_r [m]. Add a note that TI is in percent. | confirmed |
| 0 | "Turbulence length scale \| 122.5 \| 367.5" | Paper Table 2 (p. 8) and book Table 5.1 (p. 72) both say 0.1325–0.3575 m. The paper's text (p. 9) says "L_t,ω0 ≈ 245 mm … ± 50%", which gives 0.1225–0.3675. So **the published lower bound is a typo in both sources**, and the notebook is right. The data's minimum (127.0) is below 132.5, which confirms 122.5. | Keep the values, state the unit (mm), and add a footnote about the typo. | confirmed (published PDF) |
| 0 | "Stagnation pressure \| 1.6448e7 \| 1.9012e7", "Stagnation enthalpy \| 3.0551e6 \| 3.4280e6" | These match the paper (16.448–19.012 MPa, 3.0551–3.4280 MJ/kg). Units missing. | Add Pa and J/kg. | confirmed |
| 0 | "we compute this quantity for fuel-air equivalence ratios of .30 and .35" | Paper: the runs are at plenum pressures 4.8 and 5.6 bar, which correspond to φ = 0.30/0.35 *at nominal inflow*. | Reword with both. | confirmed |
| 0 | "a normalized integral of pressure over the end of the engine" | Paper: "over the last 10mm of the combustor", in bar. | Add "last 10 mm of the combustor; units bar". | confirmed |
| 0 | "Let $\mathbf f^k$ be the vector whose $i^{th}$ element is $f_{\pi^k}$ … row is $[1,\ \mathbf x_{\pi^k}^T]$" | The subscript is missing *i*. | Change to $f_{\pi_i^k}$ and $\mathbf x_{\pi_i^k}$. | confirmed |
| 0 | "[[2]] P.G. Constantine, E. Dow, and Q. Wang. _Active subspaces in theory and practice_ …" | The title is "Active subspace **methods** in theory and practice: applications to kriging surfaces" (Crossref). | Fix the title. | confirmed |
| 0 | author block | Matches the paper (CSM, Stanford CTR, UMD, Stanford). The emails are from 2015. | Keep. Optionally mark Howard as notebook author only. | confirmed |
| 6 | "we exclude those data points initially" | Cell 13's bootstrap includes them (M = 52). | Bootstrap on `[:-2]`. | confirmed |
| 11/12 | φ=0.35 w printed: 0.6996, 0.4823, −0.027, 0.1997, −0.4738, −0.0602, 0.0957 | **This equals book Table 5.2 (p. 74), φ=0.35, exactly.** Paper Table 3 (p. 14, 5.6 bar) differs: 0.7066, 0.5008, **+0.0289**, 0.2051, −0.4490, −0.0591, 0.0432. That vector is not reproducible from the shipped 14 or 16 rows; the paper's 5.6 bar runs are presumably a different data set. The φ=0.30 w matches paper Table 3 exactly. | Cite book §5.1 / Table 5.2 for φ=0.35 and the paper for φ=0.30. Add one sentence saying that the paper's 5.6 bar vector differs. | confirmed (published PDFs) |
| 13/15 | "N = int(1e5)" vs. paper "N = 100 bootstrap replicates" (p. 14) and book "10⁴ replicates" (p. 74) | Not an error, but 1e5 with `np.matrix` inverse is slow and adds nothing. | Use N = 10 000 (as in the book) with `lstsq`. | confirmed |
| 15 | "we can maximize $f$ by maximizing … $\mathbf w^T\mathbf x$ … four runs" | The stored corner rows have cowl transition = −1 at the min corner, but `w_cowl < 0` for both φ, so argmin wᵀx has cowl = +1. The corners were presumably chosen from an earlier w. The effect is negligible (|w_cowl| ≤ 0.06, and the bootstrap range covers 0). The paper (§5.3) discusses this sign sensitivity. | One sentence noting it. Compute `xmin = -np.sign(w)` for the box problem instead of reusing row −2. | confirmed |
| 17 | "an upper prediction limit that is 2.33 standard errors above our mean response" | The paper text says "upper 99% confidence bound". The Fig. 10 legend says "Prediction + 2.33 se". The code uses the **simple-linear-regression** prediction formula `1 + 1/M + (x-ȳ)²/Syy` on a **quadratic** fit, which is an approximation. | Use the exact quadratic PI `se·sqrt(1 + a(y)ᵀ(AᵀA)⁻¹a(y))`, a = [1, y, y²]. Footnote the paper's caveat that there is no noise. | confirmed (formula); naming per paper |
| 19 | "the $x_{i,min}$ are the components of xmin above" | `xmin` is never defined in the notebook. | Define xmin = argmin over [-1,1]^m of wᵀx = −sign(w) (paper eq. 27). | confirmed |
| 21 (output) | φ=0.30: all ranges full. φ=0.35: AoA ≤ 3.25, TI ≤ 0.953, H0 ≥ 3.25e6 | Paper Table 4 (p. 17): 4.8 bar AoA ≤ 4.29, TI ≤ 0.0188; 5.6 bar AoA ≤ 3.54, TI ≤ 0.013, H0 ≥ 3.15 MJ/kg. Book §5.1.3 (p. 77, 14+2 runs): φ=0.30 AoA < 3.99, TI < 0.018; φ=0.35 AoA < 3.31, TI < 0.011, H0 > "2.23" (a typo). For φ=0.30, the notebook's 50-run data never reaches the limit (prediction limit at max wᵀx is 2.78), so the whole box is safe. The book's 14-run φ=0.30 data (orphan column, without the missing corner runs) gives AoA ≤ 3.93 and TI ≤ 1.75 %, close to the book. φ=0.35 is in the book's ballpark but does not match it (book R² 0.998 vs. 0.9927 here). | Compare with the book, not the paper. State that the numbers are not reproduced exactly (the book's corner f-values for φ=0.30 are missing, and the PI details differ). | confirmed (published PDFs); residual mismatch cause unknown |
| 22 | "see also Figure 11" | The paper uses 5000 samples and a Gaussian KDE. The notebook uses 1e5 samples and an empirical CDF. | Mention the difference (fine to keep the ECDF). | confirmed |
| 0 | AS exposition: C = ∫∇f∇fᵀρ = WΛWᵀ, f ≈ g(W₁ᵀx), OLS algorithm | Consistent with book Alg. 1.3 (global linear model) and paper Algs. 1/2. | Cite book Alg. 1.3. | confirmed |
| 0 | References: only [1] the paper and [2] CDW | The φ=0.35 analysis is the book's §5.1. | Add the book (§5.1). | confirmed |

## 4. Code review
- **py2:** `print` statements (cells 11, 21). `bounds=zip(...)` (cell 20): use `[(-1, 1)]*m`. `1/M30`, `1/M35` (cell 18) are int division in py2 (= 0); py3 changes the results (see §10). `np.matrix(...).I` (cell 13).
- **pandas:** `DataFrame.from_csv` and `.as_matrix()` were removed in pandas 1.0. `in_labels[:, np.newaxis]` on an `Index` fails in modern pandas.
- **Bug:** cell 13 `M,m = XX30.shape` → 52 rows, including validation corners. Use `XX30[:-2]`.
- **Inconsistent:** surfaces (cell 18) are trained on all 52/16 rows. The paper does the same ("52 model runs"), so keep it but say so.
- **Grid root-find:** `avmaxval = x[np.argmin(abs(predlim(x)-2.8))]` on `linspace(-2,3,800)` returns the nearest grid point even when there is no crossing (φ=0.30: limit < 2.8 everywhere on the domain). Use `scipy.optimize.brentq` on [−Σ|w|, Σ|w|]. If there is no sign change, report "constraint inactive".
- **Wasted computation:** the 1e5 bootstrap with explicit inverse; `np.vectorize(lambda x: sum(s30 <= x)/N)` (Python `sum` over 1e5-element arrays × 300 points). Replace with `np.sort` + `np.searchsorted`.
- **Randomness:** `np.random.randint` and `np.random.uniform` are unseeded. Use `rng = np.random.default_rng(2015)`.
- **Hard-coded:** `range(7)`, `subplot(4,2,…)`, xl/xu duplicated from the markdown. `0.6*0.05` etc. are fine but write them as 0.03/0.07/0.087/0.203.
- Rows 14–15 of the 35 file are normalized, so cell 5 normalizes `X[:-2]` and vstacks the raw corner rows. This is fragile; see §5.

## 5. Data management review
- **Formats:** plain CSV, ≤ 10 KB, far below the size limits.
- **Mixed coordinates:** `HyShotII30.txt` is entirely normalized and has no header and no Run column. `HyShotII35.txt` has physical rows and 2 normalized rows. The normalization bounds are stored only in the notebook.
- **"Exit press 0.30" column** in `HyShotII35.txt`: the book's §5.1 φ=0.30 study, which uses the same 14 inputs as φ=0.35. OLS on these 14 values reproduces book Table 5.2 φ=0.30 exactly. These runs are **not** in `HyShotII30.txt` (nearest-neighbour distance ≥ 0.5 in normalized space). They sit ~0.08 bar above the 50-run φ=0.30 linear model, so the two φ=0.30 data sets differ systematically (different simulation setup? the paper's 4.8 bar vs. the book's φ=0.30). Rows 15–16 hold `0` because the book's φ=0.30 corner runs are not in the file. The notebook never uses the column.
- **Units:** none in the headers.
- **Provenance and license:** none in the directory. The README links only the paper; the 35 file is the book's data set.
- **Proposed (author decision, §11):** leave the files byte-identical and document everything in the notebook. Or regenerate them as `HyShotII_phi030.csv`/`HyShotII_phi035.csv` with physical inputs, a header with units, and a `validation` flag column, and drop the orphan column or set it to `nan`.
- **Tested pandas-free loader** (run in .venv):
```python
import numpy as np
names = ["AoA [deg]", "Turb int [%]", "Turb len [mm]", "Stag pres [Pa]",
         "Stag enth [J/kg]", "Cowl trans [m]", "Ramp trans [m]"]
lb = np.array([2.6, 0.1, 122.5, 1.6448e7, 3.0551e6, 0.03, 0.087])
ub = np.array([4.6, 1.9, 367.5, 1.9012e7, 3.4280e6, 0.07, 0.203])
d30 = np.loadtxt("HyShotII30.txt", delimiter=",")               # (52, 8), normalized inputs
d35 = np.loadtxt("HyShotII35.txt", delimiter=",", skiprows=1)   # (16, 10)
to_phys = lambda Z: lb + 0.5 * (ub - lb) * (Z + 1)
X30, f30 = to_phys(d30[:, :7]), d30[:, 7]
X35 = d35[:, 1:8].copy(); X35[-2:] = to_phys(X35[-2:]); f35 = d35[:, 9]
```

## 6. Migration to active_subspaces v1
| legacy call | v1 replacement | notes |
|---|---|---|
| `BoundedNormalizer(xl, xu).normalize(X)` | `space = ac.Uniform(lb, ub, names=names)`; pass physical X | `space.to_normalized(X)` when normalized coordinates are needed for plots and the box problem |
| `ss.compute(X=XX[:-2], f=f[:-2,None], sstype='OLS', nboot=500)` | `ac.active_subspace(X=X[:-2], f=f[:-2], space=space, estimator="ols", n=1, nboot=500, rng=…)` | Emits `ValidityWarning` (OLS on uniform). For φ=0.35 the bootstrap emits `AccuracyWarning` (rank-deficient replicates, M = 14). Show the warning or filter it with a comment. |
| `ss.W1` | `sub.W1[:, 0]` | Sign rule: largest component (AoA) positive, the same as 2016 here |
| `np.dot(XX, ss.W1)` | `ac.forward_map(sub, X=X)[0]` or `space.to_normalized(X) @ sub.W1` | |
| `plotters.sufficient_summary(y, f)` | `ac.plotting.plot_sufficient_summary(sub, X=X, f=f)` | Takes physical X |
| `plotters.eigenvectors(W1, w_br, …)` | matplotlib stem/errorbar from the hand-coded bootstrap | v1 has no eigenvector bootstrap bands (`W_br` removed) |
| hand bootstrap (np.matrix inverse) | keep, with numpy `lstsq` and `rng.integers` | Paper Alg. 2; pedagogical |
| `PolynomialApproximation(2).train(y, f)` | numpy `np.vander(y, 3)` + `lstsq` (needed for the PI). `ac.fit_surrogate(X=X, f=f, subspace=sub, kind="poly", degree=2)` is optional for the curve. | v1 `.r_squared`/LOO R² differs from the in-sample R² |
| `misc.atleast_2d_col` | removed; use 1-D arrays | |
| `BoundedNormalizer.unnormalize` | `space.from_normalized(Z)` | |
| `scipy.optimize.minimize(..., bounds=zip(...))` | `bounds=[(-1, 1)]*7`, `method="SLSQP"` | |

## 7. Associated .py files
None.

## 8. Update plan
1. Convert the kernel to python3. Replace pandas with the §5 loader and build physical X for both φ. Add units to the cell-0 table and add the TI/length-scale footnote.
2. Replace the subspace calls with `ac.active_subspace(..., estimator="ols", n=1)` and the plots with `ac.plotting`. Print the w table next to paper Table 3.
3. Rewrite the bootstrap: `rng`, `lstsq`, 50 rows only, N = 10 000. Histograms with matplotlib.
4. Quadratic plus exact prediction limit. Root-find with `brentq` on the active domain, and handle "no crossing" for φ=0.30. Define `xmin = -sign(w)`.
5. Safe-box SLSQP with list bounds. Print physical ranges with units.
6. CDF with `searchsorted` and seeded `rng`.
7. Text fixes from §3 (CDW title, subscripts, xmin, 4.8/5.6 bar, length-scale typo note). Add the book §5.1 as a reference. Say which data set comes from which source (paper: φ=0.30 50 runs; book: φ=0.35 14 runs, Table 5.2). Compare the φ=0.35 results with book Table 5.2 and §5.1.3, not with the paper's Tables 3/4.
7a. (Optional) add the book's φ=0.30 14-run analysis from the 0.30 column (OLS reproduces book Table 5.2), noting that its 2 corner runs are missing.
8. (Optional, author decision) restructure the data files per §5 and update the loader.

## 9. Acceptance criteria
- `jupyter nbconvert --execute` from `HyShotII/` in .venv finishes in < 60 s with no imports beyond numpy/scipy/matplotlib/active_subspaces.
- w(0.30) = [0.6506, 0.5565, −0.0002, 0.3685, −0.3566, −0.0196, 0.0607] ± 1e-4, matching both the legacy output and paper Table 3.
- w(0.35) = [0.6996, 0.4823, −0.0270, 0.1997, −0.4738, −0.0602, 0.0957] ± 1e-4 (legacy), unless the author supplies the paper's data.
- Quadratic R² (in-sample) 0.9926 (0.30) and 0.9927 (0.35). The paper gives 0.993/0.998.
- Safe box (approximate PI formula, py3, xmin = stored corner row as in legacy): φ=0.30 constraint inactive (full box); φ=0.35 AoA ≤ 3.250, TI ≤ 0.949 %, H0 ≥ 3.249e6 J/kg (±0.005 rel). The exact quadratic PI gives the same values at 4 digits. **With xmin = −sign(w)** (plan step 4) plus brentq, φ=0.35 becomes AoA ≤ 3.193, TI ≤ 0.874 %, H0 ≥ 3.265e6. Use whichever variant is implemented as the reference.
- Book φ=0.30 check (if step 7a is done): w = book Table 5.2 ± 1e-4.
- Figures: 2 summary plots, w stem plot with bootstrap range + 7 histograms (φ=0.30), 2 summary plots with corners, 2 quadratic/PI plots, 2 CDFs.

## 10. Prototype results
Scripts: `prototypes/g3/hyshot.py`, `.../g3/hyshot_safe.py`.
- v1 `active_subspace(estimator="ols", space=Uniform)` vs. legacy printed w: max |diff| 4.8e-5 (0.30) and 4.2e-5 (0.35); the difference is legacy rounding. λ₁ = 0.0309 / 0.0543. Warnings: ValidityWarning (OLS on uniform) and AccuracyWarning (rank-deficient bootstrap fit, φ=0.35).
- Bootstrap, 10 000 replicates, on 50 rows vs. 52 rows: component ranges differ by up to 0.03 (e.g. ramp min −0.022 vs −0.052).
- The safe-region reproduction matches the legacy output exactly with `1/M=0`. With py3 `1/M`: φ=0.35 ymax −0.0225 → −0.0288, AoA max 3.2533 → 3.2503, TI max 0.9528 → 0.9489. φ=0.30: the prediction limit at max wᵀx is 2.7823, so it never crosses 2.8.
- The paper's w(0.35) from Table 3 with the shipped data gives AoA ≤ 3.289 and TI ≤ 0.975 %, which also does not match Table 4.
- Follow-up against the published PDFs (inline script): OLS on the 14 "Exit press 0.30" rows equals book Table 5.2 φ=0.30 to 4 digits, and the φ=0.35 rows equal book Table 5.2 φ=0.35. Safe box with the exact PI, brentq and xmin = −sign(w): book φ=0.30 (14 rows) AoA ≤ 3.934, TI ≤ 1.753 % (book: 3.99, 1.8 %); φ=0.35 (16 rows) AoA ≤ 3.193, TI ≤ 0.874 %, H0 ≥ 3.265e6 (book: 3.31, 1.1 %, "2.23"). Quadratic R²: φ=0.30 14 rows 0.9923 (book 0.997 with 16); φ=0.35 16 rows 0.9927 (book 0.998).
- The two other HyShot II PDFs ("surrogate accelerated Bayesian inverse analysis…", CTR 2010 / flight data) are flight-condition inverse problems with different inputs. They are not sources for this data.

## 11. Open questions for the author
1. `HyShotII35.txt` is the book's §5.1 data (Table 5.2 reproduces). Where are the paper's 5.6 bar runs (Table 3 w = 0.7066, …)? Should the notebook present φ=0.35 as "book §5.1" and drop the paper comparison for it?
2. The book's φ=0.30 corner-run pressures (rows 15–16 of the 0.30 column, now `0`): are they available? Should the notebook also show the book's 14-run φ=0.30 analysis?
2a. Book p. 77 "stagnation enthalpy greater than 2.23 MJ/kg": an erratum for 3.23? Book Table 5.1 and paper Table 2 share the 0.1325 typo: erratum list?
3. May the data files be rewritten (physical units, headers, validation flag), or must they stay byte-identical?
4. Keep the paper's heuristic PI formula (to match Fig. 10), or use the exact quadratic-regression PI?
