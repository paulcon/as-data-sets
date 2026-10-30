# Ebola: update spec

Status line: `Ebola/Ebola.ipynb` | paper: Diaz, Constantine, Kalmbach, Jones, Pankavich, *A modified SEIR model for the spread of Ebola in Western Africa and metrics for resource allocation*, Appl. Math. Comput. 324 (2018) 141–155, doi:10.1016/j.amc.2017.11.039 (checked against the published PDF; arXiv:1603.04955 v3 read first; page refs below are to the published version) | m = 8 (β1, β2, β3, ρ1, γ1, γ2, ω, ψ), uniform | QoI: basic reproduction number R0 (closed form, analytic gradient), two parameter boxes (Liberia, Sierra Leone) | legacy API: `ac.utils.quadrature.gauss_legendre`, `ac.subspaces.sorted_eigh`, `ac.utils.plotters.{eigenvalues,eigenvectors,sufficient_summary}` | py2 issues: no print statements; kernelspec python2; otherwise py3-clean.

## 1. Summary

No data files: R0 and its gradient are closed form, and C is computed with an 8^8 tensor Gauss–Legendre rule (16.8M points), as in the paper. Model, R0, both parameter tables and the conclusions match the published paper (verified number by number: Eqs. (1)–(2) p.144, Eq. (4) p.145, Tables 2–3 pp.145–146, Table 7 p.150, Figs. 7–9 pp.151–152). Main problems:

1. **The eigenvalues are 256 times too small (confirmed).** `subspace_integrand` multiplies by `2**-8`, but legacy `gauss_legendre` weights already sum to 1. The legacy plot shows λ1 ≈ 4.0e-3 for Liberia; the correct value is 1.022. The eigenvectors are unaffected.
2. **Wasted computation.** A Python loop over 16.8M points builds an 8×8 outer product per point ("about 15 minutes" in cell 4) and needs several GB of RAM. A chunked `G.T @ (w*G)` at 8^8 takes 5 s. v1 `ac.active_subspace` with a 6^8 rule takes 1.2 s and about 1 GB, and its λ1 is within 0.3% of the 8^8 value.
3. **Numbers in the text.** The README says "9-parameter"; the active subspace uses 8 (the paper's "9-parameter fit" is the ODE fit). The notebook gives a wrong reference title, an off-by-one bound (`n < 7`), and typos.
4. **Questionable range in the paper.** The fitted Sierra Leone γ2 = 0.0833 (Table 2, p.145) lies outside the paper's own range [0.1236, 0.3840] (Table 7, p.150). This is in the published version too, so it is not a transcription error. Ask the author.
5. The summary-plot samples are unseeded, and the v1 eigenvector sign rule flips w1 relative to the legacy figures (ψ becomes positive).
6. **A published sentence is refuted by the paper's own figure.** Fig. 7 (p.151, natural-log axis) shows ln λ1 − ln λ2 ≈ 1.4 (L) and ≈ 2.0 (S), which is λ1/λ2 ≈ 4 and ≈ 7, matching the recomputation (3.9 and 7.1). The body text on p.151, "a notable (nearly two orders of magnitude) spectral gap occurs between λ1 and λ2", is wrong. The notebook does not repeat it.
7. **Published affiliations differ from the notebook.** Diaz is at CU Boulder Aerospace, Constantine at CU Boulder CS, Jones at UCSB Physics; Kalmbach and Pankavich are at Mines. The notebook lists all five at Mines with Mines e-mail addresses (as on arXiv).

## 2. Inventory

| file | size | format | role | loaded how | notes |
|---|---|---|---|---|---|
| Ebola.ipynb | 577 KB | nbformat 4, python2 kernel | whole case study | — | 9 PNG outputs, no printed numbers |
| (none) | — | — | — | — | README says "Each directory contains a data set as a CSV file"; not true here (analytic model) |

## 3. Text and math review (vs. literature)

| cell | quoted text | issue | fix | confidence |
|---|---|---|---|---|
| 0 | ODE system, R0 formula, both parameter tables | match published Eqs. (1)–(2) (p.144), Eq. (4) (p.145), Table 2 (p.145), Table 3 (p.146), Table 7 (p.150). The notebook omits Guinea, which the paper also leaves out of the active-subspace analysis | none | confirmed |
| 0 | "This results in basic reproduction numbers of 1.563 for Liberia and 1.445 for Sierra Leone." | matches Table 3 (p.146). Recomputing from the Table 2 values gives 1.5634 and 1.4433. The published Fig. 2c panel (p.145) gives less-rounded SL values (β3 = 0.0791, ρ1 = 0.766, γ1 = 0.0510); with those, R0 = 1.4451, so the difference is rounding | add "(computed from unrounded fits)", or print R0 at the nominal values in code | confirmed (rounding; published Fig. 2c) |
| 0 | ranges table, SL row "$\gamma_2$\|U(.1236, .384)" | matches published Table 7 (p.150), but excludes the fitted SL γ2 = .0833 (Table 2, p.145; Fig. 2c). The Table 7 caption says the γ intervals come from "data from Ref. [19]", not from the fits | ask author; at minimum add a footnote | confirmed (in published paper) |
| 0 | "Suppose $\lambda_{n+1} \ll \lambda_n$ for some $n < 7$." | m = 8, so n < 8 | "for some $n < 8$" | confirmed |
| 0 | "[[2]] P.G. Constantine, E. Dow, and Q. Wang. _Active subspaces in theory and practice: applications to kriging surfaces_" | title is "Active subspace methods in theory and practice: applications to kriging surfaces" (Crossref) | fix title | confirmed |
| 0 | "[[1]] … arXiv:1603.04955, 2016" | published version exists | cite Appl. Math. Comput. 324 (2018) 141–155, doi:10.1016/j.amc.2017.11.039; keep the arXiv link | confirmed |
| 0 | author block, all at Colorado School of Mines with mines.edu e-mails | matches arXiv v3, but the published version (p.141) lists Diaz (Aerospace Eng. Sci., CU Boulder), Constantine (Computer Science, CU Boulder), Kalmbach and Pankavich (Mines), Jones (Physics, UCSB) | either keep the 2016 affiliations with a note ("affiliations at time of writing") or update to the published ones; drop stale e-mails (ask author) | confirmed (differs from published) |
| 0 | "should recieve additional aid"; "deaths in hospitals are properly burried" | typos | receive, buried | confirmed |
| 0 | $\lambda_i = \int(\nabla R_0^T\mathbf w_i)^2\rho\,d\mathbf x$; activity scores $\alpha(n)=\sum_{j\le n}\lambda_j\mathbf w_j^2$ | consistent with Constantine 2015 and CDW 2014 | none | confirmed |
| 4 | "Gauss-Legendre quadrature with 8 points in each of the 8 dimensions … This takes about 15 minutes" | 8^8 matches the paper (p.151, and Fig. 7–9 captions); runtime is an artifact of the loop | vectorize (§4); update the runtime sentence | confirmed |
| 5 | "#un-normalized inputs for Liberia (S) and Sierra Leone (S)"; "noramlized" | comment typos | (L); normalized | confirmed |
| 9 (figure) | eigenvalue plots, λ1 ≈ 4e-3 (L) | 256× too small (`2**-8`, §4) | remove the factor | confirmed (legacy fig vs recomputation) |
| 10 | "the last 6 eigenvalues are at least an order of magnitude smaller than the first … reasonably good 2-dimensional subspace" | true (λ3/λ1 = 0.025 L, 0.020 S); the published Fig. 7 caption (p.151) uses the λ2–λ3 gap for n = 2. The body text on p.151 says "nearly two orders of magnitude … between λ1 and λ2". The paper's own Fig. 7 (natural-log axis) shows a λ1/λ2 of about 4 (L) and 7 (S), and recomputation gives 3.9 and 7.1. Fig. 7 eigenvalues are about 4× the recomputed values with a uniform offset, consistent with gradients scaled to [0,1] instead of [−1,1]; spacings agree to ±0.05 in ln | keep the notebook text; do not import the paper's sentence; mention in §11 | confirmed (notebook correct; paper sentence refuted by its own Fig. 7) |
| 14 | "ψ … is the most important parameter and several parameters (especially β2, ρ1, and ω) have weight near zero" | matches recomputed w1 (β2 −.06, ρ1 −.04, ω .04) and the published Fig. 9 caption (p.152). The published Conclusions (p.153) name "ω, as well as β2 and γ1" instead. That is inconsistent with Fig. 9 and with the recomputed scores (γ1: 0.052 L / 0.032 S vs ρ1: 0.002) | none in the notebook; do not copy the Conclusions' list | confirmed |
| 16 | "ψ is more influential on output in Sierra Leone than in Liberia" | normalized activity scores for ψ: 0.52 (L) vs 0.69 (S); matches the published text (p.151–152, "significantly greater within Sierra Leone") | none | confirmed |
| 16 | "Sierra Leone would more greatly benefit from more resources, as ψ has higher influence … " | the published Conclusions (p.153) say this only for large changes ("if the introduction of new resources can drastically alter parameters … global activity scores … indicate that Sierra Leone will derive the greater benefit"). They also say local metrics favor Liberia when resources are scarce. The notebook drops that caveat | add the paper's caveat: global scores favor Sierra Leone, local metrics favor Liberia | confirmed (caveat missing) |

## 4. Code review

- **Bug (confirmed): `subspace_integrand` multiplies by `2**-8`.** Legacy `g1d` returns `w = V[0,:]**2`, which sums to 1, so C and every eigenvalue are scaled by 1/256. The text quotes no eigenvalues, so only the y-axis of the cell 9 figures is wrong.
- **Wasted computation.** In cell 5, `integrands` is a list of 16.8M 8×8 arrays. That is about 8.6 GB of Python objects, plus arrays of shape (16.8M, 8) for `xx`, `x_L`, `x_S`, `dR_L`, `dR_S` and `hstack` temporaries. Replace it with `C = (G*w[:,None]).T @ G`, chunked over the first two axes (5 s at 8^8), or use v1 with a 6^8 rule (§6).
- `R0_grad` checked against finite differences: correct. The maximum relative error is 1e-9 for β1–ρ1, and 1e-5 for γ1 and γ2, which is FD truncation.
- `xx` (cell 5) is reused as a different variable in cell 11. `W1_L` and `W1_S` are defined in cell 11 and reused in cell 15.
- Randomness: `np.random.uniform(-1, 1, (2000, 8))` in cell 11 has no seed. Use `rng = np.random.default_rng(<seed>)` and `space.sample(2000, rng=rng)`.
- py2: none besides the kernelspec and `%matplotlib inline`, which is harmless.
- Hard-coded paths: none.

## 5. Data management review

- No data. The model is three lines of closed form, so nothing to store.
- README "Data sets" bullet says "a 9-parameter dynamical system model". The active subspace is over 8 parameters (δ is fixed at 1/9, and ρ2 does not enter R0). Change the bullet to "8-parameter (R0 of a 9-parameter SEIR fit)".
- The pandas-free loading snippet is not applicable.
- License and provenance: parameter values come from the paper's Tables 2, 3 and 7. Cite them in the notebook.

## 6. Migration to active_subspaces v1

| legacy call | v1 replacement | notes |
|---|---|---|
| `ac.utils.quadrature.gauss_legendre([8]*8)` → (x, w) in [-1,1]^8 | `t, w = scipy.special.roots_legendre(k); w = w/2`, tensor via `np.meshgrid(..., indexing='ij')` and `np.prod` | weights must sum to 1 within 1e-10 for `quad_weights` (6^8 rule sums to 1 − 7e-16: OK) |
| hand loop for C, `ac.subspaces.sorted_eigh(C)` | `space = ac.Uniform(lower=lb, upper=ub, names=[...])`; `X = space.from_normalized(T)`; `sub = ac.active_subspace(space=space, grads=R0_grad(X), quad_weights=W)` | pass **physical** gradients; v1 applies `(ub-lb)/2`. Drop the manual `dR *= (ub-lb)/2.` and the `2**-8` |
| `e_L, W_L` | `sub.eigenvalues` (m,), `sub.W` | eigenvalues 256× the legacy figure values |
| `ac.utils.plotters.eigenvalues(e, out_label=)` | `ac.plotting.plot_eigenvalues(sub)`; set title via returned `ax` | |
| `ac.utils.plotters.sufficient_summary(xx.dot(W1), R)` | `ac.plotting.plot_sufficient_summary(sub, X=X_phys, f=R, dims=1)` and `dims=2` | takes physical X; one figure per call |
| `ac.utils.plotters.eigenvectors(W[:,0][:,None], in_labels=)` | `ac.plotting.plot_eigenvectors(sub, indices=[0], labels=in_labels)` | **sign**: v1 makes the largest-magnitude component positive, so ψ is +0.76/+0.88 where the legacy plot shows −; the text does not mention sign |
| activity scores (hand) | keep the hand formula: `a = (sub.eigenvalues[:2]*sub.W[:,:2]**2).sum(1)` | v1 has no activity-score function |
| 8^8 rule | 6^8 rule (1.68M points) in v1, or 8^8 computed chunked by hand with `ac.subspace_from_basis` for plots | `subspace_from_basis` gives NaN eigenvalues, so it cannot be used for the eigenvalue plot. Recommend 6^8: λ1 1.0194 vs 1.0217 (L), 1.1014 vs 1.1049 (S); w1 agrees to 1e-3. State the rule size in the text |

## 7. Associated .py files

None.

## 8. Update plan

1. Fix the text: the reference [2] title, the published citation for [1] (Appl. Math. Comput. 324 (2018) 141–155), `n < 8`, typos, and the runtime sentence. Add a note on the SL γ2 range, add the local-vs-global caveat to the cell 16 policy sentence, and settle the affiliations (§11).
2. Convert to Python 3 (kernelspec `python3`), add a seeded `rng`, and remove the unused `subspace_integrand`.
3. Replace the quadrature and C loop with the tensor rule plus `ac.active_subspace(..., quad_weights=W)` per country, using physical gradients.
4. Replace the plot calls with `ac.plotting.*`, and keep the activity-score figure by hand. Adjust the sign wording if needed.
5. Update the README bullet ("8-parameter") and drop the "CSV in each directory" claim, or add "or closed-form model".
6. Execute headless and commit the outputs.

## 9. Acceptance criteria

- `jupyter nbconvert --execute` in `.venv` finishes in < 30 s with peak RSS < 2 GB.
- Imports are only numpy, scipy, matplotlib and active_subspaces.
- Eigenvalues (6^8 rule): Liberia λ1..3 = 1.0194, 0.2621, 0.02506 (±0.1%); SL λ1..3 = 1.1014, 0.1546, 0.02169 (±0.1%). These are **not** the legacy figure values, which are 256× smaller; say so in a markdown cell.
- w1 (sign by the v1 rule): L ≈ [−.381, −.061, −.337, −.043, .255, .299, .038, .761]; SL ≈ [−.364, −.056, −.167, −.039, .190, .177, .034, .875] (±0.005).
- Normalized activity scores for ψ: 0.52 (L) and 0.69 (S) (±0.01).
- Figures: 2 eigenvalue plots, 4 sufficient-summary plots (1-D and 2-D per country), 2 eigenvector plots, 1 activity-score figure.

## 10. Prototype results

Scripts are in `prototypes/g1/` (`ebola_proto.py`, `ebola_k8.py`, `ebola_v1k.py`, `ebola_as.py`), all run with the venv python.

- R0 at the nominal values: 1.56343 (L) and 1.44333 (S). R0_grad matches finite differences.
- Convergence of the tensor rule, λ1 (L/S):

  | k | L | S | time |
  |---|---|---|---|
  | 3 | 0.919 | 0.948 | |
  | 4 | 0.989 | 1.055 | |
  | 5 | 1.0125 | 1.0909 | |
  | 6 | 1.0194 | 1.1014 | |
  | 7 | 1.0213 | 1.1042 | |
  | 8 (chunked, hand) | 1.0217 | 1.1049 | 5 s |

- `ac.active_subspace` on 5^8 with `quad_weights` reproduces the hand C exactly. On 6^8 it takes 1.2 s with RSS of about 0.97 GB. Monte Carlo with N = 1e4 gives λ1 within 0.6%.
- Legacy figure check: λ1(L) on the legacy plot is about 4e-3. That equals 1.0194/256 = 3.98e-3, which confirms the `2**-8` bug. The legacy eigenvector plots match the recomputed w1 up to sign.
- Published-figure check, by reading the PDF p.151 at 200 dpi: in Fig. 7 (ln axis), Liberia ln λ ≈ [1.4, 0, −2.3, −3.0, −4.0, −5.2, −6.5, −8.0]. The recomputed consecutive ln-spacings [1.36, 2.35, 0.66, 1.09, 1.13, 1.35, 1.55] match the figure's [1.4, 2.3, 0.7, 1.0, 1.2, 1.3, 1.5]. The absolute offset of +1.38 means a factor of ≈4, consistent with a [0,1] normalization. This confirms λ1/λ2 ≈ 4 and refutes the p.151 "nearly two orders of magnitude" sentence.
- SL R0 using the published Fig. 2c parameter labels (β3 = 0.0791, ρ1 = 0.766, γ1 = 0.0510) is 1.4451, which matches Table 3.

## 11. Open questions for the author

1. SL γ2: the fitted value 0.0833 lies outside the analysis range [0.1236, 0.384] (paper Table 7). Is the range intended, or is it a typo?
2. The published text (p.151) says there is a "nearly two orders of magnitude" gap between λ1 and λ2. The paper's own Fig. 7 and the recomputation both give λ1/λ2 ≈ 4 (L) and 7 (S). Erratum? The notebook should simply not repeat it. Separately, Fig. 7 eigenvalues are about 4× ours: were gradients scaled to [0,1]?
3. Accept a 6^8 rule (0.3% change in λ1) in place of the paper's 8^8 for v1? The alternative is hand-chunked 8^8 C with a hand eigendecomposition and plotting.
4. Keep the "Sierra Leone would more greatly benefit" policy sentence? If so, add the paper's caveat (p.153) that local metrics favor Liberia.
5. Affiliations: keep the 2016 Mines affiliations and e-mails, or switch to the published ones (CU Boulder, UCSB, Mines)?
6. The published Conclusions (p.153) name γ1 among the negligible parameters, but Fig. 9 and the recomputation say ρ1. Typo?
