# Steel_Arch_Inplane_Buckling: data

Kind: **Python model**. The model is `arch_buckling_model.py`, a closed-form formula, and `generate_data.py` draws the inputs with a fixed seed and writes all four CSVs from it.

## Files

| file | contents | role |
|---|---|---|
| `Steel_Arch_Inplane_Buckling.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2020; outputs committed |
| `arch_buckling_model.py` | `buckling_load(X, case, grad)`, `slenderness`, `branch`, `coefficient`, `section`, `implied_span`, `sample`; `NAMES`, `UNITS`, `LOWER`, `UPPER`, `CASES` | the model |
| `generate_data.py` | writes the four CSVs (`--check` compares a rerun with them) | regenerates the data |
| `inputs.csv` | 1000 × 7 physical inputs S (m), Φ (rad), E (MPa), D, B, t_w, t_f (m), columns in the order of `NAMES` | model input |
| `outputs.csv` | 1000 × 3: N_acr of the fixed arch (kN), N_acr of the pin-ended arch (kN), slenderness λ | model output |
| `gradients_fixed.csv` | 1000 × 7: ∂N_acr/∂x_j for the fixed arch, kN per unit of input j (**physical** inputs) | model output |
| `gradients_pinned.csv` | 1000 × 7: the same for the pin-ended arch | model output |

The CSVs are plain comma-separated numbers after two `#` lines (a description and the column names with units), written with 17 significant digits, so `np.loadtxt(path, delimiter=',')` reads them and a rerun reproduces them to rounding error (see below).

## Model

**Citation.** T.-H. Nguyen, "Global sensitivity analysis of in-plane elastic buckling of steel arches", *Engineering, Technology & Applied Science Research* 10(5):6476–6480, 2020, doi:[10.48084/etasr.3833](https://doi.org/10.48084/etasr.3833). Read from the journal's open-access PDF (journal header and page numbers; the file is named `_oa`, but it is the published version). The formulas are those of Y.-L. Pi, M. A. Bradford and B. Uy, "In-plane stability of arches", *Int. J. Solids Struct.* 39(1):105–125, 2002, doi:[10.1016/S0020-7683(01)00209-8](https://doi.org/10.1016/S0020-7683(01)00209-8) (the paper's ref. [14]); Pi et al. was not read, so the formulas are as the 2020 paper states them.

**Equations** (paper pp. 6476–6477). In-plane elastic buckling load of a shallow (Φ < 90°) circular steel I-section arch under uniform compression:

- Eq. (1): λ = SΦ/(4 r_x), r_x = √(I/A).
- Eq. (6): N_cr = π²EI/(kS)², k = 0.5 (pin-ended), 0.35 (fixed): second-mode flexural buckling load of a column of length S.
- Eq. (2), pin-ended: N_acr = (0.15 + 0.006 λ²) N_cr for 3.88 < λ < 9.38; (0.26 + 0.74 √(1 − 0.63 π⁴/λ²)) N_cr for λ > 9.38.
- Eq. (3), fixed: N_acr = (0.36 + 0.0011 λ²) N_cr for 9.87 < λ < 18.6; (0.6 + 0.4 √(1 − 3.109 π⁴/λ²)) N_cr for λ > 18.6.
- Eqs. (4)–(5), deep arches (Φ > 90°), are not needed: Φ ≤ 44° here.

The square roots in the second branches are visible in the typeset equations (page image) but missing from the PDF's text layer; the paper card had flagged these terms as garbled. With the square roots, the fixed-arch branches meet at λ = 18.6 (c = 0.7406 and 0.7412), which supports the reading. The pin-ended branches do not meet: at λ = 9.38, c = 0.6779 on the first branch and 0.6670 on the second, a drop of 1.6 %, presumably from rounded constants. The model reproduces the jump.

**Choices made here** (the paper does not settle them; each is also in the notebook):

1. **Independent parameterization.** The paper samples S, L and Φ independently (Table II). But L does not enter Eqs. (1)–(6), and for a circular arch of radius R = S/Φ the span is L = 2R sin(Φ/2), fixed by S and Φ. The Table II range of L (9–10 m) is also impossible with those of S and Φ: the chord of an arc is shorter than the arc, and S ≤ 7.679 m; S and Φ imply L ∈ [6.13, 7.55] m. (Table II's S range equals 10 Φ, i.e. R = 10 m at the end points, which suggests a common radius was intended, but S and Φ are sampled independently.) We take the inputs as the seven quantities in the formulas, (S, Φ, E, D, B, t_w, t_f), independent and uniform on the Table II ranges; L is left out. Its Sobol' indices are then 0 by construction, as they would be for the paper's model too.
2. **Young's modulus.** Table II prints 18E5–22E5 MPa (1800–2200 GPa), ten times the modulus of steel. We use 1.8e5–2.2e5 MPa. The active subspace and Sobol' indices do not depend on this choice (the relative range is the same, and N_acr is proportional to E); the loads in kN do.
3. **Section properties.** The paper does not give A and I. We use a doubly symmetric welded I-section without root fillets: A = 2Bt_f + (D − 2t_f)t_w, I = [BD³ − (B − t_w)(D − 2t_f)³]/12.
4. **Validity range of Eq. (3).** Over the input box λ ∈ [8.06, 15.23] (corners). For the fixed arch the first branch is stated for λ > 9.87 only; we apply the same polynomial below 9.87. The paper does not say what it did there. The fixed arch never reaches its second branch (λ < 18.6).

**Kink and branch fractions** (notebook, check cell):

| case | below lower limit | first branch | second branch |
|---|---|---|---|
| fixed, 1000 data samples | 11.3 % (λ ≤ 9.87, extended polynomial) | 88.7 % | 0 |
| fixed, 10⁶ fresh samples | 12.2 % | 87.8 % | 0 |
| pin-ended, 1000 data samples | 0 (limit 3.88) | 4.1 % (λ < 9.38) | 95.9 % |
| pin-ended, 10⁶ fresh samples | 0 | 4.48 % | 95.52 % |

So the fixed-arch QoI is a single smooth formula over the whole box (one polynomial branch, partly applied outside its stated range), and the pin-ended QoI has one switch inside the box, with a 1.6 % downward jump that a gradient-based estimate does not see.

**Inputs** (Table II, p. 6478; all uniform; no nominal values given; E corrected as above):

| name | symbol | units | range | source |
|---|---|---|---|---|
| arch length | S | m | U[6.283, 7.679] | Table II, p. 6478 |
| included angle | Φ | rad | U[0.628, 0.768] | Table II, p. 6478 |
| Young's modulus | E | MPa | U[1.8e5, 2.2e5] (printed 18E5–22E5) | Table II, p. 6478 |
| section depth | D | m | U[0.235, 0.287] | Table II, p. 6478 |
| flange width | B | m | U[0.136, 0.166] | Table II, p. 6478 |
| web thickness | t_w | m | U[0.007, 0.008] | Table II, p. 6478 |
| flange thickness | t_f | m | U[0.011, 0.014] | Table II, p. 6478 |
| span (not used) | L | m | U[9.00, 10.00] | Table II, p. 6478 |

**Quantity of interest.** The in-plane elastic buckling load N_acr (paper Eq. 13), in kN (E in MPa, lengths in m), for the fixed arch (paper's Case 1, Fig. 5) and the pin-ended arch (Case 2, Fig. 7). The paper does not state units. The slenderness λ is stored too.

**Gradients.** Exact, by the chain rule through λ, A, I and N_cr (`buckling_load(..., grad=True)`), with respect to the physical inputs. The model is smooth on each branch; at the pin-ended switch λ = 9.38 the gradient is that of the branch the point is on.

**Sampling design.** Here: M = 1000 independent uniform samples, `numpy.random.default_rng(2020)`; the gradient estimator needs ≥ 10m = 70 and ≥ 300. In the paper: Monte Carlo estimation of Sobol' indices with 100,000 simulations per case in MATLAB (67.0 min fixed, 69.0 min pin-ended, pp. 6478–6479); no seed, no data or code published.

**Published results** (no active subspace; Sobol' indices only):

- Table III (fixed): first-order / total: S 0.2942 / 0.2819, E 0.0795 / 0.0690, D 0.6577 / 0.6510, B, t_w, t_f, Φ, L 0.0425 / – each. The text (p. 6479) quotes S, E, D as 24 %, 6 %, 53 % (first) and 28 %, 7 %, 65 % (total), and 3 % for the rest.
- Table IV (pin-ended): S 0.2337 / 0.2788, E 0.0472 / 0.0738, D 0.6611 / 0.6844, B 0.2337 / –, others –. The text quotes 26 %, 6 %, 60 % and 28 %, 7 %, 64 %, and 2 % for the rest.
- The tables are inconsistent (first-order above total for S, E and D in Table III; five equal values 0.0425; Table III first-order indices sum to 1.24; B in Table IV repeats S; text and tables disagree), so only the ranking D > S > E is usable.

## Regenerating

```bash
python generate_data.py                          # writes the four CSVs here
python generate_data.py --out /tmp/arch --check  # writes elsewhere and compares with the files here
```

The shipped files were written on 2026-10-30 in a Linux container (Python 3, numpy 2). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-10-31) the run takes 0.04 s for the computation (0.2 s with interpreter start-up), and `--check` prints maximum relative differences of 0 (inputs), 9.1e-16 (outputs), 1.4e-15 (fixed gradients) and 4.5e-15 (pin-ended gradients): rounding error. The notebook runs in about 12 s under nbconvert.

## Accuracy

- **Gradients vs. complex step** (scratch script, not committed; an independent re-implementation of the formulas in complex arithmetic, step 10⁻²⁰ of each input's range): maximum relative error of the gradient vector over the 1000 samples 7.4e-16 (fixed) and 1.2e-15 (pin-ended).
- **Gradients vs. central differences** (notebook check cell; step 10⁻⁶ of each input's range; samples within 10⁻⁴ of the pin-ended switch would be left out, none are): maximum relative error 6.3e-9 (fixed) and 6.0e-9 (pin-ended), median 1.5e-9 for both, the truncation and rounding error of the difference.
- **Outputs**: the notebook re-evaluates the model at the 1000 inputs and finds a relative difference from `outputs.csv` of 9.1e-16 (fixed) and 7.6e-16 (pin-ended), rounding error.
- **Formula reading**: continuity of the fixed-arch branches at λ = 18.6 (to 0.0006 in c) checks the square-root reading of Eq. (3); Eq. (2) has the 1.6 % jump noted above, which Pi et al. (2002) could settle.

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from its own MATLAB code (not published) on its own 100,000 random samples per case, with its own (unstated) section formulas and the 8-input design including L. The files here are an independent numpy implementation of the paper's Eqs. (1)–(3) and (6) with the four choices above.
- **What agrees.** The ranking of the two most influential inputs: D, then S, in both cases. The notebook computes Sobol' indices of this model (Saltelli et al. 2010 first-order and Jansen total estimators, 2¹⁷ base samples, 9 × 131,072 runs per case): fixed first-order D 0.3717, S 0.2709, E 0.1257, t_f 0.0941, B 0.0861, Φ 0.0380, t_w 0.0020; pin-ended D 0.3406, S 0.2314, E 0.1396, t_f 0.1013, B 0.0907, Φ 0.0773, t_w 0.0026.
- **What does not.** The paper gives D about twice our share (0.66 vs 0.34–0.37) and puts every minor input at 0.0425 or "–", where we find t_f and B (0.09–0.10) almost as influential as E (0.13–0.14). We cannot locate the cause: the paper's tables are internally inconsistent and its section formulas unstated.
- **Active subspaces** (notebook, gradient estimator, 500 bootstrap replicates): fixed arch λ₁/λ₂ = 401.7 (n = 1, leave-one-out R² of a quadratic in y₁ 0.99837); pin-ended arch λ₁/λ₂ = 30.9, λ₂/λ₃ = 53.1 (a two-dimensional subspace whose second direction follows the slenderness; quadratic in (y₁, y₂) LOO R² 0.99851). Not in the paper.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`); they are computed here, not copied from the paper, which publishes no data. Model and study design: T.-H. Nguyen (2020), after Y.-L. Pi, M. A. Bradford and B. Uy (2002). The paper states it was funded by Vinh University. Cite T.-H. Nguyen, "Global sensitivity analysis of in-plane elastic buckling of steel arches", Eng. Technol. Appl. Sci. Res. 10(5):6476–6480, 2020, doi:10.48084/etasr.3833.
