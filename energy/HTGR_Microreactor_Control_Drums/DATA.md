# HTGR_Microreactor_Control_Drums: data

Kind: **stored data**. The values come from the Serpent Monte Carlo runs of Price, Radaideh and Kochunas, "Multiobjective optimization of nuclear microreactor reactivity control system operation with swarm and evolutionary algorithms" (Nucl. Eng. Des. 393:111776, 2022). The authors posted them in the repository named in the paper's Data Availability section, [deanrp2/MicroControl](https://github.com/deanrp2/MicroControl) (whose only licence is MIT, a software licence that does not mention data; see License and credit). `convert_data.py` copied the file `pmdata/qpower_full.csv` into `microreactor_runs.csv`, losslessly. Nothing was rerun, refitted, rescaled or corrected, and no rows were dropped. The Serpent model itself is not public and cannot be run here (Serpent is export-controlled; the core model is not released, paper p. 5).

## Files

| file | contents | role |
|---|---|---|
| `HTGR_Microreactor_Control_Drums.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2022; outputs committed |
| `microreactor_runs.csv` | 751 × 37, one row per Serpent run (columns below) | stored data (source `pmdata/qpower_full.csv`) |
| `convert_data.py` | writes the CSV from the source file and checks it bit for bit (`--check`); with the pyMAISE file as a further argument, also checks that file against the source; source URLs and checksums | lossless conversion |
| `MicroControl_LICENSE.txt` | MIT licence of the source repository (copyright 2021 deanrp2) | notice kept with the hosted data file; licence of the eight angle offsets copied into the notebook |

`microreactor_runs.csv` has two `#` header lines (a description with the credit, and the column names with units), then plain comma-separated numbers, so `np.loadtxt(path, delimiter=',')` reads it. Columns, in the source's order:

| columns | name | units | meaning |
|---|---|---|---|
| 0 | `sample` | – | run number; the source label `sample_00012` is written as 12 |
| 1, 2 | `k`, `k_uncert` | –, relative sd | $k_\mathrm{eff}$ and its Serpent relative standard deviation |
| 3, 4 | `cpu_time`, `runtime` | not stated (presumably minutes) | CPU time and wall time of the run |
| 5–8 | `fluxQ1`–`fluxQ4` | n/cm²/s (pyMAISE docs) | neutron flux tallied in quadrants 1–4: the paper's $P_i$ |
| 9–12 | `fissQ1`–`fissQ4` | not stated | fission-rate tallies per quadrant |
| 13–16 | `fissEQ1`–`fissEQ4` | not stated | fission-energy tallies per quadrant (their sum is 1.1e7 in every run, relative sd 3e-4) |
| 17–28 | `flux_runcertQ*`, `fiss_runcertQ*`, `fissE_runcertQ*` | relative sd | Serpent relative standard deviations of the tallies |
| 29–36 | `theta1`–`theta8` | rad | drum orientations in the Serpent model's global convention, in [0, 2π) |

The numbers keep the source's text (k to six digits, fluxes to six significant digits, angles to nine). Row order is the source's.

**Angles.** The stored angles are not the paper's. The paper's local angle is 0° when "the absorbing material is fully facing the core", with $|\theta|$ the degree of withdrawal (p. 17), in [−180°, 180°]. The authors' function `g2l` (MicroControl `pmdata/utils.py`) converts: θ_local = s_i (θ_global − o_i) mod 2π, mapped to (−π, π], with offsets o = (3.6820187359906447, 4.067668586955522, 2.2155167202240653 − π, 2.6011665711889425 − π, 0.5404260824008517, 0.9260759333657285, 5.3571093738138575 − π, 5.742759224778734 − π) and signs s = (+, +, −, −, −, −, +, +). The notebook applies this once, in a stated cell; the constants are copied from the MIT-licensed source code (licence text beside the data). The offsets place the drums at azimuths 31°, 53°, 127°, 149°, 211°, 233°, 307°, 329° if the global angle is the direction the absorber faces, which matches Fig. 1 (two drums at the outer corner of each quadrant) and the README's numbering (quadrants counterclockwise from top right, drums counterclockwise from the lower drum of quadrant 1).

**The second copy (pyMAISE, Zenodo).** The pyMAISE benchmark data, [doi:10.5281/zenodo.20140559](https://doi.org/10.5281/zenodo.20140559) (CC BY 4.0), contain the same runs as `microreactor.csv` (MD5 `a548c9ddc384bcb08823a67f33076523`, SHA-256 `032d0227…f9401`, equal to the pyMAISE registry; obtained by the repository owner's browser on 2026-10-31). pyMAISE's loader `load_HTGR` says the data "was taken from" Price et al. (2022), without symmetry augmentation. `convert_data.py --check` shows, column by column, that it is the authors' file reordered, with the flux, fission-rate and time columns rounded half-up to three significant digits and the angles to nine; k, the fission-energy tallies and all uncertainties are identical; labels and row order are the same. Three digits quantize each flux in steps of about 0.4 % of its value, more than its Monte Carlo error (0.10–0.21 %), and leave only 27–29 distinct values per flux column, so the authors' full-precision file is the one shipped. The pyMAISE notebook also says the data were "built using MCNP simulations"; the paper (Sect. 2.3, p. 7) says Serpent.

**Source files.** `pmdata/qpower_full.csv` in https://github.com/deanrp2/MicroControl, commit `7a58e183b51238c9121ba690f2c930fd62308fa1` (2022-01-03, the repository's latest), cloned 2026-10-31; 279,609 bytes; SHA-256 `dda6ebc38c31c747de7b063d289a2815d66ec03d5e6eba8d6327d99816c7d003`. `pmdata/qpower_211018.csv` is byte-identical; `pmdata/qpower_small.csv` holds the first 284 runs (an earlier subset, not used).

## Model

Content of the paper card, checked against the paper. Page, section, equation, figure and table numbers refer to the accepted manuscript (OSTI, Elsevier user licence, 34 pages) of D. Price, M. I. Radaideh and B. Kochunas, Nucl. Eng. Des. 393:111776, 2022, doi:10.1016/j.nucengdes.2022.111776; numbers should be checked against the version of record.

- **Reactor** (Sect. 2.1, pp. 5–6, Fig. 1). HOLOS-Quad, a 22 MWt high-temperature gas-cooled microreactor of four transportable subcritical modules (the quadrants): TRISO fuel in hexagonal graphite blocks, helium cooling, BeO reflector of radius about 1 m. Eight cylindrical control drums (two per quadrant) carry B₄C on part of their surface and control reactivity by rotation; each can turn a full 360° (Sect. 2.2, p. 6).
- **Simulator** (Sects. 2.2–2.3, pp. 6–7). Serpent, continuous-energy Monte Carlo neutron transport, with the TRISO particles modelled explicitly; full-core model of the drum configuration. Serpent version, nuclear data library, neutron population, temperatures and burnup state are not given. The paper's Fig. 1 and Ref. [41] describe the design; "any additional information … is currently not available to the public" (p. 5).
- **Inputs** (m = 8, Sect. 2.3, p. 7):

| name | symbol | units | nominal | range or distribution | source |
|---|---|---|---|---|---|
| drum angles 1–8 | θ₁ … θ₈ | deg | 0 = absorber fully facing the core (p. 17) | independent uniform on [−180°, 180°] | p. 7 |

- **Quantities of interest.** The quadrant fluxes P₁–P₄ ("P_i indicates the flux in the i-th core quadrant", p. 7), modelled as "relative quadrant power" (p. 14), i.e. scaled to sum to 1 (the authors' `qpower_preprocess` does this); the quadrant power tilt ratio QPTR = 4 max(P_i)/ΣP_i (Eq. 1), with a limit of 1.02 (p. 7). $k_\mathrm{eff}$ is stored too; in the paper, reactivity comes from a separate hybrid drum-worth model (Sect. 2.2, Appendix A), not from these runs. The notebook analyses $k_\mathrm{eff}$ and $P_q = \phi_q/\sum_j\phi_j$.
- **Gradients:** none. The paper's hybrid drum-worth model is differentiable (Appendix A), but it is not the QoI here and is not public as code with its training data in this form.
- **Sampling design.** Independent uniform random angles (Monte Carlo), 751 runs with run numbers 0–818 (68 numbers absent; reason not given). The paper used "about 3000 samples after using reactor symmetries to multiply samples" (p. 7): 2404 training and 600 test rows (Table 1) = 4 × 751. Checks in the notebook: every drum covers [−180°, 180°] to within 2.2°, Kolmogorov–Smirnov p-values ≥ 0.32, largest pairwise correlation 0.069.
- **Published results.** No active subspace. Table 2 (test set, 3004-row augmented data): DNN MAE 4.62e-4, RMSE 6.05e-4, R² 0.98; random forest R² 0.75; decision tree 0.29; linear regression 0.00. Verification at two optima (Sect. 4.5, p. 25): QPTR 1.010 (Serpent) vs 1.011 (surrogate), k_eff errors 74 ± 13 and 54 ± 13 pcm, differential worth 31.85 ± 0.55 vs 29.24 pcm/°.

**Analysis choices in the notebook** (documented there): inputs u_i = cos θ_i (the angles are periodic, and in the angles themselves no active subspace exists for a function of Σc_i cos θ_i); input space `ac.Gaussian(0, 0.5 I)`, which has the mean and covariance of the arcsine law of u (QPHD uses only these two moments), the same scaling for every input. QPHD's C = ggᵀ + HΣH is the average of ∇q∇qᵀ for the fitted quadratic q, which for a quadratic depends on the input density only through its mean and covariance, so this declaration gives C for the actual (arcsine) sampling distribution. The space is a device for these two moments and must not be used to sample or integrate (its support is unbounded, while u ∈ [−1, 1]). Sensitivity, printed in the notebook: λ₁/λ₂ = 18.83 (k_eff) and 35.70 (P₁) as declared; 19.34 and 36.32 with the sample covariance of the 751 runs; 28.16 and 52.34 if u were declared uniform on [−1, 1], which is wrong for these data (variance 1/3 instead of 1/2, so the curvature term is underweighted and the gap inflated); the first directions agree to |w·w′| ≥ 0.99997 in every case. The estimator thus finds the active subspace of the best quadratic in u, i.e. of the part of f that depends on u only (even in every θᵢ); the part odd in θ (the turning direction) is not seen (see Accuracy). QPHD on the 751 raw runs with 300 bootstrap replicates; the symmetry augmentation to 3004 rows only as a point check, since augmented rows are not independent.

**Noise.** Serpent's relative standard deviations are stored per run: k_eff 0.00019–0.00027 (24 pcm on average, against a k_eff sd of 586 pcm across runs), flux tallies 0.100–0.214 %. They are taken as one standard deviation (Serpent's convention); a note in the authors' repository (`pmdata/majdi_notes.txt`) says "I think they are 2*the coefficient of variation", which would halve the noise. The notebook treats them as one standard deviation and the four quadrant tallies' errors as independent, which gives a P₁ noise sd of 3.17e-4; this is an **upper bound**, not an estimate. A 145-term second-order Fourier fit of P₁ reaches a leave-one-out RMSE of 2.77e-4, below the 3.51e-4 that independent 1-sd noise alone would give (with leverage) even for an exact fit (1.76e-4 if the uncertainties are 2 sd). The quadrant flux residuals of that fit are correlated at 0.98–0.99, so a common-mode error cancels in P_q: either the tally errors are correlated, or the uncertainties are 2 sd. For k_eff the same test is inconclusive (LOO RMSE 4.15e-4 against 2.68e-4 expected from noise), so model error remains. The notebook's noise floor (QPHD on synthetic noise with these standard deviations) is therefore conservative, and it lies more than 1000 times below λ₁ of both QoIs.

**Fission tallies.** The quadrant fission-energy fractions vary by about 1.8e-4 (sd), close to their own Monte Carlo error, are uncorrelated with the flux fractions (|r| ≤ 0.10), and a linear fit in (cos θ, sin θ) explains at most 3.8 % of their variance: they do not resolve the quadrant power and are not analysed. The fission-rate tallies are nearly constant as well.

## Regenerating

The data cannot be regenerated: the Serpent model is not public, and Serpent 2 is distributed under export control (see the EMA notes kept outside the repository). The CSV is rebuilt from the source file with

```
python convert_data.py <path>/qpower_full.csv                         # writes microreactor_runs.csv (< 1 s)
python convert_data.py <path>/qpower_full.csv --check <path>/microreactor.csv   # bit-for-bit check, and the Zenodo copy
```

The cost of the original runs, from the stored times (units not stated; if minutes, as in Serpent's result files): wall time 92–200 per run (median 129), CPU time about 19.9 times the wall time, i.e. about two hours on about 20 threads per run. The notebook runs in 5–9 s (in-notebook timer 4.9–8.3 s over three runs on 2026-10-31; 8.7 s wall for `nbconvert`).

## Accuracy

There are no gradients to check. What can be checked:
- The conversion is lossless (`--check` compares the written file with a fresh conversion, byte for byte) and the Zenodo copy is a rounding of the source (`--check` with its path).
- The authors' symmetry images (`mult_sym`), mapped to local angles, are permutations of the drums (mirror `h`) or permutations with all signs reversed (`r`, `v`) to 2e-13 degrees, so in u = cos θ they are exact permutations (notebook).
- Monte Carlo noise is below the structure found: noise-only λ₁ ≤ 9.4e-9 for k_eff (λ₁ = 3.5e-5) and ≤ 1.4e-8 for P₁ (λ₁ = 1.7e-5), with noise levels that are upper bounds (Noise, above). The residuals of the surrogates are larger than the noise (k_eff 4-variable quadratic LOO RMSE 74.8 pcm vs 24.1 pcm; P₁ 2-variable quadratic 8.00e-4, at least 2.9 times the noise, whose sd is bounded by the best fit's LOO RMSE of 2.77e-4), so they measure model structure not captured, chiefly the turning direction, not noise.
- The cost of analysing u = cos θ (dropping the sign of θ, i.e. the turning direction), from a full second-order Fourier basis (45 terms in u only + 100 odd terms): the odd terms raise the leave-one-out R² by 0.0116 for k_eff and by 0.0312 for P₁ (0.0307–0.0335 for P₂–P₄); in the full fit the odd part carries 1.2 % of the variance of k_eff and about 3 % of the quadrant powers' (0.0306 for P₁, 0.0298–0.0328 for P₂–P₄).

## How these files relate to the paper

The 751 rows are the Serpent runs behind the paper's quadrant-power surrogate (Sect. 2.3, Table 1): the paper's 3004 rows are these runs and their three symmetry images. The authors' preprocessing (`qpower_preprocess`) applies the images, keeps the flux columns, scales them to sum to 1 and converts the angles to local coordinates; the notebook does the same in numpy (augmentation only as a check). The paper does not report k_eff from these runs, the Monte Carlo uncertainties, the missing run numbers, or the fission tallies. A September 2021 note in the repository reports a linear-regression R² of 0.947 for an earlier model, against Table 2's 0.00; the notebook gets −0.015 (LOO) for a linear fit in the angles and 0.94 for one in cos θ, which suggests the two used different input coordinates.

## Open questions for the authors

1. The rotation convention: the README says positive rotation is counterclockwise in quadrants 1 and 3 and clockwise in 2 and 4, while `g2l` reverses the angles of drums 3–6 (quadrants 2 and 3). Which is right for the paper's figures (e.g. Fig. 5)? The notebook's results use cos θ and do not depend on it.
2. Are the stored relative uncertainties one standard deviation (Serpent's convention) or two (the repository note)?
3. Why are 68 run numbers (of 0–818) absent: failed runs, or runs removed by a rule?
4. The fission-rate and fission-energy tallies do not respond to the drums. Were their detectors restricted to the quadrants?
5. Units of `cpu_time` and `runtime`, the Serpent version, nuclear data, neutron population, and the reactor state (temperatures, burnup) of the runs.
6. Are the four quadrant flux tallies' Monte Carlo errors correlated (e.g. through a shared normalization)? Their residuals correlate at 0.98–0.99, and the lack of fit for P₁ is already below the independent-error noise level.
7. Is the MIT licence of MicroControl meant to cover the data files in `pmdata/`?

## License and credit

The data are the work of D. Price, M. I. Radaideh and B. Kochunas. Their repository's only licence is MIT (copyright 2021 deanrp2; text in `MicroControl_LICENSE.txt`), which grants rights to "this software and associated documentation files" and says nothing about data; whether it covers `pmdata/qpower_full.csv` is therefore not stated (open question 7). The file is hosted here with credit, under the repository author's decision of 2026-10-30 on outside data without a stated licence, and with the MIT notice kept beside it. The MIT licence does cover the eight angle offsets copied from the authors' code into the notebook. The same runs, with the flux, fission-rate and time columns rounded to three significant digits, are also published in the pyMAISE benchmark data on Zenodo under CC BY 4.0 (record metadata, read 2026-10-31: licence `cc-by-4.0`, title "pyMAISE Benchmark Data"). Cite the article: D. Price, M. I. Radaideh and B. Kochunas, Nucl. Eng. Des. 393:111776, 2022, doi:10.1016/j.nucengdes.2022.111776. Cite the pyMAISE copy as doi:10.5281/zenodo.20140559 (P. A. Myers et al., Prog. Nucl. Energy 180:105568, 2025, doi:10.1016/j.pnucene.2024.105568). The notebook and `convert_data.py` are MIT (code) and the notebook's text and figures CC BY 4.0, like the rest of as-data-sets. The paper acknowledges an Integrated University Program Graduate Fellowship and the DOE Office of Nuclear Energy's Nuclear Energy University Program, contract DE-NE0008887.
