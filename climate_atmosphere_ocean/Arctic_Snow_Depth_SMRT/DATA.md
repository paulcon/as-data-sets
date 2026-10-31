# Arctic_Snow_Depth_SMRT: data

Kind: **stored data, generated with SMRT 1.5.1; regeneration script included**. The data come from an outside code,
SMRT (Snow Microwave Radiative Transfer, LGPL-3.0), which is not a numpy/scipy model, so this data set
follows the outside-tool path of the build rules: the tool was installed in a separate environment, the
data were generated and shipped, and the provenance is recorded below. Unlike most stored data sets, the
files can be regenerated: `snow_smrt_model.py` sets up the SMRT model and `generate_data.py` draws the
inputs with a saved seed and writes all four data files. The notebook reads the files and needs only
numpy, matplotlib and active_subspaces. No SMRT code is in this directory.

**Source version.** The published paper, The Cryosphere 20(11):6157–6180, 30 October 2026
(doi:10.5194/tc-20-6157-2026, CC BY 4.0), and its Supplement (doi:10.5194/tc-20-6157-2026-supplement).
Page, table and figure numbers below are those of the published paper. An earlier build used the
EGUsphere preprint (doi:10.5194/egusphere-2026-1680); Tables 1 and 4 and the numbers printed in Fig. 3a
are the same in both versions.

## Files

| file | contents | role |
|---|---|---|
| `Arctic_Snow_Depth_SMRT.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 424; outputs committed |
| `snow_smrt_model.py` | `SnowOnFYI` (SMRT model: X (N, 5) → TB (N, 12)), `gr`, `fd_stencil`, `fd_gradient`; `NAMES`, `UNITS`, `LOWER`, `UPPER`, `CHANNELS`, `DENSE_THRESHOLD`, `CONFIG` | the model set-up; imports SMRT only when `SnowOnFYI` is built |
| `generate_data.py` | draws the inputs and writes the four CSVs | regenerates the data (needs SMRT 1.5.1) |
| `inputs.csv` | 400 × 5 physical inputs: depth (m), density (kg m⁻³), temperature (K), radius (mm), stickiness (–) | model input |
| `outputs.csv` | 400 × 12: TB (K) at 6V, 6H, 10V, 10H, 18V, 18H, 23V, 23H, 36V, 36H, 89V, 89H | model output |
| `gradients.csv` | 400 × 60: column 5 i + j is ∂TB_i/∂x_j in physical units (K per unit of input j) | model output (finite differences) |
| `fd_jumps.csv` | 400 × 12 integers: bit j of column i set if the central stencil of input j straddled a jump in channel i (DORT stream count or dense-snow switch) and a one-sided slope was used | gradient flag |

The CSVs have two `#` lines (a description and the column names with units), so
`np.loadtxt(path, delimiter=',')` reads them. Channel names use the paper's short labels (6 = 6.925 GHz,
10 = 10.65, 18 = 18.7, 23 = 23.8, 36 = 36.5, 89 = 89.0 GHz).

## Model

**Citation.** Z. Yan, Y. Ye, G. Heygster, B. Cheng, C. Gabarró, F. Hernández-Macià and X. Cheng,
"Assessment of snow depth retrievability from passive microwave observations over Arctic sea ice: a global
sensitivity analysis", The Cryosphere 20(11):6157–6180, 2026, doi:10.5194/tc-20-6157-2026 (CC BY 4.0;
checked against Crossref). Model: SMRT, G. Picard, M. Sandells and H. Löwe, Geosci. Model Dev.
11(7):2763–2788, 2018, doi:10.5194/gmd-11-2763-2018, <https://github.com/smrt-model/smrt>.

**What the paper does.** SMRT simulations of AMSR2 brightness temperatures for single- and multi-layer, dry
and wet snow on Arctic sea ice, ranked by EFAST (Saltelli et al. 1999, via SALib) first-order (SSI) and total
(TSI) indices, then an averaging-kernel "retrievable range" analysis for snow depth. This data set is the
**single-layer dry snow** case (m = 5), the spreadsheet's target; the paper's multi-layer cases (m = 14 dry,
m = 8 wet) and the wet single-layer case are not generated.

**Configuration from the paper and its Supplement.** Sticky hard sphere (SHS) microstructure with the
improved Born approximation (IBA), Sect. 2.1 (p. 6159); upwelling TB at the snow–air interface for the 12
AMSR2 channels at 55° (Sect. 2.1, p. 6159; Sect. 3.1, p. 6161); dry snow on first-year ice (FYI) of two
layers, Table 4 (p. 6166): 0.5 m and 1.0 m thick, density 910 kg m⁻³, 265 K, scatterer radius 0.1 mm,
brine volume fraction 5 %; all interfaces smooth (Sect. 4.3, p. 6175). Supplement Table S1 ("SMRT model
setups"): SMRT version 1.5.1; "Improved born approximation (with dense snow correction)"; sticky hard
spheres; "Default flat/Fresnel interfaces were used; A water substrate was added below the sea ice";
the Discrete Ordinate Radiative Transfer solver (DORT); and five permittivity entries
(`ice_permittivity_matzler87`, `ice_permittivity_matzler06`, `brine_permittivity_stogryn85`,
`ice_permittivity_matzler06`, "polder van staten") that reproduce the defaults named in SMRT 1.5.1's
docstrings, so SMRT's default permittivity models are used here (for dry snow SMRT 1.5.1's code uses
`ice_permittivity_maetzler06`, although the docstring names matzler87). Table S1 does not say whether it applies to every sensitivity run; its caption
is generic and Sect. S4 keeps "the same SMRT configurations", so it is taken as the baseline set-up here.

**Configuration chosen here** (still not stated; `snow_smrt_model.CONFIG`, all listed in `unknowns`):

| setting | value used | why |
|---|---|---|
| DORT stream settings | SMRT 1.5.1 defaults (`n_max_stream = 32`, `m_max = 2`, `stream_mode = "most_refringent"`) | Table S1 names DORT but no settings |
| dense-snow correction mode | `dense_snow_correction="auto"` (invert the medium for ice fraction > 0.5) | the only mode SMRT 1.5.1 implements |
| snow salinity | 0 | dry winter snow above the basal 10 cm is fresh (Sect. 3.3, p. 6162; Fig. 2f, p. 6164) |
| ice microstructure | sticky hard spheres, radius 0.1 mm, stickiness 0.3, spherical brine inclusions | Table 4 gives a radius; τ = 0.3 is the paper's value for superimposed ice (Table 3, p. 6165) |
| ice density | 910 kg m⁻³ passed, unused by SMRT for first-year ice | Table 4 |
| water substrate properties | SMRT's default sea water below first-year ice (`add_water_substrate=True`, 271.15 K) | Table S1 says only "a water substrate" |
| atmosphere | none | paper: TB at the snow–air interface (the Supplement adds an atmosphere only for its comparison with AMSR2, Sect. S2) |
| GR polarization | V | paper and Supplement do not say; Markus and Cavalieri (1998) use V |

**Inputs** (independent uniform; the paper states that its EFAST sampling assumes uniform distributions on
the ranges, Sect. 3.3, p. 6162):

| name | symbol | units | range | nominal (EMA) | source |
|---|---|---|---|---|---|
| depth | d | m | [0, 0.6] | 0.3 (midpoint) | Table 1, p. 6163 |
| density | ρ | kg m⁻³ | [150, 500] | 325 | Table 1, p. 6163 |
| temperature | T | K | [240, 270] | 255 | Table 1, p. 6163 |
| radius | r | mm | [0.1, 0.4] | 0.25 | Table 1, p. 6163 |
| stickiness | τ | – | [0.1, 0.3] | 0.2 | Table 1, p. 6163 |

The paper gives no nominal point; the midpoints are used where one is needed (EMA). SMRT's lower limit on
stickiness, τ_min(ρ/916.7), is 0.096 at 150 kg m⁻³ and negative above 409 kg m⁻³ (and for the inverted
medium above the dense-snow threshold), so every point of the box, and every stencil point, is valid. The
Supplement (Sect. S3, Table S3) repeats the EFAST analysis with bounded Gaussian inputs fitted to MOSAiC and
finds the same leading parameter in each channel; that variant is not generated here.

**Quantities of interest.** All 12 TBs are stored. The notebook's headline QoI is TB at 36.5 GHz V; the
second is GR(36/18) = (TB36V − TB18V)/(TB36V + TB18V) (paper Eq. 12, p. 6160), formed in the notebook from
the stored TBs, with its gradient by the chain rule from the stored TB gradients.

**Sampling design.** Plain Monte Carlo: N = 400 points from `numpy.random.default_rng(424).random((400, 5))`
mapped to the box. N meets the brief's minimum for gradients (≥ 10 m = 50 and ≥ 300). The paper used EFAST
(interference factor M = 4, Eq. 2, p. 6159) with an unreported sample size and no seed.

**Gradients.** Finite differences of SMRT, which is not differentiable here: step h = 1e-4 of each input's
half-range (δ = 3e-5 m, 0.0175 kg m⁻³, 1.5e-3 K, 1.5e-5 mm, 1e-5), central where x ± δ is in the box and
second-order one-sided otherwise (no shipped sample needed a one-sided stencil). 11 SMRT runs per sample.
Stencils that straddle a jump in TB (a DORT stream-count change or the dense-snow switch; see Accuracy) are
detected by the disagreement of the two one-sided slopes (|g⁺ − g⁻| > 0.05 max(|g⁺|, |g⁻|) + 0.5 K per unit
normalized input), and the one-sided slope of smaller magnitude is used: this happened in 2 of the 4800
(sample, channel) entries, sample 175 (0-based), the density derivative at 89V and 89H (a stream-count
jump). No entry in 18V or 36V is flagged.

**The dense-snow step and the gradients.** TB is discontinuous at density 458.35 kg m⁻³ (the jump at 36V
is −6.83 K at the midpoint of the other inputs; see Accuracy). **No finite-difference stencil straddles
it:** the nearest sample (321, 0-based) is 0.246 kg m⁻³ above the threshold, 14 times the density step
δ = 0.0175 kg m⁻³; no sample is within 10 steps and 6 are within 100 steps (printed by the notebook; the
next nearest are samples 95 and 172, 0.477 and 0.699 kg m⁻³ below). The expected number of straddling
stencils in 400 uniform draws is 400 × 2δ/350 = 0.04. Every stored gradient is therefore a one-sided,
pathwise derivative: the derivative of the smooth branch on the sample's own side of the threshold. The
notebook's matrix C is built from these gradients on each side, so it does not capture the jump itself (whose
derivative is a delta function on the plane ρ = 458.35 kg m⁻³; smoothed over any width it would dominate C
along density); the notebook prints how C would change for ramp widths from 87.5 down to 0.0175 kg m⁻³. If a
future design put a stencil across the threshold, the jump detector would flag it (the re-review checked this
at a synthetic point 0.005 kg m⁻³ above the threshold: flagged in all 12 channels, and the kept slope matched
sample 321's stored gradient).

**Samples outside physical meaning.** None in the inputs (all ranges are physical). 41 of the 400 samples
have density above 458.35 kg m⁻³ (ice fraction above 0.5); there the dense-snow correction treats the snow
as air bubbles in ice (Picard, Löwe and Mätzler 2022, The Cryosphere 16:3861–3866,
doi:10.5194/tc-16-3861-2022), as the paper did. They are kept and counted in the notebook. 9 samples have
d < 1 cm and 19 have d < 3 cm (see the thin-layer note under Accuracy). All TBs are finite and in
[91.7, 265.1] K. The smallest depth is 6.8e-5 m.

**Published results.** No active subspace. EFAST indices in Fig. 3a (p. 6166), printed for the top parameter
of each channel (SSI / TSI), identical in the preprint: 6V density 0.98 / 0.99, 6H density 0.98 / 0.99, 10V
density 0.85 / 0.90, 10H density 0.97 / 0.99, 18V density 0.25 / 0.54, 18H density 0.74 / 0.84, 23V radius
0.19 / 0.48, 23H density 0.40 / 0.60, 36V radius 0.27 / 0.51, 36H radius 0.44 / 0.54, 89V radius 0.41 / 0.48,
89H radius 0.49 / 0.56. The text gives radius TSI "ranging from 0.48 to 0.56" at 36 and 89 GHz (p. 6165).
The notebook compares these with Sobol' indices of a surrogate fitted to the shipped data (degree-5 Legendre
expansion plus a step at the dense-snow threshold): the top parameter agrees in all 12 channels and 22 of
the 24 printed indices agree to within 0.1.

## Regenerating

```bash
# a separate venv for SMRT (the notebook's venv does not need it)
python3.13 -m venv svenv && svenv/bin/pip install smrt==1.5.1
svenv/bin/python generate_data.py                                         # 400 samples, 2 processes
svenv/bin/python generate_data.py --rows 173:178 --out /tmp/snow --check  # a subset, compared with the files here
```

`generate_data.py` stops if the installed SMRT is not 1.5.1 (override with `--any-smrt`).

**Provenance of the shipped files.** Generated 2026-10-31, 19:40:06–19:49:39 UTC, 573 s wall time on two
processes of a shared 2-core Linux x86-64 container (about 0.2 s per SMRT run of 12 channels on one core).
Python 3.13.18; smrt 1.5.1 from PyPI (wheel `smrt-1.5.1-py3-none-any.whl`, sha256
53d29ffbd6c23001775712e320298190e377f840f84e0b872d90249465fbac64; PyPI release, so no git commit is
recorded), numpy 2.5.4, scipy 1.18.1, xarray 2026.9.0, pandas 3.1.0, numba 0.68.1, llvmlite 0.50.0, joblib
1.6.0. SMRT 1.5.1 emits an xarray `FutureWarning` about the default `join` of `xr.concat` in
`smrt/core/result.py`; it concerns SMRT's auxiliary result data, not the TBs. Seed 424 (inputs only; SMRT and
the differences are deterministic). `--rows 173:178 --check` after the run (it covers sample 175, the only
one with jump flags) reproduced all four files bit for bit (max relative difference 0).

**Effect of the switch from the earlier build** (SMRT 1.7.3, plain IBA; same inputs): below the dense-snow
threshold TB changed by at most 0.3 K up to 36 GHz and 1.3 K at 89 GHz (SMRT version), and the normalized
gradients by a median 0.2 % (at most 8 %); above it TB changed by up to 31 K at 36 GHz and 70 K at 89 GHz
(dense-snow correction). Another SMRT version, or other numerical libraries, may change TB in the last digits,
by tenths of a kelvin (as 1.5.1 against 1.7.3), or, through the stream-count jumps, by up to about 0.1 K near
a jump.

## Accuracy

- **Step study** (scratch `step_study.py`, rerun on the shipped data: 8 samples, 4 drawn with seed 2026 plus
  samples 10, 71 and 339, two dense samples and the smallest depth; normalized gradients for all 12 channels
  compared with the stored h = 1e-4 values). With h = 3e-5 the per-channel relative difference is at most
  1.3e-4 (sample 71, 36H; 8.4e-5 at 36V), median over the samples 8.2e-7; at 18V at most 1.4e-8. With
  h = 3e-4 it is at most 1.2e-3 (again sample 71), median 1.1e-5. At sample 71 the density derivative has
  strong, smooth curvature at 36 GHz (the difference grows linearly with h), below the jump detector's
  threshold; it is immaterial for the subspace. At the smallest-depth sample (d = 6.8e-5 m) h = 3e-5 agrees
  to 6.4e-7. The stored gradients are therefore accurate to about 1e-4 relative or better per channel (often
  1e-6), and the TBs reproduce to 5e-10 K.
- **Dense-snow step.** With the correction, SMRT models the snow as ice spheres in air below 458.35 kg m⁻³
  and as air spheres in ice above it, and TB jumps at the switch. Scratch `refvals.py` and (re-review fix
  round) `smrt_jump.py`, TB(458.36) − TB(458.34 kg m⁻³): at the midpoint of the other inputs −0.46 K (18V),
  **−6.83 K (36V)**, −41.7 K (89V); at (d, T, r, τ) = (0.6 m, 240 K, 0.4 mm, 0.1) −7.80 K (18V), −54.03 K
  (36V), −56.34 K (89V). Over 40 random points of the threshold face (`default_rng(7)`): 36V mean −8.56 K,
  rms 12.85 K, range −46.23 to −0.17 K; 18V mean −0.67 K, rms 1.13 K; 89V mean −36.94 K, rms 40.72 K.
  Below 18 GHz the step is under 1 K. The step is part of the paper's configuration and lies inside the
  paper's range, so it is kept. It contributes to the variance (and the paper's EFAST indices) but not to the
  gradients (see "The dense-snow step and the gradients" above). The notebook estimates the 36V jump from the
  stored data alone, by the step term of its Sobol' surrogate: −7.50 K at the midpoint, −59.55 K at the
  corner, face mean −10.06 K and rms 14.20 K; at the 40 face points above the estimate correlates 0.990 with
  the SMRT jumps (rms difference 1.7 K; scratch `pce_jump.py`).
- **Effect of the samples above the step on the subspace** (notebook). TB36V, all 400 samples: λ1/λ2 = 7.54,
  λ2/λ3 = 1.41, w1 = (−0.339, +0.379, +0.066, −0.533, +0.673). The 359 samples below the step: λ1/λ2 = 7.53,
  λ2/λ3 = 2.14, |w1·w1(all)| = 0.9995. So the 41 samples above the step change λ2/λ3 (1.41 → 2.14) but not
  w1; the bootstrap intervals of λ2/λ3 overlap ([1.08, 2.18] and [1.19, 3.48]). Above the step the medium is
  inverted and the mean normalized density derivative of TB36V changes sign (+9.1 K below, −6.1 K above), so
  averaging the two branches fills in λ2 and λ3. The first build (plain IBA, no step) had λ2/λ3 = 2.15.
- **Stream-count jumps.** DORT takes 32 Gauss–Legendre streams in the most refringent layer (the ice) and
  carries them into the snow by Snell's law; the number of streams that are not totally reflected changes
  in steps with the snow's refractive index, so TB is piecewise smooth with small jumps across surfaces in
  (density, temperature). Example from the earlier build (SMRT 1.7.3): at depth 0.558 m, density
  236.06 kg m⁻³, 249.3 K, r 0.217 mm, τ 0.154, TB18V drops by 0.114 K between densities 236.06 and
  236.11 kg m⁻³ with 32 streams; with 64 streams there is no jump in that interval. The detector handles the
  gradients; the jumps remain in the TB values (about 0.1 K, against channel standard deviations of
  0.57 K at 6V to 41.1 K at 89V in the shipped data).
- **Stream count (discretization)**, from the earlier build (SMRT 1.7.3, plain IBA, samples 0–39 with 64
  streams against 32): TB differs by a median of 0.27–0.41 K per channel, at most 0.36–0.61 K below 36 GHz
  and 3.2–4.6 K at 36 and 89 GHz; gradients differ by a median of 0.6–6.5 %. The active subspace did not
  move (TB36V λ1/λ2 = 14.15 with either stream count on those 40 samples, |w1(32)·w1(64)| = 1.0000). This
  was not repeated with SMRT 1.5.1; the DORT defaults are the same in both versions, its code is not.
- **Zero depth.** SMRT drops a zero-thickness layer (bare ice), and TB is discontinuous at d = 0: at the
  midpoint of the other inputs (scratch `refvals.py`), 6H is 197.58 K for d = 0 and 228.47 K for
  d = 1e-7 m; 18H 198.34 and 228.95 K; 36V 262.42 and 260.92 K. A snow layer of any thickness adds two flat
  interfaces (air–snow, snow–ice) combined incoherently. The shipped samples and stencils have d > 0, so they
  describe d → 0⁺; the paper does not say how its EFAST samples near d = 0 were treated.
- **Thin layers.** Because the layers are combined incoherently, TB for d from 1e-7 to 1e-3 m is the same as
  for d = 1e-7 m within 0.02 K in every channel below 89 GHz (0.46 K at 89V), so the +30 K H-pol offset
  over bare ice persists for any thin layer. This is unphysical for layers thinner than about a wavelength
  (centimetres), but it is what SMRT computes in the paper's configuration, so the data are consistent with
  the paper's set-up. 9 of the 400 samples have d < 1 cm and 19 have d < 3 cm.

The notebook does not rerun SMRT; `generate_data.py --check` is the reproduction test.

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from its own SMRT 1.5.1 runs on EFAST samples
  (size not stated); no input/output data were deposited. These files are an independent SMRT 1.5.1
  computation of the same single-layer dry-snow case on the same ranges, with the paper's stated
  configuration and the remaining choices tabulated above.
- **Agreement.** The Sobol' indices estimated from these data (notebook) give the same leading parameter as
  Fig. 3a in all 12 channels, and 22 of the 24 printed SSI/TSI values agree to within 0.1; the two
  exceptions are first-order indices 0.12 apart, density at 23H (0.40 against 0.52) and radius at 36H (0.44
  against 0.32).
- **Not covered.** The wet-snow and multi-layer cases (Tables 2–3, Figs. 3b, 4–5), the multiyear-ice runs,
  the Gaussian-input and exponential-microstructure variants of the Supplement, and the averaging-kernel
  retrievability analysis (Sect. 4.2).

## License and credit

Code (`snow_smrt_model.py`, `generate_data.py`, the notebook): MIT (`LICENSE`). Data (the CSVs): CC BY 4.0
(`LICENSE-DATA`). The data were generated with SMRT (G. Picard, M. Sandells, H. Löwe and contributors),
which is licensed under the GNU LGPL v3; SMRT is used as an installed library and no SMRT code is included
here. The model configuration and parameter ranges are from Yan et al. (2026, CC BY 4.0). Cite Z. Yan et al.,
The Cryosphere 20:6157–6180, 2026, doi:10.5194/tc-20-6157-2026, and G. Picard, M. Sandells and H. Löwe, Geosci.
Model Dev. 11:2763–2788, 2018, doi:10.5194/gmd-11-2763-2018.
