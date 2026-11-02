# Kaesbach_Catchment_Subsurface_Flow: data

**Kind: stored data.** The runs were made by Erdal and Cirpka with HydroGeoSphere, a commercial code (Aquanty Inc.), on a 20-node cluster (about 96,000 CPU hours for 10,000 runs, paper p. 3793). The model files are not published and the model cannot be rerun here. The files in this directory are a lossless, scripted conversion of the data in the article's supplement; no value was regenerated or corrected.

## Files

| File | Contents |
|---|---|
| `Kaesbach_Catchment_Subsurface_Flow.ipynb` | The analysis: behavioral filter, active subspaces of gauge-C discharge and net boundary outflow, checks against the paper's Figs. 4, 6, 8 and 9 |
| `inputs_unit.csv` | 10,050 × 32: the inputs exactly as stored by the authors, coordinate u in [0, 1] (variable `Rorg`) |
| `inputs.csv` | 10,050 × 32: the same inputs in physical units, computed by `kaesbach_inputs.to_physical` (map below) |
| `outputs.csv` | 10,050 × 9: the outputs exactly as stored by the authors (variable `obsMTRX`); row i matches row i of the input files |
| `kaesbach_inputs.py` | Names, units, Table 1 bounds, the u ↔ physical map, the analysis coordinates, and the behavioral rule (numpy only) |
| `extract_data.py` | Writes the three CSV files from the supplement's `.mat` file; `--check` compares them with it |

Each CSV has two `#` header lines (description; column names with units) and reads with `np.loadtxt(path, delimiter=",")`. Values are written with 17 significant digits.

Columns of `outputs.csv` (order of `obsMTRX`; identified from the supplement's app code, `activeSubspacePilot_v3.mlapp`, and checked against the paper's figures):

| Column | Units | Description |
|---|---|---|
| `flooding` | m³/s | outflow through the top drain outside the streams (0.2 m above ground) |
| `discharge_gauge_C` | m³/s | stream discharge at gauge C |
| `flow_stream_A` | m³/s | flow in stream A |
| `flow_stream_B` | m³/s | flow in stream B |
| `ratio_stream_to_recharge` | – | total stream discharge / total recharge |
| `net_boundary_outflow` | m³/s | net flow out across the bottom and side subsurface boundaries |
| `residence_time_ku`, `_km1`, `_km1w` | stored unit, not stated (yr = value / 17,520; probably half-hours) | mean particle residence time (199 particles) in the Lettenkeuper, Gipskeuper and weathered Gipskeuper |

Zero flows are stored as −0.0 (MATLAB sign convention; 393 flooding, 24 gauge C, 1,089 stream A, 1,062 stream B and 4 ratio values) and are kept as stored. One run has a negative net boundary outflow (−2.1e-3 m³/s, a net inflow); it fails the behavioral rule. **Residence-time unit.** The authors do not state it, and the header of `outputs.csv` says so (`[stored unit; yr = value/17520]`); the stored values are unchanged. The paper's Fig. 8 (p. 3799, in years) equals the stored values divided by **17,520**. Evidence: the colour bars of Fig. 8 span the min–max of the plotted behavioral runs (as those of Fig. 6 do: gauge C 0.0040–0.0328 m³/s against a bar of about 0.004–0.033), and stored/17,520 gives 0.24–79.1 yr (ku), 0.51–303.6 yr (km1) and 2.13–37.6 yr (km1-w) against bars of about 0.3–79.5, 0.5–305 and 2.2–37.7 yr read from the figure; stored/8,760 (hours) would give tops of 158.3, 607.2 and 75.2 yr, twice the bars. 17,520 is the number of half-hours in a 365-day year, so the stored unit is probably half-hours, but this is an inference. The eight Fig. 8 activity scores do not distinguish the two readings: Fig. 8 used inputs on [0, 1] (see Accuracy), which doubles √a, and dividing by 17,520 rather than 8,760 halves it, so [0, 1] with stored/17,520 and [−1, 1] with stored/8,760 give the same scores (within 3.5 yr of the figure, most within 1 yr). The notebook's median residence times on the paper's scale are 12.8 yr (ku), 48.2 yr (km1) and 10.9 yr (km1-w). The constant is `kaesbach_inputs.RESIDENCE_PER_YEAR`.

## Model

**Citation.** D. Erdal and O. A. Cirpka, Global sensitivity analysis and adaptive stochastic sampling of a subsurface-flow model using active subspaces, *Hydrology and Earth System Sciences* 23(9):3787–3805, 2019, https://doi.org/10.5194/hess-23-3787-2019 (CC BY 4.0). Supplement: https://doi.org/10.5194/hess-23-3787-2019-supplement.

**Model.** Steady-state, three-dimensional, variably saturated integrated subsurface/surface flow (Richards equation with Mualem–van Genuchten functions, Eqs. 1–3, p. 3789) in HydroGeoSphere, for a virtual catchment drawn from the Käsbach (Ammer valley, SW Germany): about 4 km × 6 km, 1,001,760 prism elements, 523,083 nodes, five homogeneous geological units (mo Upper Muschelkalk, ku Lettenkeuper, km1 Gipskeuper, km1-w its weathering zone, Q Quaternary fill) and a north–south fault; run transient with constant forcing to steady state (10¹⁰ s). Boundaries: Dirichlet bottom head from a regional model (no head higher than 5 m below the model top), stream drains, a flooding drain 0.2 m above ground, a Cauchy boundary on the southern wall, recharge by land use on a top surface 1 m below ground (pp. 3792–3793). Driven from MATLAB.

**Inputs (m = 32).** All uniform within the bounds of Table 1, p. 3794 (Sect. 3.3, p. 3793). Stored as u ∈ [0, 1]; the map from u to physical units:

| # | Name | Units | Lower | Upper | Scale | Stored u | Map status | Source |
|---|---|---|---|---|---|---|---|---|
| 0 | bottom_head_offset | m | −5 | 5 | linear | u | verified (Fig. 3) | Table 1 |
| 1 | fault_offset | m | 0 | 100 | linear | u | verified (Fig. 3) | Table 1 |
| 2 | contact_offset_ku_km1 | m | −20 | 20 | linear | u | assumed | Table 1 |
| 3 | km1w_thickness | m | 5 | 50 | linear | u | verified (Fig. 3) | Table 1 |
| 4 | K_km1w | m/s | 1e-7 | 5e-5 | log | 1 − u | verified (Fig. 3) | Table 1 |
| 5 | aniso_km1w | – | 1 | 50 | linear | u | sign checked | Table 1 |
| 6 | K_ku | m/s | 1e-8 | 1e-5 | log | 1 − u | verified (Fig. 3) | Table 1 |
| 7 | aniso_ku | – | 1 | 50 | linear | u | assumed | Table 1 |
| 8 | K_km1 | m/s | 1e-9 | 1e-7 | log | 1 − u | verified (Fig. 3) | Table 1 |
| 9 | aniso_km1 | – | 1 | 50 | linear | u | assumed | Table 1 |
| 10 | K_mo | m/s | 1e-7 | 1e-5 | log | 1 − u | verified (Fig. 3) | Table 1 |
| 11 | K_Q | m/s | 1e-7 | 1e-5 | log | 1 − u | verified (Fig. 3) | Table 1 |
| 12–16 | alpha_km1w, _ku, _km1, _mo, _Q | 1/m | 0.5 | 5 | linear | u | assumed | Table 1 |
| 17–21 | n_km1w, _ku, _km1, _mo, _Q | – | 1.5 | 9 | linear | u | assumed | Table 1 |
| 22–26 | Ss_km1w, _ku, _km1, _mo, _Q | 1/m | 1e-6 | 1e-4 | log | u | assumed | Table 1 |
| 27 | riverbed_thickness | m | 0.005 | 0.2 | linear | u | assumed | Table 1 |
| 28 | cauchy_head | m | 335 | 355 | linear | u | verified (Fig. 3) | Table 1 |
| 29 | recharge_crop | mm/yr | 100 | 150 | linear | u | sign checked | Table 1 |
| 30 | recharge_grass | mm/yr | 80 | 130 | linear | u | sign checked | Table 1 |
| 31 | recharge_forest | mm/yr | 100 | 150 | linear | u | sign checked | Table 1 |

K is the horizontal saturated conductivity; the anisotropy ratio (horizontal/vertical) is fixed at 1 for mo and Q (Table 1). "Verified": the stored marginals of all 10,050 runs and of the behavioral runs match the histograms of the paper's Fig. 3 (p. 3796), which are on physical axes (ln K for conductivities); the conductivities are log-uniform and stored reversed (u = 1 is the lower bound; e.g. the stored K_km1w coordinate has 2,859 of 10,050 runs in its lowest 5%, where Fig. 3 has its tallest bar at the highest ln K). "Sign checked": the Spearman correlation with a flow output has the physically expected sign (recharge with gauge-C discharge, +0.04 to +0.15, and net boundary outflow, +0.09 to +0.30; the km1-w anisotropy ratio with flooding, +0.09, and stream A, +0.15); the linear scale is assumed. "Assumed": linear (log for Ss, which spans two decades), not reversed; neither the paper nor the supplement says, and these inputs have negligible influence on every stored output (paper p. 3797; notebook: √a ≤ 2.3e-4 m³/s for α, n and Ss against 1.04e-2 for the top input). **Input 27 is uncertain.** Table 1 calls it "Riverbed thickness (m), 0.005–0.2", and the supplement's print label agrees, but its internal label is `River-Elev.`, and the paper says on p. 3793 that "the head at the stream drain boundaries are drawn from uniform distributions", a parameter Table 1 does not otherwise list. The 1 cm stream-drain exit head of p. 3792 lies inside 0.005–0.2 m. Input 27 may therefore be the stream-drain exit head (m above surface) rather than a riverbed thickness; we keep the Table 1 name and list the question among the open points. It has negligible influence on the analysed outputs. **The analysis does not depend on any of these choices**: the notebook's analysis coordinates (ln x for log inputs, x otherwise) are affine in u, so eigenvalues, activity scores and NSE values are those of the stored data; only `inputs.csv` and the physical labels depend on the map.

**Quantities of interest.** Discharge at gauge C and net boundary outflow (paper Fig. 6), plus residence times in ku, km1 and km1-w (Fig. 8). The other outputs define the behavioral targets.

**Gradients.** None. Following the paper's standard approach ("gradient fit 1", p. 3790), gradients are those of a full quadratic in the normalized inputs (561 coefficients) fitted by least squares to the behavioral runs, averaged as in Eq. (7). **This estimator is QPHD with the moments of the behavioral sample:** with ∇q = g + Hx, the averaged matrix is C = ggᵀ + gμᵀH + Hμgᵀ + HSH, with μ and S the mean and second-moment matrix of the behavioral runs (the notebook checks the identity to a relative 3.8e-13), so all the information in C is in the 561 regression coefficients, not in 4,528 independent gradients (8.1 runs per coefficient; M ≥ (m+1)(m+2)/2 = 561 is met). v1's bootstrap over the gradient rows therefore holds the quadratic fixed and shows Monte Carlo error only; the notebook also reports a refit bootstrap (500 resamples of the runs with replacement, quadratic refitted each time), which is the sampling uncertainty of the estimate: for gauge C, λ1/λ2 = 4.62 with 95% interval [4.07, 5.28] refitted against [4.34, 4.89] with the quadratic fixed, and mean subspace distances 0.0251, 0.0687, 0.0875 (n = 1, 2, 3) against 0.0079, 0.0139, 0.0151; for the net outflow λ1/λ2 = 6.60, [5.87, 7.32] against [6.20, 6.96]. (The paper card for this data set said fit 2 was the one used; the paper uses fit 2 only when there are fewer than about twice 561 runs, as for its first 500 and 1000 runs.)

**Sampling design.** Rows 1–500: the initial Latin hypercube (each input occupies 499 of 500 strata). Rows 501–10,050: proposals from the uniform prior pre-screened by cubic meta-models on two-dimensional active subspaces, recomputed every 100 runs, with a soft margin and 10% unconditional acceptance (step 4, p. 3791). Row order is sampling order (the first-500 and first-1000 NSE values of Fig. 4 are reproduced). The supplement has 10,050 rows; the paper says 10,000 runs. **The sample density is not the uniform prior.**

**Behavioral (stage-two) runs.** Each of five targets must be on the permitted side of the midpoint between its limit and its outer point (pp. 3793–3795): flooding ≤ 3e-3 m³/s, gauge C ≥ 4e-3 m³/s, stream A ≤ 1.5e-3 m³/s, stream B ≥ 7.5e-7 m³/s, 0.225 ≤ ratio ≤ 0.67 (thresholds as hard-coded in the supplement's app; the paper's midpoint for the upper ratio limit is 0.675, which selects the same runs). This gives **4,528** runs; the paper reports 4,533 (p. 3795). No subset of the stored runs can give 4,533: the count over all 10,050 rows is below it, and every window of 10,000 consecutive rows gives 4,499–4,527. Under the stated midpoints the paper's count must come from different output values or thresholds; no clean cause shows in the data (loosening one limit reaches 4,533 only for flooding 3e-3 → 3.0051e-3, stream A 1.5e-3 → 1.5056e-3, lower ratio 0.225 → 0.22478 or gauge C 4e-3 → 3.8245e-3; stream B and the upper ratio limit cannot reach it within 10%; rounding all outputs to 6 decimals gives 4,531). It stays unresolved. The app's adjustable limits (knob defaults 3, 4, 1.5 and 0.00075 l/s, 22.5 % and 67 %; its help text says the reset button restores "the original one (as in the paper)") confirm the rule used here. The sensitivity analysis, like the paper's, uses the behavioral runs as the density ρ (p. 3797). Only 10 of the 500 Latin hypercube runs are behavioral (the paper's separate pure Monte Carlo test: 588 of 10,000, p. 3797).

**Published results.** Fig. 4 (meta-model NSE on 500, 1000 and all runs for flooding, ratio, stream A); Fig. 6 (2-D subspaces and √activity scores for gauge C and net boundary outflow; NSE 0.67 and 0.75); Fig. 7 (199 wells; head data not in the supplement); Fig. 8 (residence times in years; NSE 0.87, 0.87, 0.31); Fig. 9 (2,500 of the 4,533 runs drawn without replacement, 1,000 times, three gradient fits; linear-fit NSE histogram about 0.45–0.56). The paper prints no eigenvalues or eigenvectors.

## Regenerating

The supplement ZIP (3,284,010 bytes, sha256 `5bce5f2885f8bb4675b2d5aea4fe559bde5a4a476c2aa33ac1c2d232f7c99943`, downloaded 2026-11-01) holds `Active Subspace Pilot.mlappinstall` (sha256 `af1cdfce…9a4c3a`), itself a ZIP whose `fsroot/activeSubSpPilotData.mat` (MATLAB v5, 2,987,972 bytes, sha256 `41d407fa152bc65809aabc5f2ef95b6747d84920a86667b021f2c57cc0e6cb14`) holds `Rorg` (10,050 × 32), `obsMTRX` (10,050 × 9), `xlbl` and `xlblPRINT`. Unzip both (nothing needs to be run) and:

```bash
python extract_data.py "<unzipped>/fsroot/activeSubSpPilotData.mat"          # writes the three CSVs (2 s)
python extract_data.py "<unzipped>/fsroot/activeSubSpPilotData.mat" --check  # compares only
```

The `.mat` file is read with `scipy.io.loadmat`. The notebook runs in 1.5–3 minutes on one core (79 s printed in the last run on a quiet machine, 162 s under load; the refit bootstrap and the 500-repetition Fig. 9 study take most of it).

## Accuracy

- Conversion: `--check` reports `inputs_unit.csv` and `outputs.csv` bit-identical to `Rorg` and `obsMTRX` (including the signs of zeros), and `inputs.csv` mapping back to u with max |difference| 1.6e-15.
- No gradient check is possible (no model). The quadratic whose gradients define C explains in-sample R² = 0.815 of the gauge-C discharge and 0.875 of the net boundary outflow; the eigenvalues describe that quadratic.
- The stored data reproduce the paper's published numbers: all nine Fig. 4 NSE values to two digits; Fig. 6 NSE 0.672 and 0.753 (paper 0.67, 0.75) and all fourteen √activity scores within 0.0006 m³/s of the values read from the figure (with the [0, 1] scaling the figure used, see below); Fig. 8 NSE 0.871, 0.870, 0.315 (paper 0.87, 0.87, 0.31), its activity scores and the ends of its colour bars (with years = stored / 17,520); Fig. 9 NSE ranges of the two quadratic fits and score bands (the linear-fit histogram of the paper, about 0.45–0.56, has a heavier upper tail than our 0.457–0.506, 95%).
- Scaling: **Figs. 6 and 8 used inputs on [0, 1]** (the stored u; √a twice the [−1, 1] value); **Fig. 9 used [−1, 1]** (Eq. 4). The supplement's app rescales to [−1, 1] only when the mean stored coordinate is within 0.01 of 0.5; over the behavioral runs it is 0.4851. The axes of the scatter plots confirm it: with [0, 1] inputs the active variables span gauge C y1 [−0.06, 1.17], y2 [−0.68, 0.58] (Fig. 6 about [−0.07, 1.17], [−0.58, 0.68], up to sign) and, for the ku residence time, y1 [−0.50, 0.90], y2 [−0.52, 0.79] (Fig. 8 about [−0.5, 0.9], [−0.5, 0.78]); with [−1, 1] inputs they span about twice as much (gauge C y1 [−0.60, 1.87]; ku y1 [−1.02, 1.78]).

## How these files relate to the paper

The CSVs are the complete data of the supplement, which the paper offers as its data ("This code is written as a MATLAB app and also includes the data", p. 3801). They contain all 10,050 runs, behavioral or not, and nine outputs; the 199 well heads of Fig. 7 are not in the supplement. Open points: the 4,528 vs 4,533 behavioral runs (no subset of the stored runs gives 4,533), the 10,050 vs 10,000 rows, the input map for the 19 inputs not in Fig. 3, the identity of input 27 (riverbed thickness or stream-drain head), the unscaled [0, 1] inputs of Figs. 6 and 8, and the residence-time unit (not stated; Fig. 8 years = stored / 17,520, probably half-hours).

## License and credit

The supplement is distributed under CC BY 4.0 ("The copyright of individual parts of the supplement might differ", supplement title page; no other licence is stated for the data). Credit: D. Erdal and O. A. Cirpka, University of Tübingen; the work was supported by the Collaborative Research Center 1253 CAMPOS (Project 7), funded by the German Research Foundation (DFG, SFB 1253/1). The converted files here are CC BY 4.0 as the repository's data; `kaesbach_inputs.py` and `extract_data.py` are MIT. The MATLAB app itself is not redistributed.
