# Residential_Building_Stock_Energy_Use: data

Kind: **stored data**, a documented **subset** of a deposited table. The values were computed by Ali et al.
(2024) with EnergyPlus driven by jEPlus and DesignBuilder construction templates, and deposited at Mendeley
Data, [doi:10.17632/m6vv9k9gcd.1](https://doi.org/10.17632/m6vv9k9gcd.1) (version 1, CC BY 4.0). The full
table (1,030,022 rows × 32 columns, 182.5 MB) is too large to ship here. `convert_data.py` keeps the 49,298
rows of one building type, one weather file and no renewables, and copies every kept number character for
character. The values were not regenerated (the archetype models are not published) and are not altered.

## Files

| file | contents | role |
|---|---|---|
| `Residential_Building_Stock_Energy_Use.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 241; outputs committed |
| `building_inputs.csv` | 49,298 × 15: the 15 numeric inputs, physical units (names and units in the second `#` line) | subset of the deposited CSV |
| `building_outputs.csv` | 49,298 × 13: `source_row` (1-based data row of the deposited CSV) and the 12 numeric outputs (kWh/yr, m², factors, EUI in kWh/(m² yr)); row i matches row i of the inputs | subset of the deposited CSV |
| `convert_data.py` | reads the deposited zip without extracting it, checks its SHA-256, selects the rows, writes the two CSVs; `--check` compares | the conversion (standard library only) |
| `check_accuracy.py` | recomputes the numbers of this file that describe the full deposited table (needs the deposited zip) | check |

The CSVs are plain comma-separated numbers after two `#` lines, so `np.loadtxt(path, delimiter=',')` reads
them. Deposited label → column name: `Floor_Insulation_U-Value` → `U_floor`, `Door_…` → `U_door`, `Roof_…` →
`U_roof`, `Window_…` → `U_window`, `Wall_…` → `U_wall`, `HVAC_Efficiency` → `HVAC_efficiency`,
`Domestic_Hot_Water_Usage` → `DHW_use`, `Building_Orientation` → `orientation`, `Lighting_Density` →
`lighting_density`, `Occupancy_Level` → `occupancy`, `Equipment_Density` → `equipment_density`,
`Heating_Setpoint_Temperature` → `heating_setpoint`, `Heating_Setback_Temperature` → `heating_setback`,
`Air_Change_Rate` → `air_change_rate`, `Window_to_Wall_Ratio` → `window_to_wall_ratio`; outputs
`Total_Building_Area` → `floor_area`, `Heating_Usage` → `heating_usage`, `Water_Systems_Energy` →
`water_systems_energy`, `Interior_Lighting_Energy` → `lighting_energy`, `Interior_Equipment_Energy` →
`equipment_energy`, `Total_Heating_Energy` → `space_heating_energy`, `Photovoltaic_Power` → `pv_energy`,
`Total_Electricity_Energy` → `electricity_energy`, `Heating_Conversion_Factor` → `heating_conversion_factor`,
`Electricity_Primary_Conversion_Factor` → `electricity_primary_factor`, `Heating_Primary_Conversion_Factor` →
`heating_primary_factor`, `Energy_Use_Intensity` → `EUI`. The name `space_heating_energy` is this data set's:
in every row `Total_Heating_Energy` = `Heating_Usage` − `Water_Systems_Energy` to 0.01 kWh (notebook), so
`Heating_Usage` holds space plus water heating.

**The subset.** Rows with `Building_Type` = "Semi Detached", `Weather_File` = "Historical" and
`Renewable_Energy_Usage` = "No", in their deposited order. This rule is this data set's choice: it fixes the
four categorical inputs of Table 4 (building type, location, weather, renewables), so the 15 numeric inputs
are the only inputs that vary, and it leaves out the photovoltaic export (PV energy is 0 in every kept row).
Not shipped: the four categorical columns (constant in the subset) and the two rating labels
(`Simple_Building_Energy_Rating`, `Building_Energy_Rating`, text classes of the EUI).
The other 15 cells of building type × weather × renewables hold 47,646–86,007 rows each
(`check_accuracy.py`) and can be extracted by changing `ROW_RULE` in `convert_data.py`.

## Model

**Citation.** U. Ali, S. Bano, M. H. Shamsi, D. Sood, C. Hoare, W. Zuo, N. Hewitt, J. O'Donnell, "Urban
residential building stock synthetic datasets for building energy performance analysis", *Data in Brief*
53:110241 (2024), doi:10.1016/j.dib.2024.110241 (publisher PDF read, CC BY 4.0; checked on Crossref).
Related research article (cited on p. 2, not read): Ali et al., *Energy and Buildings* 303:113768 (2024),
doi:10.1016/j.enbuild.2023.113768.

**Simulator** (Sects. 4.2–4.3, pp. 7–9). jEPlus runs EnergyPlus with DesignBuilder construction templates,
which represent each construction by its U-value (p. 8). Four archetypes (terraced, detached, semi-detached,
bungalow; geometry in Fig. 2, p. 7); Dublin EPW weather, historical and 2030 projections from Meteonorm
(p. 7). The paper names no EnergyPlus or jEPlus version, no run time and no simulation settings, and the
archetype models are not published. The semi-detached floor area in the data is 107.69 m².

**Inputs** (m = 15 numeric inputs; Table 4, p. 8, gives the ranges; Table 2, p. 4, the sample statistics of
the full table). All 49,298 rows lie inside the Table 4 box (notebook).

| column | Table 4 name | units | Table 4 range | levels in the subset | source |
|---|---|---|---|---|---|
| U_floor | Floor U-value | W/(m² K) | [0.15, 1.6] | 28 | Table 4, p. 8 |
| U_door | Door U-value | W/(m² K) | [0.81, 5.7] | 10 | Table 4 |
| U_roof | Roof U-value | W/(m² K) | [0.07, 2.28] | 21 | Table 4 |
| U_window | Window U-value | W/(m² K) | [0.73, 5.75] | 34 | Table 4 |
| U_wall | Wall U-value | W/(m² K) | [0.09, 2.4] (data 0.1–2.4) | 25 | Table 4 |
| HVAC_efficiency | HVAC Efficiency/COP | 1 (Table 2 prints "%") | [0.3, 4.5] (subset 0.3–4.4) | 20 | Table 4 |
| DHW_use | Domestic Hot Water | L/(m² day) (Table 2) | [0.5, 3.5] | 4 | Table 4 |
| orientation | Orientation | deg | [0, 315] | 8 | Table 4 |
| lighting_density | Lighting Density | W/m² | [1, 9] | 5 | Table 4 |
| occupancy | Occupancy | persons | [1, 6] | 6 | Table 4 |
| equipment_density | Equipment Density | W/m² | [1, 21] | 5 | Table 4 |
| heating_setpoint | Heating Setpoint | °C | [18, 23] | 6 | Table 4 |
| heating_setback | Heating Setback | °C | [10, 14] | 5 | Table 4 |
| air_change_rate | Air Change Per Hour | 1/h | [0.35, 3] | 7 | Table 4 |
| window_to_wall_ratio | Window-to-Wall Ratio | % | [0, 70] | 4 | Table 4 |

Units of Table 4 are not printed; they are those of Table 2 (p. 4). Fixed by the subset: building type
semi-detached, location Dublin, weather historical, renewables no (Table 4).

**Quantity of interest.** Energy use intensity, "annual primary energy usage per square meter of total floor
space" in kWh/(m² yr) (p. 8). Secondary: annual space heating energy (`space_heating_energy`, kWh/yr), the
EnergyPlus output through which the building inputs act. In every row the stored EUI satisfies (notebook for
the subset; `check_accuracy.py` item 7 for all rows, max deviation of order 1e-13 kWh/(m² yr), rounding)

EUI = (p_H · c_H · Q_heat + p_E · E_el) / A,

with Q_heat = `heating_usage`, c_H = `heating_conversion_factor` = round(1.084/η, 3) for η = `HVAC_efficiency`,
p_H = `heating_primary_factor` = 1.1 for η ≤ 0.9 and 2.08 for η ≥ 2, p_E = 2.08, E_el = `electricity_energy`
(= lighting + equipment − 0.95 × PV to 0.011 kWh/yr over all rows, `check_accuracy.py`) and A the floor area.
This identity is this data set's finding from the stored values; the paper describes the conversion factors
only by their means (Table 3, p. 5; p. 4) and does not give the formula or the meaning of 1.084, 1.1 and 2.08
(1.1 and 2.08 read like primary-energy factors of gas and electricity; an inference). The water-heating,
lighting and equipment energies take 4, 5 and 5 values (one per DHW, lighting and equipment level; notebook),
so the only output that varies with the building envelope and schedule is the space heating.

**Gradients.** None: EnergyPlus gives no derivatives, and no runs can be made here. The notebook estimates
gradients by local linear fits (`ac.local_linear_gradients`, 60 nearest distinct rows). The rows are sparse in
15 dimensions: in the normalized box (side 2) the median nearest-neighbour distance is 1.216 and the median
radius of the 60 rows of a fit is 1.80, and two thirds of the neighbourhoods mix boiler and heat-pump rows
(notebook). The fitted slopes are therefore averages of the EUI's variation over wide windows (an
outer-product-of-gradients estimate), not local derivatives, and their lack-of-fit variance is large (it
dominates the eigenvalues after the first; notebook). Gradients are in the normalized coordinates of the
Table 4 box (each input mapped linearly from its Table 4 range to [−1, 1]); the box sets only this scaling.
The input density of the analysis is the empirical distribution of the 49,298 rows: the matrix C is the
average over the rows. The notebook checks the direction with a global quadratic whose gradient is averaged
over the rows, with the library's QPHD (which averages over the uniform box instead: the gradient at the box
centre and the box covariance I/3) and with OLS, and repeats the OLS fit with each input through its empirical
CDF.

**Sampling design.** The paper says "This study uses the LHS method to generate a sample of 1 million
buildings" (p. 9) and states no distributions. In the data every input takes a few discrete levels. In the
subset each level of an input appears about equally often, except where the absence of high-EUI rows
(below) thins the levels that raise the EUI: η = 0.3 has 469 rows against a heat-pump mean of 2874 per level
(notebook). The marginals are not uniform on the Table 4 box: 27 of the 28 floor U-value levels lie in
[0.15, 0.84] W/(m² K) and one at 1.6, and the HVAC efficiency has no level between 0.9 and 2 (notebook).
Whether the 49,298 rows are part of one Latin hypercube cannot be told from the data: with
discrete levels the strata are not identifiable. In the subset the inputs are nearly uncorrelated (largest
|r| 0.141, between η and DHW use, which the truncation induces; notebook). Over the full table they are not
(largest |r| 0.315, roof U-value against orientation; `check_accuracy.py`), and the level sets differ between
cells (HVAC_efficiency has 26 levels in the rows with renewables, 20 without; air change rates above 3, up to
7.35, occur only in the detached rows, 110,777 rows; `check_accuracy.py`). The full table is therefore a union
of designs, not one Latin hypercube; the subset is described here as "a subset of the deposited design".

**Truncation.** No row of the deposited table has EUI ≥ 650 kWh/(m² yr), while 3,477 rows lie in [640, 650);
the cutoff holds in every one of the 16 building type × weather × renewables cells (largest EUI per cell
649.8527 to 649.9996; `check_accuracy.py` item 4). In the subset the maximum is 649.97 (notebook). Table 3
(p. 5) prints the maximum as 650, and the paper does not mention a cutoff. The removal is visible in the level
counts: if every η level had the heat-pump mean count, the subset would have 57,475 rows, so a fraction 0.142
is missing, mostly at low η (notebook; the heat-pump levels η = 2 and 2.2 hold a few rows near the cutoff, 51
and 21 with EUI ≥ 600, so they may be slightly thinned too and the estimate is slightly low). The cutoff also
selects within a level: the mean space heating is 6,234 kWh/yr at η = 0.3 against 12,485–12,883 at the
heat-pump levels the notebook prints, although η does not enter the space heating. The notebook tests this
reading: it repeats each heat-pump row three times with η redrawn from its 20 levels, recomputes the EUI from
the identity below (the space heating has no η slope among the heat-pump rows: 6.7 ± 28.9 kWh/yr per unit),
and cuts at 650. The rebuilt cell loses a fraction 0.141 of its rows, keeps 0.166, 0.474 and 0.886 of the
rows at η = 0.3, 0.5 and 0.8 (stored: 0.163, 0.461, 0.891) and reproduces the space-heating trend over the
boiler levels, so the runs at or above 650 were removed after the simulations (by the authors or their tool;
the paper does not say). Without the cutoff, the OLS direction of the rebuilt cell turns by |cos| 0.9934
(η entry −0.859 instead of −0.818) and the leading eigenvalue of a quadratic fit averaged over the rows is
4.57 times larger. Every analysis of these files describes the model on the set EUI < 650.

**Templates, not U-values.** The space heating of the subset is not monotone in the wall and roof U-values:
the mean space heating is 8,871 kWh/yr at U_wall = 0.1 and 8,596 at 2.4 W/(m² K), while the level means reach
19,842 kWh/yr at 1.95; the roof level means range from 8,712 (at 1.14) to 14,874 (at 0.1) (notebook). The
paper says that the templates represent the thickness, conductivity, density and heat capacity of the building
materials through U-values (p. 8); in the data the U-value columns behave like labels of templates whose
thermal effect is not ordered by the label. Smooth fits in these inputs therefore leave part of the variation
unexplained (LOO R² below).

**Published results.** No active subspace. The paper reports correlations of each input with the EUI over the
full table (Fig. 1, p. 6; text p. 5): HVAC efficiency −0.68, DHW 0.45, ACH 0.43, U-values 0.24–0.35,
orientation 0.35, equipment 0.36, setpoint 0.35, lighting 0.24, setback 0.19, occupancy −0.058, WWR 0.00085.
`check_accuracy.py` reproduces every Fig. 1 value from version 1 of the deposit (item 2, three significant
digits; for example the window-to-wall ratio 0.000852 against the printed 0.00085), which confirms that the
paper describes version 1. Table 2 (p. 4) and Table 3 (p. 5) also agree with version 1 to their printed digits,
with sample standard deviations (ddof = 1; items 3, 4), except Table 2's maximum air change rate: it prints 3, while
the full table reaches 7.35 (its printed mean 1.53 and sd 1.57 match the full table's 1.533 and 1.568, so the
maximum is a misprint or describes a different subset). Table 3 prints the maximum heating usage as
"10,7361.3" (107,361.27 in the data) and the maximum total heating energy as "10,6752.9" (106,752.87). The text on p. 4 gives the "Heating and Electricity Primary Conversion
Factors (averaging 0.65 and 2.08 respectively)"; in Table 3, 0.65 is the mean of the heating conversion
factor, and the heating primary conversion factor averages 1.87.

## Regenerating

The data cannot be regenerated from the repository (the archetype models are not published, and no EnergyPlus
runs are made). The two CSVs are regenerated from the deposited archive:

```
python convert_data.py "<folder>/Urban Building Energy Stock Datasets.zip" --out DIR
python convert_data.py "<folder>/Urban Building Energy Stock Datasets.zip" --out DIR --check
```

`--out` is required and must lie outside this directory and its subdirectories (the script stops otherwise);
the check prints, for each file, whether the text is identical and whether every value agrees
within relative 1e-12. In the analysis venv (2026-11-08) the check ran in 4–13 s and printed "text
identical; values equal within relative 1e-12" for both files.

**Source.** Mendeley Data, "Urban Building Energy Stock Datasets", U. Ali, S. Bano, J. O'Donnell, version 1,
DOI [10.17632/m6vv9k9gcd.1](https://doi.org/10.17632/m6vv9k9gcd.1), CC BY 4.0: the DOI the paper gives (p. 2,
Specifications Table). Archive `Urban Building Energy Stock Datasets.zip`, 52,851,275 bytes, SHA-256
`1f6bb3e1dfcb7ebefbd1a75f93d56dd12db2923aab58717f077ab4a643c42c29`, downloaded 2026-11-03; it holds one file,
`Urban Building Energy Stock Datasets/urban_building_stock_datasets.csv` (182,513,398 bytes, dated 2023-11-30
in the archive). Versions 2–5 of the record exist (the latest, version 5, dated 16 April 2024); they were not
downloaded or compared. The paper's Fig. 1 and Tables 2–3 match version 1 (above).

The notebook takes about 5 to 6 minutes in the repository's `.venv` (319 s in the committed run, of which 121 s for the
local linear gradients at all 49,298 rows and 76 s for the neighbourhood diagnostics); on a busy machine it
takes considerably longer.

## Accuracy

- Conversion: lossless for the kept rows and columns (`convert_data.py --check`, above).
- Simulation accuracy: not reported by the paper ("The developed archetypes are validated in existing studies
  using the current Irish EPC software", p. 9; no numbers). External runs, not reproducible from the repository.
- Stored precision: energies with 2 decimals (kWh/yr; electricity with 3), conversion factors with 3; the EUI
  is stored with about 16 significant digits and equals the identity above to rounding level (notebook).
- No gradients, so no gradient check. The local linear slopes are smoothed over wide neighbourhoods, and the
  notebook estimates their lack-of-fit variance (trace 6222, of which 5813 outside w₁, against an eigenvalue
  tail λ₂ + … + λ₁₅ = 7103); the bootstrap intervals resample the slopes without refitting them and do not
  include this variance. The notebook's fits have leave-one-out R² of 0.8286 (cubic in w₁ᵀx) to
  0.8781 (cubic in all 15 inputs) for the EUI, and 0.6927 (cubic in w₁ᵀx) to 0.7310 (cubic in two active
  variables; a quadratic in all 15 inputs gives 0.7244) for the space heating; the remainder is variation
  that none of these smooth fits captures (the template effect above, and the truncation, account for at
  least part of it).
- Full-table numbers in this file (row counts, Fig. 1 correlations, Table 2 and 3 statistics, the cutoff at
  650, the identity over all rows, the input correlations, the level sets): `check_accuracy.py` with the
  deposited zip, about 30–40 s on a shared two-core machine (37 s on 2026-11-08).

## How these files relate to the paper

The rows are 49,298 of the paper's 1,030,022 runs ("1 million", p. 1), unchanged. The paper's analysis is a
correlation table of all rows (Fig. 1) and a machine-learning study in the related article; the notebook
estimates an active subspace of the EUI of one cell (semi-detached, historical weather, no renewables) with
local linear gradients and compares the leading direction with the paper's correlations, which mix the four
building types, both weathers and both renewables options. The subset correlations differ from Fig. 1 for
several inputs (notebook table): in the full table orientation and the U-values correlate with the EUI at
0.24–0.35, while within the subset their correlations are between −0.012 and 0.071. The full table's
correlations between inputs (up to 0.315, `check_accuracy.py` item 5) and the differences between cells can
produce such values; the full-table correlations were not decomposed further.

Notebook results (seed 241): local linear gradients (60 nearest distinct rows, all 49,298 rows, wide
neighbourhoods) give λ₁/λ₂ = 39.65, with w₁ led by HVAC efficiency (−0.807) and DHW use (+0.422); the
quadratic averaged over the rows, the library QPHD and OLS give the same w₁ (|cosine| 0.9990, 0.9963 and
0.9995). The later eigenvalues (ratios 1.06–1.26) are mostly lack-of-fit variance of the local fits: λ₂ and
its direction w₂ (the floor U-value alone) are almost entirely that. A weaker smooth structure exists in w₃
(equipment density, HVAC efficiency, hot water, air change rate) and w₄ (the wall U-value), which lie in the
three-dimensional subspace of the global quadratic (gap 6.75 after its third eigenvalue; 8.26 for the
library QPHD). The chosen dimension is n = 1 (λ₁/λ₂ ≈ 40): a cubic in w₁ᵀx has leave-one-out R² 0.8286,
a cubic in w₁, w₃ 0.8567 and in w₁, w₃, w₄ 0.8658 (a quadratic in all 15 inputs 0.8664, a cubic 0.8781). The
space heating has a two-dimensional subspace (ratios 4.07, 21.94 for the quadratic averaged over the rows),
with the wall U-value alone as w₂.

## License and credit

Data: © Ali, Bano, O'Donnell, Mendeley Data 10.17632/m6vv9k9gcd.1, CC BY 4.0; the article is CC BY 4.0. Cite
the paper and the data record when using these files. `convert_data.py`, `check_accuracy.py` and the notebook:
MIT (this repository). No code of the authors is included (none is published).

## Unknowns

- EnergyPlus and jEPlus versions, simulation settings (time step, HVAC model), run time per building, and the
  archetype models themselves: not stated or published.
- The sampling distributions and the mapping from the Latin hypercube to the discrete levels: not stated; the
  levels and their counts are reported from the data.
- Why the runs with EUI ≥ 650 were removed: not stated. That they were removed is shown by the rebuilt cell
  in the notebook (above); whether on purpose, and by whom, is not.
- The meaning of the post-processing constants 1.084 (c_H = 1.084/η), 1.1 and 2.08: not stated.
- Why the space heating is not monotone in the wall and roof U-values: not stated; the U-values may be
  labels of templates that differ in other ways (for example thermal mass); the paper does not say.
- Why air change rates above 3 occur only in the detached rows (110,777 rows, up to 7.35), against the
  maximum of 3 in Tables 2 and 4.
- Whether versions 2–5 of the deposit change the values: not checked.
