# Offshore_Wind_Turbine_Fatigue: data

Kind: **stored data**. FAST simulations run at NREL produced `NRELwind_testing.txt`. The file is the only record of those runs in this repository, and they cannot be rerun here. Its values must not be changed or regenerated.

## Files

| file | contents | role |
|---|---|---|
| `Offshore_Wind_Turbine_Fatigue.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2016; outputs committed |
| `NRELwind_testing.txt` | CSV, one header row, 50,000 rows × 13 columns, `%.16e`; 15,074,183 bytes, md5 883763ac1da7b0ea48ee0e1227751e95 | the stored data |

The file has no missing values and no duplicate input rows. Column `Run` runs from 1 to 50,000 in order.

## Provenance

- **Who ran it**: Peter Graf, National Renewable Energy Laboratory (NREL) (author, 2026-10-30), for P. A. Graf, G. Stewart, M. Lackner, K. Dykes and P. Veers, "High-throughput computation and the applicability of Monte Carlo integration in fatigue load estimation of floating offshore wind turbines", *Wind Energy* 19(5):861–872, 2016, doi:10.1002/we.1870 (NREL/JA-2C00-62929). Page and section numbers below refer to the accepted manuscript.
- **Generating code**:
  - FAST version 7 for the turbine response;
  - TurbSim for the turbulent wind;
  - HydroDyn, inside FAST 7, for the waves (a JONSWAP spectrum set by Hs and Tp);
  - rainflow counting routines ported to C from NREL's MLife.

  NREL's WISDEM framework (on openMDAO) generated the samples and ran the cases (§3). No other versions and no run dates are recorded.
- **Model**: the OC3 Hywind spar-buoy floating turbine under IEC 61400-3 design load case 1.2 (normal operation). Each run is one 10-minute simulation with the IEC class A normal turbulence model and power-law wind shear with exponent 0.14. The wind and wave distributions were fitted to NOAA National Data Buoy Center data for a representative East Coast site (§2).
- **Sample**: the 50,000 rows are a subset of the 500,000 Monte Carlo samples of the paper's 5-input study (§4.3). The author confirmed this on 2026-10-30. How the 50,000 were selected is not recorded. The first version of this directory shipped a 500,000-sample MATLAB file, `windData500k.mat` (64.5 MB). It was added in 79a3271 (2016-07-12) and deleted in 554642d (2016-07-22), and it stays in the git history (no history rewrite, author decision 2026-10-30).

## Columns

Inputs (columns 1–5). The file stores encoded values, not physical ones. The author confirmed the encoding on 2026-10-30:

| column | input | physical unit | distribution (paper §2) | stored value |
|---|---|---|---|---|
| `Vhub` | hub-height wind speed | m/s | Weibull | 2u − 1, u = CDF value; uniform on [−1, 1] |
| `Hs` | significant wave height | m | gamma conditioned on Vhub and WaveDir | 2u − 1, u = conditional CDF value |
| `WaveDir` | wave direction with respect to the wind | rad | von Mises conditioned on Vhub | θ/π, unwrapped; 435 values < −1, 6 > 1 |
| `Tp` | peak spectral wave period | s | gamma conditioned on Vhub and Hs | 2u − 1, u = conditional CDF value |
| `PlatformDir` | wind direction with respect to the fixed platform orientation | rad | von Mises conditioned on Vhub | θ/π, unwrapped; 284 values > 1 |

In all, 724 rows have an input outside [−1, 1]. The physical values cannot be recovered, because the paper gives the distribution parameters of its 4-input study only (p. 3), not those of the 5-input study.

Outputs (columns 6–12). Each is a short-term damage equivalent load (DEL) of one 10-minute simulation: the amplitude of a 1 Hz sinusoid that does the same fatigue damage, from rainflow counting with a fixed mean of zero and material exponent m = 4 (`rf4`) (paper §2–3). The author confirmed m = 4 for all outputs. The paper uses m = 10 for composite blades; the blade-root outputs here are the m = 4 variants.

| column | load | unit |
|---|---|---|
| `rf4_RootMxc1` | blade 1 root in-plane ("edgewise") bending moment | presumably kN·m |
| `rf4_RootMyc1` | blade 1 root out-of-plane (flapwise) bending moment | presumably kN·m |
| `rf4_TwrBsMxt` | tower-base side-to-side bending moment | presumably kN·m |
| `rf4_TwrBsMyt` | tower-base fore-aft bending moment | presumably kN·m |
| `rf4_Anch1Ten`, `rf4_Anch2Ten`, `rf4_Anch3Ten` | tension at the anchor of mooring line 1, 2, 3 | presumably kN |

The paper names `RootMxc1`, `TwrBsMxt` and `Anch1Ten`; the other meanings follow FAST's channel names. The units are FAST 7's channel units by inference; neither the paper nor the data state them.

## Gradients

None are stored. The notebook estimates them by local linear fits (60 nearest neighbors, at 2,000 seeded centers) in the normalized coordinates of the smallest box that contains [−1, 1]⁵ and the data. For `Vhub`, `Hs` and `Tp` those coordinates are the stored values. For `WaveDir` (box [−1.1026, 1.0137]) and `PlatformDir` (box [−1, 1.0465]) they are the stored values scaled by 0.945 and 0.977 and shifted.

## Unknowns

- The output units (kN·m and kN are inferred).
- The distribution parameters of the 5-input study, and so the physical input values.
- How the 50,000 rows were selected from the 500,000.
- Software versions other than FAST 7, and the run dates.
- Why the paper's Fig. 7 does not match these data. The m = 4 long-term DEL (mean of f⁴)^(1/4) of `rf4_RootMxc1` is 5,327, while Fig. 7 shows about 7,200, close to the largest stored value (7,394.5). The value for `rf4_Anch1Ten` (67.4) does agree with Fig. 6 (about 66). Both checks are printed by the notebook.

## Loading

```python
import numpy as np
fname = "NRELwind_testing.txt"
with open(fname) as fh:
    names = fh.readline().strip().split(",")
d = np.loadtxt(fname, delimiter=",", skiprows=1)      # (50000, 13)
X, F = d[:, 1:6], d[:, 6:]
in_labels, out_labels = names[1:6], names[6:]
```

## Regenerating

The data cannot be regenerated here. The notebook computes everything else from the file in about 40 seconds. All draws (gradient centers, bootstrap, plotted subset) come from one `np.random.default_rng(2016)`, and two runs give identical printed output.

## License and credit

Code: MIT (`LICENSE`). Data (`NRELwind_testing.txt`): CC BY 4.0 (`LICENSE-DATA`); redistribution confirmed by the author (2026-10-30). The data were generated by Peter Graf at the National Renewable Energy Laboratory; cite Graf et al. 2016 (above) when you use them. Notebook by Andrew Glaws, Peter Graf and Ryan Howard.
