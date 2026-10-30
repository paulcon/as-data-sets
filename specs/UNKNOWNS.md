# Unknowns: questions the author could not answer

Every item here is a question the author answered as "unknown" or "not sure", in either the first or second round of the 2026-10-30 interview ([TODO.md](TODO.md) §0). The notebook and `DATA.md` must state each one as unknown, without guessing. If a fact turns up later (from a data provider, an old archive or a coauthor), record it here, update the matching TODO item and spec, and strike the line.

Add new unknowns from later interviews to this file.

## Data provenance and setup

- [ ] **NACA0012**: flow conditions (Mach, α, Euler or RANS, SU2 version); where the second batch (runs 945–1756) came from; which Hicks–Henne bumps x01–x18 are (upper/lower surface, chord stations) ([NACA0012.md](NACA0012.md) §11.1, 1a, 2). Lead, not used: Lukaczyk's 2015 dissertation (§2.4.1, p. 23) runs SU2 on NACA 0012 at Mach 0.80, α = 1.25°, inviscid Euler, but nothing ties that 2-parameter study to this file ([#39](https://github.com/paulcon/as-data-sets/pull/39))
- [ ] **ONERA-M6**: which 3 of the 300 LHS runs are missing; which FFD control points and directions x01–x50 are. The book's 60-parameter data are unavailable ([ONERA-M6.md](ONERA-M6.md) §11.1, 2). Also: the SU2 version, mesh and solver settings; where the three PNGs were rendered ([#44](https://github.com/paulcon/as-data-sets/pull/44))
- [ ] **Subsurface_Permeability**: the KL basis B (100 PCs) and the domain grid, so no spatial plots of w; what "GW02" means; data license and provenance (ParFlow version, run date). Known: "ShortCorr" = short correlation length of the permeability model ([Subsurface_Permeability.md](Subsurface_Permeability.md) §11.3, 4, 7)
- [ ] **MHD**: the duplicate u_avg rows 47/48 (labels 49, 50) turned out to start a one-label shift: u_avg labels 50–99 hold runs 49–98 ([#45](https://github.com/paulcon/as-data-sets/pull/45)). Unknown: why that block was shifted; whether u_avg outputs for runs 71 and 99 exist anywhere; whether all 17 label gaps are failed runs (run 72 is: its gradient is a placeholder). Also unknown for the generator: units, Drekar version, who ran the simulations, the sampling seed, and the region u_avg and B_ind are averaged over. The original Drekar output is not recoverable ([MHD.md](MHD.md) §11.2)
- [ ] **Stomatal_Resistance**: data license and PF-CLM version ([Stomatal_Resistance.md](Stomatal_Resistance.md) §11.7); who ran PF-CLM; how the 18 continuous inputs were sampled ([#47](https://github.com/paulcon/as-data-sets/pull/47))
- [ ] **Elliptic_PDE**: the script that generated `kl-short.npz` and `kl-long.npz` is lost ([Elliptic_PDE.md](Elliptic_PDE.md) §11.6)
- [ ] **LM104X**: why the file has 310 rows when the dissertation reports 315 sampled and 305 retained; which columns are wing (50), tail (24), aft deck (16) and fuselage (15) ([LM104X.md](LM104X.md) §11.3, 4b); what "104X" stands for ([#50](https://github.com/paulcon/as-data-sets/pull/50))
- [ ] **HyShotII**: the book's two φ = 0.30 corner runs are unavailable (stored as `nan` in `HyShotII_phi035.csv`; rows 15–16 of the 0.30 column in the old `HyShotII35.txt`) ([HyShotII.md](HyShotII.md) §11.2). Also: the Joe version and meshes; how the design points were drawn; why the book's φ = 0.30 runs sit 0.06–0.10 bar above the JCP 4.8 bar runs; where the JCP 5.6 bar runs are ([#49](https://github.com/paulcon/as-data-sets/pull/49))
- [ ] **NREL_Wind**: how the 50,000 rows were selected from Graf et al.'s 500,000-sample study ([#43](https://github.com/paulcon/as-data-sets/pull/43))
- [ ] **Lithium_Battery**: which t* each voltage column holds; the units and positions of the concentration columns ([#48](https://github.com/paulcon/as-data-sets/pull/48))

## Units and scales

- [ ] **Lithium_Battery**: I_1C = 30 A m⁻² and `Cap*` columns = elapsed time in seconds are inferred, not confirmed ([Lithium_Battery.md](Lithium_Battery.md) §11.3)
- [ ] **Stomatal_Resistance**: tran_veg in mm s⁻¹ and rs in s m⁻¹ are inferred from JHM Fig. 9, not confirmed ([Stomatal_Resistance.md](Stomatal_Resistance.md) §11.6)
- [ ] **LM104X**: FFD variable bounds, ±0.3 in the data vs ±3.0 in dissertation Eq. 2.21 ([LM104X.md](LM104X.md) §11.4)
- [ ] **Hydrology**: meaning of output columns 3–4 (0-based) and the flux units ([Hydrology.md](Hydrology.md) §11). The ratio of the data's fitted C₀ to the paper's Table 4 C₀ is not one factor ("~171×") but 103–617, different for each flux ([#40](https://github.com/paulcon/as-data-sets/pull/40))
- [ ] **NREL_Wind**: output units; the notebook says "presumably kN·m / kN (FAST 7 channel units)" ([#43](https://github.com/paulcon/as-data-sets/pull/43))
- [ ] **LM104X**: along w₁ the EA adjoint derivatives follow the slope of the values (correlation 0.94) but are 15.1× smaller, so values and gradients may be on different scales; the stored ΔA_e values run 5.5–1666 while dissertation Eq. 2.21 constrains ΔA_e ≤ 1e-4 ([#50](https://github.com/paulcon/as-data-sets/pull/50))

## Inconsistencies in the published sources the author cannot settle

- [ ] **Ebola**: the fitted Sierra Leone γ2 = 0.0833 (Table 2) lies outside the range U(0.1236, 0.384) (Table 7). It is a typo, but which value is wrong is unknown ([Ebola.md](Ebola.md) §11)
- [ ] **MHD**: how published Fig. 3 (Hartmann B_ind) was computed: its w1 components 3 and 5 and eigenvalue scale differ from `hartmann_model.py` with ℓ = μ0 = 1, and neither paper version states μ0 ([MHD.md](MHD.md) §11.1)
- [ ] **Lithium_Battery**: r_a, r_c are Normal in the published table (p. 250) but uniform on [−1, 1] in the stored data ([Lithium_Battery.md](Lithium_Battery.md) §11.1)
- [ ] **Hydrology**: default rl_n = 0.58 lies outside its range U(0.35, 0.55) ([Hydrology.md](Hydrology.md) §11)
- [ ] **Stomatal_Resistance**: JHM Fig. 8 labels local time at Tonzi UTC−6, but California summer time is UTC−7 ([Stomatal_Resistance.md](Stomatal_Resistance.md) §11.4)
- [ ] **NREL_Wind**: Graf et al. Fig. 7 shows a long-term m = 4 DEL of about 7,200 for RootMxc1; these data give 5,327 (Anch1Ten matches Fig. 6). The notebook no longer compares them (author, 2026-10-30; [#43](https://github.com/paulcon/as-data-sets/pull/43))
- [ ] **HyShotII**: JCP Table 4's 4.8 bar safe box (AoA ≤ 4.29, TI ≤ 1.88 %), the book's φ = 0.35 box (AoA ≤ 3.31) and its R² 0.998 are not reproduced by the stored data ([#49](https://github.com/paulcon/as-data-sets/pull/49))
- [ ] **HIV**: at t = 1700 the data give λ₂ = 509; paper Fig. 2 shows about 2000 ([#51](https://github.com/paulcon/as-data-sets/pull/51))
- [ ] **SingleDiodePV**: the cause of the constant 4.7e-5 bias in dISC (possibly slightly different I_SC bounds than Table 1) ([SingleDiodePV.md](SingleDiodePV.md) §11.3)
