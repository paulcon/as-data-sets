# Unknowns: questions the author could not answer

Every item here is a question the author answered as "unknown" or "not sure", in either the first or second round of the 2026-10-30 interview ([TODO.md](TODO.md) §0). The notebook and `DATA.md` must state each one as unknown, without guessing. If a fact turns up later (from a data provider, an old archive or a coauthor), record it here, update the matching TODO item and spec, and strike the line.

Add new unknowns from later interviews to this file.

## Data provenance and setup

- [ ] **NACA0012**: flow conditions (Mach, α, Euler or RANS, SU2 version); where the second batch (runs 945–1756) came from; which Hicks–Henne bumps x01–x18 are (upper/lower surface, chord stations) ([NACA0012.md](NACA0012.md) §11.1, 1a, 2)
- [ ] **ONERA-M6**: which 3 of the 300 LHS runs are missing; which FFD control points and directions x01–x50 are. The book's 60-parameter data are unavailable ([ONERA-M6.md](ONERA-M6.md) §11.1, 2)
- [ ] **Subsurface_Permeability**: the KL basis B (100 PCs) and the domain grid, so no spatial plots of w; what "GW02" means; data license and provenance (ParFlow version, run date). Known: "ShortCorr" = short correlation length of the permeability model ([Subsurface_Permeability.md](Subsurface_Permeability.md) §11.3, 4, 7)
- [ ] **MHD**: why u_avg rows 47 and 48 (labels 49, 50) are identical; the original Drekar output is not recoverable ([MHD.md](MHD.md) §11.2)
- [ ] **Stomatal_Resistance**: data license and PF-CLM version ([Stomatal_Resistance.md](Stomatal_Resistance.md) §11.7)
- [ ] **Elliptic_PDE**: the script that generated `kl-short.npz` and `kl-long.npz` is lost ([Elliptic_PDE.md](Elliptic_PDE.md) §11.6)
- [ ] **LM104X**: why the file has 310 rows when the dissertation reports 315 sampled and 305 retained; which columns are wing (50), tail (24), aft deck (16) and fuselage (15) ([LM104X.md](LM104X.md) §11.3, 4b)
- [ ] **HyShotII**: the book's two φ = 0.30 corner runs (rows 15–16 of the 0.30 column in `HyShotII35.txt`) are unavailable ([HyShotII.md](HyShotII.md) §11.2)

## Units and scales

- [ ] **Lithium_Battery**: I_1C = 30 A m⁻² and `Cap*` columns = elapsed time in seconds are inferred, not confirmed ([Lithium_Battery.md](Lithium_Battery.md) §11.3)
- [ ] **Stomatal_Resistance**: tran_veg in mm s⁻¹ and rs in s m⁻¹ are inferred from JHM Fig. 9, not confirmed ([Stomatal_Resistance.md](Stomatal_Resistance.md) §11.6)
- [ ] **LM104X**: FFD variable bounds, ±0.3 in the data vs ±3.0 in dissertation Eq. 2.21 ([LM104X.md](LM104X.md) §11.4)
- [ ] **Hydrology**: meaning of output columns 3–4 (0-based) and the flux units ([Hydrology.md](Hydrology.md) §11)

## Inconsistencies in the published sources the author cannot settle

- [ ] **Ebola**: the fitted Sierra Leone γ2 = 0.0833 (Table 2) lies outside the range U(0.1236, 0.384) (Table 7). It is a typo, but which value is wrong is unknown ([Ebola.md](Ebola.md) §11)
- [ ] **MHD**: how published Fig. 3 (Hartmann B_ind) was computed: its w1 components 3 and 5 and eigenvalue scale differ from `gradients.py` with ℓ = μ0 = 1, and neither paper version states μ0 ([MHD.md](MHD.md) §11.1)
- [ ] **Lithium_Battery**: r_a, r_c are Normal in the published table (p. 250) but uniform on [−1, 1] in the stored data ([Lithium_Battery.md](Lithium_Battery.md) §11.1)
- [ ] **Hydrology**: default rl_n = 0.58 lies outside its range U(0.35, 0.55) ([Hydrology.md](Hydrology.md) §11)
- [ ] **Stomatal_Resistance**: JHM Fig. 8 labels local time at Tonzi UTC−6, but California summer time is UTC−7 ([Stomatal_Resistance.md](Stomatal_Resistance.md) §11.4)
- [ ] **SingleDiodePV**: the cause of the constant 4.7e-5 bias in dISC (possibly slightly different I_SC bounds than Table 1) ([SingleDiodePV.md](SingleDiodePV.md) §11.3)
