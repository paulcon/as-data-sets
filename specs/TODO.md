# TODO: update paulcon/as-data-sets for active_subspaces v1

Compiled 2026-10-29 from the per-notebook specs in this directory. Each item links to the spec section with details. Paste the sections into GitHub issues (one issue per heading works well); `- [ ]` items render as task lists.

Paths: on 2026-10-30 the data sets moved into category directories with new names (map in [README.md](README.md#paths)). The items below use the old names, which are also the spec names.

Target for every notebook: Python 3.11+, runs headless with `jupyter nbconvert --execute` from its own directory, imports only numpy, scipy, matplotlib and `active_subspaces` v1. No pandas, no numpad.

## 0. Decisions for the author (block the items below)

- [x] LM104X: can the Lockheed/NASA N+2 data be released? **Cleared: include it** (author, 2026-10-30) ([LM104X.md](LM104X.md) §11)
- [x] Atacamac: was there a reason other than "no paper" for its removal in 9c1b8a7, e.g. data permission? **No permission issue. The author does not want the Atacamac model in main** (author, 2026-10-30). It staying in history (from 7c26e3a, and on dev) is fine ([BRANCH_MERGE.md](BRANCH_MERGE.md) §5)
- [x] History rewrite, decided once for all three: **No rewrite** (author, 2026-10-30). purge `SNL-CLM/IJUQ-6821.pdf` (publisher PDF), the deleted 64.5 MB `windData500k.mat`, and old `Vanilla_Chain.mat` versions? Rewriting breaks 12 forks and every SHA; must happen before tagging. ([BRANCH_MERGE.md](BRANCH_MERGE.md) §4.4, [NREL_Wind.md](NREL_Wind.md) §5, [MCMC.md](MCMC.md) §5)
- [x] Data format standard: plain CSV with `#` header for small data, compressed `.npz` for large, Zenodo for anything over ~40 MB? **Yes, as stated** (author, 2026-10-30): full originals go to Zenodo with a DOI, and the repo ships the used columns. ([Lithium_Battery.md](Lithium_Battery.md) §5, [MCMC.md](MCMC.md) §5)
- [x] `.idea/` (7 files staged now, two with a machine-specific path): ignore all, or commit only `inspectionProfiles/` and `vcs.xml`? Commit `CLAUDE.md` and `specs/`? **Ignore `.idea/` (unstage it: `git restore --staged .idea`); commit `CLAUDE.md` and `specs/`** (author, 2026-10-30) ([BRANCH_MERGE.md](BRANCH_MERGE.md) §4.3)
- [x] Mixed uniform × Gaussian input space: added to v1 as `ac.Product` (V1-06, D-198; see §5)
- [x] Science questions only the authors can settle:
  - [x] Ebola: the paper's fitted Sierra Leone γ2 = 0.0833 (Table 2) lies outside its published range [0.1236, 0.384] (Table 7). Note it in the notebook? ([Ebola.md](Ebola.md) §11) **Decision (2026-10-30):** a typo, but the author is not sure which value is wrong. Footnote in cell 0 that the fitted SL γ2 = 0.0833 (Table 2) and the range U(0.1236, 0.384) (Table 7, used for the shipped samples) disagree, without saying which is wrong.
  - [x] MHD: the notebook follows the arXiv dimensional analysis (3 Π groups); the published version has 4 (adds Rm, Eqs. 22–24, 40). Which to present? Hartmann B_ind does not reproduce published Fig. 3 (w1 components 3 and 5); μ0 is stated in neither version ([MHD.md](MHD.md) §11) **Decision (2026-10-30):** present the published version: 4 Π groups (Re, Ha, Rm, ℓ∇p0/(v²ρ); published §3.1, Eqs. 22–24), the full governing equations (published Eq. 21), cite SAM 2017. The Hartmann B_ind / Fig. 3 and μ0 questions stay open (MHD.md §11.1).
  - [x] Elliptic_PDE: the shipped KL basis is exp(−‖s−t‖₁/(2β)); CDW eq. (5.3), the book eq. (3.80) and the MCMC paper eq. (5.5) all say exp(−‖s−t‖₁/β). Describe the data as shipped, or regenerate (which changes the meaning of every x_i and the stored chains)? ([Elliptic_PDE.md](Elliptic_PDE.md) §3) **Decision (2026-10-30):** describe the data as shipped: kernel exp(−‖s−t‖₁/(2β)) with β = 1 and .01, and a note that the papers write 1/β. No regeneration; the KL files and MCMC chains keep their meaning.
  - [x] Lithium_Battery: r_a, r_c are Normal in the published table (p. 250) but uniform in the stored data ([Lithium_Battery.md](Lithium_Battery.md) §11) **Decision (2026-10-30):** unknown. Describe the data as shipped (r_a, r_c uniform on [−1, 1] like the other inputs) and note the mismatch with the paper's Normal entries without resolving it.
  - [x] NREL_Wind: confirm the input encoding inferred from Graf et al.: Vhub, Hs, Tp as probability coordinates 2u−1; the two directions as θ/π from unwrapped von Mises draws (hence 724 rows outside [−1, 1]). Is the shipped 50,000-row file a subset of the paper's 500,000-sample 5D study? Output units? ([NREL_Wind.md](NREL_Wind.md) §11) **Decision (2026-10-30):** the inferred encoding and the 50,000-of-500,000 subset are correct. Document them as fact in the notebook and DATA.md. Output units still to document (NREL_Wind.md §11.4).
  - [x] HyShotII: the φ = 0.30 data are the JCP 2015 paper's (50 runs); `HyShotII35.txt` is the book's §5.1 data set (φ = 0.35 w equals book Table 5.2; its "Exit press 0.30" column is the book's φ = 0.30, two corner runs missing). Present both sources, or split? ([HyShotII.md](HyShotII.md) §11) **Decision (2026-10-30):** one notebook, both sources cited: φ = 0.30 as JCP 2015 (50 runs), φ = 0.35 as book §5.1 (Table 5.2). Drop the paper comparison for φ = 0.35.
  - [x] Hydrology: the shipped "updated runs" differ from the paper more than "slightly" (grass LH R² 0.82 vs 0.89, outputs ~171× scale, ranking changes); the notebook's wt default −1 has no source ([Hydrology.md](Hydrology.md) §11) **Decision (2026-10-30):** state the measured differences (R², ~171× scale, ranking changes) in place of "slightly", and call the data later model runs; no claim to reproduce the paper's figures. Keep wt = −1 m in cell 13 and say it is an illustrative choice. Output columns 3–4 (0-based) and the flux units are unknown: DATA.md says so. rl_n default 0.58 vs range U(0.35, 0.55): unresolved, note the inconsistency without picking a side
  - [x] MCMC: regenerate the chains with saved seeds and synthetic data (~1.5 h) or keep the legacy chains as unreproducible figures? ([MCMC.md](MCMC.md) §8) **Decision (2026-10-30):** regenerate with saved seeds and synthetic data (~1.5 h) at the notebook's scale (10⁵ / 10⁴ steps). Ship one `pde_mcmc_chains.npz` ≤ 10 MB and delete the 5 `.mat` files from HEAD (no history purge).

### Author interview, 2026-10-30

Every item marked "(author, 2026-10-30)" in this file was decided in the interview. Those decisions supersede the matching per-spec §11 questions and §6 rows; the specs themselves were not edited. The spec §11 fact questions below were asked in a second round the same day; "unknown" means the author does not know, and the notebook and DATA.md must say so rather than guess. Every unknown from both rounds is listed in [UNKNOWNS.md](UNKNOWNS.md); add new ones there:

- [x] NACA0012 §11.1, 1a, 2: **Unknown** (author, 2026-10-30). Flow conditions (Mach, α, Euler/RANS, SU2 version), the origin of the second batch (runs 945–1756) and the Hicks–Henne mapping of x01–x18 are not recorded; DATA.md says so. Rows 1–122 for book Fig. 5.11 stay as inferred (§3 item below)
- [x] ONERA-M6 §11.1, 2: **Unavailable / unknown** (author, 2026-10-30). The book's 60-parameter data will not be added; the 3 missing LHS runs and the FFD mapping of x01–x50 are documented as unknown
- [x] Subsurface_Permeability §11.2–4, 7 (author, 2026-10-30): keep the **Gaussian scale**. "ShortCorr" = the short correlation length of the permeability correlation model. The KL basis B and grid, "GW02", license and provenance are **unknown**: no spatial plots of w
- [x] MHD §11.2, 3 (author, 2026-10-30): the cause of the duplicate rows 47/48 (labels 49/50) is **unknown** and the Drekar output is not recoverable; dedupe and say so. Redistribution is fine; credit Sandia National Laboratories (Drekar) in the header and DATA.md
- [x] Lithium_Battery §11.3, 4, 6, 10 (author, 2026-10-30): I_1C = 30 A m⁻² and `Cap*` = seconds are **unconfirmed**: label them as inferred. The 150 concentration columns go to Zenodo only (the repo ships the used columns). Credit Hadigol et al. in the header and DATA.md. Add an errata note in the notebook (§6 battery rows)
- [x] Stomatal_Resistance §11.4, 6, 7: **Unknown** (author, 2026-10-30). Note the JHM Fig. 8 UTC−6 label vs UTC−7 (PDT) without settling it; label tran_veg mm s⁻¹ and rs s m⁻¹ as inferred from JHM Fig. 9; license and PF-CLM version unknown
- [x] HyShotII §11.2, 2a (author, 2026-10-30): the book's two φ = 0.30 corner runs are **unavailable** (mask the `0` placeholders in rows 15–16 of the 0.30 column). Book p. 77 "2.23 MJ/kg" **should read 3.23** (§6)
- [x] SingleDiodePV §11.3, 6 (author, 2026-10-30): one sentence on the constant 4.7e-5 dISC bias (cause unknown; consistent with slightly different I_SC bounds than Table 1), and a footnote listing the paper's two errors (§6)
- [x] HIV §11.3: fixed initial conditions (nominal uninfected equilibrium for every sample) **are intended** (author, 2026-10-30); state it in the notebook
- [x] Elliptic_PDE §11.6 (author, 2026-10-30): header authors **Ryan Howard and Paul Constantine**. The KL generation script is lost: describe `kl-short.npz`/`kl-long.npz` as shipped
- [x] NREL_Wind §11.3 (author, 2026-10-30): data generated by **Peter Graf, NREL**; redistribution under CC BY 4.0 is fine; credit Graf/NREL in the header and DATA.md
- [x] LM104X §11.3, 4, 4b: **Unknown** (author, 2026-10-30). Document the 310-row vs 305-retained discrepancy and the ±0.3 (data) vs ±3.0 (dissertation Eq. 2.21) bounds; no column-to-component mapping, so eigenvector plots are unlabeled by component
- [x] MCMC §11.3, 4 (author, 2026-10-30): the §6 table is the record; add the eq. (3.5) gradient-sign row. No notebook note

## 1. Branches: master + dev → main ([BRANCH_MERGE.md](BRANCH_MERGE.md))

- [x] Tag `legacy-master` (32e5839) and push it. Annotated tag pushed 2026-10-30. **No `legacy-dev` tag** (author, 2026-10-30): dev stays reachable through the merge commit's second parent
- [x] In a separate worktree: `git merge -s ours --no-commit origin/dev`, restore only `LM104X/LM104X.txt` and `LM104X/active-subspace-LM104X.ipynb` from dev (not Atacamac: author decision), commit. Fix the BRANCH_MERGE.md §2 commit message, which names Atacamac. Done 2026-10-30: branch `merge-dev` (no upstream set), merge commit f910d06; BRANCH_MERGE.md §2.1 updated. The worktree was then removed and `merge-dev` is checked out in the main checkout, so `specs/` and `CLAUDE.md` exist once, tracked
- [x] Remove `Reentry_freestream_conditions/` from main (author decision 2026-10-29; it stays in `legacy-master`). Done 2026-10-30 on `merge-dev` (8ec9b43), with its README bullet
- [x] Verify: exactly 2 added files vs master (LM104X), plus the Reentry removal; blame on `LM104X/LM104X.txt` points to 7c26e3a; dev listed as merged. Verified 2026-10-30 (also: README.md modified, LM104X differs from dev only by the omitted `.DS_Store`)
- [x] Hygiene commit: `git rm --cached ONERA-M6/figs/.DS_Store`; `git restore --staged .idea`; extend `.gitignore` (`papers/` already added locally 2026-10-29; `.venv/`, `__pycache__/`, `.mcp.json`, `.claude/settings.local.json`, `.idea/`); commit `CLAUDE.md` and `specs/` (minus `specs/prototypes/**/__pycache__`), which also commits the D-212 edit to `accept_surrogates.py`. Done 2026-10-30 on `merge-dev` as two commits (hygiene; CLAUDE.md + specs/). `.idea` unstaged in the main checkout
- [x] Open PR into master with a merge commit (never squash or rebase). [#32](https://github.com/paulcon/as-data-sets/pull/32), merged 2026-10-30 as merge commit 56aee01 (parents 32e5839, 12fa6a2)
- [x] Rename `master` → `main` on GitHub (keeps redirects and forks); update local clones. Done 2026-10-30; default branch is `main`; the local checkout tracks `origin/main`, `origin/HEAD` → `main`
- [x] Check redirects of `blob/master`, `raw.githubusercontent.com/.../master`, nbviewer URLs. Checked 2026-10-30: `github.com/…/blob/master/…` and `tree/master` redirect to `main` (200); `raw.githubusercontent.com/…/master/…` serves main's files (200). **`nbviewer.org/…/blob/master/…` returns 404**, because nbviewer uses the GitHub contents API, and `contents?ref=master` does not follow the rename. That breaks the HIV paper's link (p. 710). `nbviewer…/blob/main/…` works. **Leave it broken** (author, 2026-10-30): no `master` branch is recreated
- [x] Delete `dev`. Deleted 2026-10-30 (merged into main); `merge-dev` deleted too
- [x] Protect `main` (author, 2026-10-30): require a PR to merge, block force-pushes and deletion, no required reviewers. Done 2026-10-30, with `enforce_admins` on so the rule also applies to the owner
- [ ] Close issue #1 when v1 is released
- [ ] Author: update the external links you control from `blob/master` to `blob/main` and to the new category paths (map in [README.md](README.md#paths); Github does not redirect moved files), especially nbviewer links, which no longer resolve (activesubspaces.org, the active_subspaces README, course pages). The HIV paper's printed nbviewer link (p. 710) stays dead

## 2. Repo-wide

- [x] Directory layout (author, 2026-10-30): 12 category directories (snake_case, each with a `README.md`); data sets renamed after their papers and moved in; each notebook renamed after its directory. HyShotII in aerospace, MHD in plasma_fusion, Stomatal_Resistance in hydrology, Elliptic_PDE and MCMC in uq_benchmarks. Map in [README.md](README.md#paths). Done in [#35](https://github.com/paulcon/as-data-sets/pull/35)
- [x] README parameter counts: Ebola 9 → 8, HIV 21 → 27, Hydrology 20 → 19, Lithium_Battery 18 → 19, ONERA-M6 60 → 50. Done 2026-10-30 ([#34](https://github.com/paulcon/as-data-sets/pull/34))
- [x] README typo: "fatige" → "fatigue". Done 2026-10-30 ([#34](https://github.com/paulcon/as-data-sets/pull/34)); also "off-shore" → "offshore" and the NREL link labeled [model] (NREL_Wind.md §8.5)
- [x] README: add the missing Elliptic_PDE bullet; add an `N2_Supersonic_Jet` (ex-LM104X) bullet; remove the Reentry_freestream_conditions bullet. Done 2026-10-30 ([#34](https://github.com/paulcon/as-data-sets/pull/34)). The bullet was named **LM104X** at first; it became **N2_Supersonic_Jet** with the directory move ([#35](https://github.com/paulcon/as-data-sets/pull/35)). The Reentry bullet went in 8ec9b43
- [x] README: replace "Each directory contains a data set as a CSV file" (data are .txt, .mat, .npz, or closed-form); `nbviewer.jupyter.org` → `nbviewer.org`; any nbviewer link must use `blob/main` (`blob/master` 404s since the rename); say that the HIV paper's printed nbviewer link is dead and give the `main` URL (done 2026-10-30 in the README HIV bullet and HIV notebook cell 0); add a "2016 version" note pointing to the `legacy-master` tag. Done 2026-10-30 ([#34](https://github.com/paulcon/as-data-sets/pull/34)), with a note that unported notebooks do not run under `requirements.txt`, and Installation and License sections
- [x] README: cite published versions — Ebola: Appl. Math. Comput. 324 (2018), doi:10.1016/j.amc.2017.11.039; HIV: Math. Biosci. Eng. 14(3) (2017), doi:10.3934/mbe.2017040; Lithium_Battery: Stat. Anal. Data Min. 10(5) (2017), doi:10.1002/sam.11347; MHD: Stat. Anal. Data Min. 10(5) (2017), doi:10.1002/sam.11355; Stomatal_Resistance: J. Hydrometeorol. 18(3) (2017), doi:10.1175/JHM-D-16-0053.1. Done 2026-10-30 ([#34](https://github.com/paulcon/as-data-sets/pull/34)): DOI links, arXiv kept as a second link; all `dx.doi.org` links → `doi.org`
- [x] All notebooks: fix the Constantine–Dow–Wang title to "Active subspace methods in theory and practice: applications to kriging surfaces". Done 2026-10-30 ([#34](https://github.com/paulcon/as-data-sets/pull/34)) (13 notebooks; LM104X already had it)
- [x] All notebooks: "Palacois" → "Palacios" (NACA0012, ONERA-M6). Done 2026-10-30 ([#34](https://github.com/paulcon/as-data-sets/pull/34))
- [ ] Add `LICENSE` (author, 2026-10-30): MIT for code (notebooks, `.py`), CC BY 4.0 for data, with per-directory data credit in each `DATA.md` (Hadigol et al. for Lithium_Battery, NREL/Graf for NREL_Wind, Sandia/Drekar for MHD, Lukaczyk for LM104X, …). Confirm redistribution with the data providers named in the spec §11 questions
  - [x] `LICENSE` (MIT, so Github detects it) and `LICENSE-DATA` (CC BY 4.0 notice and link). Done 2026-10-30 ([#34](https://github.com/paulcon/as-data-sets/pull/34))
  - [ ] Per-directory credit: with the `DATA.md` item below
  - [x] Author: confirm redistribution with the providers not yet confirmed (NREL, Sandia and LM104X are confirmed). **Assume all data can be redistributed** (author, 2026-10-30)
- [ ] Notebook headers (author, 2026-10-30): authors and affiliations as in each published paper, no emails. Where there is no published paper (NREL_Wind's analysis, the Elliptic_PDE notebook), ask the author. Answered: NREL_Wind: **Andrew Glaws, Peter Graf and Ryan Howard**; Elliptic_PDE: Ryan Howard and Paul Constantine (author, 2026-10-30). Each header is set in its port PR. Done so far: Elliptic_PDE ([#38](https://github.com/paulcon/as-data-sets/pull/38))
- [x] Add `requirements.txt` (numpy, scipy, matplotlib, jupyter, active_subspaces ≥ 1.0). 1.0.0 is not on PyPI yet (account recovery): until it is, install from the `upstream-v1/src` checkout. Done 2026-10-30 ([#34](https://github.com/paulcon/as-data-sets/pull/34)): pins `active_subspaces @ git+https://github.com/paulcon/active_subspaces@ca41b3d` (`pip install --dry-run -r requirements.txt` resolves). Switch to `active_subspaces>=1.0` when 1.0 is on PyPI
- [ ] When porting, rename any use of the RBF field `f_variance` to `RBFSurrogate.training_variance` (D-206; the `fit_surrogate` keyword stays `f_variance`). Remove any warning filter added for `accuracy.flat_objective` (D-212)
- [ ] Add a `DATA.md` per directory: files, shapes, columns with units, coordinates of gradients (physical or normalized), provenance, license. It opens with the data set's kind (stored data or Python model; [README.md](README.md#two-kinds-of-data-sets)). Stored data: generating code and version, who ran it, credit, unknowns. Python model: the model module, how to regenerate, seeds, runtime, tolerance, and how the files relate to the paper's original runs. Done so far: Elliptic_PDE ([#38](https://github.com/paulcon/as-data-sets/pull/38))
- [x] Shared PDE solver: one numpy/scipy module (`PDESolver`, `PDEMisfit`) replacing numpad, copied byte-identically into Elliptic_PDE/ and MCMC/ ([numpad_replacement.md](numpad_replacement.md)). Done 2026-10-30 (branch `shared-pde-solver`): the spec's Appendix A prototype with `indz = [19, …, 79]`; `uq_benchmarks/check_pde_model.py` runs the spec §8 criteria on both copies, plus the identity and sensor-position checks (all pass, ~30 s). The callers (notebooks, `pde_study.py`, `pde_mcmc_study.py`, `generate_data.py`) are not changed: they go with each port. The direct numpad-vs-replacement comparison (spec §4c) is still not run
- [x] Two kinds of data sets (author, 2026-10-30): **stored data** (outside model; the files are the record) vs **Python model** (the model is code in the directory; data regenerable). Definitions, rules and the per-data-set table in [README.md](README.md#two-kinds-of-data-sets); a `Kind:` line at the top of each spec; tags in the top-level README; CLAUDE.md section
- [ ] Output policy: commit notebooks executed top-to-bottom with fixed seeds; optional `nbstripout --keep-output`. Done so far: Elliptic_PDE ([#38](https://github.com/paulcon/as-data-sets/pull/38))
- [x] ~~Optional: GitHub Actions job executing every notebook on PRs~~ **No CI** (author, 2026-10-30); the per-port spec §9 checks are the gate

## 3. Per-notebook ports

Each port is one PR: Python 3 kernel, numpy-only loading (tested snippets in each spec §5), v1 calls (spec §6), seeded `rng`, fixes below, executed outputs. Acceptance criteria in each spec §9.

The spec §6 call maps predate the v1 additions at `ca41b3d`. Where a spec §6 row shows a hand workaround, use the v1 call in the "v1:" item below instead (from [ACTIVE_SUBSPACES_V1_RESPONSE.md](ACTIVE_SUBSPACES_V1_RESPONSE.md), "Actions for the as-data-sets port").

### Ebola ([Ebola.md](Ebola.md))
- [ ] Remove the extra `2**-8` factor (eigenvalues are 256× too small; λ1 = 1.02, not 4e-3)
- [ ] Replace the 15-minute 8^8 loop with the paper's 8^8 tensor rule passed as `quad_weights` to `ac.active_subspace` (author, 2026-10-30); λ1 = 1.0217, peak memory ≈ 1.25 × G (V1-08). Say in markdown that it needs ~1.5 GB free
- [ ] Python model: move R0 and its gradient from the notebook into `ebola_model.py` so they can be called outside it (author, 2026-10-30)
- [ ] `n < 7` → `n < 8`; typos; runtime sentence; affiliations as in the published paper
- [ ] Cell 16 policy sentence: restore the paper's caveat that local metrics favor Liberia
- [ ] v1: replace the activity-score formula with `ac.activity_scores(sub, n=)` (V1-07) and the hand-built C with `ac.subspace_from_matrix(C, space=)` or `active_subspace(grads=, quad_weights=)` (V1-08; both give λ1 = 1.0217 at 8^8)

### HIV ([HIV.md](HIV.md))
- [ ] State that the initial conditions are fixed at the nominal uninfected equilibrium for every sample, by design (author, 2026-10-30)
- [ ] Rewrite `HIV_model.py` with `solve_ivp` (LSODA, tol 1e-10) and a T = 0.1 event; legacy vode setup fails silently in Python 3
- [ ] Use the shipped CSVs (verified to 1e-5) for the analysis; keep only a small v1 finite-difference demo at t = 24
- [ ] Mask out the frozen samples (T at the 0.1 floor; 474/1000 at t = 3400) when computing late-time subspaces, and report how many remain at each time (author, 2026-10-30)
- [ ] Load CSVs with `np.loadtxt(skiprows=1, delimiter=",")[:, 1:]`, then re-save without the pandas index column
- [ ] Port `generate_data.py` to `solve_ivp` (LSODA, tol 1e-10, T = 0.1 event) (author, 2026-10-30); it must reproduce the shipped CSVs to 1e-5. Document that the data are a Python regeneration, not the paper's MATLAB runs (ode23s, forward FD step 1e-6 on normalized parameters)
- [ ] Times (author, 2026-10-30): the paper's 18 (Appendix B: 24 … 3400) plus t = 1700 as the worked example (Figs. 2–6); drop 5, 15 and 750. The shipped CSVs have no t = 1700, so add it with the ported `generate_data.py` (~45 min) and re-save the CSVs
- [x] ~~Keep the path `HIV/HIV.ipynb`: the published paper links to it on master (p. 710); after the rename, check that the redirect works~~ Moot: the paper's link is an nbviewer `blob/master` URL, which 404s since the rename, and the author moved the notebook to `epidemiology_systems_biology/HIV_Long_Term_Dynamics/HIV_Long_Term_Dynamics.ipynb` (2026-10-30). The README and notebook cell 0 give the new URL
- [ ] Negligible parameters (author, 2026-10-30): state the list the data support (K11, K12, K13 plus K2, K8, δ3, α1 with max |w1| ≤ 0.0014; recheck on the final time set) and note that the paper (p. 726) names only K11–K13
- [ ] Do not repeat the paper's t = 2000 "2-D subspace" claim (data give λ1/λ2 = 75); describe the late-time trend as the paper's "arctangent × step" (p. 723) in cell 10
- [ ] v1: the finite-difference demo makes one `ac.finite_difference_gradients_multi(model, X=, space=, outputs=21)` call, not 21 scalar calls (V1-09); replace the hand sign flips with `ac.orient`, chaining `by="reference", reference=sub_prev` across time steps (V1-05)

### MHD ([MHD.md](MHD.md))
- [ ] Remove the extra `2**-5` factor (Hartmann eigenvalues 32× too small)
- [ ] Text: inner (not outer) product in the eigenvalue-sum identity; ∇_z not ∇_x; momentum equation as published Eq. 21 (arXiv Eq. 20); u/U notation; affiliations as in the published paper
- [ ] `gradients.py`: `np.zeros` not `0.0*np.empty`; gradients w.r.t. log inputs; no star import / module globals. Rename it `hartmann_model.py` (author, 2026-10-30; the Python-model naming rule, [README.md](README.md#two-kinds-of-data-sets)) and import it by name; the generator part stays stored data
- [ ] Dedupe u_avg rows 47/48 (cause unknown, author 2026-10-30); document the index gaps; credit Sandia (Drekar) in the header
- [ ] Hartmann B_ind (author, 2026-10-30): show the notebook's result with ℓ = μ0 = 1 stated, and note that it does not match published Fig. 3 (w1 components 3 and 5, eigenvalue scale); how Fig. 3 was computed is unknown
- [ ] v1: replace the hand sign flips with `ac.orient` (V1-05)

### Elliptic_PDE ([Elliptic_PDE.md](Elliptic_PDE.md))
- [x] ~~Replace `pde_model.py` with the shared solver~~ (done in §2); delete `pde_study.py`. Done 2026-10-30 ([#38](https://github.com/paulcon/as-data-sets/pull/38)); the notebook's solver-check cell replaces it. Added `PDESolver.qoi_and_grad` (one LU for value and gradient) to both copies, checked by `check_pde_model.py`
- [x] Fix the SENS off-by-one (β = 1 picks x1 twice; β = .01 uses x5 not x6) by choosing SENS from the gradient at the origin as in the paper (author, 2026-10-30). Done ([#38](https://github.com/paulcon/as-data-sets/pull/38)): ∇f(0) gives x1, x2 (β = 1; the paper's x3 is the other member of the degenerate pair γ2 = γ3) and x1, x6 (β = .01, as the paper)
- [x] Keep the legacy QoI weights (end nodes half-weighted, column s1 = 0.99) (author, 2026-10-30); the shared solver must reproduce them. Done ([#38](https://github.com/paulcon/as-data-sets/pull/38)): f(0) = 0.07608
- [x] Refit instead of setting `.ell = 1` after training. **Superseded for β = 1** (author, 2026-10-30, [#38](https://github.com/paulcon/as-data-sets/pull/38)): with `f_variance` the likelihood puts η at the upper end of the CDW interval (ℓ1 = 2.0), and the β = 1, n = 2 AS surface then loses to the corrected SENS surface (mean relative error at the 300 samples 0.294 vs 0.227), against the paper's "in all cases". So β = 1 fixes ℓ1 = 1 with `log10_eta = -log10(Σ|λ|/λ1)` and refits (0.154 / 0.160 at the 300 samples, paper 0.178 / 0.149; 0.133 / 0.154 at the 500 test points); β = .01 uses `f_variance` (7.96e-3 / 8.32e-3, `accuracy.empty_bounds` printed). The original item: v1: `fit_surrogate(..., kind="rbf", f_variance=f0.var())` uses the CDW (4.13) bounds (V1-01) and replaces the hand `log10_eta` of spec §6 line 138 and §8 line 161, and settles open question §11.3. Targets: β = 1 mean relative error 0.095 (n = 1) and 0.300 (n = 2); β = .01 0.008 with `accuracy.empty_bounds`. The n = 1 default no longer warns `accuracy.flat_objective` (D-212). C₁ = 4/π² answers §11.4 (D-193, D-204), and the docs now cite CDW 2014 §4.2–4.3, which answers §11.5
- [x] v1: replace the hand sign flips with `ac.orient` (V1-05). Done ([#38](https://github.com/paulcon/as-data-sets/pull/38)): the legacy notebook had no hand flips; `ac.orient(by="reference")` against e1 restores the 2016 sign rule (first component positive) for w1, w2, so the eigenvector figure matches the 2016 one (v1's default mirrors the β = 1 w2)
- [x] Text: correlation function / β factor, √γ_i, finite-volume (not finite-element) description, w_i/E typos. Done ([#38](https://github.com/paulcon/as-data-sets/pull/38)), with C1 = 1 (Gaussian) and 4/π² (uniform), the v1 kernel parametrization (η = 2α), the 3M + P equal-cost rule, the cell-21 claim about the paper's kriging code removed, and references as DOIs. New finding: the shipped `kl-short` KL ratios do not match CDW Table 1 (0.9894 vs 0.9946 for λ2/λ1), while `kl-long` does; stated in the notebook and DATA.md
- [x] Header: Ryan Howard and Paul Constantine (author, 2026-10-30); say that the KL generation script is lost and the `kl-*.npz` files are described as shipped. Done ([#38](https://github.com/paulcon/as-data-sets/pull/38)); names only, no affiliations or emails. `DATA.md` added; outputs committed from a seeded run (two runs give identical output, ~2 min)

### MCMC ([MCMC.md](MCMC.md))
- [ ] Shared solver (installed in §2; switch the callers from `PDESolver` to `PDEMisfit`); port the sampler (cached ĝ per Algorithm 1, log acceptance, `rng`, burn-in)
- [ ] Rewrite `generate_data.py`; save seeds, x_true, d, σ², sample points; replace the .mat files with one npz ≤ 10 MB (notebook needs < 4 MB of the 80 MB chain)
- [ ] Cell 35: reconstruct with the stored W, not one recomputed from different data; pass `n=2`
- [ ] Quadratic example: Gauss–Hermite ĝ; fix the gradient sign. Keep σ = 0.1 and proposal std 0.1: the code is right and the paper's "σ² = 0.1" and "variance" are typos (§6)
- [ ] Sensors at the paper's s2 = 0.2 … 0.8 (author, 2026-10-30): `indz = [19, 29, …, 79]` (`indz` indexes unknown rows, which are offset by one vertex; numpad_replacement.md §2). Change `PDEMisfit.indz` before regenerating (done in §2: the shared solver has `[19, …, 79]`). The sensors stay at s1 = 0.99, not the paper's s1 = 1: say so
- [ ] Text: c_pos vs c_pos⁻¹; L·C; author email as in the paper
- [ ] v1: replace the hand sign flips with `ac.orient` (V1-05)

### HyShotII ([HyShotII.md](HyShotII.md))
- [ ] Exclude the 2 validation runs from the bootstrap (50 rows, not 52)
- [ ] The book's φ = 0.30 corner runs are unavailable (author, 2026-10-30): store them as missing, not `0`, in the rewritten HyShotII35 file, and say so
- [ ] Rewrite the data files (author, 2026-10-30): physical units, a `#` header with names and units, a validation-run flag column; record the normalization bounds in DATA.md (HyShotII30.txt is now fully normalized with no header; HyShotII35.txt mixes physical rows with 2 normalized rows)
- [ ] Python 3 division in `1/M30`; define `xmin = -sign(w)`; exact quadratic prediction limit via root-find; vectorize the CDF
- [ ] Cite the book (§5.1, Table 5.2) for the φ = 0.35 data alongside JCP 2015; note the Table 2 / book Table 5.1 typo (0.1225 m, not 0.1325); add units
- [ ] Safe box: with `xmin = -sign(w)` the φ = 0.35 box becomes AoA ≤ 3.193; spec §9 lists both reference values
- [ ] v1: bootstrap with `active_subspace(..., nboot=B, bootstrap_vectors=k)` and plot with `plot_eigenvectors(..., interval=)`, replacing the hand bootstrap and alignment (V1-02). Get the prediction limit from `ac.predict_interval(poly, X, level=0.99, side="upper")` (V1-03)
- [ ] v1 uses Student's t (AoA ≤ 3.179); the paper uses the normal quantile 2.33 (AoA ≤ 3.193). Rescale the half-width by `st.norm.ppf(0.99) / st.t.ppf(0.99, iv.dof)` as `accept_surrogates.py` does, and explain the difference in markdown (D-211)

### NACA0012 ([NACA0012.md](NACA0012.md))
- [ ] Fix the cell-3 comment: gradients are physical, not normalized (pass with the physical space)
- [ ] DATA.md: flow conditions, the second batch (runs 945–1756) and the bump mapping of x01–x18 are unknown (author, 2026-10-30)
- [ ] Drop the duplicate nominal run 945 from the fits and say so (author, 2026-10-30); drag w₂ sign flip under v1; "last 9 parameters" holds for drag only
- [ ] The book (ch. 5) used the first M = 122 rows; say so, and match book Fig. 5.11 on those rows
- [ ] v1: replace the hand sign flips with `ac.orient` (V1-05)

### ONERA-M6 ([ONERA-M6.md](ONERA-M6.md))
- [ ] Rewrite the R² text with v1 leave-one-out values (drag 0.70 / 0.88 / 0.93 for n = 1/2/3, not ≈ .965); print `ac.dimension_criterion`'s choice beside the eigenvalue plot, then set n = 2 for drag explicitly (author, 2026-10-30)
- [ ] Reference [1]: author order as on the paper's title page (Lukaczyk, Palacios, Alonso, Constantine; the cell-0 author block is already right) and "Palacios"; add the missing [R3] link
- [ ] README: 50 parameters; the "60" is book §5.3.2, a different study not in this repo (its data are unavailable, author 2026-10-30)
- [ ] DATA.md: the 3 missing LHS runs and the FFD mapping of x01–x50 are unknown (author, 2026-10-30)

### Reentry_freestream_conditions: excluded from main
Not ported (author decision 2026-10-29). The spec stays for reference: [Reentry_freestream_conditions.md](Reentry_freestream_conditions.md).

### Hydrology ([Hydrology.md](Hydrology.md))
- [ ] Fix the quadratic confidence/prediction bands (currently a straight-line leverage formula). v1: `ac.predict_interval(..., kind="confidence" | "prediction")` (V1-03)
- [ ] v1: replace the hand normalization and Gaussian-reference workaround (spec §6 line 91, §10 line 126, §11 Q1) with `ac.Product` blocks in column order. log z0m is column 16: `[Uniform(YL[:16], YU[:16]), Gaussian([-5.11], [0.52**2]), Uniform(YL[16:], YU[16:])]` (V1-06). No `active_domain`, integration or optimization on a Product
- [ ] Flag the nominal point outside the sampled box (rl_n 0.58 vs U(0.35, 0.55): say the paper is inconsistent, not which side is wrong); remove duplicate z0m row in the table; document output columns 3–4 (0-based) and the flux units as unknown
- [ ] Orient w = â/‖â‖ as the paper does (legacy bare GH, grass SH and grass GH panels are mirrored); place annotations from the data. v1 OLS now returns this sign, so drop the `np.sign(w @ a_hat)` of spec §6 line 93 (V1-04)

### Stomatal_Resistance ([Stomatal_Resistance.md](Stomatal_Resistance.md))
- [ ] Drop row 0 (the separate default run) from the fits; wj1 ∈ {1, 4, 4.5}, wj2 ∈ {2, 8, 10.5} are discrete (JHM Table 1); ppcd 355–400 ppm in the paper, stored as mol/mol
- [ ] Replace the hard-coded October sign flip with orientation w = â/‖â‖ (the paper's convention; every legacy resistance panel is mirrored against JHM Fig. 9); compute each subspace once (234 fits → 78). v1 OLS returns this sign, so drop the `np.sign(w @ a_hat)` of spec §6 line 97 (V1-04)
- [ ] Temper "fairly tight trends" (linear R² 0.37–0.99); cite JHM 2017 as the primary reference, the thesis as secondary; no new sites or years (author, 2026-10-30)
- [ ] Add a boxplot of each output by wj1 / wj2 level; say in markdown that their linear weight only summarizes a discrete effect (author, 2026-10-30)
- [ ] Note the JHM Fig. 8 UTC−6 label vs UTC−7 (PDT) without settling it; label tran_veg mm s⁻¹ and rs s m⁻¹ as inferred from JHM Fig. 9; DATA.md: license and PF-CLM version unknown (author, 2026-10-30)

### Subsurface_Permeability ([Subsurface_Permeability.md](Subsurface_Permeability.md))
- [ ] Mask the 3,137 zero Hortonian runoffs before log10, and fit the Hortonian OLS on the 6,863 non-zero runs only, as the paper did (author, 2026-10-30); report the count
- [ ] Orient the Hortonian w by â (paper Fig. 7a, w₁ ≈ −0.65); note that the paper's figures appear to rescale inputs to [−1, 1] by sample min/max (spec §3). v1 OLS returns this sign, so drop the `np.sign(w @ a_hat)` of spec §6 line 91 (V1-04)
- [ ] Replace the meaningless 3-D partition of rank-1 OLS with 1-D OLS vs 3-D QPHD comparison, labeled as an extension not in the paper (author, 2026-10-30); rewrite cell 15 (v1 QPHD: eigenvalues [12252.52, 630.56, 424.28, 363.64], ~18 s; V1-11)
- [ ] Inputs are standard Gaussian: pass with `ac.Gaussian.reference(100)`
- [ ] No spatial plot of w: the KL modes and grid are unavailable (author, 2026-10-30). DATA.md: "ShortCorr" = short correlation length of the permeability model; GW02, license and provenance unknown

### Lithium_Battery ([Lithium_Battery.md](Lithium_Battery.md))
- [x] Rewrite `battery_figs.py` (~1,170 lines) as ~150 lines of Python 3/numpy with seeded rng. v1: OLS returns the signed direction, so drop the `sign(cov(Xw, f))` alignment (V1-04); orient QPHD with `ac.orient` (V1-05); bootstrap with `bootstrap_vectors=` and `plot_eigenvectors(..., interval=)` (V1-02); build spaces with `Uniform.reference(m, names=)` (V1-12). Done ([#48](https://github.com/paulcon/as-data-sets/pull/48)): 151 lines; v1 OLS signs need no flip (weights = 2016 lstsq to 2.6e-13); QPHD (paper Fig. 8, new) oriented with `orient(by="covariance")`; `Uniform.reference(19, names=)`. Bootstrap: v1 `bootstrap_vectors=` at the 18 A/B/C points and in a `plot_eigenvectors(..., interval="percentile")` figure; the Fig. 9 curves use a shared-resample `lstsq` bootstrap (`bootstrap_se`), because v1 at all 294 points took 271 s on a loaded machine (notebook now ~40 s). **Keep the shared-resample bootstrap with the v1 cross-check at 18 points** (author, 2026-10-30); keep the band figure: 15 figures, spec §9 updated (author, 2026-10-30); the QPHD covariance orientation and the ±2.2 summary limits are accepted (author, 2026-10-30)
- [x] Replace the silent drop of 4C row 1537 with one stated rule that drops both cutoff overshoots (author, 2026-10-30): rows 1537 (Run 1538, 2.746 V, z = 59) and 2968 (Run 2969, 2.791 V, z = 9.5). For example, |z| > 5 on 4C `Vol50` selects exactly these two (spec §10). Store `dropped_runs` in the npz, print N = 3598 for 4C, and note the paper's N = 3600. Done ([#48](https://github.com/paulcon/as-data-sets/pull/48)): rule "final voltage more than 1 mV below 2.8 V" (asserted equal to |z| > 5); `dropped_runs_4C` = [1538, 2969] in the npz; N = 3598 printed; paper N = 3600 and the 2016 silent drop noted. The > 1 mV rule is accepted (author, 2026-10-30)
- [x] Keep the coded t* grid, column k at t* = 99 − 2k, and the current axis labels (author, 2026-10-30). Done ([#48](https://github.com/paulcon/as-data-sets/pull/48))
- [x] Fix the bootstrap "all below 0.010" and 4C porosity claims; the active-subspace definition. Done ([#48](https://github.com/paulcon/as-data-sets/pull/48)): statements computed from the run (max SE 0.0099 here, unclipped; 4C voltage: ε_s, then D, largest)
- [x] Convert 67 MB of text to an ~8.6 MB npz of the used columns (concentration columns to Zenodo only); cite and credit Hadigol et al. 2015 as data source in the header and DATA.md (author, 2026-10-30). Done ([#48](https://github.com/paulcon/as-data-sets/pull/48)): `battery_data.npz` 8.6 MB by `make_battery_npz.py` (checked value by value; `Cap*` kept raw, in s, and converted in the notebook (author, 2026-10-30)). Header: published affiliations, Ryan Howard on a separate "(2016 notebook)" line (author, 2026-10-30); Acknowledgments add the published paper's funding text verbatim (author, 2026-10-30). The text files stay until the Zenodo DOI exists; nothing uploaded
- [x] Label I_1C = 30 A m⁻² and `Cap*` = seconds as inferred, not confirmed; add an errata note for the paper (p. 259 region c → b; pp. 253–254 ε_a sign and brugg_a/L_a; p. 250) (author, 2026-10-30). Done ([#48](https://github.com/paulcon/as-data-sets/pull/48)): the errata note also lists p. 252 N = 3600, Table 1 r_a/r_c, and a new one: eq. (14) prints ggᵀ + H², but Fig. 8 matches ggᵀ + HH/3 (uniform covariance)

### NREL_Wind ([NREL_Wind.md](NREL_Wind.md))
- [ ] "nearly 500,000 pairs" → 50,000, a subset of the paper's 500,000-sample 5D study (Graf et al. p. 7)
- [ ] Space from data bounds (inputs are not on [−1, 1]); 2,000 seeded gradient centers; choose each output's n with `ac.dimension_criterion`, print it, then pass n explicitly. This replaces the unexplained (1, 1, 3, 3, 2, 2, 2) (author, 2026-10-30)
- [ ] Document the input encoding, outputs (edgewise blade-root moment, side-to-side tower-base moment, anchor tensions; DELs with m = 4 for all outputs) and units. Keep m = 4 and note that Graf et al. use m = 10 for the composite blade root (author, 2026-10-30)
- [ ] Credit Peter Graf / NREL in the header and DATA.md; redistribution under CC BY 4.0 confirmed (author, 2026-10-30). Header authors: Andrew Glaws, Peter Graf, Ryan Howard (author, 2026-10-30)
- [ ] State that the active-subspace analysis is unpublished: Graf et al. 2016 is the source of the model and data only; compare with its grid-coarsening findings (Figs. 6–7)
- [ ] Framing: the contrast is fixed-bottom vs floating, not land vs offshore
- [ ] v1: the `Uniform.reference(5)` failure (spec §10 line 117) now names the input, value, side and bound; `details["n_outside"] == 724`, `details["name"] == "PlatformDir"` (V1-10). Update the expected message

### SingleDiodePV ([SingleDiodePV.md](SingleDiodePV.md))
- [ ] I_SC upper bound typo `0.23598` → `0.23958`
- [ ] Bootstrap on a seeded M = 1000 subsample with `interval="percentile", level=0.99` to match the paper's figures (author, 2026-10-30)
- [ ] Add `single_diode.py` (~25 lines, from `prototypes/g5/pv_model.py`: `pmax` via Lambert W, `to_physical`, `to_normalized`) and a cell that checks it against the shipped data (author, 2026-10-30)
- [ ] Stored gradients are normalized: divide by (xu − xl)/2 before passing with the physical space (or use `Uniform.reference(5)`)
- [ ] Gap statement: λ_n − λ_{n+1}, not λ_{n+1} − λ_n; add units and V_th
- [ ] One sentence on the constant 4.7e-5 dISC bias (cause unknown); a footnote listing the paper's gap-wording (p. 266) and cosine/sine (p. 269) errors (author, 2026-10-30)
- [ ] v1: bootstrap with `bootstrap_vectors=` and `plot_eigenvectors(..., interval=)` (V1-02); `physical_directions()` now keeps the signs of W (V1-05), so recheck spec §6 line 77

### Atacamac ([Atacamac.md](Atacamac.md)): excluded from main
Not ported: the author does not want the model in main (2026-10-30). It stays in history. The items below stay for reference only.

- [ ] Fix the chain-rule sign error in cell 13 (strong-case λ₂/λ₁ .457 → .318)
- [ ] f_Im > 0 means unstable; "S. Nicoud" → "F. Nicoud"
- [ ] Document that the data are Gaussian while both papers state uniform inputs (JFM Table 2; CTR 2014 p. 213); report data, surrogate and paper risk factors side by side
- [ ] Add Acknowledgments; also cite Bauerheim et al., CTR Summer Program 2014, pp. 209–218
- [ ] v1: `Uniform.reference(m, names=)` for any reference box (V1-12); a diagonal `Gaussian` cov is stored as (m,) variances (D-214)

### LM104X → N2_Supersonic_Jet ([LM104X.md](LM104X.md)) — after merge (data release cleared 2026-10-30)
- [ ] 105 inputs (wing 50, tail 24, aft deck 16, fuselage 15; dissertation fig. 2.20, p. 42), not 50; the column-to-component mapping is unknown (author, 2026-10-30), so do not label eigenvector components
- [ ] Cite the final dissertation (June 2015, purl.stanford.edu/xx611nd3190, CC BY-NC 3.0 US); the final is now in `papers/`
- [ ] Compare with dissertation fig. 4.14 (p. 108): inputs normalized to [−1, 1] and each output min–max scaled to [−1, 1]; the data then give λ₁ lift 0.19, drag 0.084, EA 1.4e-3. State the output scaling, since v1 does not apply it
- [x] Rename the directory to `N2_Supersonic_Jet` (author, 2026-10-30) with `git mv` in the port PR, not in the merge commit, so the §1 merge still restores dev's paths. Rename the notebook to match and check that `git log --follow` crosses the rename. Done 2026-10-30 in the directory move ([#35](https://github.com/paulcon/as-data-sets/pull/35)), not the port PR: `aerospace_cfd/N2_Supersonic_Jet/N2_Supersonic_Jet.ipynb`. `LM104X.txt` keeps its name
- [ ] State that the 310 rows vs 305 retained (dissertation) and the ±0.3 data vs ±3.0 Eq. 2.21 bounds are unexplained (author, 2026-10-30)
- [ ] Add a gradient-quality cell (author, 2026-10-30) (adjoint direction ~25° off the exact linear regression direction; the dissertation reports 9.9 % lift vs 2.0 % drag adjoint-gradient error on NACA0012, Table 3.4 p. 72, and no N+2-specific gradient check)

## 4. Verification before closing each port

- [ ] Reproduce the legacy printed numbers from shipped data first (exact match), then apply fixes and record each change
- [ ] Every number quoted in markdown is printed by the run
- [ ] Spec §9 acceptance criteria all pass

## 5. Upstream: active_subspaces v1

Done. All of V1-01 to V1-13 from [ACTIVE_SUBSPACES_V1_CHANGES.md](ACTIVE_SUBSPACES_V1_CHANGES.md) landed on `paulcon/active_subspaces` `main` at `ca41b3d` (PRs #58–#61, 2026-10-30). The editable install in `.venv` imports that commit. What shipped, and where it differs from the proposal: [ACTIVE_SUBSPACES_V1_RESPONSE.md](ACTIVE_SUBSPACES_V1_RESPONSE.md). The changes for each notebook are the "v1:" items in §3.

- [x] V1-01 RBF bounds: `fit_surrogate(..., f_variance=)`; C₁ = 4/π² (uniform), 1 (Gaussian, Product)
- [x] V1-02 eigenvector bootstrap replicates: `bootstrap_vectors=`, `plot_eigenvectors(..., interval=)`
- [x] V1-03 polynomial intervals: `ac.predict_interval` (Student's t)
- [x] V1-04 OLS returns the signed û/‖û‖
- [x] V1-05 `ac.orient`; `physical_directions` keeps the signs of W
- [x] V1-06 `ac.Product` (no `order=`; blocks in column order)
- [x] V1-09 `ac.finite_difference_gradients_multi` (a separate function, not `outputs=`)
- [x] V1-07, V1-08, V1-10 – V1-13
- [x] Commit the D-212 edit to `prototypes/v1report/accept_surrogates.py` (in the §1 hygiene commit with the rest of `specs/`). Done 2026-10-30 on `merge-dev` (PR #32)
- [ ] Update the per-notebook spec expectations that relied on the old behavior: the Elliptic `flat_objective` warning (D-212), the out-of-space message text (V1-10, NREL_Wind §10), the signs of physical directions (V1-05), and the shape of `Gaussian.cov` for a diagonal matrix (D-214). Alternatively, treat the §3 "v1:" items as superseding the spec rows
- [ ] Re-run the three acceptance scripts in `prototypes/v1report/` after any upstream change. Expected: subspaces 20 passed, 1 skipped; surrogates 17 passed; spaces 21 passed. Since Reentry left main (8ec9b43), spaces gives 19 passed, 1 skipped (`V1-06 reentry`: data file missing); last run 2026-10-30 after the directory move

## 6. Errata in the published sources (for the author's records)

Found while checking the notebooks against the local PDFs. None needs a notebook change unless noted; the notebooks must not copy these.

| Source | Location | Error | Evidence |
|---|---|---|---|
| Glaws et al., SAM 10(5) 2017 (MHD) | Fig. 2A, p. 320; p. 319 | Hartmann eigenvalues carry the same 1/32 factor as the notebook (λ1 ≈ 1.4, true 36.6); the "three orders of magnitude" gap is ≈ 28× | [MHD.md](MHD.md) §3 |
| Glaws et al. (MHD) | p. 316 | "at least n+1" should be "at most" | [MHD.md](MHD.md) §3 |
| Diaz et al., AMC 324 2018 (Ebola) | λ1/λ2 sentence; Fig. 7 p. 151 | "nearly two orders" gap is ≈ 4× (Liberia), 7× (Sierra Leone) | [Ebola.md](Ebola.md) §3 |
| Diaz et al. (Ebola) | Conclusions p. 153 | names γ1 as negligible; Fig. 9 and recomputation say ρ1 | [Ebola.md](Ebola.md) §3 |
| Diaz et al. (Ebola) | Tables 2 and 7 | fitted SL γ2 = 0.0833 outside the stated range [0.1236, 0.384] (a typo per the author; which value is unknown) | [Ebola.md](Ebola.md) §3 |
| Constantine, Kent, Bui-Thanh, SISC 2016 (MCMC) | p. A2789 | quadratic example "σ² = 0.1" should be σ = 0.1 (only σ reproduces Fig. 5.1 and 12% acceptance) | [MCMC.md](MCMC.md) §3 |
| Constantine, Kent, Bui-Thanh (MCMC) | eq. (3.5), p. A2784 | wrong sign of ∇f for f = ‖d − m‖²/(2σ²); the linear case eq. (3.15), p. A2786, has the right sign | [MCMC.md](MCMC.md) §6 row 3 |
| Constantine, Kent, Bui-Thanh (MCMC) | proposal; Table 5.1 | proposal 0.1 is a standard deviation, not a variance (60% acceptance); Table 5.1 column labels swapped | [MCMC.md](MCMC.md) §3 |
| Constantine, Dow, Wang, SISC 2014 | eq. (5.3) | missing minus sign in the correlation function | [Elliptic_PDE.md](Elliptic_PDE.md) §3 |
| Constantine, Emory, Larsson, Iaccarino, JCP 2015; book Table 5.1 | Table 2 | turbulence length-scale lower bound 0.1325 m should be 0.1225 m | [HyShotII.md](HyShotII.md) §3 |
| Constantine, *Active Subspaces*, SIAM 2015 | p. 77 | "stagnation enthalpy greater than 2.23 MJ/kg" is outside the input range (starts at 3.0551); should read 3.23 (author, 2026-10-30) | [HyShotII.md](HyShotII.md) §3 |
| Constantine, Doostan, SAM 2017 (battery) | p. 259 | "region c in Figures 9A and 9C" should be region b | [Lithium_Battery.md](Lithium_Battery.md) §3 |
| Constantine, Doostan (battery) | pp. 253–254 | ε_a weight called positive and brugg_a named; Fig. 2A shows ε_a ≈ −0.77, L_a ≈ +0.64 | [Lithium_Battery.md](Lithium_Battery.md) §3 |
| Constantine, Doostan (battery) | p. 250; p. 252 | "capacity inversely proportional to cell voltage" unsupported; "N = 3600" while the 4C script drops run 1538 | [Lithium_Battery.md](Lithium_Battery.md) §3 |
| Constantine, Zaharatos, Campanelli, SAM 2015 (PV) | p. 266; p. 269 | gap written λ_{n+1} − λ_n (should be λ_n − λ_{n+1}); subspace error called the cosine of the principal angle (it is the sine) | [SingleDiodePV.md](SingleDiodePV.md) §3 |
| Jefferson et al., Comput. Geosci. 2015 (Hydrology) | Table A1; p. 13; Table 2 | quadratic R² below linear R² for bare GH (impossible for nested in-sample fits); "all 19 inputs uniform" contradicts eq. 14; default rl_n = 0.58 outside range 0.35–0.55 (unresolved: author unsure which is wrong) | [Hydrology.md](Hydrology.md) §3 |
| Jefferson, Maxwell, Constantine, JHM 2017 | Fig. 8 | local time labeled UTC−6; California summer time is UTC−7 (not confirmed by the author) | [Stomatal_Resistance.md](Stomatal_Resistance.md) §3 |
| Loudon, Pankavich, MBE 14(3) 2017 (HIV) | Fig. 7 p. 721; Fig. 2 p. 717; pp. 726–727 | λ2 ≈ λ1 at t = 2000 (data: λ1/λ2 = 75); eigenvalue tail 1–2 orders too high; negligible-parameter list incomplete; K3 early weight 0.1256 vs 0.203 (likely from noisier FD gradients) | [HIV.md](HIV.md) §3b |
| Loudon, Pankavich (HIV) | pp. 717, 723 | "vanish" / "T = 0" impossible under Eq. 1 (dT/dt = s1 > 0 at T = 0) | [HIV.md](HIV.md) §3b |
| Lukaczyk, PhD dissertation, final June 2015 (LM104X) | Figs. 4.15–4.16 pp. 109, 112; Eq. 2.21 p. 39; figs. 2.16 p. 38, 2.19 p. 41 | figs. 4.15–4.16 use raw ±0.3 inputs, contradicting §4.2 (pp. 78–79) and fig. 4.14; Eq. 2.21 bounds ±3.0 vs ±0.3 in data; "Passengjer" in two captions | [LM104X.md](LM104X.md) §3 |
| Graf et al., Wind Energy 2016 (NREL_Wind; accepted manuscript) | p. 2; p. 5 | unresolved "TurbSim (reference)"; "σ² is the variance of α"; placeholder report number and DOI | [NREL_Wind.md](NREL_Wind.md) §3 |
| Crossref metadata for AIAA 2014-1171 (ONERA-M6) | author list | order differs from the paper's title page | [ONERA-M6.md](ONERA-M6.md) §3 |
