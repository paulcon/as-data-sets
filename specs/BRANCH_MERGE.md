# Branch merge plan: master + dev → main

Repo `paulcon/as-data-sets`, checked 2026-10-29 with read-only commands.
- Refs: `origin/master` = `32e5839`, `origin/dev` = `d3d31e6`, merge-base `da61ac5`.
- GitHub: default branch `master`, unprotected. 12 forks, no tags, **no open PRs** (#2–#31 all merged or closed). One open issue: #1 "Cannot access python library `active_subspaces`" (2015).

## 0. Corrections to the facts as briefed
- **README.** Dev did not replace README.md. The one-line `README` is the original file, present at the fork point `da61ac5`. Master renamed it to `README.md` and expanded it in `e75c732`. Keep master's `README.md`.
- **Dev-only directories are unchanged since the fork.** `Atacamac`, `ICS-4reac`, `LM104X`, `Reentry`, `SNL-CLM`, `SNL-LES` on dev are byte-identical to master at `9c1b8a7^` (the commit before master deleted them). The only exception is `ICS-4reac/.DS_Store`.
  - Dev's own commits only touch `ICS-3reac` (8dd5812) and `Zonotope_Vertices` (0b0aa19, ec30b53, d1053dc).
- **Dev's `ICS-3reac/active-subspace-ICS3.ipynb` is not a notebook.** It is a saved GitHub HTML page (`<!DOCTYPE html>…`, links to `blob/master`, `blob/dev`), and `json.load` fails on it. The last valid version is at `da61ac5` / `f158474`.
- **`SNL-CLM/IJUQ-6821.pdf` is already in master's history.** It was added in the initial commit `7c26e3a` and deleted in `9c1b8a7`, so leaving it out of main's tree does not remove it from the repository (§4.4).
- **No in-repo link points to a branch.** `git grep` for `blob/master|tree/master|/master/|raw/master` found nothing on master or dev, apart from the broken ICS-3 HTML file.

## 1. Inventory and decisions

| path | master | dev | decision |
|---|---|---|---|
| `.gitignore` | yes | yes (same) | keep master's and extend it (§4.2) |
| `README.md` | yes (full) | — | keep master's; add bullets for directories that get included |
| `README` | — (renamed) | yes (1 line, original) | drop |
| `Ebola`, `HIV`, `HyShotII`, `Hydrology`, `Lithium_Battery`, `MCMC`, `MHD`, `NACA0012`, `NREL_Wind`, `ONERA-M6`, `SingleDiodePV`, `Stomatal_Resistance`, `Subsurface_Permeability` | yes (newer) | yes (older) | keep master's |
| `Reentry_freestream_conditions` | yes (newer) | yes (older) | **remove from main** (author decision 2026-10-29; stays reachable through `legacy-master`) |
| `Elliptic_PDE` | yes | — | keep |
| `ONERA-M6/figs/.DS_Store` | tracked | tracked | delete (§4.1) |
| `Atacamac` | removed in 9c1b8a7 | yes | **include** (spec `specs/Atacamac.md`) |
| `LM104X` (without `.DS_Store`) | removed in 9c1b8a7 | yes | **include, gated on data-release confirmation** (spec `specs/LM104X.md`) |
| `ICS-3reac` | removed in bf005dd | yes (broken HTML notebook) | drop; include later only with a citable source |
| `ICS-4reac` | removed in 9c1b8a7 | yes | drop; include later only with a citable source |
| `Reentry` | removed in 9c1b8a7 | yes | drop (same study as `Reentry_freestream_conditions`, which is also excluded) |
| `SNL-CLM` (incl. `IJUQ-6821.pdf`) | removed in 9c1b8a7 | yes | drop; never bring the PDF |
| `SNL-LES` | removed in 9c1b8a7 | yes | drop; include later only after an AS paper |
| `Zonotope_Vertices` | — | yes (dev only) | drop from this repo; superseded by v1 `ac.active_domain` (§1.6) |
| `.idea/` (7 files **staged** in the local index) | untracked | — | **author decides** (§4.3) |
| `CLAUDE.md` | untracked | — | author decides; commit it if it should guide contributors (update its branch wording after the rename) |
| `.mcp.json` | untracked | — | ignore it (machine-specific: `http://127.0.0.1:64462/stream`) |
| `.claude/` | ignored | — | keep ignored |
| `.venv/` | self-ignored (its own `.gitignore: *`) | — | also add to `.gitignore` |
| `specs/` | untracked | — | author decides whether the specs are committed; a `docs/specs/` or a separate branch is fine |

### 1.1 ICS-3reac / ICS-4reac: drop, include later only with a source
- Model: in-situ combustion (ISC) for oil recovery, with 3 or 4 reaction mechanisms.
  - Inputs: the activation energies (3 or 4), uniform, ±20 % around the nominal values.
  - Outputs: cumulative oil, water, CO, CO₂ and front velocity.
  - Data: 500 × 3 and 1 000 × 4 runs, no gradients. The legacy flow is OLS, then local linear gradients, then AS.
- People: Constantine (contact). The notebooks cite only "Folake's MS Thesis (reference?)". The closest match is O. Ogunbanwo, *Uncertainty Assessment on In-Situ Combustion Simulations Using Experimental Design*, Stanford MS thesis (https://purl.stanford.edu/mb768dc1016). That thesis uses experimental design, not AS, and I could not confirm the data come from it.
- No AS paper found.
- Defects:
  - The ICS-4 description says "a chemical mechanism with three reactions" while the table lists four.
  - The dev ICS-3 notebook is corrupt HTML.
- Action: leave them out of main; they stay reachable through `legacy-dev`. If the author wants them back, restore the notebook from `da61ac5`, cite the thesis, and give them a full spec.

### 1.2 Reentry: drop, superseded
- Model: the EXPERT re-entry vehicle stagnation point, with simulations by P. Congedo (INRIA Bordeaux). 7 inputs (p∞, M∞, catalytic γ, plus 4 log reaction rates); outputs are wall pressure and heat flux. 150 runs.
- Cites Tryoen, Congedo, Abgrall, Villedieu, Magin (AIAA J., doi:10.2514/1.J052831).
- Master's `Reentry_freestream_conditions` covers the same model and 7 inputs with 7 460 runs. Its cited Cortesi et al. "Submitted, 2016" paper was never published, and Cortesi et al., JCP 2020 is a different (cylinder) study.
- The dev version also has a bug in cell 6: `UnboundedNormalizer(xmean, xstd)` passes standard deviations where a covariance is expected. Its σ vector `[.12,.12,.1,.1]` also contradicts its own table, which gives σ = 0.12, 0.10, 0.10, 0.10.
- Moot since 2026-10-29: the author excluded `Reentry_freestream_conditions` from main too, so neither Reentry directory is ported.

### 1.3 SNL-CLM: drop
- Model: Community Land Model ensemble by K. Sargsyan (Sandia). 68 inputs normalized to [−1, 1]; QoI is the leaf area index; 7 434 runs, no gradients.
- Source paper: Sargsyan, Safta, Najm, Debusschere, Ricciuto, Thornton, *Dimensionality reduction for complex models via Bayesian compressive sensing*, Int. J. Uncertainty Quantification 4(1):63–93 (2014), doi:10.1615/Int.J.UncertaintyQuantification.2013006821. That paper does not use AS, and no AS paper on this data was found.
- `IJUQ-6821.pdf` is the publisher PDF of that paper (Begell House). Do not redistribute it.
- Data release permission from Sandia/ORNL is unknown.

### 1.4 SNL-LES: drop
- Model: LES of a turbulent bluff-body stabilized flame (Sandia CRF). 3 inputs (C_s, Pr_t, Sc_t, uniform); 30 outputs (mean/RMS of T, w, mixture fraction, P, ρ at 3 centerline points); **35 runs**.
- Cites Khalil, Lacaze, Oefelein, Najm, Proc. Combust. Inst. 35 (2015). That is a PCE study, not AS; its DOI was not verified.
- With 3 inputs, 35 runs and local-linear gradients, this is a weak AS demonstration, and no AS paper was found.

### 1.5 Atacamac and LM104X: include
See the specs.
- Atacamac: J. Fluid Mech. 789 (2016), doi:10.1017/jfm.2015.730, with Constantine as co-author.
  - Companion paper: Bauerheim, Ndiaye, Constantine, Iaccarino, Moreau, Nicoud, *Uncertainty quantification of thermo-acoustic instabilities in annular combustors*, Proc. CTR Summer Program 2014, pp. 209–218.
  - The data are most likely the 500-sample gradient set that both papers use.
  - Re-checked against both PDFs: the data distribution (Gaussian, σ 8.66 % / 4.33 %) still contradicts both papers (uniform, σ 10 % / 5 %).
- LM104X: Lukaczyk's Stanford dissertation (June 2015, http://purl.stanford.edu/xx611nd3190), §4.3.3.
  - Confirmed as the source: the data's eigenvalues, with inputs and outputs scaled to [−1, 1], reproduce fig. 4.14 (p. 108).
  - The work was funded by NASA N+2 and Lockheed Martin (acknowledgments, p. vii).
  - No export-control or distribution statement in the dissertation, so the release still needs the author's confirmation.
- Both were already in master before 9c1b8a7. That commit's message ("without associated papers") does not fit Atacamac: the CTR 2014 paper and the JFM paper (Jan 2016) both predate the removal in Sep 2016. Ask the author whether there was another reason, such as data permission.

### 1.6 Zonotope_Vertices: drop from this repo
- Content: K. Stinson, D. F. Gleich, P. G. Constantine, *A randomized algorithm for enumerating zonotope vertices*, arXiv:1602.06620 (2016). The local PDF is arXiv v1 (22 Feb 2016) in a Springer "Noname manuscript" template with "Received: date / Accepted: date", so it lists no journal version, and none was found online.
  - `zonotopes.py` has `nzv`, the randomized `zonotope_vertices`, and Hausdorff error tools.
  - `Polytopes.py` is a Python port of MinkSum reverse search.
  - `qp_solver.py` has LP/QP wrappers with optional Gurobi.
  - There are 2 notebooks. They contain no data, need PyMINKSUM (C++/GMP build) and Gurobi, and are Python 2 (`except ImportError, e`, `print`, `scipy.misc.comb`, a float `size=` in `np.random.normal`).
- v1 already supersedes the code:
  - `ac.zonotope_vertex_count(m, n)` replaces `nzv`.
  - `ac.active_domain(sub).vertices` enumerates the vertices exactly and deterministically through facet enumeration (`geometry._verts`). It replaces both the randomized `zonotope_vertices` (migration map, "Nsamples, maxcount removed; exact") and the MinkSum ground truth.
- Prototype (`…/g6/zono_proto.py`): v1 exact vertices match the count formula for (m,n) = (5,2), (10,2), (10,3), (10,4), (20,3), in ≤ 8 ms each. With 10⁶ samples, the randomized algorithm finds 92/92, 244/260 and 374/382 vertices, always a subset of the exact set.
- Optional follow-up: a short v1 tutorial in the active_subspaces docs, "randomized vs exact zonotope vertices", that reproduces the arXiv convergence figures with `ac.active_domain` as ground truth. It would need no PyMINKSUM or Gurobi.
- The paper's authors appear in dev history (PR #31, ryan-kelley-howard), which is preserved by the merge in §2.
- The notebook misspells names: "Kerrick Stinson" should be "Kerrek" (confirmed on the paper's title page), and "dglecih@purdue.edu" is a typo. This does not matter if the notebook is dropped.

## 2. Producing `main`: recommended strategy

**Recommendation: a `-s ours` merge of `origin/dev` plus a selective restore of the included directories, all in one merge commit. Then rename `master` to `main` on GitHub.** Do not run a plain `git merge origin/dev`: it would conflict with, or revert, master's notebook updates, the README rename and the deliberate deletions.

Why this way, and not a plain `git checkout origin/dev -- <dir>` on top of master:
1. **Blame survives.** Verified in a scratch clone: after the merge, `git blame` on `Atacamac/atacamac-weak.txt` attributes its lines to `7c26e3a` (initial commit), and the notebooks to Ryan Howard's 2016 commits. After a plain checkout, every line would be attributed to the restore commit.
2. **Dev becomes an ancestor of main.** GitHub and `git branch --merged` then show dev as merged, so deleting the dev branch loses nothing. That includes Zonotope_Vertices and PR #31.
3. **Master's tree wins everywhere else.** The `ours` strategy takes no dev content except what is explicitly restored.

The cost is one merge commit whose tree differs from both parents. The commit message states this. The alternative, a plain restore from `9c1b8a7^` with no merge, gives a linear history but loses (1) and (2).

### 2.1 Commands
Run these in a **separate worktree**, so that the `.idea/` files staged in the main checkout are not swept into the commit and the working tree is untouched.

```bash
cd /Users/paulconstantine/PycharmProjects/as-data-sets
git fetch origin --prune --tags

# 1. Legacy tags (annotated), mirroring active_subspaces D-01
git tag -a legacy-master 32e5839 -m "as-data-sets master before the v1 port (2016 API, Python 2)"
# no legacy-dev tag (author, 2026-10-30): dev stays reachable through the merge's second parent
git push origin legacy-master

# 2. Work branch in its own worktree
git worktree add ../as-data-sets-merge -b merge-dev origin/master
cd ../as-data-sets-merge

# 3. Record dev as merged, keep master's tree, restore only the included directories
git merge -s ours --no-commit origin/dev
git checkout origin/dev -- LM104X/LM104X.txt LM104X/active-subspace-LM104X.ipynb   # not Atacamac (author, 2026-10-30)
git commit -m "Merge dev into master: restore LM104X

Records origin/dev (d3d31e6) as merged with the 'ours' strategy, so master's
notebooks, README.md and the 2016 removals are kept. Only LM104X/LM104X.txt
and LM104X/active-subspace-LM104X.ipynb (without .DS_Store) are taken from
dev; both were on master before 9c1b8a7 and are byte-identical on dev.
Atacamac, ICS-3reac, ICS-4reac, Reentry, SNL-CLM (incl. IJUQ-6821.pdf),
SNL-LES and Zonotope_Vertices stay out of the tree; they remain reachable
through this merge's second parent.

Co-Authored-By: ..."

# 4. Verify
git diff --name-status origin/master HEAD      # expect exactly 2 'A' lines (LM104X)
git diff --name-status origin/dev HEAD -- Atacamac LM104X   # expect only 'D LM104X/.DS_Store'
git blame -s HEAD -- LM104X/LM104X.txt | head -3   # expect 7c26e3a…
git branch -r --merged HEAD                     # lists origin/dev and origin/master

# 5. Hygiene in the same pass (separate commits, §4)
git rm --cached ONERA-M6/figs/.DS_Store
# edit .gitignore (§4.2), then:
git add .gitignore && git commit -m "Remove tracked .DS_Store; extend .gitignore"
git rm -r Reentry_freestream_conditions
# remove its README bullet in the same commit
git commit -am "Remove Reentry_freestream_conditions from main (author decision)"
# README bullets for Atacamac and LM104X: commit together with, or after, their v1 ports

# 6. Publish: fast-forward master (sole maintainer), or open a PR and merge it
#    with "Create a merge commit". Never use squash or rebase: they drop the dev parent.
git push origin merge-dev
gh pr create --repo paulcon/as-data-sets --base master --head merge-dev --title "Merge dev into master; prepare main" --body "..."
```

The hygiene commit and the PR are separate from the v1 notebook ports. Each port then lands as its own PR into `main`.

## 3. GitHub steps (after the merge is on master)
Order matters: tags first (done in §2.1), then the rename, then deleting dev.

1. **Rename `master` → `main`**, as in active_subspaces D-187. Use Settings → Branches → rename, or `gh api -X POST repos/paulcon/as-data-sets/branches/master/rename -f new_name=main`.
   - GitHub then makes `main` the default branch, retargets open PRs (there are none now), and redirects web URLs `…/tree/master/…` and `…/blob/master/…` to `main`.
   - This is better than creating a new `main` and deleting `master`. External links (activesubspaces.org, papers, the active_subspaces README, nbviewer bookmarks) would then 404, and there are 12 forks.
2. **Check the redirects after the rename** (unverified for non-web URLs):
   - `https://github.com/paulcon/as-data-sets/blob/master/HIV/HIV.ipynb`
   - `https://raw.githubusercontent.com/paulcon/as-data-sets/master/HIV/HIV.ipynb`
   - `https://nbviewer.org/github/paulcon/as-data-sets/blob/master/HIV/HIV.ipynb`

   If the raw or nbviewer URLs break, list the known external referrers for the author to update. In-repo links need no change (none point to a branch).
3. **Local clones:**
   ```bash
   git branch -m master main
   git fetch origin
   git branch -u origin/main main
   git remote set-head origin -a
   git worktree remove ../as-data-sets-merge   # done 2026-10-30, before the push: merge-dev is checked out in the main checkout
   git branch -d merge-dev
   ```
4. **Delete dev:** `git push origin --delete dev`. It is fully merged and also tagged as `legacy-dev`.
5. **Post-rename text:**
   - README: add a "2016 version" note pointing to `legacy-master` / `legacy-dev`, as in the active_subspaces README.
   - README: change `nbviewer.jupyter.org` to `nbviewer.org`.
   - Update CLAUDE.md if it gets committed.
   - Close issue #1 when v1 is released.
6. Optional: protect `main` (require PRs).

## 4. Repo hygiene (same pass)

### 4.1 .DS_Store
- On master, `ONERA-M6/figs/.DS_Store` is tracked; it is the only file in `figs/`. Remove it with `git rm --cached`; the directory disappears from the tree.
- Dev's `ICS-4reac/.DS_Store`, `ICS-4reac/figs/.DS_Store` and `LM104X/.DS_Store` are not brought over.
- `.gitignore` already lists `.DS_Store`. These files were committed before the rule existed.

### 4.2 Proposed `.gitignore`
```
# notebooks
.ipynb_checkpoints/
*.ipynb_checkpoints*
# Python
__pycache__/
*.py[cod]
.venv/
venv/
# publisher PDFs used for reviews (already added locally 2026-10-29)
papers/
# OS / tools
.DS_Store
gurobi.log
.mcp.json
.claude/settings.local.json
# JetBrains: see 4.3 (either the whole directory or only per-user files)
.idea/workspace.xml
.idea/shelf/
```

### 4.3 `.idea/`: flag for the author, not decided here
Seven `.idea` files are currently **staged (`A`) in the local index** of the main checkout: `.gitignore`, `as-data-sets.iml`, `inspectionProfiles/*`, `misc.xml`, `modules.xml`, `vcs.xml`.
- `misc.xml` and `as-data-sets.iml` contain the machine-specific SDK name `~/PycharmProjects/as-data-sets/.venv`.
- Option A: unstage them (`git restore --staged .idea`) and add `.idea/` to `.gitignore`. This is the usual choice for a data/notebook repo with outside contributors.
- Option B: commit only `inspectionProfiles/` and `vcs.xml`, and ignore the rest.
- Whichever option is chosen, the merge in §2 runs in a separate worktree and is not affected.

### 4.4 `SNL-CLM/IJUQ-6821.pdf` (Begell House publisher PDF)
- It will not be in main's tree. It remains in history: `7c26e3a`..`9c1b8a7^` on master, and the tip of `legacy-dev`, where it is browsable at `/tree/legacy-dev/SNL-CLM/`.
- Option A (recommended; consistent with active_subspaces D-07/D-175 "do not rewrite history"): do not rewrite history. Skip the `legacy-dev` tag, or accept that it exposes the file. Dev is reachable from main through the §2 merge, so the tag is only a label.
- Option B: purge with `git filter-repo --path SNL-CLM/IJUQ-6821.pdf --invert-paths`, force-push all branches and tags, and ask GitHub Support to purge cached views and `refs/pull/*`. This rewrites every SHA (breaking the 12 forks, the PR references and the tags above) and must happen **before** step 1 of §2.1.
- The author decides. If a takedown request arrives, Option B becomes necessary.

### 4.5 Notebook output policy
- Keep outputs. The repo exists to be read on GitHub/nbviewer.
- Do not install `nbstripout` in its default mode. The v1 library's tutorials strip outputs because CI rebuilds them; this repo has no docs build.
- Policy: each notebook is committed after `jupyter nbconvert --to notebook --execute --inplace` in `.venv` from its own directory, with a fixed seed. Diffs then come only from real changes.
- Optional: `nbstripout --keep-output --keep-count` as a git filter, to drop volatile metadata only.

### 4.6 Size (outside the scope of this merge, noted for the MCMC owner)
`MCMC/Vanilla_Chain.mat` is 80.0 MB. That is above GitHub's 50 MB warning, below the 100 MB limit.

## 5. Open questions for the author
1. Include LM104X now, or wait for release confirmation (Lockheed/NASA N+2 data)?
2. Was there a reason other than "no paper" for removing Atacamac in 9c1b8a7, such as data permission?
3. `.idea/`: Option A or B (§4.3)? Commit `CLAUDE.md` and `specs/`?
4. IJUQ PDF: Option A or B (§4.4)? Should the `legacy-dev` tag be created at all?
5. Publish the merge with a PR or a direct fast-forward push?
