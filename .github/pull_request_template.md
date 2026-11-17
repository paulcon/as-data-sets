<!-- For a new or changed data set. For other changes (a typo, a guide), delete what does not apply. -->

## Data set

- Directory: `<category>/<Name>/`
- Claim issue: #<number>
- Article: <authors, title, venue, year, DOI>
- Kind: stored data (<outside code>) | Python model (`<name>_model.py`)
- Reproduction | new study with the paper's model (CONTRIBUTING.md, section 6)
- Other data sets here for this article: none | `<directory>`
- Run times: notebook <...>; `generate_data.py` <...>; `check_accuracy.py` <...> (machine: <...>)

## For the listing (the maintainer writes it)

- Number of inputs: <m>
- Active subspace dimension: <n> (with several quantities of interest: the dimension for each, or the range and the exceptions)
- Summary: <one sentence on the model and the quantity studied>

## Checklist (CONTRIBUTING.md, sections 6, 10 and 11)

- [ ] The notebook runs top to bottom from its directory in a fresh kernel, and twice gives the same numbers.
- [ ] `generate_data.py --check` (where there is one) runs, and `DATA.md` reports the differences it prints.
- [ ] `check_accuracy.py` (where there is one) reproduces the numbers in `DATA.md`.
- [ ] Every claim about the paper cites a page, table, figure or equation, and my own choices are stated as mine.
- [ ] Every number in the text is either cited to the paper or printed by the code; disagreements with the paper are reported with both values.
- [ ] The self-review in section 10 is done; anything it could not settle is in `DATA.md`'s unknowns.
- [ ] `DATA.md` has every section of the template.
- [ ] The license of every file is clear, `LICENSE-DATA` lists any exception, and no PDF, publisher supplement or copied GPL code is included.
- [ ] I have not edited the READMEs, `docs/` or `CONTRIBUTORS.md`, and "Allow edits by maintainers" is on.

## AI use

- [ ] No AI agent was used.
- [ ] An AI agent was used: <which, and for what: literature reading, code, text, checks>.

## Review by a person

- [ ] I, <name>, read the whole notebook and `DATA.md` myself and checked every attribution against the paper.

## Unknowns and open questions

<What you could not verify, and questions for the article's authors. Write "none" if there are none.>

## For the maintainer (contributors: leave this section unticked)

Before merging:

- [ ] The PR matches its claim issue and changes only its data set's directory.
- [ ] The notebook, `generate_data.py --check` and `check_accuracy.py` rerun in the maintainer's environment, and the re-executed notebook is committed on this branch.
- [ ] Spot checks (CONTRIBUTING.md, section 10): gradients against finite differences at 3 samples, eigenvalues from the shipped files, input marginals against the declared space, and a few attributions against the paper.
- [ ] The license of every file is checked: no PDFs, publisher supplements or copied GPL code.
- [ ] The listing is on this branch: README bullet, category README section, `SIZES` in `docs/make_notebook_index.py` and the regenerated index, and `CONTRIBUTORS.md`.
- [ ] If another data set uses the same article, each notebook's introduction points to the other, and the PR is labelled `same-article`.

After merging: close the claim issue, and remove the article from the suggested list.

<!-- Delete anything a tool added to this description that is not about the data set, such as a link to an agent session. -->
