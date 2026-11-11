<!-- For a new or changed data set. For other changes (a typo, a guide), delete what does not apply. -->

## Data set

- Directory: `<category>/<Name>/`
- Claim issue: #<number>
- Article: <authors, title, venue, year, DOI>
- Kind: stored data (<outside code>) | Python model (`<name>_model.py`)
- Already a data set here for this article? none | `<directory>`: how this one differs
- Run times: notebook <...>; `generate_data.py` <...>; `check_accuracy.py` <...> (machine: <...>)

## Checklist (CONTRIBUTING.md, sections 6, 10 and 11)

- [ ] The notebook runs top to bottom from its directory in a fresh kernel, and twice gives the same numbers.
- [ ] `generate_data.py --check` (where there is one) reproduces the shipped files.
- [ ] `check_accuracy.py` (where there is one) reproduces the numbers in `DATA.md`.
- [ ] Every claim about the paper cites a page, table, figure or equation, and my own choices are stated as mine.
- [ ] Every number in the text is printed by the code; disagreements with the paper are reported with both values.
- [ ] The self-review in section 10 is done; anything it could not settle is in `DATA.md`'s unknowns.
- [ ] `DATA.md` has every section of the template.
- [ ] The license of every file is clear, `LICENSE-DATA` lists any exception, and no PDF, publisher supplement or copied GPL code is included.
- [ ] The README, the category README and `docs/notebook-index.md` list the data set, and `CONTRIBUTORS.md` lists me.

## AI use

- [ ] No AI agent was used.
- [ ] An AI agent was used: <which, and for what: literature reading, code, text, checks>.

## Review by a person

- [ ] I, <name>, read the whole notebook and `DATA.md` myself and checked every attribution against the paper.

## Unknowns and open questions

<What you could not verify, and questions for the article's authors. Write "none" if there are none.>
