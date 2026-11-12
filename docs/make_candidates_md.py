"""Write docs/candidates.md, the list of suggested articles, from docs/candidates.csv.

    python docs/make_candidates_md.py          # rewrite docs/candidates.md
    python docs/make_candidates_md.py --check  # exit 1 if the file is out of date

docs/candidates.csv has one row per article: category, title, authors, year, doi, url (url only
when there is no DOI). The table shows the first three authors; the CSV has them all. It uses only
the standard library.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'docs' / 'candidates.csv'
OUT = ROOT / 'docs' / 'candidates.md'

CATEGORIES = {  # directory: heading, in the README's order
    'aerospace_cfd': 'Aerospace and CFD',
    'hydrology_subsurface_geoscience': 'Hydrology, subsurface and geoscience',
    'climate_atmosphere_ocean': 'Climate, atmosphere and ocean',
    'ecology_agriculture': 'Ecology and agriculture',
    'energy': 'Energy',
    'plasma_fusion': 'Plasma and fusion',
    'epidemiology_systems_biology': 'Epidemiology and systems biology',
    'biomedical_cardiovascular': 'Biomedical and cardiovascular',
    'combustion_chemical_kinetics': 'Combustion and chemical kinetics',
    'structural_materials_manufacturing': 'Structural, materials and additive manufacturing',
    'electronics_circuits': 'Electronics and circuits',
    'uq_benchmarks': 'UQ benchmark collections',
}
MAX_AUTHORS = 3

HEADER = """# Suggested articles

These {n} published articles might each support a new data set. They are suggestions, not approved projects: you may build a data set from any suitable article, on this list or not. [Contributing a data set](../CONTRIBUTING.md) explains how. Each entry gives only the article's year, title, authors and DOI; [candidates.csv](candidates.csv) has the same entries with every author.

## Where the list came from

An AI agent working under Paul Constantine's direction drafted the list in October and November 2026 by searching the literature. The searches covered works that cite active subspaces papers, applied global sensitivity analysis studies, papers on open-source scientific software, collections of benchmark functions with physical meaning, and the application areas below.

The agent screened each article, mostly from its abstract and any open text, for:

- a simulation model with at least one scalar quantity of interest;
- inputs with stated ranges or distributions;
- a route to input/output data: data that can be downloaded, an executable model, or data the authors might share.

It also noted what makes an article a stronger candidate: an existing active subspace, ridge or gradient-based dimension reduction analysis; open data or code; a global sensitivity analysis; calibration or inverse uncertainty quantification; forward uncertainty propagation; a surrogate for a scalar output; or a well-specified parameter study. Articles that seemed to fall short are listed too, because a closer reading may reach a different conclusion.

## Before you start

- **Check the article yourself.** The screening relied mostly on abstracts, and a person has not checked every entry. An article may lack the data, input ranges or model it seemed to have, and another may be better than the screening suggested.
- **Articles that already have a data set here are left off.** You may still build another data set from one of them.
- **Report errors** in an entry (a wrong year, a DOI that does not resolve, a missing author) in an issue.

Written by `docs/make_candidates_md.py` from `docs/candidates.csv`; do not edit by hand.

{contents}
"""


def esc(s):
    return s.replace('|', '\\|')


def authors(s):
    names = [a.strip() for a in s.split(';') if a.strip() and a.strip() != 'et al.']
    short = '; '.join(names[:MAX_AUTHORS])
    return short + ('; et al.' if len(names) > MAX_AUTHORS or 'et al.' in s else '')


def link(row):
    if row['doi']:
        return f"[{esc(row['doi'])}](https://doi.org/{row['doi']})"
    if row['url']:
        return f"[link]({row['url']})"
    return 'none found'


def build():
    rows = list(csv.DictReader(SRC.open(newline='')))
    unknown = sorted({r['category'] for r in rows} - set(CATEGORIES))
    if unknown:
        sys.exit(f'unknown categories in {SRC.name}: {unknown}')
    groups = {c: sorted((r for r in rows if r['category'] == c),
                        key=lambda r: (r['year'], r['title'].lower())) for c in CATEGORIES}
    contents = '\n'.join(f"- [{h}](#{h.lower().replace(',', '').replace(' ', '-')}) ({len(groups[c])})"
                         for c, h in CATEGORIES.items() if groups[c])
    out = [HEADER.format(n=len(rows), contents=contents)]
    for c, h in CATEGORIES.items():
        if not groups[c]:
            continue
        out += [f'## {h}', '', '| Year | Title | Authors | DOI or link |', '|---|---|---|---|']
        out += [f"| {r['year']} | {esc(r['title'])} | {esc(authors(r['authors']))} | {link(r)} |"
                for r in groups[c]]
        out.append('')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true', help='exit 1 if candidates.md is out of date')
    text = build()
    if ap.parse_args().check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f'{OUT.relative_to(ROOT)} is out of date: run python docs/make_candidates_md.py')
        print(f'{OUT.relative_to(ROOT)} is up to date')
    else:
        OUT.write_text(text)
        print(f'wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
