"""Write docs/notebook-index.md: every notebook, with the methods it uses.

    python docs/make_notebook_index.py          # rewrite docs/notebook-index.md
    python docs/make_notebook_index.py --check  # exit 1 if the file is out of date

Run it from the repository root after adding or changing a notebook. It reads the code cells of
every */*/*.ipynb for active_subspaces calls, and the kind of each data set from the end of its
entry in README.md. It uses only the standard library.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / 'notebook-index.md'

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

# How the subspace is estimated: (label, test on the code)
SOURCES = [
    ('gradients', lambda c, est: 'as' in est or (not est and 'grads=' in c)),
    ('local linear gradients', lambda c, est: 'local_linear_gradients' in c or 'opg' in est),
    ('finite differences', lambda c, est: 'finite_difference_gradients' in c),
    ('linear fit', lambda c, est: 'ols' in est),
    ('quadratic fit', lambda c, est: 'qphd' in est),
    ('SIR', lambda c, est: 'sir' in est),
    ('given matrix', lambda c, est: 'subspace_from_matrix' in c),
    ('given basis', lambda c, est: 'subspace_from_basis' in c),
]
INPUTS = [
    ('uniform', r'ac\.Uniform\b'),
    ('Gaussian', r'ac\.Gaussian\b'),
    ('mixed', r'ac\.Product\b'),
]
USES = [
    ('surrogate', r'ac\.fit_surrogate\b'),
    ('prediction intervals', r'\.predict_interval\b|ac\.predict_interval\b'),
    ('activity scores', r'ac\.activity_scores\b'),
    ('active domain', r'ac\.active_domain\b'),
    ('inverse map', r'ac\.inverse_map\b'),
    ('MCMC', r'(?i)\bmcmc\b|metropolis'),
]


def kinds():
    """Data set name -> kind, from the italic note that ends each README entry."""
    out = {}
    for line in (ROOT / 'README.md').read_text().splitlines():
        m = re.match(r'\* \*\*\[([^\]]+)\]', line)
        if not m:
            continue
        note = re.findall(r'(?:^|\s)_([A-Z][^_]*?)[._]', line)
        text = note[-1] if note else ''
        if text.startswith('Both'):
            out[m.group(1)] = 'both'
        elif text.startswith('Python model'):
            out[m.group(1)] = 'Python model'
        elif text.startswith('Stored data'):
            out[m.group(1)] = 'stored data'
        else:
            out[m.group(1)] = '?'
    return out


def row(nb_path, kind):
    nb = json.loads(nb_path.read_text())
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    est = set(re.findall(r"estimator\s*=\s*['\"]([a-z_]+)", code))
    sources = [label for label, test in SOURCES if test(code, est)]
    inputs = [label for label, pat in INPUTS if re.search(pat, code)]
    uses = [label for label, pat in USES if re.search(pat, code)]
    name = nb_path.parent.name
    link = f'[{name}](../{nb_path.parent.parent.name}/{name}/{nb_path.name})'
    return f'| {link} | {kind} | {", ".join(sources)} | {", ".join(inputs)} | {", ".join(uses)} |'


def build():
    kind = kinds()
    lines = [
        '# Notebooks by method',
        '',
        'Every notebook, with how it estimates the subspace, how it models the inputs, and what it does '
        'with the subspace beyond the eigenvalue, eigenvector and sufficient summary plots. '
        '[using-the-notebooks.md](using-the-notebooks.md) explains the terms. '
        'Written by `docs/make_notebook_index.py` from the notebooks\' code; do not edit by hand.',
        '',
        '- **Subspace from**: *gradients* (the gradient-based estimator, with gradients from files or the model); '
        '*local linear gradients* (gradients estimated from values by local linear fits); '
        '*finite differences*; *linear fit* or *quadratic fit* (a global least-squares model of the values); '
        '*SIR* (sliced inverse regression); *given matrix* or *given basis* (computed in the notebook, e.g. by quadrature).',
        '- **Inputs**: *uniform* on a box, *Gaussian* (possibly correlated), or *mixed* (a product of uniform and Gaussian blocks).',
        '',
    ]
    nbs = sorted(ROOT.glob('*/*/*.ipynb'))
    for cat, heading in CATEGORIES.items():
        rows = [row(p, kind.get(p.parent.name, '?')) for p in nbs
                if p.parent.parent.name == cat and p.stem == p.parent.name]
        lines += [f'## {heading}', '',
                  '| Notebook | Kind | Subspace from | Inputs | Also |',
                  '|---|---|---|---|---|', *rows, '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true', help='exit 1 if the index is out of date')
    text = build()
    if ap.parse_args().check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f'{OUT.relative_to(ROOT)} is out of date: run python docs/make_notebook_index.py')
        print(f'{OUT.relative_to(ROOT)} is up to date')
        return
    OUT.write_text(text)
    print(f'wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
