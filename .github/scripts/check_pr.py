"""Check the files a pull request changes.

    python .github/scripts/check_pr.py                     # changes since origin/main
    python .github/scripts/check_pr.py --base upstream/main
    python .github/scripts/check_pr.py --all               # every data set and tracked file
    python .github/scripts/check_pr.py --datasets-json     # print the changed data-set directories

A data-set directory is <category>/<Name>/ under one of the category directories. The run fails on:

- a changed or added file larger than 10 MB;
- an added PDF;
- a machine path (/Users/<name>/, /home/<name>/, C:\\Users\\, /var/folders/, or a temporary
  directory in a notebook output) in a changed .md, .ipynb, .py or .json file or a .csv header;
- a reference-style link in a notebook's markdown cell that is not defined in that cell
  (Jupyter renders each cell on its own; write links inline, [[1]](https://doi.org/...));
- a changed data set whose DATA.md lacks a section of docs/templates/DATA.md;
- a changed data set without exactly one notebook, named after its directory.

Changes outside data-set directories only give a warning. Standard library only.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = {
    'aerospace_cfd', 'biomedical_cardiovascular', 'climate_atmosphere_ocean',
    'combustion_chemical_kinetics', 'ecology_agriculture', 'electronics_circuits', 'energy',
    'epidemiology_systems_biology', 'hydrology_subsurface_geoscience', 'plasma_fusion',
    'structural_materials_manufacturing', 'uq_benchmarks',
}
MAX_BYTES = 10 * 1024 * 1024
TEXT_SUFFIXES = {'.md', '.ipynb', '.py', '.json', '.csv'}

# DATA.md sections: the template's "## " headings. A heading counts when it starts with the
# template's heading or with one of the accepted variants below.
TEMPLATE = ROOT / 'docs' / 'templates' / 'DATA.md'
VARIANTS = {
    'Unknowns': ('Open questions', 'Provenance and unknowns'),
    'How these files relate to the paper': ('Relation to the paper',),
}
# Data sets written before the DATA.md template whose DATA.md lacks one of its sections. Their
# DATA.md files are checked only for the sections they already have; a data set leaves this
# list when its DATA.md is brought up to the template.
DATA_MD_EXEMPT = {
}

MACHINE_PATH = re.compile(
    r'/Users/[A-Za-z0-9._-]+/'
    r'|/home/[A-Za-z0-9._-]+/'
    r'|[A-Za-z]:\\\\?Users\\'
    r'|/var/folders/'
    r'|/private/var/')
# Temporary directories, looked for in notebook outputs only.
TEMP_PATH = re.compile(r'(?<![\w.])/tmp/|AppData\\\\?Local\\\\?Temp')

# Reference-link check. Spans removed first: fenced code, inline code, display and inline math.
STRIP = re.compile(
    r'^\s*(```|~~~).*?^\s*\1'
    r'|`+[^`]*?`+'
    r'|\$\$.*?\$\$'
    r'|\\\[.*?\\\]|\\\(.*?\\\)'
    r'|(?<!\\)\$[^$\n]+?(?<!\\)\$',
    re.S | re.M)
LINK_USE = re.compile(r'\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\[([^\[\]]*)\]')
LINK_DEF = re.compile(r'^ {0,3}\[([^\]]+)\]:[ \t]*\S', re.M)

GITHUB = os.environ.get('GITHUB_ACTIONS') == 'true'
errors = []
warnings = []


def error(path, msg):
    errors.append((path, msg))


def warn(path, msg):
    warnings.append((path, msg))


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout


def changed_files(base):
    """(status, path) for each file changed between the merge base with `base` and HEAD."""
    out = git('diff', '--name-status', '--no-renames', f'{base}...HEAD')
    return [tuple(line.split('\t', 1)) for line in out.splitlines() if line]


def dataset_of(path):
    parts = Path(path).parts
    if len(parts) >= 3 and parts[0] in CATEGORIES:
        return f'{parts[0]}/{parts[1]}'
    return None


def all_datasets():
    return sorted(f'{c}/{d.name}' for c in CATEGORIES if (ROOT / c).is_dir()
                  for d in (ROOT / c).iterdir() if d.is_dir() and not d.name.startswith('.'))


# ---- file checks -------------------------------------------------------------------------

def check_size(path):
    size = (ROOT / path).stat().st_size
    if size > MAX_BYTES:
        error(path, f'{size / 2**20:.1f} MB is over the 10 MB limit; put large originals in an '
                    'archive such as Zenodo and ship a reduced file (CONTRIBUTING.md, section 4)')


def text_to_scan(path):
    """The parts of a text file to search for machine paths: (text, text of outputs)."""
    p = ROOT / path
    raw = p.read_text(encoding='utf-8', errors='replace')
    if p.suffix == '.csv':
        lines = raw.splitlines()
        head = [lines[0]] if lines else []
        head += [l for l in lines[1:50] if l.lstrip().startswith('#')]
        return '\n'.join(head), ''
    if p.suffix == '.ipynb':
        try:
            cells = json.loads(raw)['cells']
        except (ValueError, KeyError):
            error(path, 'the notebook is not valid JSON')
            return '', ''
        src = '\n'.join(''.join(c.get('source', [])) for c in cells)
        outs = json.dumps([c.get('outputs', []) for c in cells])
        return src, outs
    return raw, ''


def check_machine_paths(path):
    src, outs = text_to_scan(path)
    for text, where in ((src, ''), (outs, ' (in an output)')):
        m = MACHINE_PATH.search(text)
        if m:
            error(path, f'machine path "{m.group(0)}"{where}; use paths relative to the '
                        'data set\'s directory')
    m = TEMP_PATH.search(outs)
    if m:
        error(path, f'temporary path "{m.group(0)}" in an output; rerun the notebook from its '
                    'own directory')


def check_links(path):
    cells = json.loads((ROOT / path).read_text(encoding='utf-8'))['cells']
    md = [(i, ''.join(c['source'])) for i, c in enumerate(cells) if c['cell_type'] == 'markdown']
    norm = lambda s: ' '.join(s.split()).lower()
    defined = {}
    for i, src in md:
        for label in LINK_DEF.findall(STRIP.sub('', src)):
            defined.setdefault(norm(label), set()).add(i)
    for i, src in md:
        for m in LINK_USE.finditer(STRIP.sub('', src)):
            label = norm(m.group(2) or m.group(1))
            if i not in defined.get(label, ()):
                error(path, f'cell {i}: reference link {m.group(0)} is not defined in that cell, '
                            'so it renders as plain text; write it inline, [text](https://...)')


def check_file(path):
    p = ROOT / path
    if not p.is_file():
        return
    check_size(path)
    if p.suffix in TEXT_SUFFIXES and p != Path(__file__).resolve():   # this file names the patterns
        check_machine_paths(path)
    if p.suffix == '.ipynb' and not any(part.startswith('.') for part in Path(path).parts):
        try:
            check_links(path)
        except (ValueError, KeyError):
            pass  # reported by check_machine_paths


# ---- data-set checks ---------------------------------------------------------------------

def template_sections():
    return [l[3:].strip() for l in TEMPLATE.read_text(encoding='utf-8').splitlines()
            if l.startswith('## ')]


def check_dataset(d):
    p = ROOT / d
    if not p.is_dir():
        return  # removed
    name = p.name
    notebooks = sorted(x.name for x in p.glob('*.ipynb'))
    if notebooks != [f'{name}.ipynb']:
        found = ', '.join(notebooks) or 'none'
        error(d, f'a data set has exactly one notebook, {name}.ipynb, in its directory '
                 f'(found: {found})')
    data_md = p / 'DATA.md'
    if not data_md.is_file():
        error(d, 'DATA.md is missing; start from docs/templates/DATA.md')
        return
    headings = [l[3:].strip() for l in data_md.read_text(encoding='utf-8').splitlines()
                if l.startswith('## ')]
    missing = [s for s in template_sections()
               if not any(h.startswith((s,) + VARIANTS.get(s, ())) for h in headings)]
    missing = [s for s in missing if s not in DATA_MD_EXEMPT.get(d, ())]
    if missing:
        error(f'{d}/DATA.md', 'missing section(s) of docs/templates/DATA.md: '
                              + '; '.join(f'"## {s}"' for s in missing))


# ---- main --------------------------------------------------------------------------------

def report():
    for kind, items in (('warning', warnings), ('error', errors)):
        for path, msg in items:
            if GITHUB:
                print(f'::{kind} file={path}::{msg}')
            else:
                print(f'{kind}: {path}: {msg}')
    print(f'{len(errors)} error(s), {len(warnings)} warning(s)')
    return 1 if errors else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--base', default='origin/main', help='the branch the changes are against')
    ap.add_argument('--all', action='store_true', help='check every data set and tracked file')
    ap.add_argument('--datasets-json', action='store_true',
                    help='print the changed data-set directories as a JSON list and exit')
    args = ap.parse_args()

    if args.all:
        files = git('ls-files').splitlines()
        added = set()
        datasets = all_datasets()
    else:
        changes = changed_files(args.base)
        files = [p for s, p in changes if s != 'D']
        added = {p for s, p in changes if s == 'A'}
        datasets = sorted({dataset_of(p) for _, p in changes} - {None})

    if args.datasets_json:
        print(json.dumps([d for d in datasets if (ROOT / d).is_dir()]))
        return 0

    for path in files:
        if path.lower().endswith('.pdf') and (args.all or path in added):
            error(path, 'do not commit article PDFs or publisher supplements; link to them with '
                        'a DOI or a stable URL')
        if not args.all and dataset_of(path) is None:
            warn(path, 'outside a data-set directory; contributors change only their data '
                       'set\'s directory (CONTRIBUTING.md, section 9)')
        check_file(path)
    for d in datasets:
        check_dataset(d)
    if not args.all and not files and not datasets:
        print('no changed files')
    return report()


if __name__ == '__main__':
    sys.exit(main())
