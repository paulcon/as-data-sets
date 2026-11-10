"""Check that every reference link in a notebook's markdown cells resolves.

Jupyter renders each markdown cell on its own, so a reference link such as
``[[1]][R1]`` is a link only in the cell that holds the definition
``[R1]: https://...``. A reference whose definition sits in another cell (or
nowhere) renders as plain text. Write such links inline instead:
``[[1]](https://doi.org/...)``.

Usage:
    python check_notebook_links.py              # every notebook in the repository
    python check_notebook_links.py a.ipynb ...  # the given notebooks

Math ($...$, $$...$$, \\(...\\), \\[...\\]) and code (`...`, fenced blocks)
are skipped, so LaTeX such as [\\mathrm{H^+}][\\mathrm{CO_3^{2-}}] is not
reported. Exits 1 if any reference does not resolve, 0 otherwise.
Standard library only.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SKIP_DIRS = {'.venv', '.git', '.ipynb_checkpoints', 'node_modules'}

# Spans removed before looking for links: fenced code, inline code, display
# and inline math. Order matters: fences and $$ before ` and $.
STRIP = re.compile(
    r'^\s*(```|~~~).*?^\s*\1'      # fenced code block
    r'|`+[^`]*?`+'                  # inline code
    r'|\$\$.*?\$\$'                 # display math
    r'|\\\[.*?\\\]|\\\(.*?\\\)'     # \[...\] and \(...\)
    r'|(?<!\\)\$[^$\n]+?(?<!\\)\$',  # inline math
    re.S | re.M)
# [text][label] (text may hold one level of brackets, as in [[1]][R1]) and
# the collapsed form [label][].
USE = re.compile(r'\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\[([^\[\]]*)\]')
DEFN = re.compile(r'^ {0,3}\[([^\]]+)\]:[ \t]*\S', re.M)


def norm(label):
    return ' '.join(label.split()).lower()


def notebooks(args):
    if args:
        return [pathlib.Path(a) for a in args]
    return sorted(p for p in ROOT.rglob('*.ipynb')
                  if not SKIP_DIRS.intersection(p.relative_to(ROOT).parts))


def check(path):
    """Return a list of (cell index, reference text, label, where defined)."""
    cells = json.loads(path.read_text(encoding='utf-8'))['cells']
    md = [(i, ''.join(c['source'])) for i, c in enumerate(cells)
          if c['cell_type'] == 'markdown']
    defined_in = {}
    for i, src in md:
        for label in DEFN.findall(STRIP.sub('', src)):
            defined_in.setdefault(norm(label), []).append(i)
    bad = []
    for i, src in md:
        for m in USE.finditer(STRIP.sub('', src)):
            label = norm(m.group(2) or m.group(1))
            where = defined_in.get(label, [])
            if i not in where:
                bad.append((i, m.group(0), label, where))
    return bad


def main(args):
    n_bad = n_nb = 0
    for path in notebooks(args):
        bad = check(path)
        n_nb += 1
        if not bad:
            continue
        try:
            name = path.resolve().relative_to(ROOT)
        except ValueError:
            name = path
        print(f'{name}: {len(bad)} unresolved reference link(s)')
        for i, text, label, where in bad:
            hint = (f'defined in cell {", ".join(map(str, where))}' if where
                    else 'not defined in any cell')
            print(f'  cell {i}: {text}  ([{label}] {hint})')
        n_bad += len(bad)
    print(f'{n_nb} notebook(s) checked, {n_bad} unresolved reference link(s)')
    return 1 if n_bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
