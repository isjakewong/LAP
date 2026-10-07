"""Recompute the main-text table means and sample SDs from released per-seed metrics."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', help='experiment group, e.g. teachers, recipes, tokenizers, mechanism, aerial')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    registry = json.loads((ROOT/'configs/experiments.json').read_text())
    groups = {g for e in registry.values() for g in e['groups']}
    if a.group and a.group not in groups:
        p.error(f'unknown group; choose from {sorted(groups)}')
    rows = {}
    for name, e in registry.items():
        if not e.get('metrics') or (a.group and a.group not in e['groups']):
            continue
        base = name.rsplit('_s', 1)[0] if e['training_seed'] else name
        rows.setdefault(base, []).append(e['metrics'])
    f = a.output.open('w', newline='') if a.output else sys.stdout
    writer = csv.writer(f)
    writer.writerow(['experiment', 'seeds', 'FID mean', 'FID SD', 'IS mean', 'IS SD',
                     'precision mean', 'precision SD', 'recall mean', 'recall SD'])
    for name, metrics in sorted(rows.items()):
        values = [name, len(metrics)]
        for key in ['fid','inception_score','precision','recall']:
            xs = [m[key] for m in metrics]
            values += [f'{statistics.mean(xs):.6f}', f'{statistics.stdev(xs):.6f}' if len(xs)>1 else '']
        writer.writerow(values)
    if a.output:
        f.close()


if __name__ == '__main__':
    main()
