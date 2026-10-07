"""List and run the paper recipes. Run from the repository root."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)  # pass other flags through intact
    p.add_argument('action', choices=['list', 'train', 'sample', 'purifier'])
    p.add_argument('experiment', nargs='?')
    p.add_argument('--group')
    p.add_argument('--data-dir')
    p.add_argument('--checkpoint')
    p.add_argument('--output-dir', default='outputs')
    p.add_argument('--gpus', type=int, default=1, help='Sampling GPUs; training recipes use one GPU')
    p.add_argument('--dry-run', action='store_true')
    a, extra = p.parse_known_args()
    experiments = json.loads((ROOT / 'configs/experiments.json').read_text())
    if a.action == 'list':
        for name, v in experiments.items():
            if a.group is None or a.group in v['groups']:
                print(f"{name:42s} {','.join(v['groups']):30s} {v['training_steps']:7d} steps")
        return
    if a.experiment not in experiments:
        p.error('Select an experiment from: python reproduce.py list')
    e = experiments[a.experiment]
    c = json.loads((ROOT / e['config']).read_text())
    if a.action == 'train':
        if a.gpus != 1:
            p.error('Paper training uses a physical batch of 64 on one GPU, with gradient accumulation for batch 256.')
        cmd = [sys.executable, 'train.py', '--config', e['config'], '--exp-name', a.experiment,
               '--seed', str(e['training_seed']), '--max-train-steps', str(e['training_steps']),
               '--output-dir', a.output_dir]
        if a.data_dir:
            cmd += ['--data-dir', a.data_dir]
    elif a.action == 'purifier':
        asset = c.get('purifier_ckpt')
        if not asset:
            p.error('This recipe does not use a nonlinear purifier.')
        configs = json.loads((ROOT / 'configs/purifiers.json').read_text())
        if Path(asset).name not in configs:
            p.error('No verified purifier training configuration for this asset.')
        pc = dict(configs[Path(asset).name])
        pc['data_dir'] = a.data_dir or c.get('data_dir', 'data/in1k_256')
        pc['out'] = asset
        if (ROOT / asset).exists() and '--out' not in extra:
            p.error(f'{asset} already exists; remove it or pass --out <path>.')
        cmd = [sys.executable, 'train_purifier.py']
        for key, value in pc.items():
            if value is None:
                continue
            flag = '--' + key.replace('_', '-')
            cmd += ([flag] if value else []) if isinstance(value, bool) else [flag, str(value)]
    else:
        ckpt = a.checkpoint or str(Path(a.output_dir) / a.experiment / 'checkpoints' / f"{e['training_steps']:07d}.pt")
        cmd = [sys.executable]
        if a.gpus > 1:
            cmd += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={a.gpus}']
        cmd += ['sample.py', '--ckpt', ckpt, '--out', f'samples/{a.experiment}',
                '--num-samples', str(e['num_samples']), '--sample-num-classes', str(e['sample_num_classes'])]
        if 'development' in e['groups']:
            cmd += ['--mode', 'ode', '--steps', '50']
    cmd += extra
    print(shlex.join(cmd), flush=True)
    if not a.dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
