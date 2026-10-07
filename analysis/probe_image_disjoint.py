"""Image-disjoint presence, sufficiency, and causal-use probes (inference only).

Probes are fit on 1,024 images and evaluated on 256 disjoint images at t in [0.3, 0.7);
see docs/REPRODUCING.md. Checkpoints are read from outputs/in1k_<name>/checkpoints/
0400000.pt, falling back to the released checkpoints/diffusion/in1k_<name>.pt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image, PngImagePlugin
import torch
import torch.nn.functional as F

MODELS = ['vanilla', 'mae_res', 'mae_lapk5', 'repa', 'aa_res', 'lapk3',
          'uca_ls2.0', 'repa_s1', 'uca_ls2.0_s1']
BUNDLE_MODELS = ['vanilla_s1', 'mae_res', 'mae_lapk5_s1', 'repa_s1']
SEED = 20260916
NTRAIN, NTEST, NTOK, CTOK = 1024, 256, 80, 48
BOOT = 2000


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_digest(x):
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


def save_json(value, path):
    temp = path.with_suffix('.json.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def selected(x, idx):
    return x[torch.arange(len(x), device=x.device)[:, None], idx.to(x.device)]


def setup(root):
    os.chdir(root)
    sys.path.insert(0, str(root))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available(), 'Use a scheduled GPU allocation.'


@torch.inference_mode()
def prepare(args):
    from utils import load_encoders
    from train_purifier import encoder_tokens
    from models.purifier import patchify_latent, linear_residual

    data = args.data
    labels_path = data / 'vae-sd/dataset.json'
    labels = sorted(json.loads(labels_path.read_text())['labels'])
    ids = torch.randperm(len(labels), generator=torch.Generator().manual_seed(SEED))[:NTRAIN + NTEST]
    assert len(set(ids.tolist())) == len(ids)
    assert not set(ids[:NTRAIN].tolist()) & set(ids[NTRAIN:].tolist())
    manifest = [{
        'position': j, 'dataset_index': int(i), 'name': labels[i][0],
        'class': int(labels[i][1]), 'split': 'fit' if j < NTRAIN else 'evaluation'
    } for j, i in enumerate(ids)]
    save_json({'data_dir': str(data), 'manifest_sha256': digest(labels_path),
               'sampling_seed': SEED, 'images': manifest}, args.out / 'manifest.json')
    g = torch.Generator().manual_seed(SEED + 1)
    posterior_noise = torch.randn(len(ids), 4, 32, 32, generator=g)
    noise = torch.randn(len(ids), 4, 32, 32, generator=g)
    times = .3 + .4 * torch.rand(len(ids), generator=g)
    token_ids = torch.stack([torch.randperm(256, generator=g)[:NTOK] for _ in ids])
    encs, types, _ = load_encoders('dinov2-vit-b', 'cuda', 256)
    encoder = encs[0].eval().requires_grad_(False)
    PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024
    fields = {k: [] for k in ['x', 'z', 'xp', 'res']}
    for start in range(0, len(ids), 64):
        rows = manifest[start:start + 64]
        raw, moments = [], []
        for row in rows:
            name = Path(row['name'])
            # Fail on an invalid image, rather than silently substituting a
            # neighboring image that might cross the held-out boundary.
            with Image.open(data / 'images' / name.with_suffix('.png')) as im:
                raw.append(np.array(im.convert('RGB')))
            moments.append(np.load(data / 'vae-sd' / name))
        raw = torch.from_numpy(np.stack(raw)).permute(0, 3, 1, 2)
        mom = torch.from_numpy(np.stack(moments)).squeeze(1)
        mean, std = mom.chunk(2, 1)
        x = (mean + std * posterior_noise[start:start + len(rows)]) * .18215
        z = torch.cat([encoder_tokens(encoder, types[0], r, 'cuda', autocast=False)
                       for r in raw.split(args.batch_size)])
        assert z.shape[1:] == (256, 768)
        pred = linear_residual(z, patchify_latent(x.cuda(), 2), 'xpredictive')
        idx = token_ids[start:start + len(rows)]
        fields['x'].append(x)
        fields['z'].append(selected(z, idx).cpu())
        fields['xp'].append(selected(pred, idx[:, :CTOK]).cpu())
        fields['res'].append(selected(z - pred, idx[:, :CTOK]).cpu())
        # Full component tensors are small enough to cache per batch and allow
        # evaluation cosines over all 256 tokens, as in the old component test.
        if start >= NTRAIN:
            torch.save({'z': z.cpu(), 'xp': pred.cpu(), 'res': (z - pred).cpu()},
                       args.out / f'components_{start:04d}.pt')
        print(f'PREPARE {start + len(rows)}/{len(ids)}', flush=True)
    cache = {k: torch.cat(v) for k, v in fields.items()}
    cache.update(noise=noise, times=times, token_ids=token_ids,
                 labels=torch.tensor([r['class'] for r in manifest]))
    for k, v in cache.items():
        assert torch.isfinite(v).all(), k
    torch.save(cache, args.out / 'inputs.pt')
    sources = ['analysis/diagnostics.py', 'analysis/run_diagnostics.py', 'models/sit.py',
               'models/purifier.py', 'train_purifier.py', 'utils.py']
    source_dir = args.out / 'source_snapshot'
    source_dir.mkdir(exist_ok=True)
    for path in sources:
        target = source_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((args.root / path).read_bytes())
    metadata = {
        'protocol': 'Image-disjoint probes, mid-noise',
        'fit_images': NTRAIN, 'evaluation_images': NTEST,
        'tokens_per_image': NTOK, 'component_tokens_per_image': CTOK,
        'fit_tokens': NTRAIN * NTOK, 'evaluation_tokens': NTEST * NTOK,
        'image_disjoint_assertion': True,
        'input_tensor_sha256': {k: tensor_digest(v) for k, v in cache.items()},
        'source_sha256': {p: digest(args.root / p) for p in sources},
        'audit_script_sha256': digest(Path(__file__).resolve()),
        'software': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                     'python': platform.python_version(), 'numpy': np.__version__},
        'gpu': torch.cuda.get_device_name(),
        'inference_dtype': 'float32, TF32 disabled', 'ridge_solve_dtype': 'float64',
        'bootstrap_replicates': BOOT,
        'scope': 'Probe fit/evaluation images disjoint; denoiser was trained on the dataset.'
    }
    save_json(metadata, args.out / 'metadata.json')


@torch.inference_mode()
def fit_ridge(H, Y):
    H, Y = H.cuda().double(), Y.cuda().double()
    hm, ym = H.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    H -= hm
    Y -= ym
    gram = H.T @ H
    lam = .001 * gram.diagonal().mean().clamp_min(1e-8)
    W = torch.linalg.solve(gram + lam * torch.eye(H.shape[1], device='cuda'), H.T @ Y)
    b = ym - hm @ W
    assert torch.isfinite(W).all()
    return W, b


def bootstrap_weights(n):
    rng = np.random.default_rng(SEED + 99)
    draws = rng.integers(0, n, size=(BOOT, n))
    return np.stack([np.bincount(row, minlength=n) for row in draws]).astype(np.float64)


def interval(samples):
    return [float(x) for x in np.quantile(samples, [.025, .975])]


@torch.inference_mode()
def evaluate_probe(W, b, H, Y, weights):
    H, Y = H.cuda().double(), Y.cuda().double()
    pred = H @ W + b
    # One row per image, preserving dependence between its patch tokens.
    sse = ((Y - pred) ** 2).sum((1, 2))
    sy = Y.sum(1)
    sy2 = (Y * Y).sum((1, 2))
    n = torch.full((len(Y),), Y.shape[1], dtype=torch.float64, device='cuda')
    sst = sy2.sum() - sy.sum(0).square().sum() / n.sum()
    r2 = float(1 - sse.sum() / sst)
    w = torch.as_tensor(weights, device='cuda')
    bst = w @ sy2 - (w @ sy).square().sum(1) / (w @ n)
    boot = (1 - (w @ sse) / bst).cpu().numpy()
    stats = {k: v.cpu().numpy() for k, v in {'sse': sse, 'sy': sy, 'sy2': sy2, 'n': n}.items()}
    return {'r2': r2, 'ci95_images': interval(boot)}, stats, boot


@torch.inference_mode()
def paired_probes(H, Y, weights, token_seed):
    W, b = fit_ridge(H[:NTRAIN].flatten(0, 1), Y[:NTRAIN].flatten(0, 1))
    image, stats, boot = evaluate_probe(W, b, H[NTRAIN:], Y[NTRAIN:], weights)
    perm = torch.randperm(H.shape[0] * H.shape[1], generator=torch.Generator().manual_seed(token_seed))
    nfit = NTRAIN * H.shape[1]
    hf, yf = H.flatten(0, 1), Y.flatten(0, 1)
    wt, bt = fit_ridge(hf[perm[:nfit]], yf[perm[:nfit]])
    he, ye = hf[perm[nfit:]].cuda().double(), yf[perm[nfit:]].cuda().double()
    token = float(1 - ((ye - he @ wt - bt) ** 2).sum() / ((ye - ye.mean(0)) ** 2).sum())
    train_ids = set((perm[:nfit] // H.shape[1]).tolist())
    eval_ids = set((perm[nfit:] // H.shape[1]).tolist())
    result = {'image_split': image, 'paired_token_split_r2': token,
              'token_minus_image_r2': token - image['r2'],
              'paired_token_split_overlapping_images': len(train_ids & eval_ids)}
    return result, stats, boot, W


@torch.inference_mode()
def projector(W, rank=256):
    U, _, _ = torch.linalg.svd(W, full_matrices=False)
    q = U[:, :rank]
    pi = (q @ q.T).float()
    assert (pi @ pi - pi).abs().max() < 1e-5
    return pi


@torch.inference_mode()
def run_model(args, name):
    from utils import build_model_from_ckpt
    from diagnostics import capture_hidden, linear_interpolant, patchify_velocity, ablate_forward
    ckpath = args.root / f'outputs/in1k_{name}/checkpoints/0400000.pt'
    if not ckpath.exists():
        ckpath = args.root / f'checkpoints/diffusion/in1k_{name}.pt'
    checkpoint_hash = digest(ckpath)
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    model, _ = build_model_from_ckpt(ck, 'cuda')  # strict load
    layer = model.encoder_depth - 1
    assert layer == 7 and model.projector_type == 'mlp'
    model.requires_grad_(False)
    del ck
    c = torch.load(args.out / 'inputs.pt', map_location='cpu', weights_only=False)
    Hs, As, Vs, cosines = [], [], [], []
    hidden_sum = torch.zeros(768, dtype=torch.float64, device='cuda')
    component_cosines = {k: [] for k in ['raw', 'xp', 'res']}
    for start in range(0, NTRAIN + NTEST, args.batch_size):
        stop = min(start + args.batch_size, NTRAIN + NTEST)
        x, no, t, lab = [c[k][start:stop].cuda() for k in ['x', 'noise', 'times', 'labels']]
        xt, vel = linear_interpolant(x, t, no)
        h, _ = capture_hidden(model, xt, t, lab, layer)
        a = model.projectors[0](h)
        if start < NTRAIN:
            hidden_sum += h.double().sum((0, 1))
        idx = c['token_ids'][start:stop]
        hs, aa = selected(h, idx).cpu(), selected(a, idx).cpu()
        Hs.append(hs)
        As.append(aa)
        Vs.append(selected(patchify_velocity(vel, 2), idx).cpu())
        if start >= NTRAIN and a.shape[-1] == 768:
            cosines.append(F.cosine_similarity(aa, c['z'][start:stop], dim=-1).mean(1))
        if name == 'repa' and start >= NTRAIN:
            block = (start // 64) * 64
            comps = torch.load(args.out / f'components_{block:04d}.pt', weights_only=False)
            offset = start - block
            for key, ck in [('raw', 'z'), ('xp', 'xp'), ('res', 'res')]:
                comp = comps[ck][offset:offset + len(x)].cuda()
                component_cosines[key].append(F.cosine_similarity(a, comp, dim=-1).mean(1).cpu())
        if stop % 256 == 0:
            print(f'EXTRACT {name} {stop}/{NTRAIN+NTEST}', flush=True)
    H, A, V = torch.cat(Hs), torch.cat(As), torch.cat(Vs)
    del Hs, As, Vs
    weights = bootstrap_weights(NTEST)
    result = {'model': name, 'checkpoint': str(ckpath), 'checkpoint_sha256': checkpoint_hash,
              'layer': layer, 'noise_bin': [.3, .7], 'training_seed': 1 if name.endswith('_s1') else 0}
    saved, boots = {}, {}
    for key, inp, target in [('presence', H, c['z']), ('sufficiency', A, V)]:
        res, stat, boot, W = paired_probes(inp, target, weights, SEED + 2)
        result[key] = res
        saved.update({key + '_' + k: v for k, v in stat.items()})
        boots[key] = boot
        if key == 'presence':
            pi_target = projector(W)
        print('PROBE', name, key, json.dumps(res), flush=True)
    if cosines:
        vals = torch.cat(cosines).numpy()
        saved['raw_dino_cosine'] = vals
        result['raw_dino_cosine'] = {'mean': float(vals.mean()), 'ci95_images': interval(weights @ vals / NTEST)}
    Wa, _ = fit_ridge(H[:NTRAIN].flatten(0, 1), A[:NTRAIN].flatten(0, 1))
    pis = {'aligned': projector(Wa), 'target': pi_target, 'full': torch.eye(768, device='cuda')}
    if name == 'repa':
        for i in range(5):
            q, _ = torch.linalg.qr(torch.randn(768, 256, generator=torch.Generator().manual_seed(9100 + i), dtype=torch.float64).cuda())
            pis[f'random_{i}'] = (q @ q.T).float()
    hbar = (hidden_sum / (NTRAIN * 256)).float()
    effects = {k: [] for k in pis}
    base_errors = []
    for start in range(NTRAIN, NTRAIN + NTEST, args.batch_size):
        stop = start + args.batch_size
        x, no, t, lab = [c[k][start:stop].cuda() for k in ['x', 'noise', 'times', 'labels']]
        xt, vel = linear_interpolant(x, t, no)
        baseline = model(xt, t, lab)[0]
        err = (baseline.double() - vel.double()).square().flatten(1).mean(1)
        base_errors.append(err.cpu())
        for key, pi in pis.items():
            v = ablate_forward(model, xt, t, lab, layer, pi, hbar, 'mean')
            e = (v.double() - vel.double()).square().flatten(1).mean(1)
            effects[key].append((e - err).cpu())
    effects = {k: torch.cat(v).numpy() for k, v in effects.items()}
    base_errors = torch.cat(base_errors).numpy()
    saved['baseline_mse'] = base_errors
    result['baseline_velocity_mse'] = float(base_errors.mean())
    full_boot = weights @ effects['full']
    assert effects['full'].mean() > 0 and (full_boot > 0).all()
    result['use'] = {}
    for key, values in effects.items():
        boot = 100 * (weights @ values) / full_boot
        result['use'][key] = {'delta_mse': float(values.mean()),
                              'normalized_percent': float(100 * values.mean() / effects['full'].mean()),
                              'ci95_images_percent': interval(boot)}
        saved['use_' + key] = values
        boots['use_' + key] = boot
    if name == 'repa':
        rand = np.mean([effects[f'random_{i}'] for i in range(5)], axis=0)
        result['use']['random_mean'] = {'normalized_percent': float(100 * rand.mean() / effects['full'].mean()),
                                       'ci95_images_percent': interval(100 * (weights @ rand) / full_boot)}
        saved['use_random_mean'] = rand
        boots['use_random_mean'] = 100 * (weights @ rand) / full_boot
        result['components'] = {}
        for key, target in [('raw', c['z'][:, :CTOK]), ('xp', c['xp']), ('res', c['res'])]:
            res, stat, boot, _ = paired_probes(H[:, :CTOK], target, weights, SEED + 3)
            vals = torch.cat(component_cosines[key]).numpy()
            res['readout_cosine'] = {'mean': float(vals.mean()), 'ci95_images': interval(weights @ vals / NTEST)}
            result['components'][key] = res
            saved.update({'component_' + key + '_' + k: v for k, v in stat.items()})
            saved['component_' + key + '_cosine'] = vals
            boots['component_' + key] = boot
            boots['component_' + key + '_cosine'] = weights @ vals / NTEST
    for v in saved.values():
        assert np.isfinite(v).all()
    np.savez_compressed(args.out / f'{name}_per_image.npz', **saved)
    np.savez_compressed(args.out / f'{name}_bootstrap.npz', **boots)
    save_json(result, args.out / f'{name}.json')
    print('MODEL DONE', name, json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--phase', choices=['prepare', 'models', 'all'], default='all')
    ap.add_argument('--models', nargs='+', default=MODELS, choices=sorted(set(MODELS + BUNDLE_MODELS)))
    args = ap.parse_args()
    assert 64 % args.batch_size == 0
    args.root, args.out = args.root.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    setup(args.root)
    if args.phase in ['prepare', 'all']:
        if (args.out / 'inputs.pt').exists():
            raise FileExistsError('Use a fresh output directory or --phase models.')
        prepare(args)
        gc.collect()
        torch.cuda.empty_cache()
    if args.phase in ['models', 'all']:
        for name in args.models:
            if (args.out / f'{name}.json').exists():
                print('SKIP completed', name, flush=True)
                continue
            start = time.monotonic()
            run_model(args, name)
            print(f'ELAPSED {name} {time.monotonic()-start:.1f}s', flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    print('AUDIT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
