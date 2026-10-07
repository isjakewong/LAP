"""Sample EMA checkpoints with the paper's SDE-250 protocol (single GPU or torchrun)."""
import argparse
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from diffusers import AutoencoderKL
from tqdm import trange

from run_diagnostics import build_model_from_ckpt
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_latents_stats


def pack_samples(folder, count):
    """Write ADM arr_0.npy into NPZ without materializing 10 GB of pixels in RAM."""
    first = np.asarray(Image.open(folder / '000000.png').convert('RGB'))
    npy = folder / 'arr_0.npy'
    arr = np.lib.format.open_memmap(npy, mode='w+', dtype=np.uint8, shape=(count, *first.shape))
    for i in trange(count, desc='Packing samples'):
        arr[i] = np.asarray(Image.open(folder / f'{i:06d}.png').convert('RGB'))
    arr.flush()
    del arr
    output = Path(str(folder) + '.npz')
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as f:
        f.write(npy, 'arr_0.npy')
    npy.unlink()
    return output


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--num-samples', type=int, default=50000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--sample-num-classes', type=int)
    p.add_argument('--steps', type=int, default=250)
    p.add_argument('--mode', choices=['sde', 'ode'], default='sde')
    p.add_argument('--cfg-scale', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--vae-path')
    p.add_argument('--latents-stats')
    p.add_argument('--trusted-legacy', action='store_true', help='Allow pickle only for your own original training checkpoints')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if a.num_samples <= 0 or a.batch_size <= 0 or a.steps < 2:
        p.error('Sample count and batch size must be positive; steps must be at least 2.')
    if a.cfg_scale < 1:
        p.error('--cfg-scale must be at least 1; 1 means no guidance.')
    if not torch.cuda.is_available():
        p.error('Sampling requires a CUDA GPU.')
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl')
    if ((a.out.exists() and any(a.out.iterdir())) or Path(str(a.out) + '.npz').exists()) and not a.overwrite:
        p.error('Output directory is not empty; choose a new directory or use --overwrite.')
    if rank == 0:
        a.out.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(a.seed * world + rank)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=not a.trusted_legacy)
    model, get = build_model_from_ckpt(ck, device)
    vae_path = a.vae_path or get('vae_path', 'stabilityai/sd-vae-ft-mse')
    stats = a.latents_stats or get('latents_stats')
    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
    scale, bias = load_latents_stats(stats, device) if stats else (0.18215, 0.0)
    nclasses = a.sample_num_classes or get('num_classes', 1000)
    if not 0 < nclasses <= get('num_classes', 1000):
        p.error('--sample-num-classes is outside the checkpoint label vocabulary.')
    size = get('resolution', 256) // 8
    total = math.ceil(a.num_samples / (world * a.batch_size))
    sampler = euler_maruyama_sampler if a.mode == 'sde' else euler_sampler
    for it in trange(total, disable=rank != 0):
        z = torch.randn(a.batch_size, 4, size, size, device=device)
        y = torch.randint(0, nclasses, (a.batch_size,), device=device)
        x = sampler(model, z, y, num_steps=a.steps, cfg_scale=a.cfg_scale,
                    path_type=get('path_type', 'linear'))
        pixels = vae.decode((x.float() - bias) / scale).sample
        pixels = (127.5 * (pixels + 1)).clamp(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for j, pixel in enumerate(pixels):
            idx = it * world * a.batch_size + j * world + rank
            if idx < a.num_samples:
                Image.fromarray(pixel).save(a.out / f'{idx:06d}.png')
    if world > 1:
        dist.barrier()
    if rank == 0:
        path = pack_samples(a.out, a.num_samples)
        (a.out / 'sampling.json').write_text(json.dumps(dict(vars(a), out=str(a.out), world_size=world), indent=2) + '\n')
        print(path)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
