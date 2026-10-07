"""Per-channel latent statistics for a swapped frozen VAE, in the REPA-E stats format.

A released VAE's `scaling_factor` is not a reliable normalization (zelaki/eq-vae and
zelaki/eq-vae-ema ship 0.18215 and 0.0683 for the same architecture), so the statistics
are measured on training images. Posterior samples z = mean + std * eps are drawn exactly
as on train.py's --vae-encode-online path (images from dataset.CustomDataset scaled by
x/127.5 - 1), and per-channel mean/std are accumulated in float64.

Output: latents_scale = 1/std, latents_bias = mean, each (1,C,1,1); utils.load_latents_stats
converts this to the training convention z * scale + bias.

  python preprocessing/compute_latent_stats.py --vae-path zelaki/eq-vae \
      --data-dir data/in1k_256 --n-images 8192 --out assets/eqvae-latents-stats.pt
"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ChannelAccumulator:
    """Streaming per-channel mean/std over (N,C,H,W) batches, accumulated in float64."""

    def __init__(self, c):
        self.n = 0
        self.s = torch.zeros(c, dtype=torch.float64)
        self.ss = torch.zeros(c, dtype=torch.float64)

    def update(self, z):
        z = z.double()
        self.n += z.shape[0] * z.shape[2] * z.shape[3]
        self.s += z.sum(dim=(0, 2, 3)).cpu()
        self.ss += (z ** 2).sum(dim=(0, 2, 3)).cpu()

    def mean_std(self):
        mean = self.s / self.n
        var = (self.ss / self.n) - mean ** 2
        return mean, var.clamp_min(0).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-path", required=True, help="HF id or local dir of the diffusers AutoencoderKL")
    ap.add_argument("--data-dir", default="data/in1k_256",
                    help="dataset with images/ and <latents-dir>/dataset.json")
    ap.add_argument("--latents-dir", default="vae-sd", help="only used to find the label manifest")
    ap.add_argument("--n-images", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", required=True, help="stats .pt to write (REPA-E format)")
    args = ap.parse_args()

    from dataset import CustomDataset
    from diffusers.models import AutoencoderKL
    from torch.utils.data import DataLoader, Subset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CustomDataset(args.data_dir, latents_dir=args.latents_dir)
    assert args.n_images <= len(ds), f"--n-images {args.n_images} > dataset size {len(ds)}"
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(ds), size=args.n_images, replace=False))
    vae = AutoencoderKL.from_pretrained(args.vae_path).eval().requires_grad_(False).to(device)
    dl = DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, drop_last=False)

    C = int(vae.config.latent_channels)
    acc = ChannelAccumulator(C)
    torch.manual_seed(args.seed)
    for raw_image, _cached_moments, _y in dl:
        x = raw_image.to(device).float() / 127.5 - 1.0
        with torch.no_grad():
            d = vae.encode(x).latent_dist
            mean, std = d.mean.float(), d.std.float()
            acc.update(mean + std * torch.randn_like(mean))
    z_mean, z_std = acc.mean_std()
    print(f"per-channel mean {[round(float(v), 4) for v in z_mean]}, "
          f"std {[round(float(v), 4) for v in z_std]} ({acc.n:,} positions per channel)")

    scale = (1.0 / z_std).float().view(1, C, 1, 1).contiguous()
    bias = z_mean.float().view(1, C, 1, 1).contiguous()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"latents_scale": scale, "latents_bias": bias}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
