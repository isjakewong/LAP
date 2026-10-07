"""
Development generation metric: Frechet distance between final normalized DINOv2-B CLS
features of generated and real images (FID-DINOv2). Not Inception FID.

Samples EMA weights with the Euler ODE sampler, decodes with the run's VAE, and compares
against a fresh shuffled real reference of the same size.

Example:
  python analysis/gen_fid.py --ckpt outputs/dev200_mae_raw/checkpoints/0050000.pt \
    --data-dir data/in256_dev200 --num-samples 10000 --num-classes 200 \
    --steps 50 --batch 50 --out results/dev_mae_new.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

import argparse, json
import numpy as np
import torch


def frechet(mu1, s1, mu2, s2):
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean = sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


@torch.no_grad()
def dino_feats(x255, encoder, enc_type, device):
    """x255: (B,3,256,256) in [0,255] -> DINOv2 CLS features (B, D)."""
    from utils import preprocess_raw_image
    x = preprocess_raw_image(x255.to(device), enc_type)
    z = encoder.forward_features(x)
    if "dinov2" in enc_type:
        z = z["x_norm_clstoken"]
    return z.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--num-classes", type=int, default=200)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--out", default="fid.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from utils import build_model_from_ckpt
    from samplers import euler_sampler
    from diffusers.models import AutoencoderKL
    from utils import load_encoders
    from torch.utils.data import DataLoader
    from dataset import CustomDataset

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, get = build_model_from_ckpt(ckpt, device)
    vae = AutoencoderKL.from_pretrained(get("vae_path", "stabilityai/sd-vae-ft-mse")).eval().to(device)
    encoders, enc_types, _ = load_encoders("dinov2-vit-b", device, 256)
    enc, et = encoders[0], enc_types[0]
    scale = 0.18215
    bias = 0.
    if get("latents_stats", None):
        from utils import load_latents_stats
        scale, bias = load_latents_stats(get("latents_stats"), device)

    # generated features
    G, n = [], 0
    while n < args.num_samples:
        b = min(args.batch, args.num_samples - n)
        xT = torch.randn(b, 4, get("resolution", 256) // 8, get("resolution", 256) // 8, device=device)
        y = torch.randint(0, args.num_classes, (b,), device=device)
        with torch.no_grad():
            samp = euler_sampler(model, xT, y, num_steps=args.steps, cfg_scale=args.cfg,
                                 guidance_low=0., guidance_high=1., path_type="linear", heun=False).float()
            imgs = vae.decode((samp - bias) / scale).sample   # (b,3,256,256) in [-1,1]
            imgs = ((imgs + 1) / 2).clamp(0, 1) * 255.0
        G.append(dino_feats(imgs, enc, et, device))
        n += b
        if n % 1000 == 0:
            print(f"  generated {n}/{args.num_samples}", flush=True)
    G = torch.cat(G).numpy()

    # real features
    dl = DataLoader(CustomDataset(args.data_dir), batch_size=args.batch, shuffle=True, num_workers=4)
    R, m = [], 0
    for raw, _xm, _yl in dl:
        if m >= args.num_samples:
            break
        R.append(dino_feats(raw.float(), enc, et, device))
        m += raw.shape[0]
    R = torch.cat(R)[:args.num_samples].numpy()

    mu_g, s_g = G.mean(0), np.cov(G, rowvar=False)
    mu_r, s_r = R.mean(0), np.cov(R, rowvar=False)
    fd = frechet(mu_g, s_g, mu_r, s_r)
    print(f"FD-DINOv2 = {fd:.4f}  (n_gen={len(G)}, n_real={len(R)})")
    with open(args.out, "w") as f:
        json.dump({"ckpt": args.ckpt, "fd_dinov2": fd, "n": len(G)}, f)


if __name__ == "__main__":
    main()
