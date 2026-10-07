"""
Encoder utility R^2(X | target): how well clean-image teacher tokens linearly predict the
patchified clean latent. Per-token ridge probe with a contiguous 80/20 fit/evaluation
split; no diffusion checkpoint is needed because this is a property of the target alone.

Usage:
  python target_utility.py --enc mae-vit-l --data-dir data/in256_dev200 \
      --out results/mae_utility_new.json
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

import diagnostics as dg
from dataset import CustomDataset
from train_purifier import sample_posterior
from utils import load_encoders, preprocess_raw_image


def r2_of(features, targets, device):
    n = int(0.8 * len(features))
    w, b = dg.fit_ridge(features[:n].to(device), targets[:n].to(device))
    return dg.r2_score(w, b, features[n:].to(device), targets[n:].to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--enc", required=True, help="e.g. mae-vit-l, dinov2-vit-b, jepa-vit-h")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    encoders, encoder_types, _ = load_encoders(args.enc, device, 256)
    enc, enc_type = encoders[0], encoder_types[0]
    p = 2  # SiT-B/2 patch size; 32/2 = 16x16 tokens, matches all 256-token encoders
    dl = DataLoader(CustomDataset(args.data_dir), batch_size=32, shuffle=True, num_workers=4)

    Z, X = [], []
    for bi, (raw, xm, yl) in enumerate(dl):
        if bi >= args.batches:
            break
        raw = raw.to(device)
        x = sample_posterior(xm.squeeze(1).to(device))
        with torch.no_grad():
            z = enc.forward_features(preprocess_raw_image(raw, enc_type))
        if 'mocov3' in enc_type:
            z = z[:, 1:]
        if 'dinov2' in enc_type:
            z = z['x_norm_patchtokens']
        xp = dg.patchify_velocity(x, p)
        assert z.shape[1] == xp.shape[1], f"token grids differ: {z.shape} vs {xp.shape}"
        Z.append(z.reshape(-1, z.shape[-1]).float().cpu())
        X.append(xp.reshape(-1, xp.shape[-1]).float().cpu())
    Z, X = torch.cat(Z), torch.cat(X)

    out = args.out or f"target_utility_{args.enc}.json"
    res = {f"R2(X|{enc_type}_clean)": r2_of(Z, X, device), "n_tokens": int(Z.shape[0])}
    for k, v in res.items():
        print(f"{k:32s} = {v}")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
