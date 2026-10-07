"""
Exact constant-readout baseline C_0 for each encoder and alignment target.

C_0(y) = || E[ (1/N) sum_i yhat_i ] ||, yhat_i = y_i / ||y_i||: the best expected
cosine attainable by ONE fixed unit direction shared across tokens and images
(Appendix B, Proposition 5). Also reports C_0^pos = (1/N) sum_i || E[yhat_i] ||, the
best expected cosine of a position-dependent constant readout (>= C_0), the mean-token
readout cosine (a lower bound on C_0), token-norm statistics, and the inverse-norm
moment E[1/||y_i||] used by the smoothed bounds.

Targets follow train.py exactly, on consecutive batches of 64 images:
  raw, centered (per-batch mean token removed), lapl (per-batch affine residual),
  lapn (frozen purifier residual), and for DINOv2 also xpred (predictable component).

Usage:
  python analysis/constant_baseline.py --data-dir data/in256_dev200 --tag dev200 \
      --purifiers mae-vit-l=assets/purifier_mae_k5.pt,dinov2-vit-b=assets/purifier_dinov2_k3.pt \
      --batches 32 --out results/c0_new.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

import argparse, json, time
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from train import patchify_latent, split_target_by_latent
from utils import load_encoders, preprocess_raw_image
from models.purifier import load_purifier


def get_tokens(enc, enc_type, raw):
    with torch.no_grad():
        z = enc.forward_features(preprocess_raw_image(raw, enc_type))
    if "mocov3" in enc_type:
        z = z[:, 1:]
    if "dinov2" in enc_type:
        z = z["x_norm_patchtokens"]
    return z.float()


class Acc:
    """Streaming accumulators for one target."""
    def __init__(self, D, device, N):
        self.unit_sum = torch.zeros(D, device=device, dtype=torch.float64)
        self.unit_sum_pos = torch.zeros(N, D, device=device, dtype=torch.float64)  # per token position
        self.n_img = 0
        self.raw_sum = torch.zeros(D, device=device, dtype=torch.float64)
        self.n = 0
        self.norm_sum = 0.0
        self.inv_norm_sum = 0.0
        self.min_norm = float("inf")
        self.tokens = []          # subsample kept for the mean-token cosine
    def add(self, y):
        y3 = y.double()                                   # (B, N, D)
        self.unit_sum_pos += (y3 / y3.norm(dim=2, keepdim=True).clamp_min(1e-12)).sum(0)
        self.n_img += y3.shape[0]
        y = y.reshape(-1, y.shape[-1]).double()
        nrm = y.norm(dim=1)
        self.unit_sum += (y / nrm.clamp_min(1e-12)[:, None]).sum(0)
        self.raw_sum += y.sum(0)
        self.n += y.shape[0]
        self.norm_sum += nrm.sum().item()
        self.inv_norm_sum += (1.0 / nrm.clamp_min(1e-12)).sum().item()
        self.min_norm = min(self.min_norm, nrm.min().item())
        self.tokens.append(y[::8].float().cpu())   # every 8th token
    def summary(self):
        m_unit = (self.unit_sum / self.n).cpu()
        m_raw = (self.raw_sum / self.n).cpu()
        toks = torch.cat(self.tokens)
        cos_meantoken = torch.nn.functional.cosine_similarity(
            toks, m_raw.float().expand_as(toks), dim=1).mean().item()
        cos_bestconst = torch.nn.functional.cosine_similarity(
            toks, m_unit.float().expand_as(toks), dim=1).mean().item()
        m_unit_pos = (self.unit_sum_pos / self.n_img)     # (N, D): E[yhat_i] per position
        return {"C0": m_unit.norm().item(),
                "C0_pos": m_unit_pos.norm(dim=1).mean().item(),
                "C0_pos_max": m_unit_pos.norm(dim=1).max().item(),
                "cos_mean_token_readout": cos_meantoken,
                "cos_best_constant_direction_check": cos_bestconst,
                "mean_token_norm": self.norm_sum / self.n,
                "E_inv_norm": self.inv_norm_sum / self.n,
                "min_token_norm": self.min_norm,
                "n_tokens": self.n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--purifiers", default="",
                    help="Comma-separated enc=purifier[;purifier2] pairs")
    ap.add_argument("--encoders", default="", help="Comma-separated encoders to measure without LAP-N")
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    dl = DataLoader(CustomDataset(args.data_dir), batch_size=64, shuffle=True,
                    num_workers=4, generator=g, drop_last=True)
    batches = []
    for bi, (raw, xm, yl) in enumerate(dl):
        if bi >= args.batches:
            break
        batches.append((raw, xm))
    # one fixed posterior sample per image, shared by every encoder/target
    lat_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    latents = []
    for raw, xm in batches:
        mom = xm.squeeze(1).to(device)
        mean, std = torch.chunk(mom, 2, dim=1)
        latents.append(((mean + std * torch.randn(mean.shape, generator=lat_gen, device=device)) * 0.18215))
    res = {"tag": args.tag, "data_dir": args.data_dir, "batches": len(batches),
           "images": 64 * len(batches), "seed": args.seed, "targets": {}}
    for spec in ([s for s in args.purifiers.split(",") if s] + [s + "=" for s in args.encoders.split(",") if s]):
        enc_name, ckpts = spec.split("=")
        ckpts = ckpts.split(";")
        t0 = time.time()
        encoders, encoder_types, _ = load_encoders(enc_name, device, 256)
        enc, enc_type = encoders[0], encoder_types[0]
        purs = {c: load_purifier(c, device)[0] for c in ckpts if c}
        accs = {}
        for (raw, _xm), x in zip(batches, latents):
            raw = raw.to(device)
            z = get_tokens(enc, enc_type, raw)
            B, N, D = z.shape
            side = int(round(N ** 0.5))
            xp = patchify_latent(x, x.shape[2] // side).float()
            targets = {"raw": z,
                       "centered": z - z.mean(dim=(0, 1), keepdim=True),
                       "lapl": split_target_by_latent(z, x, "residual")}
            if "dinov2" in enc_type:
                targets["xpred"] = split_target_by_latent(z, x, "xpredictive")
            for c, pur in purs.items():
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    targets["lapn:" + c.split("/")[-1]] = z - pur(xp)
            for k, y in targets.items():
                accs.setdefault(k, Acc(D, device, N)).add(y)
        res["targets"][enc_name] = {k: a.summary() for k, a in accs.items()}
        res["targets"][enc_name]["_seconds"] = time.time() - t0
        print(enc_name, json.dumps({k: (round(v["C0"], 4), round(v["C0_pos"], 4)) for k, v in res["targets"][enc_name].items() if not k.startswith("_")}), flush=True)
        del encoders, enc, purs, accs
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("DONE ->", args.out)


if __name__ == "__main__":
    main()
