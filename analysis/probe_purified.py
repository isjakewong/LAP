"""
Class-information audit of purified targets (no diffusion training needed).

For each encoder and target mode
    raw            z
    linres         z - lin_proj_X(z)        (LAP-L, affine residual)
    linxpred       lin_proj_X(z)            (affine reconstruction part)
    pur_k{rf}      z - p_phi(Xp)            (LAP-N residual, per purifier)
    pred_k{rf}     p_phi(Xp)                (what the purifier removes)
measure the token-level variance fraction each mode keeps, linear class probes on
token-mean-pooled features and on individual tokens (logits averaged per image), and
linear CKA against pooled raw DINOv2.

Leak-aware split: train_purifier.py fits on perm[4096:] and validates on perm[:4096]
(same seed/permutation). Here
    probe train      = perm[8192 : 8192+train_images]   (purifier-training images)
    probe val honest = perm[:4096]                       (purifier-held-out images)
    probe val leaked = perm[4096:8192]                   (purifier-training images)
The honest/leaked difference measures purifier memorization; decisions use the honest
split. A purifier size k is feasible when its residual keeps class accuracy within 2 SE
of linres and beats its own predicted component by more than 2 SE.

Example:
  python analysis/probe_purified.py --data-dir data/in256_dev200 \
      --purifiers assets/purifier_mae_k5.pt assets/purifier_dinov2_k3.pt --out probe.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from models.purifier import load_purifier, patchify_latent, linear_residual
from models.purifier import purifier_latent_signature, latent_signature_mismatch
from train_purifier import encoder_tokens
from train_purifier import add_latent_args, check_latent_args, make_latents_fn


def linear_cka(X, Y):
    """Linear CKA between (N,Dx) and (N,Dy), centered."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    num = (X.t() @ Y).pow(2).sum()
    den = torch.sqrt((X.t() @ X).pow(2).sum() * (Y.t() @ Y).pow(2).sum())
    return float(num / (den + 1e-12))


def train_probe(Ftr, ytr, evals, nclass, iters=2500, lr=1e-2, wd=1e-4,
                device="cuda", sd_ref=None, group=None):
    """
    Linear softmax probe. Standardizes with the mode's own mean but a REFERENCE
    per-dim std (sd_ref, e.g. the raw target's) so well-purified dims are not
    noise-amplified. evals: dict name -> (F, y[, group]); if `group` sizes are
    given for an eval set, logits are averaged per group (token-vote probing).
    Returns dict with train_top1 and per-eval top1/top5.
    """
    mu = Ftr.mean(0, keepdim=True)
    sd = (sd_ref if sd_ref is not None else Ftr.std(0, keepdim=True)) + 1e-6
    Ftr = ((Ftr - mu) / sd).to(device)
    ytr = ytr.to(device)
    lin = torch.nn.Linear(Ftr.shape[1], nclass).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    prev = None
    for it in range(iters):
        opt.zero_grad()
        loss = F.cross_entropy(lin(Ftr), ytr)
        loss.backward()
        opt.step()
        if it % 500 == 499:  # plateau early-stop
            cur = loss.item()
            if prev is not None and prev - cur < 1e-4:
                break
            prev = cur
    out = {}
    with torch.no_grad():
        out["train_top1"] = float((lin(Ftr).argmax(-1) == ytr).float().mean())
        for name, pack in evals.items():
            Fe, ye = pack[0], pack[1]
            logits = lin(((Fe - mu) / sd).to(device))
            if len(pack) > 2 and pack[2] is not None:      # token-vote: (N*g, D)
                logits = logits.reshape(-1, pack[2], nclass).mean(dim=1)
            ye = ye.to(device)
            k = min(5, nclass)
            out[f"{name}_top1"] = float((logits.argmax(-1) == ye).float().mean())
            out[f"{name}_top5"] = float(
                (logits.topk(k, dim=-1).indices == ye[:, None]).any(-1).float().mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--purifiers", nargs="+", required=True)
    ap.add_argument("--extra-encoders", default="dinov2-vit-b,mae-vit-l")
    ap.add_argument("--train-images", type=int, default=21000)
    ap.add_argument("--pur-val-images", type=int, default=4096,
                    help="must match train_purifier.py --val-images (split contract)")
    ap.add_argument("--tokens-per-image", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--probe-iters", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="purified_probe.json")
    add_latent_args(ap)
    args = ap.parse_args()
    check_latent_args(ap, args)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from torch.utils.data import DataLoader, Subset
    from dataset import CustomDataset
    from utils import load_encoders

    purifiers = defaultdict(list)   # enc_type_str -> [(rf, purifier, ckpt), ...]
    for path in args.purifiers:
        p, ck = load_purifier(path, device)
        purifiers[ck["enc_type"]].append((ck["config"]["rf"], p, ck))
        print(f"loaded {path}: enc={ck['enc_type']} rf={ck['config']['rf']} "
              f"rho_val={ck.get('rho_val')} rho_train={ck.get('rho_train')}")
    enc_names = sorted(set(list(purifiers.keys()) +
                           [e for e in args.extra_encoders.split(",") if e]))
    encoders = {}
    for name in enc_names:
        encs, etypes, _ = load_encoders(name, device, 256)
        encoders[name] = (encs[0], etypes[0])
        encs[0].eval()

    # every purifier must have been fitted on the latent space this probe feeds it
    latents_fn, lat_sig = make_latents_fn(args, device)
    print(f"latent space: {lat_sig}")
    for plist in purifiers.values():
        for rf, _p, ck in plist:
            why = latent_signature_mismatch(purifier_latent_signature(ck)[0], lat_sig)
            if why is not None:
                raise SystemExit(f"purifier (enc {ck['enc_type']}, k={rf}) was fitted on a "
                                 f"different latent space than this probe: {why}")

    dataset = CustomDataset(args.data_dir, latents_dir=args.latents_dir)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    V = args.pur_val_images
    ntr = min(args.train_images, len(dataset) - 2 * V)
    idx = perm[2 * V: 2 * V + ntr] + perm[:V] + perm[V: 2 * V]
    seg = {"train": (0, ntr), "honest": (ntr, ntr + V), "leaked": (ntr + V, ntr + 2 * V)}
    print(f"dataset {len(dataset)}: probe train {ntr}, val honest {V} "
          f"(purifier-held-out), val leaked {V} (purifier-train)")
    loader = DataLoader(Subset(dataset, idx), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    pooled = defaultdict(list)      # (enc, mode) -> [(B,D)]
    toks = defaultdict(list)        # (enc, mode) -> [(B,g,D) fp16]
    ss = defaultdict(float)
    labels = []
    ps, gtok = None, args.tokens_per_image
    with torch.no_grad():
        for bi, (raw, mom, lab) in enumerate(loader):
            labels.append(lab.clone())
            x = latents_fn(raw, mom)
            xp = None
            tok_idx = None
            for name, (enc, etype) in encoders.items():
                z = encoder_tokens(enc, etype, raw, device)
                if ps is None:
                    side = int(round(z.shape[1] ** 0.5))
                    ps = x.shape[2] // side
                if xp is None:
                    xp = patchify_latent(x, ps).float()
                if tok_idx is None:  # same tokens across modes/encoders per batch
                    tok_idx = torch.randint(z.shape[1], (z.shape[0], gtok), device=device)
                zx = linear_residual(z, xp, "xpredictive")
                modes = {"raw": z, "linres": z - zx, "linxpred": zx}
                for rf, p, _ in purifiers.get(name, []):
                    pred = p(xp)
                    modes[f"pur_k{rf}"] = z - pred
                    modes[f"pred_k{rf}"] = pred
                for m, feat in modes.items():
                    pooled[(name, m)].append(feat.mean(dim=1).cpu())
                    sel = feat.gather(1, tok_idx[:, :, None].expand(-1, -1, feat.shape[-1]))
                    toks[(name, m)].append(sel.half().cpu())
                    c = feat - feat.mean(dim=(0, 1), keepdim=True)
                    ss[(name, m)] += float(c.pow(2).sum())
            if (bi + 1) % 20 == 0:
                print(f"batch {bi+1}/{len(loader)}", flush=True)

    y = torch.cat(labels).long()
    nclass = int(y.max()) + 1
    sl = {s: slice(a, b) for s, (a, b) in seg.items()}
    print(f"{y.shape[0]} images, {nclass} classes")

    dinoraw = None
    for name in enc_names:
        if "dinov2" in name:
            dinoraw = torch.cat(pooled[(name, "raw")])

    raw_sd_pool, raw_sd_tok = {}, {}
    for name in enc_names:
        Rp = torch.cat(pooled[(name, "raw")])
        raw_sd_pool[name] = Rp[sl["train"]].std(0, keepdim=True)
        Rt = torch.cat(toks[(name, "raw")]).float()
        raw_sd_tok[name] = Rt[sl["train"]].reshape(-1, Rt.shape[-1]).std(0, keepdim=True)

    results = {}
    for (name, m), feats in sorted(pooled.items()):
        Fp = torch.cat(feats)
        pool = train_probe(
            Fp[sl["train"]], y[sl["train"]],
            {"honest": (Fp[sl["honest"]], y[sl["honest"]]),
             "leaked": (Fp[sl["leaked"]], y[sl["leaked"]])},
            nclass, iters=args.probe_iters, device=device, sd_ref=raw_sd_pool[name])
        Ft = torch.cat(toks[(name, m)]).float()
        D = Ft.shape[-1]
        tok = train_probe(
            Ft[sl["train"]].reshape(-1, D), y[sl["train"]].repeat_interleave(gtok),
            {"honest": (Ft[sl["honest"]].reshape(-1, D), y[sl["honest"]], gtok),
             "leaked": (Ft[sl["leaked"]].reshape(-1, D), y[sl["leaked"]], gtok)},
            nclass, iters=args.probe_iters, device=device, sd_ref=raw_sd_tok[name])
        var_frac = ss[(name, m)] / max(ss[(name, "raw")], 1e-12)
        cka = linear_cka(Fp[sl["honest"]], dinoraw[sl["honest"]]) if dinoraw is not None else None
        results.setdefault(name, {})[m] = {
            "pooled": pool, "token": tok,
            "token_var_frac": var_frac, "cka_vs_dinov2_raw": cka,
        }
        print(f"{name:16s} {m:10s} pooled(honest/leaked/train) "
              f"{pool['honest_top1']:.4f}/{pool['leaked_top1']:.4f}/{pool['train_top1']:.4f} "
              f"token(honest) {tok['honest_top1']:.4f} var {var_frac:.3f}", flush=True)

    # ---- decision quantities: k*, feasibility, memorization bias, underfit flag
    decisions = {}
    for name in enc_names:
        r = results[name]
        if not any(m.startswith("pur_k") for m in r):
            continue
        lr1 = r["linres"]["pooled"]["honest_top1"]
        se = (lr1 * (1 - lr1) / V) ** 0.5
        lin_share = 1.0 - r["linres"]["token_var_frac"]
        ks, feas = {}, []
        for m in sorted(r):
            if not m.startswith("pur_k"):
                continue
            k = int(m[5:])
            pk, dk = r[m]["pooled"]["honest_top1"], r[f"pred_k{k}"]["pooled"]["honest_top1"]
            pk_t = r[m]["token"]["honest_top1"]
            lr1_t = r["linres"]["token"]["honest_top1"]
            ok_pool = (pk >= lr1 - 2 * se) and (pk - dk > 2 * se)
            ok_tok = (pk_t >= lr1_t - 2 * se) and \
                     (pk_t - r[f"pred_k{k}"]["token"]["honest_top1"] > 2 * se)
            rho_v = rho_t = None
            for rf, _p, ck in purifiers.get(name, []):
                if rf == k:
                    rho_v, rho_t = ck.get("rho_val"), ck.get("rho_train")
            ks[k] = {"sep_pooled": pk - dk, "retention_pooled": pk / max(r["raw"]["pooled"]["honest_top1"], 1e-9),
                     "feasible_pooled": ok_pool, "feasible_token": ok_tok,
                     "memorization_bias_pooled": r[m]["pooled"]["leaked_top1"] - pk,
                     "rho_val": rho_v, "rho_gap": (rho_t - rho_v) if rho_t is not None else None,
                     "underfit_flag": (rho_v is not None and rho_v < lin_share + 0.05)}
            if ok_pool and ok_tok:
                feas.append(k)
        decisions[name] = {"se_2x": 2 * se, "linres_top1": lr1,
                           "linear_var_share": lin_share, "per_k": ks,
                           "feasible": feas, "k_star": max(feas) if feas else None,
                           "kill": len(feas) == 0}
        print(f"[decision] {name}: feasible={feas} k*={max(feas) if feas else None} "
              f"kill={len(feas) == 0}")

    out = {"results": results, "decisions": decisions,
           "n_train": ntr, "n_val": V, "nclass": nclass, "chance_top1": 1.0 / nclass,
           "args": vars(args), "latent_signature": lat_sig}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
