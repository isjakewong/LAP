"""
LAP-N pre-pass: fit the locality-bounded purifier p_phi(Xp) ~= z on frozen encoder
tokens, then freeze it for diffusion training.

  python train_purifier.py --enc-type mae-vit-l --rf 5 \
      --data-dir data/in1k_256 --out assets/purifier1k_mae_k5.pt

`python reproduce.py purifier <experiment>` supplies the paper's arguments
(configs/purifiers.json). For a swapped tokenizer, pass the same --vae-path,
--latents-stats and --vae-encode-online as the diffusion run; the checkpoint records
that latent space and train.py refuses a mismatched purifier.

Reports the held-out purification ratio rho = R^2(z | p_phi(Xp)) and the same
ratio on training images (a large gap would indicate memorization).
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from models.purifier import LocalPurifier, patchify_latent, latent_signature, SDVAE_PATH
from utils import load_encoders, preprocess_raw_image

LATENTS_SCALE = 0.18215


@torch.no_grad()
def sample_posterior(moments, scale=LATENTS_SCALE):
    mean, std = torch.chunk(moments, 2, dim=1)
    return (mean + std * torch.randn_like(mean)) * scale


def add_latent_args(ap):
    """Swapped-VAE flags; same names and defaults as train.py (defaults = cached SD-VAE)."""
    ap.add_argument("--latents-dir", type=str, default="vae-sd",
                    help="cached VAE moments dir under --data-dir (train.py --latents-dir)")
    ap.add_argument("--vae-path", type=str, default=SDVAE_PATH,
                    help="VAE for --vae-encode-online (train.py --vae-path)")
    ap.add_argument("--latents-stats", type=str, default=None,
                    help="per-channel latent stats, REPA-E format (train.py --latents-stats)")
    ap.add_argument("--vae-encode-online", action=argparse.BooleanOptionalAction, default=False,
                    help="encode the raw images with --vae-path every batch "
                         "(train.py --vae-encode-online)")
    ap.add_argument("--vae-encode-batch", type=int, default=64,
                    help="chunk size for online encoding; 64 = the diffusion runs' batch, so the "
                         "VAE sees the same batch shape as in train.py")


def check_latent_args(ap, args):
    if args.vae_path != SDVAE_PATH and not args.vae_encode_online:
        ap.error("--vae-path only takes effect with --vae-encode-online (cached moments are read "
                 "from --latents-dir); add --vae-encode-online or drop --vae-path")


def make_latents_fn(args, device):
    """(raw uint8 images, cached moments) -> normalised clean latent x, built exactly as train.py
    builds the x it feeds to the SiT and to the purifier:
        online: moments = cat(mean, std) of vae.encode(raw / 127.5 - 1) under fp16 autocast
        cached: moments = the --latents-dir file
        x = (mean + std * eps) * scale + bias, (scale, bias) = utils.load_latents_stats(
            --latents-stats) or (0.18215, 0).
    With all defaults this is the legacy sample_posterior(mom) call, bit-for-bit.
    Returns (fn, latent-space signature recorded in the checkpoint)."""
    sig = latent_signature(args.vae_path, args.latents_dir, args.vae_encode_online,
                           args.latents_stats)
    if not args.vae_encode_online and not args.latents_stats:
        return (lambda raw, mom: sample_posterior(mom.squeeze(1).to(device))), sig
    if args.latents_stats:
        from utils import load_latents_stats
        scale, bias = load_latents_stats(args.latents_stats, device)
    else:
        scale = torch.tensor([LATENTS_SCALE] * 4).view(1, 4, 1, 1).to(device)
        bias = torch.tensor([0.] * 4).view(1, 4, 1, 1).to(device)
    vae = None
    if args.vae_encode_online:
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
        vae.requires_grad_(False)

    @torch.no_grad()
    def fn(raw, mom):
        if vae is not None:
            parts = []
            for i in range(0, raw.shape[0], args.vae_encode_batch):
                r = raw[i:i + args.vae_encode_batch].to(device)
                with torch.autocast("cuda", torch.float16, enabled=r.is_cuda):
                    d = vae.encode(r.float() / 127.5 - 1.0).latent_dist
                parts.append(torch.cat([d.mean, d.std], dim=1).float())
            m = torch.cat(parts)
        else:
            m = mom.squeeze(1).to(device)
        mean, std = torch.chunk(m, 2, dim=1)
        z = mean + std * torch.randn_like(mean)
        return z * scale + bias
    return fn, sig


@torch.no_grad()
def encoder_tokens(encoder, enc_type, raw_image, device, autocast=True):
    """Frozen encoder tokens, identical to train.py's target extraction."""
    raw = preprocess_raw_image(raw_image.to(device), enc_type)
    with torch.autocast("cuda", torch.float16, enabled=autocast and raw.is_cuda):
        z = encoder.forward_features(raw)
        if 'mocov3' in enc_type:
            z = z[:, 1:]
        if 'dinov2' in enc_type:
            z = z['x_norm_patchtokens']
    return z.float()


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def eval_rho(purifier, encoder, enc_type, loader, device, ps, max_batches=None, latents_fn=None):
    """Held-out rho = 1 - E||z - p(Xp)||^2 / E||z - mean(z)||^2 (two-pass)."""
    # pass 1: per-dim mean of z (independent of the posterior sample)
    tot, cnt = 0.0, 0
    for bi, (raw, mom, _) in enumerate(loader):
        z = encoder_tokens(encoder, enc_type, raw, device)
        tot = tot + z.sum(dim=(0, 1)).double()
        cnt += z.shape[0] * z.shape[1]
        if max_batches and bi + 1 >= max_batches:
            break
    zmean = (tot / cnt).float()
    # pass 2: residual and total sum of squares
    sse, sst = 0.0, 0.0
    for bi, (raw, mom, _) in enumerate(loader):
        z = encoder_tokens(encoder, enc_type, raw, device)
        x = (sample_posterior(mom.squeeze(1).to(device)) if latents_fn is None
             else latents_fn(raw, mom))
        pred = purifier(patchify_latent(x, ps).float())
        sse += ((z - pred) ** 2).sum().double()
        sst += ((z - zmean) ** 2).sum().double()
        if max_batches and bi + 1 >= max_batches:
            break
    return float(1.0 - sse / sst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--enc-type", default="mae-vit-l")
    ap.add_argument("--rf", type=int, default=3, help="receptive field in tokens (odd)")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val-images", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--out", default=None)
    add_latent_args(ap)
    args = ap.parse_args()
    check_latent_args(ap, args)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from torch.utils.data import DataLoader, Subset
    from dataset import CustomDataset

    encoders, encoder_types, _ = load_encoders(args.enc_type, device, 256)
    encoder, enc_type = encoders[0], encoder_types[0]
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    latents_fn, lat_sig = make_latents_fn(args, device)
    print(f"latent space: {lat_sig}")

    dataset = CustomDataset(args.data_dir, latents_dir=args.latents_dir)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    val_idx, train_idx = perm[:args.val_images], perm[args.val_images:]
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"dataset {len(dataset)} images -> train {len(train_idx)} / val {len(val_idx)}")

    # infer dims from one batch
    raw0, mom0, _ = next(iter(val_loader))
    z0 = encoder_tokens(encoder, enc_type, raw0[:8], device)
    x0 = latents_fn(raw0[:8], mom0[:8])
    print("normalised latent per-channel mean %s std %s (8 images)" % (
        [round(v, 3) for v in x0.mean(dim=(0, 2, 3)).tolist()],
        [round(v, 3) for v in x0.std(dim=(0, 2, 3)).tolist()]))
    side = int(round(z0.shape[1] ** 0.5))
    ps = x0.shape[2] // side
    in_ch = x0.shape[1] * ps * ps
    print(f"tokens {z0.shape[1]} (grid {side}x{side}), latent patch {ps} -> in_ch {in_ch}, "
          f"out_dim {z0.shape[2]}")

    purifier = LocalPurifier(in_ch=in_ch, hidden=args.hidden,
                             out_dim=z0.shape[2], rf=args.rf).to(device)
    opt = torch.optim.AdamW(purifier.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    print(f"purifier params: {sum(p.numel() for p in purifier.parameters()):,} (rf={args.rf})")

    enc_short = args.enc_type.split(",")[0].split("-")[0]
    out = args.out or f"assets/purifier_{enc_short}_k{args.rf}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    def save(rho_val=None, rho_train=None, step=None):
        torch.save({"state_dict": purifier.state_dict(), "config": purifier.config(),
                    "enc_type": args.enc_type, "args": vars(args), "ps": ps,
                    "rho_val": rho_val, "rho_train": rho_train, "step": step,
                    # latent-space provenance (train.py refuses a mismatched purifier)
                    "vae_path": args.vae_path, "latents_stats": args.latents_stats,
                    "latents_dir": args.latents_dir,
                    "vae_encode_online": args.vae_encode_online,
                    "latent_signature": lat_sig}, out)

    step, run_loss, run_rho = 0, 0.0, 0.0
    while step < args.steps:
        for raw, mom, _ in train_loader:
            lr = cosine_lr(step, args.steps, args.lr, args.warmup)
            for gp in opt.param_groups:
                gp["lr"] = lr
            z = encoder_tokens(encoder, enc_type, raw, device)
            x = latents_fn(raw, mom)
            pred = purifier(patchify_latent(x, ps).float())
            loss = F.mse_loss(pred, z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                run_loss += loss.item()
                run_rho += float(1.0 - loss / (z - z.mean(dim=(0, 1))).pow(2).mean())
            step += 1
            if step % args.log_every == 0:
                print(f"step {step:6d} lr {lr:.2e} mse {run_loss/args.log_every:.4f} "
                      f"rho(batch) {run_rho/args.log_every:.4f}", flush=True)
                run_loss, run_rho = 0.0, 0.0
            if step % args.ckpt_every == 0:
                save(step=step)
            if step >= args.steps:
                break

    purifier.eval()
    rho_val = eval_rho(purifier, encoder, enc_type, val_loader, device, ps,
                       latents_fn=latents_fn)
    tr_eval = DataLoader(Subset(dataset, train_idx[:args.val_images]),
                         batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)
    rho_train = eval_rho(purifier, encoder, enc_type, tr_eval, device, ps,
                         latents_fn=latents_fn)
    save(rho_val=rho_val, rho_train=rho_train, step=step)
    print(json.dumps({"out": out, "enc_type": args.enc_type, "rf": args.rf,
                      "rho_val": rho_val, "rho_train": rho_train,
                      "latent_vae": lat_sig["latent_vae"]}, indent=2))
    print(f"PURIFIER DONE -> {out}")


if __name__ == "__main__":
    main()
