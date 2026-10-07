"""
LocalPurifier: a small, locality-bounded, nonlinear predictor of frozen-encoder
tokens z = f(I) from the (patchified) clean VAE latent X.

Used for LAP (Local-Appearance-Purified) alignment targets:
    y_pure = z - p_phi(Xp),  p_phi frozen after a short pre-pass.

The receptive field on the patchified latent grid is rf x rf tokens: the first
convolution has kernel rf, and all later convolutions are 1x1. This bounds the
local predictor's input context; it does not guarantee a purely semantic residual.
"""
import torch
import torch.nn as nn


@torch.no_grad()
def patchify_latent(x, p):
    """(B,C,H,W) -> (B, (H/p)(W/p), C p p), row-major tokens (matches train.py)."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // p, p, W // p, p)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)


@torch.no_grad()
def linear_residual(z, xp, mode="residual"):
    """
    Per-batch linear split of targets z (B,N,D) against patchified latent
    xp (B,N,F) — same math as train.py::split_target_by_latent, but taking
    the already-patchified latent so callers control the patch size.
      residual:    z - proj_X(z);   xpredictive: proj_X(z)
    """
    B, N, D = z.shape
    zf = z.reshape(B * N, D).float()
    xf = xp.reshape(B * N, -1).float()
    xf = torch.cat([xf, torch.ones(xf.shape[0], 1, device=xf.device)], dim=1)
    W = torch.linalg.lstsq(xf, zf).solution
    zpred = (xf @ W).reshape(B, N, D).to(z.dtype)
    return zpred if mode == "xpredictive" else (z - zpred)


class LocalPurifier(nn.Module):
    """Conv stack on the token grid; receptive field = rf x rf tokens."""

    def __init__(self, in_ch=16, hidden=512, out_dim=768, rf=3):
        super().__init__()
        assert rf % 2 == 1, "rf must be odd so the token grid is preserved"
        self.in_ch, self.hidden, self.out_dim, self.rf = in_ch, hidden, out_dim, rf
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=rf, padding=rf // 2),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_dim, kernel_size=1),
        )

    def forward(self, xp):
        """xp: (B, N, in_ch) row-major square token grid -> (B, N, out_dim)."""
        B, N, C = xp.shape
        side = int(round(N ** 0.5))
        assert side * side == N, f"token count {N} is not a square grid"
        g = xp.reshape(B, side, side, C).permute(0, 3, 1, 2)
        out = self.net(g)
        return out.permute(0, 2, 3, 1).reshape(B, N, self.out_dim)

    def config(self):
        return {"in_ch": self.in_ch, "hidden": self.hidden,
                "out_dim": self.out_dim, "rf": self.rf}


def load_purifier(path, device="cpu"):
    """Load a frozen purifier saved by train_purifier.py."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    p = LocalPurifier(**ckpt["config"])
    p.load_state_dict(ckpt["state_dict"])
    p = p.to(device).eval()
    p.requires_grad_(False)
    return p, ckpt


# ---------------------------------------------------------------------------
# Latent-space provenance. A purifier maps the patchified, normalized clean latent X_p to
# encoder tokens, so it is valid only for the latent space it was fitted on: the same tokenizer
# and the same per-channel normalization x = z * scale + bias. train_purifier.py records this
# signature in every checkpoint; checkpoints without one were fitted on cached SD-VAE-ft-mse
# moments ('vae-sd') with scale 0.18215, bias 0.
SDVAE_PATH = "stabilityai/sd-vae-ft-mse"
SDVAE_LATENTS_DIR = "vae-sd"
SDVAE_SCALE = 0.18215


def latent_vae_identity(vae_path, latents_dir, vae_encode_online):
    """Tokenizer that produced the latents: online encoding -> vae_path; cached moments -> the
    tokenizer that wrote the dir ('vae-sd' = SD-VAE-ft-mse, any other dir is named as such)."""
    if vae_encode_online:
        return str(vae_path)
    if latents_dir == SDVAE_LATENTS_DIR:
        return SDVAE_PATH
    return "cached:" + str(latents_dir)


def latent_signature(vae_path=SDVAE_PATH, latents_dir=SDVAE_LATENTS_DIR,
                     vae_encode_online=False, latents_stats=None):
    """Latent space as train.py builds it: tokenizer identity plus the per-channel
    (scale, bias) of x = z * scale + bias (utils.load_latents_stats, else 0.18215 / 0)."""
    if latents_stats:
        from utils import load_latents_stats
        sc, bi = load_latents_stats(latents_stats, "cpu")
        scale, bias = sc.flatten().tolist(), bi.flatten().tolist()
    else:
        scale, bias = [SDVAE_SCALE] * 4, [0.0] * 4
    return {"latent_vae": latent_vae_identity(vae_path, latents_dir, vae_encode_online),
            "latents_scale": [float(v) for v in scale],
            "latents_bias": [float(v) for v in bias]}


def purifier_latent_signature(ckpt):
    """(signature, is_legacy). Checkpoints without a record are SD-VAE (see above)."""
    sig = ckpt.get("latent_signature") if isinstance(ckpt, dict) else None
    if sig is None:
        return latent_signature(), True
    return sig, False


def latent_signature_mismatch(a, b, rtol=1e-6, atol=1e-8):
    """None if a and b describe the same latent space, else a human-readable reason."""
    if a["latent_vae"] != b["latent_vae"]:
        return "tokenizer %r vs %r" % (a["latent_vae"], b["latent_vae"])
    for k in ("latents_scale", "latents_bias"):
        x, y = a[k], b[k]
        if len(x) != len(y) or any(abs(u - v) > atol + rtol * abs(v) for u, v in zip(x, y)):
            return "%s %s vs %s" % (k, x, y)
    return None
