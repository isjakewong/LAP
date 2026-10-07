"""SatMAE ViT-L/16 (fMoW-RGB; Cong et al., NeurIPS 2022) target encoder.

The historical aerial unaligned runs built a SatMAE projector with alignment
coefficient 0; this wrapper preserves that model construction. Weights:
https://github.com/sustainlab-group/SatMAE (RGB fMoW ViT-L) -> ckpts/satmae_vitl.pth.

Same token convention as the 'mae' branch of utils.load_encoders
(models/mae_vit.VisionTransformer at img_size=256, 14x14 pos_embed resampled to
16x16, CLS dropped, no final LayerNorm): forward_features -> (B, 256, 1024).
"""
import torch
import torch.nn as nn

# fMoW-RGB statistics from SatMAE util/datasets.py (not ImageNet).
FMOW_RGB_MEAN = (0.4182007312774658, 0.4214799106121063, 0.3991275727748871)
FMOW_RGB_STD = (0.28774282336235046, 0.27541765570640564, 0.2764017581939697)

SATMAE_CKPT = "ckpts/satmae_vitl.pth"


def _resample_pos_embed(pe, grid=(16, 16)):
    import timm
    return timm.layers.pos_embed.resample_abs_pos_embed(pe, list(grid))


def load_mae_encoder_state(path):
    """Encoder-only state dict of an MAE pretraining checkpoint, with pos_embed
    resampled to a 16x16 grid (+ CLS)."""
    with open(path, "rb") as f:
        ck = torch.load(f, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("decoder_") or k == "mask_token":
            continue  # MAE decoder is not part of the target encoder
        out[k] = v
    if "pos_embed" in out:
        out["pos_embed"] = _resample_pos_embed(out["pos_embed"])
    return out


class MAEPatchEncoder(nn.Module):
    """MAE-style ViT encoder: 256x256 input -> 256 patch tokens, CLS dropped, no final norm."""

    def __init__(self, arch, ckpt_path, img_size=256):
        super().__init__()
        from models import mae_vit
        self.model = getattr(mae_vit, arch)(img_size=img_size)
        self.model.load_state_dict(load_mae_encoder_state(ckpt_path), strict=True)
        self.model.pos_embed.data = _resample_pos_embed(self.model.pos_embed.data)
        self.embed_dim = self.model.embed_dim
        self.num_tokens = (img_size // 16) ** 2

    @torch.no_grad()
    def forward_features(self, x):
        z = self.model.forward_features(x)  # (B, N, D), CLS already dropped
        assert z.shape[1] == self.num_tokens and z.shape[2] == self.embed_dim, \
            f"expected (B,{self.num_tokens},{self.embed_dim}), got {tuple(z.shape)}"
        return z

    def forward(self, x):
        return self.forward_features(x)


def build_satmae_encoder(model_config, device="cpu"):
    assert model_config == "l", "satmae: only the ViT-L/16 fMoW-RGB checkpoint is supported"
    enc = MAEPatchEncoder("vit_large_patch16", SATMAE_CKPT).to(device).eval()
    enc.requires_grad_(False)
    return enc
