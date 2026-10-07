"""Frozen DINOv3, Perception Encoder, and AIMv2 target encoders.

Every wrapper exposes ``embed_dim`` and ``forward_features(x) -> (B, 256, D)``:
patch tokens after the backbone's final norm, with CLS/register tokens dropped.
``x`` is the 256px image already normalized by ``utils.preprocess_raw_image``.
Patch-14 timm backbones (PE-Core-L) resize that normalized input bicubically to
224px inside the wrapper; AIMv2 is resized to 224px by the preprocessing itself.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# name -> (library, model id, timm img_size, backbone input size (None = as fed),
#          embed dim, prefix tokens dropped)
EXT_ENCODERS = {
    "dinov3-vit-b":     ("timm", "timm/vit_base_patch16_dinov3.lvd1689m", 256, None, 768, 5),
    "dinov3-vit-l":     ("timm", "timm/vit_large_patch16_dinov3.lvd1689m", 256, None, 1024, 5),
    "dinov3-vit-hplus": ("timm", "timm/vit_huge_plus_patch16_dinov3.lvd1689m", 256, None, 1280, 5),
    "pecore-vit-l":     ("timm", "timm/vit_pe_core_large_patch14_336.fb", 224, 224, 1024, 1),
    "pespatial-vit-b":  ("timm", "timm/vit_pe_spatial_base_patch16_512.fb", 256, None, 768, 1),
    "aimv2-vit-l":      ("hf", "apple/aimv2-large-patch14-224", None, None, 1024, 0),
}


def _offline_retry(fn):
    """Load normally; if that fails (e.g. a compute node without internet), retry from
    the local Hugging Face cache with HF_HUB_OFFLINE=1."""
    try:
        return fn()
    except Exception:
        prev = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            return fn()
        finally:
            if prev is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev


class TimmPatchEncoder(nn.Module):
    """timm ViT wrapper: forward_features minus the prefix tokens."""

    def __init__(self, model_id, img_size, expected_dim, expected_tokens=256,
                 input_size=None):
        super().__init__()
        import timm
        self.model = _offline_retry(lambda: timm.create_model(
            "hf_hub:" + model_id, pretrained=True, num_classes=0, img_size=img_size))
        self.model.eval()
        self.num_prefix = int(getattr(self.model, "num_prefix_tokens", 0))
        self.embed_dim = int(self.model.embed_dim)
        self.num_tokens = expected_tokens
        self.input_size = input_size
        self.model_id = model_id
        assert self.embed_dim == expected_dim, \
            "%s: embed_dim %d != expected %d" % (model_id, self.embed_dim, expected_dim)

    @torch.no_grad()
    def forward_features(self, x):
        if self.input_size is not None and x.shape[-1] != self.input_size:
            x = F.interpolate(x, self.input_size, mode='bicubic')
        z = self.model.forward_features(x)
        z = z[:, self.num_prefix:]
        assert z.shape[1] == self.num_tokens and z.shape[2] == self.embed_dim, \
            "%s: expected (B,%d,%d), got %s" % (
                self.model_id, self.num_tokens, self.embed_dim, tuple(z.shape))
        return z

    def forward(self, x):
        return self.forward_features(x)


class HFPatchEncoder(nn.Module):
    """transformers vision-model wrapper: last_hidden_state minus prefix tokens."""

    def __init__(self, name, num_prefix, expected_dim, expected_tokens=256):
        super().__init__()
        from transformers import AutoModel
        self.model = _offline_retry(lambda: AutoModel.from_pretrained(name)).eval()
        self.embed_dim = int(self.model.config.hidden_size)
        self.num_prefix = int(num_prefix)
        self.num_tokens = expected_tokens
        self.name = name
        assert self.embed_dim == expected_dim, \
            "%s: hidden_size %d != expected %d" % (name, self.embed_dim, expected_dim)

    @torch.no_grad()
    def forward_features(self, x):
        z = self.model(pixel_values=x).last_hidden_state
        z = z[:, self.num_prefix:]
        assert z.shape[1] == self.num_tokens and z.shape[2] == self.embed_dim, \
            "%s: expected (B,%d,%d), got %s" % (
                self.name, self.num_tokens, self.embed_dim, tuple(z.shape))
        return z

    def forward(self, x):
        return self.forward_features(x)


def build_ext_encoder(name, device="cpu"):
    """Build a frozen encoder listed in EXT_ENCODERS (used by utils.load_encoders)."""
    if name not in EXT_ENCODERS:
        raise ValueError("unsupported encoder %s; expected one of %s" % (name, sorted(EXT_ENCODERS)))
    library, model_id, img_size, input_size, dim, num_prefix = EXT_ENCODERS[name]
    if library == "timm":
        enc = TimmPatchEncoder(model_id, img_size, dim, input_size=input_size)
        assert enc.num_prefix == num_prefix, \
            "%s: expected %d prefix tokens, got %d" % (name, num_prefix, enc.num_prefix)
    else:
        enc = HFPatchEncoder(model_id, num_prefix, dim)
    enc = enc.to(device).eval()
    enc.requires_grad_(False)
    return enc
