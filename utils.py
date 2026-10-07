import torch
import torch.nn.functional as F
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from models import mocov3_vit
from models.domain_encoders import FMOW_RGB_MEAN, FMOW_RGB_STD


CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

# encoder type -> input normalization (mean, std) applied to [0, 1] RGB.
ENCODER_NORM = {
    'mae': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    'mocov3': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    'dinov2': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    'dinov3': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    'jepa': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    'clip': (CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD),
    'clip16': (CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD),
    'aimv2': (CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD),
    'pecore': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    'pespatial': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    'satmae': (FMOW_RGB_MEAN, FMOW_RGB_STD),
}


def preprocess_raw_image(x, enc_type):
    """Map uint8-range RGB (B, 3, R, R) to the teacher's input.

    ``enc_type`` is the encoder type ('mae') or full name ('mae-vit-l'). Patch-14
    teachers read a 224px image (16x16 tokens at R=256): CLIP is resized before
    normalization (REPA convention), DINOv2/I-JEPA/AIMv2 after it. All other
    teachers read the R px image directly or resize inside their wrapper.
    """
    from torchvision.transforms import Normalize
    kind = enc_type.split('-')[0]
    if kind not in ENCODER_NORM:
        raise ValueError(f'Unsupported encoder type: {enc_type}')
    resolution = x.shape[-1]
    x = x / 255.
    if kind == 'clip':
        x = F.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    x = Normalize(*ENCODER_NORM[kind])(x)
    if kind in ('dinov2', 'jepa', 'aimv2'):
        x = F.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    return x


def fix_mocov3_state_dict(state_dict):
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            if 'head' not in new_k and new_k.split('.')[0] != 'fc':
                state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if 'pos_embed' in state_dict.keys():
        state_dict['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
            state_dict['pos_embed'], [16, 16],
        )
    return state_dict


def _clip_visual(name, device):
    """OpenAI CLIP visual tower returning patch tokens (CLS dropped, no ln_post/proj).
    jit=False keeps the module attributes accessible; .float() keeps fp32 master
    weights (clip.load only up-casts on device 'cpu') so fp16 autocast stays NaN-safe."""
    import clip
    from models.clip_vit import UpdatedVisionTransformer
    visual = clip.load(name, device='cpu', jit=False)[0].visual.float()
    encoder = UpdatedVisionTransformer(visual).to(device)
    encoder.embed_dim = encoder.model.transformer.width
    encoder.forward_features = encoder.forward
    return encoder.eval()


@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    assert (resolution == 256) or (resolution == 512)

    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        # Currently, we only support 512x512 experiments with DINOv2 encoders.
        if resolution == 512:
            if encoder_type != 'dinov2':
                raise NotImplementedError(
                    "Currently, we only support 512x512 experiments with DINOv2 encoders."
                    )

        architectures.append(architecture)
        encoder_types.append(encoder_type)
        if encoder_type == 'mocov3':
            if architecture != 'vit':
                raise NotImplementedError()
            encoder = {'s': mocov3_vit.vit_small, 'b': mocov3_vit.vit_base,
                       'l': mocov3_vit.vit_large}[model_config]()
            ckpt = torch.load(f'./ckpts/mocov3_vit{model_config}.pth')
            state_dict = fix_mocov3_state_dict(ckpt['state_dict'])
            del encoder.head
            encoder.load_state_dict(state_dict, strict=True)
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()

        elif encoder_type == 'dinov2':
            encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()

        elif encoder_type == 'clip':      # ViT-L/14, fed 224px
            encoder = _clip_visual(f"ViT-{model_config}/14", device)

        elif encoder_type == 'clip16':    # ViT-B/16, fed 256px; pos-embed resampled 14x14 -> 16x16
            assert model_config.lower() == 'b', "clip16: only ViT-B/16 is supported"
            encoder = _clip_visual("ViT-B/16", device)

        elif encoder_type == 'mae':
            # Official MAE pretraining checkpoint at 256px: pos-embed 14x14 -> 16x16,
            # CLS dropped, no final LayerNorm (models/mae_vit.forward_features).
            from models.mae_vit import vit_large_patch16, vit_base_patch16
            build = {'b': vit_base_patch16, 'l': vit_large_patch16}[model_config.lower()]
            encoder = build(img_size=256).to(device)
            with open(f"ckpts/mae_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f)
            if 'pos_embed' in state_dict["model"].keys():
                state_dict["model"]['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    state_dict["model"]['pos_embed'], [16, 16],
                )
            encoder.load_state_dict(state_dict["model"])

            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [16, 16],
            )
            encoder.eval()

        elif encoder_type == 'jepa':
            from models.jepa import vit_huge
            kwargs = dict(img_size=[224, 224], patch_size=14)
            encoder = vit_huge(**kwargs).to(device)
            with open(f"ckpts/ijepa_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f, map_location=device)
            new_state_dict = dict()
            for key, value in state_dict['encoder'].items():
                new_state_dict[key[7:]] = value
            encoder.load_state_dict(new_state_dict)
            encoder.forward_features = encoder.forward
            encoder.eval()

        elif encoder_type == 'satmae':
            from models.domain_encoders import build_satmae_encoder
            encoder = build_satmae_encoder(model_config, device)

        elif encoder_type in ('dinov3', 'aimv2', 'pecore', 'pespatial'):
            from models.ext_encoders import build_ext_encoder
            encoder = build_ext_encoder(f'{encoder_type}-{architecture}-{model_config.lower()}', device)

        else:
            raise ValueError(f'Unsupported encoder: {enc_name}')

        encoders.append(encoder)

    return encoders, encoder_types, architectures


def load_latents_stats(path, device):
    """Per-channel latent normalization in the REPA-E file format
    (latents_scale = 1/std, latents_bias = mean), converted to our
    z * scale + bias convention. Decoding uses (z - bias) / scale, which
    recovers z / scale_e + bias_e exactly as in REPA-E."""
    st = torch.load(path, map_location="cpu", weights_only=True)
    scale_e = st["latents_scale"].flatten().float()
    bias_e = st["latents_bias"].flatten().float()
    c = scale_e.numel()
    return (scale_e.view(1, c, 1, 1).to(device),
            (-bias_e * scale_e).view(1, c, 1, 1).to(device))


def build_model_from_ckpt(ckpt, device):
    """Rebuild the EMA SiT from a training or exported checkpoint; returns (model, get_arg)."""
    from models.sit import SiT_models
    a = ckpt["args"]
    get = (lambda k, d=None: getattr(a, k, d)) if not isinstance(a, dict) else (lambda k, d=None: a.get(k, d))
    state = ckpt['ema'] if 'ema' in ckpt else ckpt['model']
    # The projector output dim (= target dim) and head type are read from the weights.
    zk = [k for k in state if k.startswith("projectors.0.") and k.endswith(".weight")]
    zdim = int(state[sorted(zk)[-1]].shape[0]) if zk else 768
    proj_type = "conv" if (zk and state[sorted(zk)[0]].ndim == 4) else "mlp"
    model = SiT_models[get("model")](
        input_size=get("resolution", 256) // 8,
        num_classes=get("num_classes", 1000),
        use_cfg=(get("cfg_prob", 0.1) > 0),
        projector_type=proj_type,
        z_dims=[zdim],
        encoder_depth=get("encoder_depth", 8),
        sufficiency=any(k.startswith("sufficiency_head.") for k in state),
        fused_attn=get("fused_attn", True),
        qk_norm=get("qk_norm", False),
    )
    model.load_state_dict(state, strict=True)  # REG checkpoints add their CLS modules here
    return model.to(device).eval(), get
