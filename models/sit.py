# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


def build_conv_projector(hidden_size, projector_dim, z_dim, k=3):
    """iREPA-style alignment head: one k x k conv over the token grid so
    neighboring hidden tokens interact during projection, then 1x1 convs.
    Mirrors build_mlp's depth/width so the heads differ only in the spatial
    kernel; module indices (0/2/4) match build_mlp so checkpoint zdim
    inference (projectors.0.4.weight.shape[0]) stays valid."""
    return nn.Sequential(
                nn.Conv2d(hidden_size, projector_dim, kernel_size=k, padding=k // 2),
                nn.SiLU(),
                nn.Conv2d(projector_dim, projector_dim, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(projector_dim, z_dim, kernel_size=1),
            )


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class SufficiencyHead(nn.Module):
    """Auxiliary velocity predictor from the aligned projection (prediction-loss intervention)."""
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.act = nn.SiLU()
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, a_t, t):
        # a_t: (N, T, in_dim); t: (N,)
        h = self.fc_in(a_t) + self.t_embedder(t).unsqueeze(1)
        return self.out(self.act(h))


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, cls_token_dim=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # REG (Wu et al., NeurIPS 2025, arXiv:2507.01467) -- their FinalLayer.linear_cls:
        # a second head that reads token 0 and predicts the CLS velocity. Built only when
        # REG is on, so the non-REG parameter set and init RNG stream are untouched.
        self.linear_cls = nn.Linear(hidden_size, cls_token_dim, bias=True) if cls_token_dim else None
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c, split_cls=False):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        if split_cls:
            # REG: adaLN + final norm run on the whole (1+T) sequence; only the output
            # heads are split -- token 0 -> linear_cls, tokens 1.. -> the patch head.
            return self.linear(x[:, 1:]), self.linear_cls(x[:, 0])
        x = self.linear(x)

        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        projector_dim=2048,
        sufficiency=False,                # auxiliary velocity prediction from the aligned projection
        projector_type='mlp',             # 'mlp' (REPA) | 'conv' (iREPA-style head)
        reg_cls=False,                    # REG: prepend a noised DINOv2 CLS token
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # REG: one extra sequence slot at index 0 carries the noised CLS token.
        self.reg_cls = bool(reg_cls)
        self.reg_cls_dim = int(z_dims[0]) if self.reg_cls else 0
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + (1 if self.reg_cls else 0), hidden_size),
            requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.projector_type = projector_type
        if projector_type == 'conv':
            self.projectors = nn.ModuleList([
                build_conv_projector(hidden_size, projector_dim, z_dim) for z_dim in z_dims
                ])
        else:
            self.projectors = nn.ModuleList([
                build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
                ])
        self.final_layer = FinalLayer(
            decoder_hidden_size, patch_size, self.out_channels,
            cls_token_dim=(self.reg_cls_dim if self.reg_cls else None))

        # ---- REG: project the noised DINOv2 CLS token into the SiT token space and
        # LayerNorm it, exactly as REG's `cls_projectors2` (nn.Linear, z_dim ->
        # hidden_size) followed by `wg_norm` (affine LayerNorm). Same names as upstream
        # so their checkpoints/ours stay mutually legible.
        if self.reg_cls:
            assert projector_type != 'conv', \
                "REG (--reg-cls-beta) does not compose with --conv-projector"
            assert not sufficiency, \
                "REG (--reg-cls-beta) does not compose with --sufficiency"
            self.cls_projectors2 = nn.Linear(self.reg_cls_dim, hidden_size, bias=True)
            self.wg_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)

        # Auxiliary sufficiency head: inside SiT so the optimizer, EMA, checkpointing and
        # DDP gradient sync cover it. It is not on the sampling path.
        self.sufficiency_head = None
        if sufficiency:
            self.sufficiency_head = SufficiencyHead(
                in_dim=z_dims[0], out_dim=patch_size * patch_size * self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # REG: extra_tokens=1 prepends a ZERO row for the CLS slot (their setting).
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5),
            cls_token=self.reg_cls, extra_tokens=(1 if self.reg_cls else 0)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if self.final_layer.linear_cls is not None:   # REG
            nn.init.constant_(self.final_layer.linear_cls.weight, 0)
            nn.init.constant_(self.final_layer.linear_cls.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def _enable_reg_cls(self, z_dim):
        """Build REG's CLS modules on an already-constructed SiT.

        Used when a loader built a plain SiT and then loads a REG checkpoint, so samplers
        and diagnostics work without knowing the flag.
        """
        hidden = self.pos_embed.shape[-1]
        dev, dt = self.pos_embed.device, self.pos_embed.dtype
        self.reg_cls = True
        self.reg_cls_dim = int(z_dim)
        self.cls_projectors2 = nn.Linear(self.reg_cls_dim, hidden, bias=True).to(device=dev, dtype=dt)
        self.wg_norm = nn.LayerNorm(hidden, elementwise_affine=True, eps=1e-6).to(device=dev, dtype=dt)
        self.final_layer.linear_cls = nn.Linear(
            self.final_layer.linear.in_features, self.reg_cls_dim, bias=True).to(device=dev, dtype=dt)
        if self.pos_embed.shape[1] == self.x_embedder.num_patches:
            pe = torch.zeros(1, self.x_embedder.num_patches + 1, hidden, device=dev, dtype=dt)
            pe[:, 1:] = self.pos_embed.data
            self.pos_embed = nn.Parameter(pe, requires_grad=False)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # A REG checkpoint carries cls_projectors2/wg_norm/final_layer.linear_cls and a
        # (num_patches+1)-row pos_embed. Rebuild them here so any loader in the repo can
        # restore and sample a REG model without passing reg_cls=True itself.
        if not getattr(self, "reg_cls", False) and \
                any(k.startswith("cls_projectors2.") for k in state_dict):
            self._enable_reg_cls(state_dict["cls_projectors2.weight"].shape[1])
        return super().load_state_dict(state_dict, *args, **kwargs)

    def forward(self, x, t, y, return_logvar=False, cls_token=None):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x)                   # (N, T, D), where T = H * W / patch_size ** 2
        if self.reg_cls:
            # REG: g(cls_t) = wg_norm(cls_projectors2(cls_t)) prepended at index 0, then
            # the (zero-CLS-row) positional embedding is added to the whole sequence.
            if cls_token is None:  # defensive: diagnostic forwards without a CLS token
                cls_token = x.new_zeros(x.shape[0], self.reg_cls_dim)
            x = torch.cat((self.wg_norm(self.cls_projectors2(cls_token)).unsqueeze(1), x), dim=1)
        x = x + self.pos_embed
        N, T, D = x.shape

        # timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                                # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)
            if (i + 1) == self.encoder_depth:
                if self.projector_type == 'conv':
                    side = int(round(T ** 0.5))
                    g = x.reshape(N, side, side, D).permute(0, 3, 1, 2)
                    zs = [p(g).permute(0, 2, 3, 1).reshape(N, T, -1) for p in self.projectors]
                else:
                    zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
        cls_out = None
        if self.reg_cls:
            x, cls_out = self.final_layer(x, c, split_cls=True)
        else:
            x = self.final_layer(x, c)            # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        v_suff = None
        if self.sufficiency_head is not None:
            v_suff = self.unpatchify(self.sufficiency_head(zs[0], t))

        if self.reg_cls:
            # REG models return the CLS velocity as a 4th output; samplers index [0].
            return x, zs, v_suff, cls_out

        return x, zs, v_suff


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}
