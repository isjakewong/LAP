import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[],
            accelerator=None,
            latents_scale=None,
            latents_bias=None,
            proj_weight_schedule=None,
            proj_tau=0.7,
            proj_k=20.0,
            struc_coeff=0.0,
            ):
        # VA-REPA (Yang et al., 2026, "Stable Velocity"): per-sample alignment
        # weights w(t) that emphasize the low-variance regime near the data,
        # normalized by their batch sum. None reproduces REPA bit-identically.
        self.proj_weight_schedule = proj_weight_schedule
        self.proj_tau = proj_tau
        self.proj_k = proj_k
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        # sREPA (Xu et al., arXiv:2605.16949): weight of the structural (off-diagonal
        # Gram) term. 0 (default) skips the computation entirely -> exact REPA.
        self.struc_coeff = struc_coeff

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None, cls_token=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)

        time_input = time_input.to(device=images.device, dtype=images.dtype)

        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        # ---- REG (Wu et al., NeurIPS 2025, arXiv:2507.01467) -----------------------
        # The DINOv2 CLS token is a second diffusion variable: noised with the SAME t
        # (independent noise), prepended to the latent token sequence, and denoised
        # jointly under a v-prediction loss. cls_token is None unless REG is on, in
        # which case nothing below draws extra randomness -> bit-identical to REPA.
        cls_target = None
        if cls_token is not None:
            noises_cls = torch.randn_like(cls_token)
            _sq = lambda v: v.squeeze(-1).squeeze(-1) if torch.is_tensor(v) else v
            cls_input = _sq(alpha_t) * cls_token + _sq(sigma_t) * noises_cls
            cls_target = _sq(d_alpha_t) * cls_token + _sq(d_sigma_t) * noises_cls
            model_kwargs = dict(model_kwargs, cls_token=cls_input)

        _out = model(model_input, time_input.flatten(), **model_kwargs)
        model_output, zs_tilde, v_suff = _out[0], _out[1], _out[2]
        cls_output = _out[3] if len(_out) > 3 else None
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        if cls_output is not None and cls_target is not None:
            cls_loss = mean_flat((cls_output - cls_target) ** 2)
        else:
            cls_loss = torch.zeros_like(denoising_loss)

        # projection loss
        proj_loss = 0.
        bsz = zs[0].shape[0]
        per_sample = [0.] * bsz   # VA-REPA needs the per-sample cosine losses
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1)
                z_j = torch.nn.functional.normalize(z_j, dim=-1)
                l_j = mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
                proj_loss += l_j
                per_sample[j] = per_sample[j] + l_j
        proj_loss /= (len(zs) * bsz)
        if self.proj_weight_schedule is not None:
            w = self._proj_weight(time_input.flatten().float())          # (N,)
            l = torch.stack(per_sample).float() / len(zs)                # (N,)
            wsum = w.sum()
            proj_loss = (w * l).sum() / wsum if wsum > 1e-8 else proj_loss * 0.

        # Auxiliary prediction loss ||r(a_t, t) - v*||^2 on the sufficiency head's output,
        # in the same units as the denoising loss. Zero when the head is disabled.
        if v_suff is not None:
            suff_loss = mean_flat((v_suff - model_target) ** 2)
        else:
            suff_loss = torch.zeros_like(denoising_loss)

        # ---- sREPA (Xu et al., arXiv:2605.16949, Alg. 1 / Eq. 6, mode='MSE') --------
        # Token-token cosine-similarity (Gram) matrices of the L2-normalised teacher
        # target and of the projected student features; MSE over the OFF-DIAGONAL
        # entries only (the diagonal is identically 1 and would be a free win).
        # Computed from zs, i.e. from whatever target transform train.py already
        # applied, so the LAP-L arm residualises the structure too.
        if self.struc_coeff > 0:
            zt = torch.nn.functional.normalize(zs[0].float(), dim=-1)
            zst = torch.nn.functional.normalize(zs_tilde[0].float(), dim=-1)
            A = torch.bmm(zt, zt.transpose(1, 2))
            A_til = torch.bmm(zst, zst.transpose(1, 2))
            off = ~torch.eye(A.shape[1], dtype=torch.bool, device=A.device)
            struc_loss = mean_flat(((A - A_til) ** 2)[:, off])
        else:
            struc_loss = torch.zeros_like(denoising_loss)

        # Monitor: mean cosine between the projected features and the alignment target.
        with torch.no_grad():
            a0 = torch.nn.functional.normalize(zs_tilde[0], dim=-1)
            y0 = torch.nn.functional.normalize(zs[0], dim=-1)
            align_cos = (a0 * y0).sum(dim=-1).mean(dim=-1)  # (N,)

        return denoising_loss, proj_loss, suff_loss, align_cos, cls_loss, struc_loss

    def _proj_weight(self, t):
        # VA-REPA sigmoid weighting (linear path).
        if self.proj_weight_schedule == "sigmoid":
            return torch.sigmoid(self.proj_k * (self.proj_tau - t))
        raise ValueError(self.proj_weight_schedule)
