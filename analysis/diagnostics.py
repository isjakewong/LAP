"""
Presence vs. use of the alignment target in a trained SiT, per noise bin b.

  * Presence P(b): held-out linear (optionally MLP) probe R^2 from the hidden state
    h_t^l of the aligned block to the frozen teacher tokens y.
  * Use U(b): causal dependence of the velocity output on a subspace of h_t^l. The
    subspace is ablated (mean-fill or resample) by a forward hook on block l, and
    U(b) = E||v_ablate - v*||^2 - E||v - v*||^2.
  * Sufficiency: held-out R^2 from the projected representation a_t = g_psi(h) to the
    patchified velocity target.

All model interaction goes through forward hooks. The aligned layer l is block index
encoder_depth - 1, whose output the alignment projector reads.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BINS = {"high": (0.7, 1.0), "mid": (0.3, 0.7), "low": (0.0, 0.3)}


# --------------------------------------------------------------------------- #
#  Flow-matching (linear path, v-prediction) — matches loss.SILoss            #
# --------------------------------------------------------------------------- #
def linear_interpolant(x, t, noise):
    """x_t = (1-t) x + t noise ; v* = noise - x  (linear path, v-prediction)."""
    tb = t.view(-1, *([1] * (x.dim() - 1)))
    x_t = (1 - tb) * x + tb * noise
    v_star = noise - x
    return x_t, v_star


def sample_t_in_bin(n, lo, hi, device):
    return lo + (hi - lo) * torch.rand(n, device=device)


# --------------------------------------------------------------------------- #
#  Hidden-state capture / ablation via forward hooks on a chosen block        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def capture_hidden(model, x_t, t, y, layer):
    """Return (h_layer, v) where h_layer is the output of model.blocks[layer]."""
    store = {}
    handle = model.blocks[layer].register_forward_hook(
        lambda m, i, o: store.__setitem__("h", o))
    try:
        v = model(x_t, t, y)[0]
    finally:
        handle.remove()
    return store["h"], v


@torch.no_grad()
def ablate_forward(model, x_t, t, y, layer, Pi, h_bar=None, mode="mean"):
    """
    Forward pass with the subspace Pi removed from block `layer`'s output.
      mean:     h <- h - Pi (h - h_bar)         (mean-fill)
      resample: h <- h - Pi (h - h[perm])       (replace with another image's)
    Pi: (D, D) symmetric projector. h_bar: (D,) mean hidden (mean mode).
    """
    def hook(mod, inp, out):
        if mode == "mean":
            ref = h_bar.view(1, 1, -1)
        elif mode == "resample":
            perm = torch.randperm(out.shape[0], device=out.device)
            ref = out[perm]
        else:
            raise ValueError(mode)
        return out - torch.einsum("ntd,de->nte", out - ref, Pi)

    handle = model.blocks[layer].register_forward_hook(hook)
    try:
        v = model(x_t, t, y)[0]
    finally:
        handle.remove()
    return v


# --------------------------------------------------------------------------- #
#  Presence: probe h -> y                                                      #
# --------------------------------------------------------------------------- #
def fit_ridge(H, Y, rel_ridge=1e-3):
    """
    Closed-form ridge with a SCALE-RELATIVE penalty: returns (W, b), Y ~= H@W + b.
    The penalty is rel_ridge * mean(diag(HtH)) so it is invariant to feature scale
    and well-conditioned.
    """
    Hm, Ym = H.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    Hc, Yc = H - Hm, Y - Ym
    D = Hc.shape[1]
    HtH = Hc.t() @ Hc
    lam = rel_ridge * HtH.diagonal().mean().clamp(min=1e-8)
    A = HtH + lam * torch.eye(D, device=H.device, dtype=H.dtype)
    W = torch.linalg.solve(A, Hc.t() @ Yc)          # (D, Denc)
    b = Ym - Hm @ W
    return W, b


def r2_score(W, b, H, Y):
    pred = H @ W + b
    sse = ((Y - pred) ** 2).sum()
    sst = ((Y - Y.mean(0, keepdim=True)) ** 2).sum()
    return float(1.0 - sse / (sst + 1e-12))


def fit_mlp_probe(Htr, Ytr, Hva, Yva, hidden=256, iters=300, lr=1e-2, device="cpu"):
    """Nonlinear presence probe: 1-hidden-layer MLP h->y, held-out R^2."""
    net = nn.Sequential(nn.Linear(Htr.shape[1], hidden), nn.SiLU(),
                        nn.Linear(hidden, Ytr.shape[1])).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = ((net(Htr) - Ytr) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        sse = ((net(Hva) - Yva) ** 2).sum()
        sst = ((Yva - Yva.mean(0, keepdim=True)) ** 2).sum()
    return float(1.0 - sse / (sst + 1e-12))


# --------------------------------------------------------------------------- #
#  Target-correlated subspace                                                  #
# --------------------------------------------------------------------------- #
def subspace_from_probe(W, k):
    """Top-k target-correlated directions in h-space = top-k left sing. vecs of W."""
    U, S, _ = torch.linalg.svd(W, full_matrices=False)   # W:(D,Denc) -> U:(D,r)
    Vk = U[:, :k]
    return Vk @ Vk.t()                                   # (D, D) projector


# --------------------------------------------------------------------------- #
#  Driver-level: collect data, compute P(b) and U(b)                          #
# --------------------------------------------------------------------------- #
def patchify_velocity(v, p):
    """(N,C,H,W) -> (N, (H/p)*(W/p), C*p*p) in PatchEmbed (row-major) token order."""
    N, C, H, W = v.shape
    v = v.reshape(N, C, H // p, p, W // p, p)
    v = v.permute(0, 2, 4, 1, 3, 5).reshape(N, (H // p) * (W // p), C * p * p)
    return v


@torch.no_grad()
def collect_diag_data(model, loader, layer, lo, hi, device, max_tokens=100_000):
    """
    Collect token-level (H, Y, A_t, Vp) for a bin, where A_t = g_psi(h) is the
    aligned projection and Vp is the patchified velocity target v* = noise - x.
    Enables: aligned-subspace use (h->a_t directions) and the sufficiency score
    (a_t -> v* R^2).
    """
    proj = model.projectors[0]
    p = model.x_embedder.patch_size[0]
    Hs, Ys, As, Vs = [], [], [], []
    n = 0
    for x, ylab, yfeat in loader:
        x, ylab, yfeat = x.to(device), ylab.to(device), yfeat.to(device)
        t = sample_t_in_bin(x.shape[0], lo, hi, device)
        x_t, v_star = linear_interpolant(x, t, torch.randn_like(x))
        h, _ = capture_hidden(model, x_t, t, ylab, layer)        # (N,T,D)
        a = proj(h)                                              # (N,T,zdim)
        vp = patchify_velocity(v_star, p)                        # (N,T,C*p*p)
        Hs.append(h.reshape(-1, h.shape[-1]).float().cpu())
        Ys.append(yfeat.reshape(-1, yfeat.shape[-1]).float().cpu())
        As.append(a.reshape(-1, a.shape[-1]).float().cpu())
        Vs.append(vp.reshape(-1, vp.shape[-1]).float().cpu())
        n += Hs[-1].shape[0]
        if n >= max_tokens:
            break
    cut = lambda L: torch.cat(L)[:max_tokens]
    return cut(Hs), cut(Ys), cut(As), cut(Vs)


@torch.no_grad()
def mean_hidden(model, loader, layer, lo, hi, device, max_batches=20):
    acc, m = 0.0, 0
    for bi, (x, ylab, _) in enumerate(loader):
        x, ylab = x.to(device), ylab.to(device)
        t = sample_t_in_bin(x.shape[0], lo, hi, device)
        x_t, _ = linear_interpolant(x, t, torch.randn_like(x))
        h, _ = capture_hidden(model, x_t, t, ylab, layer)
        acc = acc + h.reshape(-1, h.shape[-1]).sum(0)
        m += h.reshape(-1, h.shape[-1]).shape[0]
        if bi + 1 >= max_batches:
            break
    return acc / m


@torch.no_grad()
def compute_use(model, loader, layer, Pi, h_bar, device, lo, hi,
                mode="mean", max_batches=40):
    """U(b) = E||v_ablate - v*||^2 - E||v - v*||^2 over the bin (lo,hi)."""
    Pi = Pi.to(device)
    h_bar = h_bar.to(device) if h_bar is not None else None
    deltas = []
    for bi, (x, ylab, _) in enumerate(loader):
        x, ylab = x.to(device), ylab.to(device)
        t = sample_t_in_bin(x.shape[0], lo, hi, device)
        x_t, v_star = linear_interpolant(x, t, torch.randn_like(x))
        v_clean = model(x_t, t, ylab)[0]
        v_abl = ablate_forward(model, x_t, t, ylab, layer, Pi, h_bar, mode)
        e_clean = ((v_clean - v_star) ** 2).flatten(1).mean(1)
        e_abl = ((v_abl - v_star) ** 2).flatten(1).mean(1)
        deltas.append((e_abl - e_clean).mean().item())
        if bi + 1 >= max_batches:
            break
    return float(np.mean(deltas))


def presence_use_report(model, loader_factory, layer, device, k=(32, 128, 256),
                        bins=BINS, mode="mean", nonlinear=False):
    """
    Full per-bin P(b), U(b) for each subspace rank in `k`. `loader_factory()` returns a
    fresh iterator each call (so presence/use/mean passes are independent).
    """
    k_sweep = k
    MB = 20  # compute_use batches (speed)
    out = {}
    for b, (lo, hi) in bins.items():
        H, Y, At, Vp = collect_diag_data(model, loader_factory(), layer, lo, hi, device)
        H, Y, At, Vp = H.to(device), Y.to(device), At.to(device), Vp.to(device)
        perm = torch.randperm(H.shape[0], device=device)
        H, Y, At, Vp = H[perm], Y[perm], At[perm], Vp[perm]
        ntr = int(0.8 * H.shape[0])
        Htr, Hva = H[:ntr], H[ntr:]
        Ytr, Yva = Y[:ntr], Y[ntr:]
        Atr, Ava = At[:ntr], At[ntr:]
        Vtr, Vva = Vp[:ntr], Vp[ntr:]

        # presence: linear / nonlinear probe h -> y
        Wy, by = fit_ridge(Htr, Ytr)
        p_lin = r2_score(Wy, by, Hva, Yva)
        p_nl = fit_mlp_probe(Htr, Ytr, Hva, Yva, device=device) if nonlinear else None
        # cross-encoder diagnostics (e.g. MAE-target model probed against DINOv2):
        # a_t and y have different dims; cosine is undefined there, probes still work
        align_cos = (float((F.normalize(Atr, dim=-1) * F.normalize(Ytr, dim=-1)).sum(-1).mean())
                     if Atr.shape[1] == Ytr.shape[1] else None)

        # sufficiency: a_t -> v* held-out R^2
        Wv, bv = fit_ridge(Atr, Vtr)
        suff_r2 = r2_score(Wv, bv, Ava, Vva)

        # aligned-subspace probe: directions of h that linearly carry a_t = g_psi(h)
        Wa, ba = fit_ridge(Htr, Atr)

        h_bar = mean_hidden(model, loader_factory(), layer, lo, hi, device)
        D = H.shape[1]
        U_full = compute_use(model, loader_factory(), layer, torch.eye(D, device=device),
                             h_bar, device, lo, hi, mode, max_batches=MB)
        # use of the y-correlated subspace AND the aligned (g_psi) subspace, per k
        use_y, var_y, use_al, var_al = {}, {}, {}, {}
        for kk in k_sweep:
            Pi_y = subspace_from_probe(Wy, kk).to(device)
            Pi_a = subspace_from_probe(Wa, kk).to(device)
            var_y[str(kk)] = float((Pi_y @ H.t()).pow(2).sum() / H.pow(2).sum())
            var_al[str(kk)] = float((Pi_a @ H.t()).pow(2).sum() / H.pow(2).sum())
            use_y[str(kk)] = compute_use(model, loader_factory(), layer, Pi_y, h_bar,
                                         device, lo, hi, mode, max_batches=MB)
            use_al[str(kk)] = compute_use(model, loader_factory(), layer, Pi_a, h_bar,
                                          device, lo, hi, mode, max_batches=MB)
        out[b] = {"presence_r2": p_lin, "presence_r2_nonlinear": p_nl,
                  "align_cosine": align_cos, "sufficiency_r2": suff_r2,
                  "use_full_layer": U_full,
                  "use_by_k": use_y, "var_frac_by_k": var_y,
                  "use_aligned_by_k": use_al, "var_aligned_by_k": var_al,
                  "estimator": "probe", "mode": mode}
    return out
