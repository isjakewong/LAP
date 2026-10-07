# Reproducing the paper

## Experiment groups

`python reproduce.py list --group <group>` lists the runs of each group. Every run has an entry in `configs/experiments.json` with its recipe, seed, step count, sampling setup and, where archived, its metrics.

| Paper result | Group | Evaluation |
|---|---|---|
| Table 1: five teachers, raw / LAP-L / LAP-N | `teachers` | 400K steps, 50K samples, FID-50K against the ADM ImageNet reference |
| Table 2: iREPA-style head and normalization, VA-REPA, REG, sREPA | `recipes` | as Table 1 |
| Table 3: SD-VAE, EQ-VAE, REPA-E VAE | `tokenizers` | as Table 1, with the matching VAE |
| Table 4: RESISC45 and AID | `aerial` | 50K steps, 10K samples, dataset-specific 10K reference |
| Training curves | `teachers` (100K/200K/300K checkpoints) | 10K samples, ADM ImageNet reference; archived in `results/curves_fid10k.json` |
| Constant-readout score vs. LAP-L gain, 13 encoders | `sweep` and Table 1 raw/LAP-L | 100K: FID-10K; 400K: FID-50K |
| Predictable-only target and prediction-loss interventions | `mechanism`, `development` | as Table 1; development runs use the proxy metric below |
| Batch 256, SiT-L/XL, alignment depth 4 | `scale` | step counts per entry; guided check uses `sample.py --cfg-scale 1.5` |

## Experiment names

| Name part | Meaning |
|---|---|
| `vanilla` | no alignment (`proj_coeff` 0) |
| `repa`, `aa_raw` | DINOv2 ViT-B, raw target (REPA) |
| `aa_res`, `lapk3` | DINOv2, LAP-L and LAP-N (purifier receptive field 3) |
| `aa_xpred` | DINOv2, predictable component only |
| `mae_raw`, `mae_res`, `mae_lapk5` | MAE ViT-L: raw, LAP-L, LAP-N (receptive field 5) |
| `clip_*`, `aimv2_*`, `dinov3_*` | other teachers; `res`/`lapl` = LAP-L, `lapk3`/`lapn` = LAP-N |
| `irepa`, `varepa`, `reg`, `srepa` | composition with iREPA-style head/normalization, VA-REPA, REG, sREPA |
| `uca_ls<λ>` | auxiliary prediction loss with weight λ |
| `b256`, `L`, `XL`, `d4` | batch 256, SiT-L/2, SiT-XL/2, alignment after block 4 |
| `in1k100k_*` | 100K-step encoder sweep, raw vs. LAP-L |
| `eqvae`, `e2esd` | EQ-VAE, REPA-E VAE |
| `rs`, `aid`, `dev200` | RESISC45, AID, 200-class ImageNet development set |
| `_s1`, `_s2`, `_s3` | training seed (no suffix: seed 0) |

## Training details

The configs use AdamW (lr 1e-4, betas (0.9, 0.999), no weight decay), gradient clipping at 1, fp16 with TF32, EMA 0.9999, uniform timesteps, the linear interpolant with velocity prediction, and label dropout 0.1.

LAP-L fits its affine predictor on all tokens of each physical batch of 64 images, so changing the per-GPU batch changes the target. `reproduce.py` therefore trains on one GPU and reaches batch 256 by gradient accumulation.

The compositions use the shared block-8, batch-64 recipe rather than each method's own settings. iREPA uses spatial target standardization and a 3×3 convolutional projector; VA-REPA uses its sigmoid weighting (τ = 0.7, k = 20); REG uses a jointly noised CLS token with β = 0.03; sREPA uses the off-diagonal Gram MSE with weight 2 and alignment weight 1 (its defaults). Purification is applied before each method's loss.

## Purifiers

`python reproduce.py purifier <experiment>` trains the frozen LAP-N predictor with the arguments in `configs/purifiers.json` (20K steps, 4,096 held-out images; receptive field 3 for DINOv2 and CLIP, 5 for MAE, AIMv2 and DINOv3). `probe_purified.py` audits how much class information each target keeps (raw, LAP-L, LAP-N and the removed components), using images held out from purifier training:

```bash
python probe_purified.py --data-dir data/in256_dev200 \
  --purifiers assets/purifier_mae_k5.pt assets/purifier_dinov2_k3.pt --out results/probe_purified.json
```

## Development metric

The `development` runs (dev-200, 50K steps) are compared with a proxy FID computed on normalized DINOv2 CLS features, with Euler ODE-50 sampling and a fresh 10K real reference. It is not comparable to Inception FID.

```bash
python gen_fid.py --ckpt outputs/dev200_mae_raw/checkpoints/0050000.pt \
  --data-dir data/in256_dev200 --num-samples 10000 --num-classes 200 \
  --steps 50 --batch 50 --out results/dev_mae_raw.json
python target_utility.py --enc mae-vit-l --data-dir data/in256_dev200 \
  --out results/mae_utility.json
```

`run_diagnostics.py` gives the earlier token-split presence/use diagnostics reported for the development runs. `dev200_aa_raw` is the 50K DINOv2 reference for the encoder-utility comparison; `dev200_repa` is the 100K reference for the prediction-loss runs.

## Constant-readout score

```bash
python analysis_constant_baseline.py --data-dir data/in256_dev200 --tag dev200 --batches 32 --seed 0 \
  --purifiers mae-vit-l=assets/purifier_mae_k5.pt,dinov2-vit-b=assets/purifier_dinov2_k3.pt,clip-vit-L=assets/purifier_clip_k3.pt \
  --out results/c0_dev200.json
python analysis_constant_baseline.py --data-dir data/in256_dev200 --tag sweep \
  --encoders mae-vit-b,clip16-vit-b,dinov2-vit-s,dinov2-vit-l,pespatial-vit-b,dinov3-vit-l,dinov3-vit-hplus,pecore-vit-l \
  --out results/c0_sweep.json
python scripts/verify_sweep_statistics.py
```

Archived outputs are `results/constant_baseline_{dev200,in1k}.json` and `results/sweep_{100k,400k}.json`. The score uses 2,048 images in batches of 64 with one posterior sample per image. C₀ is the norm of the mean unit-normalized target token. The 400K sweep points continue the same runs: append `--max-train-steps 400000` to `reproduce.py train`. `verify_sweep_statistics.py` recomputes the exact permutation tests (eight new encoders) and the Monte Carlo tests (all 13) from the archived seed results.

## Hidden-state probes

```bash
python probe_image_disjoint.py --data data/in1k_256 --out outputs/probes \
  --models vanilla_s1 mae_res mae_lapk5_s1 repa_s1
```

This measures presence, velocity sufficiency and causal use of the aligned subspace, fitting on 1,024 images and evaluating on 256 disjoint images (t ∈ [0.3, 0.7), rank 256, 2,000 bootstrap replicates). It reads `outputs/in1k_<name>/checkpoints/0400000.pt`, falling back to `checkpoints/diffusion/in1k_<name>.pt`, so the command above runs on the released checkpoints. Archived results are in `results/image_disjoint_probes.json`. The paper's numbers use the seeds in the script's default model list, which also needs `aa_res`, `lapk3` and `uca_ls2.0` trained from their configs.

## Notes on exact reproduction

- Training is not bitwise reproducible: GPU kernels are nondeterministic, and some original runs were resumed without data-loader or RNG state.
- `in1k_e2esd_vanilla_s1` was trained by a collaborator; only rounded metrics exist, and its config is the seed-0 recipe with seed 1.
- Recorded preprocessing, class counts and dataset hashes are in `results/reproducibility_manifest.json`.
