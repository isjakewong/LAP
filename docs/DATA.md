# Data and Teachers

No images, teacher weights or VAE weights are distributed with this code.

## Dataset Layout

Training reads paired RGB images and cached SD-VAE posterior moments:

```text
data/in1k_256/
  images/<stem>.png
  vae-sd/<stem>.npy        # float32 [8, 32, 32]: posterior mean and standard deviation
  vae-sd/dataset.json      # {"labels": [["<stem>.npy", label], ...]}
```

Image and latent stems must match. Labels are used as stored. A fresh posterior sample is drawn every time a latent is used, with scale 0.18215. There are no random crops or flips.

## ImageNet-1k

Download the `ILSVRC/imagenet-1k` training shards (gated; accept the ImageNet terms on Hugging Face), then encode:

```bash
hf download ILSVRC/imagenet-1k --repo-type dataset --include 'data/train*' --local-dir data/imagenet_parquet
python preprocessing/encode_in1k.py --out data/in1k_256 \
  --parquet-glob 'data/imagenet_parquet/data/train*.parquet' --batch-size 64
python preprocessing/merge_labels.py --out data/in1k_256
```

Images are converted to RGB, resized bicubically so the short side is 256, and center-cropped. `SLURM_ARRAY_TASK_ID`/`SLURM_ARRAY_TASK_COUNT`, if set, split the shards across array jobs.

The corpus used in the paper contained 1,272,453 images, slightly fewer than the canonical 1,281,167; we could not establish why. Its class counts and manifest hashes are in `results/reproducibility_manifest.json`. A fresh preparation reproduces the recipe, not that exact corpus.

## dev-200 (Development Experiments)

Classes 0–199 of `evanarlian/imagenet_1k_resized_256`, all images (255,224), labels not remapped:

```bash
python preprocessing/download_dev_images.py --out data/in256_dev200 --num-classes 200
python preprocessing/encode_from_images.py --data-dir data/in256_dev200
```

## RESISC45 and AID

Generative training sets, not classification splits: RESISC45 merges all three splits (31,500 images, 45 classes); AID uses all 10,000 images (30 classes). The FID references are 10,000 training images drawn with seed 0.

```bash
hf download timm/resisc45 --repo-type dataset --local-dir data/resisc45_raw
python preprocessing/prepare_resisc45.py --snapshot data/resisc45_raw --out data/resisc45_256
python preprocessing/encode_from_images.py --data-dir data/resisc45_256
python preprocessing/make_ref.py --data-dir data/resisc45_256 --out data/resisc45_ref.npz

hf download blanchon/AID --repo-type dataset --local-dir data/aid_raw
python preprocessing/prepare_aid.py --dl data/aid_raw --out data/aid_256
python preprocessing/encode_from_images.py --data-dir data/aid_256
python preprocessing/make_ref.py --data-dir data/aid_256 --out data/aid_ref.npz
```

These models keep the 1,000-class label embedding. `reproduce.py sample` restricts sampling to the valid labels (45 or 30); with `sample.py`, pass `--sample-num-classes`. Evaluate against the dataset's own reference, not the ImageNet one.

## Swapped VAEs

The tokenizer experiments (EQ-VAE `zelaki/eq-vae`, REPA-E `REPA-E/e2e-sdvae-hf`) encode images online and normalize with per-channel statistics from `assets/eqvae-latents-stats.pt` and `assets/e2e-sdvae-400k-latents-stats.pt` (in the checkpoint bundle). Each VAE has its own purifier; `train.py` refuses a purifier fitted on a different latent space. To recompute the statistics:

```bash
python preprocessing/compute_latent_stats.py --vae-path zelaki/eq-vae \
  --data-dir data/in1k_256 --n-images 8192 --out assets/eqvae-latents-stats.pt
```

Different source images give slightly different statistics; use the released files to match the paper.

## Teachers

Teachers are loaded by `utils.load_encoders`, which also fixes each teacher's input normalization, resolution and token handling. Weights downloaded by torch.hub, timm, transformers or the CLIP package need network access on first use.

| `enc_type` | Source |
|---|---|
| `mae-vit-l`, `mae-vit-b` | [MAE](https://github.com/facebookresearch/mae) pretraining checkpoints, saved as `ckpts/mae_vitl.pth` / `ckpts/mae_vitb.pth` |
| `dinov2-vit-{s,b,l}` | [DINOv2](https://github.com/facebookresearch/dinov2) via torch.hub |
| `clip-vit-L`, `clip16-vit-b` | [OpenAI CLIP](https://github.com/openai/CLIP) ViT-L/14 and ViT-B/16 |
| `aimv2-vit-l` | [apple/aimv2-large-patch14-224](https://huggingface.co/apple/aimv2-large-patch14-224) via transformers |
| `dinov3-vit-{b,l,hplus}`, `pecore-vit-l`, `pespatial-vit-b` | timm ports of DINOv3 (LVD-1689M) and Perception Encoder; model names in `models/ext_encoders.py` |
| `mocov3-vit-b` (development) | [RCG](https://github.com/LTH14/rcg) release, saved as `ckpts/mocov3_vitb.pth` |
| `jepa-vit-h` (development) | [I-JEPA](https://github.com/facebookresearch/ijepa) ImageNet-1k ViT-H/14, saved as `ckpts/ijepa_vith.pth` |
| `satmae-vit-l` (aerial, unaligned runs only) | [SatMAE](https://github.com/sustainlab-group/SatMAE) RGB fMoW ViT-L, saved as `ckpts/satmae_vitl.pth` |

The unaligned aerial runs use alignment coefficient 0; their configs keep the SatMAE projector only so that initialization matches the original runs. Teacher weight revisions were not pinned during the original experiments.
