# Purify Before You Align

Code for **Purify Before You Align: Improving Representation Alignment for Diffusion Models**.

Local appearance purification (LAP) changes only the representation-alignment target. Instead of aligning a SiT hidden state to `teacher(image)`, it aligns to `teacher(image) - predictor(clean_latent_patch)`:

- **LAP-L** fits an affine predictor from clean-latent patches to teacher tokens on each minibatch (`--target-mode residual`).
- **LAP-N** uses a small frozen convolutional predictor with a local receptive field, trained once per teacher (`--target-mode purified`, `train_purifier.py`).

The denoising loss, architecture and sampler are unchanged, so there is no inference cost. The code builds on [REPA](https://github.com/sihyun-yu/REPA).

## Setup

Linux, Python 3.10 and a CUDA GPU. `requirements.txt` pins the tested environment (PyTorch 2.5.1).

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "setuptools<81" wheel && pip install --no-build-isolation \
  git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33
python -m unittest discover -s tests -v
```

Run all commands from the repository root. Prepare the data and teacher weights as described in [docs/DATA.md](docs/DATA.md). For the MAE experiments:

```bash
mkdir -p ckpts
curl -fL https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth -o ckpts/mae_vitl.pth
```

## Training

Each main-text experiment has a recipe in `configs/`, listed by `reproduce.py`:

```bash
python reproduce.py list --group teachers      # Table 1; also: recipes, tokenizers, aerial, sweep, scale, mechanism, development

# MAE teacher: raw target (REPA), LAP-L, LAP-N. Append _s1 for the second seed.
python reproduce.py train in1k_mae_raw
python reproduce.py train in1k_mae_res
python reproduce.py train in1k_mae_lapk5
```

`reproduce.py train` runs `train.py --config configs/<recipe>.json` with the experiment's seed and step count; `--dry-run` prints the command, `--data-dir` overrides the dataset location, and any further flags are passed to `train.py`. Configs list only the options that differ from the `train.py` defaults (batch 64 on one GPU, alignment after block 8, coefficient 0.5, 400K steps, SD-VAE latents). Effective batch 256 uses 4 gradient-accumulation steps, because LAP-L fits its predictor on each physical batch of 64.

LAP-N needs a frozen purifier in `assets/`. Download the released purifiers (below) or train one:

```bash
python reproduce.py purifier in1k_mae_lapk5    # writes assets/purifier1k_mae_k5.pt
```

## Checkpoints

The [Hugging Face bundle](https://huggingface.co/yingheng/LAP-checkpoints) contains five SiT-B/2 models (ImageNet 256×256, 400K steps; fp32 EMA weights) and the small assets needed for training: 12 purifiers and the latent statistics of the two swapped VAEs. For each model it includes the seed with the lower FID-50K; the paper reports two-seed means.

| Model | File | FID-50K |
|---|---|---:|
| No alignment | `diffusion/in1k_vanilla_s1.pt` | 53.90 |
| MAE, REPA | `diffusion/in1k_mae_raw_s1.pt` | 52.41 |
| MAE, LAP-L | `diffusion/in1k_mae_res.pt` | 44.81 |
| MAE, LAP-N | `diffusion/in1k_mae_lapk5_s1.pt` | 41.53 |
| DINOv2, REPA | `diffusion/in1k_repa_s1.pt` | 41.35 |

```bash
python scripts/download_checkpoints.py --output checkpoints   # from huggingface.co/yingheng/LAP-checkpoints, verifies SHA256
cp checkpoints/assets/*.pt assets/
```

## Sampling and evaluation

```bash
python sample.py --ckpt checkpoints/diffusion/in1k_mae_lapk5_s1.pt --out samples/mae_lapn
torchrun --standalone --nproc_per_node=4 sample.py --ckpt ... --out ...   # multi-GPU
python reproduce.py sample in1k_mae_lapk5 --trusted-legacy               # your own training checkpoint
```

The default is the paper protocol: EMA weights, Euler–Maruyama SDE with 250 steps, no guidance, 50K samples, written as PNGs and an ADM-format `.npz`. Exported checkpoints load with `weights_only=True`; `--trusted-legacy` allows the pickled arguments stored in checkpoints written by `train.py`. FID uses the [ADM evaluator](https://github.com/openai/guided-diffusion/tree/main/evaluations) in a separate TensorFlow environment:

```bash
python3.10 -m venv .venv-eval && .venv-eval/bin/pip install -r requirements-eval.txt
.venv-eval/bin/python evaluations/evaluator.py VIRTUAL_imagenet256_labeled.npz samples/mae_lapn.npz
```

Sampling noise depends on the number of processes and the batch size, so fresh scores differ slightly from the archived ones.

## Reproducing the paper

[docs/REPRODUCING.md](docs/REPRODUCING.md) maps each table and figure to its experiment group, explains the experiment names, and gives the commands for the diagnostics (constant-readout score, image-disjoint probes, development metric). Recorded per-seed metrics are in `configs/experiments.json`; `python scripts/summarize_results.py --group teachers` prints seed means.

## Citation

```bibtex
@article{lap2026,
  title   = {Purify Before You Align: Improving Representation Alignment for Diffusion Models},
  author  = {Wang, Yingheng and Li, Yaoqiang and Wu, Yaqin and Bai, Junwen and Gu, Jiatao and De Sa, Christopher and Kuleshov, Volodymyr},
  journal = {arXiv preprint arXiv:ARXIV_ID},
  year    = {2026}
}
```

## License

New code is MIT-licensed. Files adapted from MAE, MoCo v3 and I-JEPA keep their CC BY-NC 4.0 terms; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The released checkpoints are CC BY-NC 4.0. Teacher and VAE weights and the datasets are not covered and must be obtained under their own terms.
