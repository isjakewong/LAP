# Third-party notices

This is a derived research code release. Original notices are preserved; modifications include LAP targets, local predictors, experimental composition options, diagnostics, and portability tooling. New LAP code follows the MIT grant in `LICENSE`. That grant does not replace third-party terms.

| Files / component | Upstream | Notice |
|---|---|---|
| Training, SiT integration, sampling, dataset conventions, encoder integration | [REPA](https://github.com/sihyun-yu/REPA), with SiT/DiT lineage | `licenses/REPA.txt` (MIT, Sihyun Yu) |
| `models/mae_vit.py` | [MAE](https://github.com/facebookresearch/mae) | `licenses/MAE.txt`, CC BY-NC 4.0; Meta copyright retained |
| `models/mocov3_vit.py` | [MoCo v3](https://github.com/facebookresearch/moco-v3) | `licenses/MoCo-v3.txt`, CC BY-NC 4.0; Facebook copyright retained |
| `models/jepa.py` | [I-JEPA](https://github.com/facebookresearch/ijepa) | `licenses/I-JEPA.txt`, CC BY-NC 4.0; Meta copyright retained |
| `models/clip_vit.py` | [OpenAI CLIP](https://github.com/openai/CLIP), via REPA | `licenses/CLIP.txt` (MIT) |
| `evaluations/evaluator.py` | [guided-diffusion](https://github.com/openai/guided-diffusion) | `licenses/guided-diffusion.txt` (MIT, OpenAI) |

License texts were retrieved from the named upstream repositories while preparing the release. Teacher/VAE weights are downloaded separately, including sources with model-specific terms (AIMv2, DINOv3, Perception Encoder, and EQ-VAE). The code license does not license those weights or ImageNet/AID/RESISC45 data. No third-party teacher or VAE checkpoint is included in the LAP weight export.
