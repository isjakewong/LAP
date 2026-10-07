"""
Development presence/use diagnostics (token-split protocol) for a trained checkpoint.

Per noise bin, reports the linear presence of the teacher target in the hidden state of
the aligned block (default encoder_depth-1), the alignment cosine, the velocity
sufficiency of the projected representation, and the causal use of the target-correlated
and aligned subspaces (analysis/diagnostics.py). The main-text full-scale numbers use
analysis/probe_image_disjoint.py instead.

Example:
  python analysis/run_diagnostics.py --ckpt outputs/dev200_repa/checkpoints/0100000.pt \
      --data-dir data/in256_dev200 --enc-type dinov2-vit-b --out diag_repa.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

import argparse
import json

import torch

from utils import build_model_from_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--enc-type", default="dinov2-vit-b")
    ap.add_argument("--layer", type=int, default=-1, help="block index; -1 => encoder_depth-1")
    ap.add_argument("--k", type=int, nargs="+", default=[32, 128, 256],
                    help="subspace ranks for the use sweep")
    ap.add_argument("--num-batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mode", default="mean", choices=["mean", "resample"])
    ap.add_argument("--nonlinear", action="store_true")
    ap.add_argument("--out", default="diagnostics.json")
    args = ap.parse_args()

    from torch.utils.data import DataLoader
    import diagnostics as dg
    from dataset import CustomDataset
    from train_purifier import sample_posterior
    from utils import load_encoders, preprocess_raw_image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, get = build_model_from_ckpt(ckpt, device)
    layer = args.layer if args.layer >= 0 else model.encoder_depth - 1

    encoders, encoder_types, _ = load_encoders(args.enc_type, device, get("resolution", 256))
    encoder, enc_type = encoders[0], encoder_types[0]
    dataset = CustomDataset(args.data_dir)

    def loader_factory():
        """Yields (x_latent, y_label, y_feat) batches, train-faithful."""
        dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True)
        for bi, (raw_image, x_mom, y_label) in enumerate(dl):
            if bi >= args.num_batches:
                break
            raw_image = raw_image.to(device)
            x = sample_posterior(x_mom.squeeze(1).to(device))
            with torch.no_grad():
                z = encoder.forward_features(preprocess_raw_image(raw_image, enc_type))
                if 'dinov2' in enc_type:
                    z = z['x_norm_patchtokens']
            yield x, y_label.to(device), z

    report = dg.presence_use_report(model, loader_factory, layer, device, k=args.k,
                                    mode=args.mode, nonlinear=args.nonlinear)

    with open(args.out, "w") as f:
        json.dump({"ckpt": args.ckpt, "layer": layer, "report": report}, f, indent=2)
    print(f"Wrote {args.out}")
    for b, d in report.items():
        uy = {k: round(v, 5) for k, v in d["use_by_k"].items()}
        ua = {k: round(v, 5) for k, v in d["use_aligned_by_k"].items()}
        print(f"  {b:4s} P_lin={d['presence_r2']:+.3f} cos={(d.get('align_cosine') or 0):+.3f} "
              f"suff_r2={(d.get('sufficiency_r2') or 0):+.3f} U_full={d['use_full_layer']:+.4f}")
        print(f"        use_y(probe)={uy}")
        print(f"        use_aligned(g_psi)={ua}")


if __name__ == "__main__":
    main()
