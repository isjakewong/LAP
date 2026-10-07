"""
ImageNet-1k parquet shards -> paired PNGs and SD-VAE moments. Each image is converted to
RGB, its short side bicubically resized to 256, center-cropped, and encoded.

Optional SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT select a subset of the parquet files.
Out: <out>/images/<stem>.png, <out>/vae-sd/<stem>.npy, <out>/labels/labels_<task>.json
(merge the label files with preprocessing/merge_labels.py).
"""
import argparse, glob, io, json, os
import numpy as np
import torch
from PIL import Image


def center_crop_256(img):
    img = img.convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = img.size
    l, t = (w - 256) // 2, (h - 256) // 2
    return img.crop((l, t, l + 256, t + 256))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parquet-glob", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ntask = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    import pyarrow.parquet as pq
    from diffusers.models import AutoencoderKL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().requires_grad_(False).to(device)

    files = sorted(glob.glob(args.parquet_glob))
    mine = files[task::ntask]
    print(f"[task {task}] {len(mine)}/{len(files)} parquet files", flush=True)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "vae-sd"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
    labels = {}

    def flush(stems, imgs):
        x = torch.from_numpy(np.stack(imgs)).to(device).float() / 127.5 - 1.0
        with torch.no_grad():
            d = vae.encode(x)["latent_dist"]
            mom = torch.cat([d.mean, d.std], dim=1).cpu().numpy().astype(np.float32)
        for st, m in zip(stems, mom):
            np.save(os.path.join(args.out, "vae-sd", st + ".npy"), m)

    n = 0
    stems, imgs = [], []
    for fi, f in enumerate(mine):
        tbl = pq.read_table(f, columns=["image", "label"])
        imgcol = tbl.column("image").to_pylist()
        labcol = tbl.column("label").to_pylist()
        for im, lab in zip(imgcol, labcol):
            b = im["bytes"] if isinstance(im, dict) else im
            img = center_crop_256(Image.open(io.BytesIO(b)))
            stem = f"t{task:03d}_{n:07d}"
            img.save(os.path.join(args.out, "images", stem + ".png"))
            labels[stem + ".npy"] = int(lab)
            stems.append(stem); imgs.append(np.asarray(img).transpose(2, 0, 1))
            n += 1
            if len(imgs) == args.batch_size:
                flush(stems, imgs); stems, imgs = [], []
        print(f"[task {task}] file {fi + 1}/{len(mine)}, {n} images", flush=True)
    if imgs:
        flush(stems, imgs)
    json.dump(labels, open(os.path.join(args.out, "labels", f"labels_{task}.json"), "w"))
    print(f"[task {task}] DONE {n} images")


if __name__ == "__main__":
    main()
