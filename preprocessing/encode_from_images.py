"""Encode <data>/images/*.png (256x256) with a frozen VAE:
<data>/<latents-name>/<stem>.npy = concat(mean, std) (8,32,32) float32. The label manifest
<data>/vae-sd/dataset.json must already exist (written by the prepare/download scripts).
Optional SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT shard the sorted image list;
existing outputs are skipped, so the script can be resumed.
"""
import argparse, os, shutil, time
import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vae-path", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--latents-name", default="vae-sd")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args()
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ntask = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    from diffusers.models import AutoencoderKL
    from torch.utils.data import Dataset, DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(args.vae_path).eval().requires_grad_(False).to(device)
    img_dir = os.path.join(args.data_dir, "images")
    out_dir = os.path.join(args.data_dir, args.latents_name)
    os.makedirs(out_dir, exist_ok=True)
    if task == 0:
        src = os.path.join(args.data_dir, "vae-sd", "dataset.json")
        dst = os.path.join(out_dir, "dataset.json")
        if not os.path.exists(dst):
            shutil.copy(src, dst)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".png"))
    mine = stems[task::ntask]
    todo = [s for s in mine if not os.path.exists(os.path.join(out_dir, s + ".npy"))]
    print(f"[task {task}] {len(mine)} images in shard, {len(todo)} to encode", flush=True)

    class Imgs(Dataset):
        def __len__(self): return len(todo)
        def __getitem__(self, i):
            st = todo[i]
            img = Image.open(os.path.join(img_dir, st + ".png")).convert("RGB")
            return st, torch.from_numpy(np.asarray(img).transpose(2, 0, 1).copy())

    dl = DataLoader(Imgs(), batch_size=args.batch_size, num_workers=args.num_workers)
    n, t0 = 0, time.time()
    for bstems, x in dl:
        x = x.to(device).float() / 127.5 - 1.0
        with torch.no_grad():
            d = vae.encode(x)["latent_dist"]
            mom = torch.cat([d.mean, d.std], dim=1).cpu().numpy().astype(np.float32)
        for st, m in zip(bstems, mom):
            np.save(os.path.join(out_dir, st + ".npy"), m)
        n += len(bstems)
        if n % (args.batch_size * 100) == 0:
            print(f"[task {task}] {n}/{len(todo)} ({n / (time.time() - t0):.1f} img/s)", flush=True)
    print(f"[task {task}] DONE {n} images")


if __name__ == "__main__":
    main()
