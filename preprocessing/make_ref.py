"""Build an ADM-evaluator reference batch for RESISC45 or AID.

`arr_0` = uint8 images (N, 256, 256, 3), drawn uniformly without replacement from all
stems in <data>/vae-sd/dataset.json with a fixed seed (all images when N equals the
dataset size, as for AID); `arr_1` = their class labels. The sampled stems are listed
in <data>/ref_index.csv.
"""
import argparse, csv, json, os, time

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, "vae-sd", "dataset.json")) as f:
        labels = json.load(f)["labels"]
    labels = sorted(labels)  # [[stem.npy, label], ...] in stem order
    stems = [os.path.splitext(k)[0] for k, _ in labels]
    ys = np.array([int(v) for _, v in labels], dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(stems), size=args.n, replace=False))

    arr = np.empty((args.n, 256, 256, 3), dtype=np.uint8)
    t0 = time.time()
    for j, i in enumerate(idx):
        im = Image.open(os.path.join(args.data_dir, "images", stems[i] + ".png")).convert("RGB")
        a = np.asarray(im)
        assert a.shape == (256, 256, 3), (stems[i], a.shape)
        arr[j] = a
        if (j + 1) % 2000 == 0:
            print(f"[ref] {j + 1}/{args.n} ({(j + 1) / (time.time() - t0):.0f}/s)", flush=True)
    sel_labels = ys[idx]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp.npz"
    np.savez(tmp, arr_0=arr, arr_1=sel_labels)
    os.replace(tmp, args.out)
    with open(os.path.join(args.data_dir, "ref_index.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["ref_row", "stem", "label"])
        for j, i in enumerate(idx):
            wr.writerow([j, stems[i], int(ys[i])])

    z = np.load(args.out)
    print("[ref] keys:", z.files, {k: (z[k].shape, str(z[k].dtype)) for k in z.files})
    hist = np.bincount(sel_labels, minlength=int(ys.max()) + 1)
    print(f"[ref] label histogram: min={hist.min()} max={hist.max()} mean={hist.mean():.1f}")
    print(f"[ref] wrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB), seed={args.seed}, N={args.n}")
    print("[ref] DONE")


if __name__ == "__main__":
    main()
