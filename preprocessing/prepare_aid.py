"""Build the AID generative training set from the HF imagefolder copy blanchon/AID:
<dl>/data/<ClassName>/<name>.jpg, 10,000 Google-Earth images, 30 classes, 600x600 RGB.

Downscaling follows ADM's center_crop_arr: center-crop to square if needed (recorded),
BOX reduction by 2 while the side is >= 512, then BICUBIC to 256 (600 -> 300 -> 256).

Output (<out>):
  images/aid_XXXXXXX.png    256x256 RGB PNG
  vae-sd/dataset.json       {"labels": [["aid_XXXXXXX.npy", label], ...]}
  aid_meta.csv, aid_class_counts.csv, aid_classes.json

Stems follow (class folder, natural file order: airport_1 < airport_2 < ...). Labels index
the sorted class-folder list. Count mismatches are reported, not asserted. Existing valid
PNGs are skipped.
"""
import argparse, csv, json, os, re, sys, time
from collections import Counter
from multiprocessing import Pool

from PIL import Image

EXPECTED_N = 10000
EXPECTED_NC = 30
EXPECTED_SIZE = (600, 600)
RES = 256
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def downsample(img):
    # BOX while >= 2x target, then BICUBIC (dhariwal center_crop_arr; image is square at this point)
    while min(img.size) >= 2 * RES:
        img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)
    if img.size != (RES, RES):
        img = img.resize((RES, RES), resample=Image.BICUBIC)
    return img


def write_one(args):
    stem, src, out_png = args
    with Image.open(src) as im0:
        w, h = im0.size
        existed = False
        if os.path.exists(out_png):
            try:
                with Image.open(out_png) as im:
                    existed = im.size == (RES, RES) and im.mode == "RGB"
            except Exception:
                existed = False  # rewrite a truncated/corrupt file
        cropped = w != h
        if not existed:
            im = im0.convert("RGB")
            if cropped:
                s = min(w, h)
                x0, y0 = (w - s) // 2, (h - s) // 2
                im = im.crop((x0, y0, x0 + s, y0 + s))
            im = downsample(im)
            tmp = out_png + ".tmp.png"
            im.save(tmp, format="PNG", compress_level=6)
            os.replace(tmp, out_png)
    return stem, w, h, cropped, existed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", required=True, help="local download of blanchon/AID")
    ap.add_argument("--out", default="data/aid_256")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefix", default="aid_")
    args = ap.parse_args()

    root = os.path.join(args.dl, "data")
    if not os.path.isdir(root):
        sys.exit(f"{root} not found; download blanchon/AID into --dl first")
    names = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if len(names) != EXPECTED_NC:
        print(f"[prepare] WARNING: {len(names)} class folders, expected {EXPECTED_NC}", flush=True)
    recs = []
    for lab, n in enumerate(names):
        files = sorted((f for f in os.listdir(os.path.join(root, n)) if f.lower().endswith(IMG_EXT)), key=natural_key)
        for f in files:
            recs.append(dict(src_file=f"{n}/{f}", label_name=n, label=lab))
        print(f"[prepare] {lab:2d} {n}: {len(files)} images", flush=True)
    if len(recs) != EXPECTED_N:
        print(f"[prepare] WARNING: {len(recs)} images, expected {EXPECTED_N}", flush=True)

    img_dir = os.path.join(args.out, "images")
    lat_dir = os.path.join(args.out, "vae-sd")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lat_dir, exist_ok=True)
    stems = [f"{args.prefix}{i:07d}" for i in range(len(recs))]

    jobs = [(s, os.path.join(root, r["src_file"]), os.path.join(img_dir, s + ".png")) for s, r in zip(stems, recs)]
    t0, done, skipped = time.time(), 0, 0
    info = {}
    with Pool(args.workers) as pool:
        for stem, w, h, cropped, existed in pool.imap(write_one, jobs, chunksize=16):
            info[stem] = (w, h, cropped, existed)
            done += 1
            skipped += int(existed)
            if done % 1000 == 0:
                print(f"[prepare] {done}/{len(jobs)} PNGs ({done / (time.time() - t0):.0f}/s, {skipped} pre-existing)", flush=True)
    print(f"[prepare] wrote {done - skipped} PNGs, {skipped} pre-existing, {time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(args.out, "aid_meta.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["stem", "src_file", "label_name", "label", "src_w", "src_h", "nonstandard_size", "center_cropped"])
        for s, r in zip(stems, recs):
            w, h, cropped, _ = info[s]
            wr.writerow([s, r["src_file"], r["label_name"], r["label"], w, h, int((w, h) != EXPECTED_SIZE), int(cropped)])
    counts = Counter(r["label"] for r in recs)
    with open(os.path.join(args.out, "aid_class_counts.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["label", "label_name", "count"])
        for i, n in enumerate(names):
            wr.writerow([i, n, counts.get(i, 0)])
    with open(os.path.join(args.out, "aid_classes.json"), "w") as f:
        json.dump(names, f, indent=1)
    with open(os.path.join(lat_dir, "dataset.json"), "w") as f:
        json.dump({"labels": [[s + ".npy", r["label"]] for s, r in zip(stems, recs)]}, f)

    sizes = Counter((v[0], v[1]) for v in info.values())
    n_nonstd = sum(1 for v in info.values() if (v[0], v[1]) != EXPECTED_SIZE)
    n_crop = sum(1 for v in info.values() if v[2])
    print(f"[prepare] classes ({len(names)}): {names}")
    print(f"[prepare] per-class counts: min={min(counts.values())} max={max(counts.values())}")
    print(f"[prepare] source sizes: {dict(sizes)}; nonstandard(!=600x600)={n_nonstd}; center_cropped(non-square)={n_crop}")
    n_png = len([f for f in os.listdir(img_dir) if f.endswith(".png")])
    print(f"[prepare] PNGs on disk: {n_png} (expected {EXPECTED_N}, records {len(recs)})")
    print("[prepare] DONE" if n_png == len(recs) == EXPECTED_N and len(names) == EXPECTED_NC else "[prepare] DONE WITH COUNT MISMATCH")


if __name__ == "__main__":
    main()
