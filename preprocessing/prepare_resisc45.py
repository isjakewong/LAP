"""Build the NWPU-RESISC45 generative training set from the HF parquet copy timm/resisc45.

All three splits (train 18,900 + validation 6,300 + test 6,300 = 31,500 images, 256x256,
45 classes, 700 per class) are merged; no held-out split is needed for class-conditional
generation.

Output (<out>):
  images/rs_XXXXXXX.png      256x256 RGB PNG (bicubic-resized only if the source is not 256x256)
  vae-sd/dataset.json        {"labels": [["rs_XXXXXXX.npy", label], ...]}
  rs_meta.csv, rs_class_counts.csv, rs_classes.json

Stems are assigned after sorting all records by image_id (airplane_001 ... wetland_700),
so the order is class-grouped and independent of the HF split shuffle. Labels index the
alphabetically sorted class names and are cross-checked against the parquet labels and
the image_id prefix. Existing valid PNGs are skipped.
"""
import argparse, csv, io, json, os, time
from collections import Counter
from multiprocessing import Pool

import pyarrow.parquet as pq
from PIL import Image

SPLITS = ["train", "validation", "test"]
EXPECTED_N = 31500
EXPECTED_NC = 45
RES = 256


def load_records(snap):
    """Return (class_names, records). records: dict(split,row,image_id,src_file,label,bytes)."""
    recs, names = [], None
    for split in SPLITS:
        t = pq.read_table(os.path.join(snap, "data", f"{split}-00000-of-00001.parquet"))
        meta = json.loads(t.schema.metadata[b"huggingface"].decode())
        n = meta["info"]["features"]["label"]["names"]
        if names is None:
            names = list(n)
        assert names == list(n), f"class names differ across splits ({split})"
        imgs = t.column("image").to_pylist()
        labels = t.column("label").to_pylist()
        ids = t.column("image_id").to_pylist()
        for i, (im, lab, iid) in enumerate(zip(imgs, labels, ids)):
            recs.append(dict(split=split, row=i, image_id=iid, src_file=im["path"] or "", label=int(lab), bytes=im["bytes"]))
        print(f"[prepare] {split}: {len(imgs)} rows", flush=True)
    return names, recs


def write_one(args):
    stem, data, out_png = args
    if os.path.exists(out_png):
        try:
            with Image.open(out_png) as im:
                if im.size == (RES, RES) and im.mode == "RGB":
                    return stem, None, None, False, True
        except Exception:
            pass  # rewrite a truncated/corrupt file
    im = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = im.size
    resized = (w, h) != (RES, RES)
    if resized:
        im = im.resize((RES, RES), Image.BICUBIC)
    tmp = out_png + ".tmp"
    im.save(tmp, format="PNG", compress_level=6)
    os.replace(tmp, out_png)
    return stem, w, h, resized, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/resisc45_256")
    ap.add_argument('--snapshot', required=True, help='Local download of timm/resisc45 (contains data/*.parquet)')
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefix", default="rs_")
    args = ap.parse_args()

    snap = args.snapshot
    print("[prepare] snapshot:", snap, flush=True)
    names, recs = load_records(snap)
    assert len(names) == EXPECTED_NC, len(names)
    assert names == sorted(names), "class names are not alphabetical"
    assert len(recs) == EXPECTED_N, f"expected {EXPECTED_N} records, got {len(recs)}"
    name2id = {n: i for i, n in enumerate(names)}

    # deterministic global order: by original NWPU file id (class name + 3-digit index)
    recs.sort(key=lambda r: (r["image_id"], r["split"], r["row"]))
    ids = [r["image_id"] for r in recs]
    assert len(set(ids)) == EXPECTED_N, "duplicate image_id across splits"
    for r in recs:
        cls = r["image_id"].rsplit("_", 1)[0]
        assert cls in name2id, r["image_id"]
        assert name2id[cls] == r["label"], (r["image_id"], r["label"], name2id[cls])
        r["label_name"] = cls

    img_dir = os.path.join(args.out, "images")
    lat_dir = os.path.join(args.out, "vae-sd")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lat_dir, exist_ok=True)
    stems = [f"{args.prefix}{i:07d}" for i in range(len(recs))]

    # PNGs
    jobs = [(s, r["bytes"], os.path.join(img_dir, s + ".png")) for s, r in zip(stems, recs)]
    t0, done, skipped = time.time(), 0, 0
    src_size = {}
    with Pool(args.workers) as pool:
        for stem, w, h, resized, existed in pool.imap(write_one, jobs, chunksize=64):
            src_size[stem] = (w, h, resized, existed)
            done += 1
            skipped += int(existed)
            if done % 2000 == 0:
                print(f"[prepare] {done}/{len(jobs)} PNGs ({done / (time.time() - t0):.0f}/s, {skipped} pre-existing)", flush=True)
    print(f"[prepare] wrote {done - skipped} PNGs, {skipped} pre-existing, {time.time() - t0:.0f}s", flush=True)

    # metadata
    with open(os.path.join(args.out, "rs_meta.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["stem", "split", "row_in_split", "image_id", "src_file", "label_name", "label", "src_w", "src_h", "resized"])
        for s, r in zip(stems, recs):
            w, h, resized, existed = src_size[s]
            wr.writerow([s, r["split"], r["row"], r["image_id"], r["src_file"], r["label_name"], r["label"],
                         "" if w is None else w, "" if h is None else h, "" if existed else int(resized)])
    counts = Counter(r["label"] for r in recs)
    with open(os.path.join(args.out, "rs_class_counts.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["label", "label_name", "count"])
        for i, n in enumerate(names):
            wr.writerow([i, n, counts.get(i, 0)])
    with open(os.path.join(args.out, "rs_classes.json"), "w") as f:
        json.dump(names, f, indent=1)
    # dataset.json in the REPA layout (stems listed in sorted order)
    with open(os.path.join(lat_dir, "dataset.json"), "w") as f:
        json.dump({"labels": [[s + ".npy", r["label"]] for s, r in zip(stems, recs)]}, f)

    n_resized = sum(1 for v in src_size.values() if v[2])
    print(f"[prepare] classes ({len(names)}): {names}")
    print(f"[prepare] per-class counts: min={min(counts.values())} max={max(counts.values())}; resized={n_resized}")
    print(f"[prepare] PNGs on disk: {len([f for f in os.listdir(img_dir) if f.endswith('.png')])}")
    print("[prepare] DONE")


if __name__ == "__main__":
    main()
