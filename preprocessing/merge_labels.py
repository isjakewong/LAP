"""Merge encode_in1k.py per-task label maps into <out>/vae-sd/dataset.json."""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
args = ap.parse_args()

merged = {}
for f in glob.glob(os.path.join(args.out, "labels", "labels_*.json")):
    merged.update(json.load(open(f)))
ds = {"labels": [[k, v] for k, v in sorted(merged.items())]}
json.dump(ds, open(os.path.join(args.out, "vae-sd", "dataset.json"), "w"))
print(f"merged {len(ds['labels'])} labels -> {args.out}/vae-sd/dataset.json")
