"""
Download the dev-200 ImageNet subset (labels 0-199, no per-class cap) from the ungated
256x256 mirror evanarlian/imagenet_1k_resized_256 and write <out>/images/<stem>.png plus
<out>/vae-sd/dataset.json. Then encode with preprocessing/encode_from_images.py.

  python preprocessing/download_dev_images.py --out data/in256_dev200 --num-classes 200
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="evanarlian/imagenet_1k_resized_256")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-images", type=int, default=0, help="0 => use per-class/num-classes caps")
    ap.add_argument("--num-classes", type=int, default=0, help="0 => all 1000 classes")
    ap.add_argument("--per-class", type=int, default=0, help="0 => no per-class cap")
    args = ap.parse_args()

    from datasets import load_dataset
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    ds = load_dataset(args.repo, split=args.split, streaming=True)

    labels, per_class, idx = {}, {}, 0
    for ex in ds:
        lab = int(ex["label"])
        if args.num_classes and lab >= args.num_classes:
            continue  # Do not assume upstream examples are class ordered.
        if args.per_class and per_class.get(lab, 0) >= args.per_class:
            continue
        img = ex["image"].convert("RGB")
        if img.size != (256, 256):
            img = img.resize((256, 256))
        name = f"img{idx:08d}.png"
        img.save(os.path.join(args.out, "images", name))
        labels[f"img{idx:08d}.npy"] = lab     # key matches the future .npy
        per_class[lab] = per_class.get(lab, 0) + 1
        idx += 1
        if args.num_images and idx >= args.num_images:
            break
        if idx % 2000 == 0:
            print(f"  saved {idx} images ({len(per_class)} classes)", flush=True)

    os.makedirs(os.path.join(args.out, "vae-sd"), exist_ok=True)
    with open(os.path.join(args.out, "vae-sd", "dataset.json"), "w") as f:
        json.dump({"labels": [[k, v] for k, v in sorted(labels.items())]}, f)
    print(f"DONE: {idx} images across {len(per_class)} classes -> {args.out}/images")


if __name__ == "__main__":
    main()
