"""
Merges all 8 Roboflow dataset folders into ONE final YOLO detection dataset
with just 2 classes: "package" and "damage".

WHAT IT DOES:
    1. Reads each dataset's data.yaml to get its original class names.
    2. Matches each dataset's class set against the known mapping rules
       below (based on the visual audit already done) and remaps every
       box's class to either "package" or "damage".
    3. Converts any OBB/polygon-format labels into plain axis-aligned boxes.
    4. For the two "condition-only" datasets (Damaged package v1i/v2i,
       classes Damaged/Destroyed/Intact), every box becomes "package" —
       the original condition word is saved separately in
       condition_metadata.csv instead of becoming a damage box.
    5. Removes near-duplicate images (perceptual hash) so the same or a
       near-identical photo can't end up in both train and val/test.
    6. Does a fresh random 80/10/10 train/val/test split across everything.
    7. Writes the final merged dataset + a ready-to-use data.yaml.

REQUIREMENTS (one-time):
    pip install pillow pyyaml imagehash

HOW TO RUN:
    Put this file in the SAME parent folder that contains your 8 dataset
    folders (same place as count_datasets.py / visualize_boxes.py), then:
        python merge_datasets.py

    Output appears in a new folder: <parent>/merged_dataset/
        merged_dataset/
            images/train, images/val, images/test
            labels/train, labels/val, labels/test
            data.yaml
            condition_metadata.csv

    If your dataset folders live somewhere else, pass the path:
        python merge_datasets.py "C:\\path\\to\\parent_folder"
"""

import os
import sys
import csv
import random
import shutil

random.seed(42)  # fixed seed so the split is reproducible if you rerun

try:
    from PIL import Image
    import imagehash
except ImportError:
    print("Missing packages. Run:  pip install pillow pyyaml imagehash")
    sys.exit(1)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FINAL_CLASSES = ["package", "damage"]  # class 0, class 1
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test

# ---------------------------------------------------------------------------
# Mapping rules, based on the visual audit. Each rule is matched by the
# EXACT SET of original class names found in that dataset's data.yaml
# (case-insensitive). "map" gives old_class_name -> "package" or "damage".
# "condition_only": True means don't produce any "damage" box for this
# dataset even if a rule mapping says so elsewhere -- everything becomes
# "package", and the original class name is logged as metadata instead.
# ---------------------------------------------------------------------------
RULES = [
    {
        "classes": {"damaged package", "undamaged"},
        "map": {"damaged package": "damage", "undamaged": "package"},
        "condition_only": False,
    },
    {
        "classes": {"damaged", "destroyed", "intact"},
        "map": {"damaged": "package", "destroyed": "package", "intact": "package"},
        "condition_only": True,
    },
    {
        "classes": {"dent", "hole", "package"},
        "map": {"dent": "damage", "hole": "damage", "package": "package"},
        "condition_only": False,
    },
    {
        "classes": {"damagedpackage", "damagedpart"},
        "map": {"damagedpackage": "package", "damagedpart": "damage"},
        "condition_only": False,
    },
    {
        "classes": {"damagedpart", "normalbox"},
        "map": {"damagedpart": "damage", "normalbox": "package"},
        "condition_only": False,
    },
    {
        "classes": {"damaged", "package"},
        "map": {"damaged": "damage", "package": "package"},
        "condition_only": False,
    },
]


def read_classes(data_yaml_path):
    try:
        import yaml
        with open(data_yaml_path, "r", encoding="utf-8", errors="ignore") as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        if isinstance(names, list):
            return names
    except Exception:
        pass
    # fallback manual parse
    names = []
    in_block = False
    with open(data_yaml_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("names:"):
                after = s.split("names:", 1)[1].strip()
                if after.startswith("["):
                    inline = after.strip("[]")
                    return [n.strip().strip("'\"") for n in inline.split(",") if n.strip()]
                in_block = True
                continue
            if in_block:
                if s.startswith("-"):
                    names.append(s.lstrip("-").strip().strip("'\""))
                elif ":" in s:
                    break
    return names


def match_rule(class_names):
    lower_set = {c.strip().lower() for c in class_names}
    for rule in RULES:
        if rule["classes"] == lower_set:
            return rule
    return None


def parse_label_line_to_aabb(line):
    """Returns (class_id, xc, yc, w, h) all normalized 0-1, regardless of
    whether the input was a standard box or an OBB/polygon."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    class_id = int(float(parts[0]))
    nums = [float(p) for p in parts[1:]]

    if len(nums) == 4:
        xc, yc, w, h = nums
        return class_id, xc, yc, w, h

    # polygon / OBB: pairs of x,y -> compute axis-aligned bounding box
    xs = nums[0::2]
    ys = nums[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return class_id, xc, yc, w, h


def gather_dataset_records(dataset_dir, dataset_index):
    """Returns a list of dicts, one per image, with converted+remapped
    label lines ready to write, plus metadata for condition-only datasets."""
    class_names = read_classes(os.path.join(dataset_dir, "data.yaml"))
    rule = match_rule(class_names)
    ds_name = os.path.basename(dataset_dir)

    if rule is None:
        print(f"⚠️  {ds_name}: class set {class_names} did not match any known rule — SKIPPING this dataset.")
        return []

    # build old_class_id -> (new_class_name, original_name_for_metadata)
    id_to_new = {}
    for idx, name in enumerate(class_names):
        new_name = rule["map"].get(name.strip().lower())
        id_to_new[idx] = (new_name, name)

    records = []
    for split in ("train", "valid", "test"):
        images_dir = os.path.join(dataset_dir, split, "images")
        labels_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.isdir(images_dir):
            continue
        for fname in os.listdir(images_dir):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in IMG_EXTS:
                continue
            img_path = os.path.join(images_dir, fname)
            label_path = os.path.join(labels_dir, stem + ".txt") if os.path.isdir(labels_dir) else None

            out_lines = []
            condition_tag = None

            if label_path and os.path.isfile(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        parsed = parse_label_line_to_aabb(line)
                        if parsed is None:
                            continue
                        old_id, xc, yc, w, h = parsed
                        if old_id not in id_to_new:
                            continue
                        new_name, orig_name = id_to_new[old_id]
                        if new_name is None:
                            continue
                        if w < 0.002 or h < 0.002:
                            continue  # drop degenerate boxes

                        if rule["condition_only"]:
                            # all boxes become "package"; remember condition
                            condition_tag = orig_name
                            new_class_id = FINAL_CLASSES.index("package")
                        else:
                            new_class_id = FINAL_CLASSES.index(new_name)

                        xc = min(max(xc, 0.0), 1.0)
                        yc = min(max(yc, 0.0), 1.0)
                        w = min(max(w, 0.0), 1.0)
                        h = min(max(h, 0.0), 1.0)
                        out_lines.append(f"{new_class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            records.append({
                "img_path": img_path,
                "ext": ext,
                "label_lines": out_lines,
                "dataset_index": dataset_index,
                "dataset_name": ds_name,
                "condition_tag": condition_tag,
            })

    return records


def main():
    parent = sys.argv[1] if len(sys.argv) > 1 else "."
    parent = os.path.abspath(parent)

    dataset_dirs = []
    for entry in sorted(os.listdir(parent)):
        full = os.path.join(parent, entry)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "data.yaml")):
            dataset_dirs.append(full)

    if not dataset_dirs:
        print(f"No dataset folders with data.yaml found in: {parent}")
        return

    print(f"Found {len(dataset_dirs)} dataset folder(s). Reading and converting labels...\n")

    all_records = []
    for i, ds in enumerate(dataset_dirs, start=1):
        recs = gather_dataset_records(ds, i)
        print(f"  {os.path.basename(ds)}: {len(recs)} images")
        all_records.extend(recs)

    print(f"\nTotal images before dedup: {len(all_records)}")

    # ---- Dedup by perceptual hash ----
    print("Hashing images to remove duplicates/near-duplicates (this may take a few minutes)...")
    seen_hashes = set()
    deduped = []
    for rec in all_records:
        try:
            with Image.open(rec["img_path"]) as img:
                h = imagehash.average_hash(img)
        except Exception as e:
            print(f"   could not open {rec['img_path']}: {e}")
            continue
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        deduped.append(rec)

    print(f"Total images after dedup: {len(deduped)} (removed {len(all_records) - len(deduped)} duplicates)\n")

    # ---- Shuffle and split ----
    random.shuffle(deduped)
    n = len(deduped)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    split_assignment = (["train"] * n_train) + (["val"] * n_val) + (["test"] * (n - n_train - n_val))

    # ---- Write output ----
    out_root = os.path.join(parent, "merged_dataset")
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", split), exist_ok=True)

    condition_rows = []
    class_counts = {"package": 0, "damage": 0}
    split_counts = {"train": 0, "val": 0, "test": 0}

    for rec, split in zip(deduped, split_assignment):
        new_stem = f"ds{rec['dataset_index']:02d}_{os.path.splitext(os.path.basename(rec['img_path']))[0]}"
        new_img_name = new_stem + rec["ext"]
        dst_img = os.path.join(out_root, "images", split, new_img_name)
        dst_label = os.path.join(out_root, "labels", split, new_stem + ".txt")

        shutil.copy2(rec["img_path"], dst_img)
        with open(dst_label, "w") as f:
            f.write("\n".join(rec["label_lines"]))

        for line in rec["label_lines"]:
            cid = int(line.split()[0])
            class_counts[FINAL_CLASSES[cid]] += 1
        split_counts[split] += 1

        if rec["condition_tag"]:
            condition_rows.append([new_img_name, rec["dataset_name"], rec["condition_tag"]])

    # data.yaml
    with open(os.path.join(out_root, "data.yaml"), "w") as f:
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n")
        f.write(f"nc: {len(FINAL_CLASSES)}\n")
        f.write(f"names: {FINAL_CLASSES}\n")

    # condition metadata
    if condition_rows:
        with open(os.path.join(out_root, "condition_metadata.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "source_dataset", "original_condition_label"])
            writer.writerows(condition_rows)

    print("=" * 70)
    print(f"DONE. Merged dataset written to: {out_root}")
    print(f"Split counts: {split_counts}")
    print(f"Final box counts -> package: {class_counts['package']}, damage: {class_counts['damage']}")
    if condition_rows:
        print(f"Condition metadata saved for {len(condition_rows)} images (from the condition-only datasets) -> condition_metadata.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
