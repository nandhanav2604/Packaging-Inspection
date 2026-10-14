"""
Evaluates best_yolov8s.pt on data it never saw during training:
  1. Kaggle folders (have known damaged/intact ground truth) -> real accuracy numbers
  2. Your own photos (no ground truth) -> visual sanity check only
  
"""
import os
import csv
from ultralytics import YOLO

# ============================== CONFIG ==============================
MODEL_PATH = "best_yolov8s.pt"          # path to your downloaded model weights

# List every root folder that contains "damaged"/"intact" subfolders
# anywhere in their structure (side/top nesting is fine - it just needs
# "damaged" or "intact" to appear somewhere in the path).
KAGGLE_FOLDERS = [
   r"external_data/kaggle_damaged-and-intact-packages",
   r"external_data/kaggle_package",
]

OWN_PHOTOS_FOLDER = r"external_data/Package_images"   # folder with your 1,2,3...n images

DAMAGE_CONF_THRESHOLD = 0.05    # lower threshold for damage, as discussed (favor catching real damage)
PACKAGE_CONF_THRESHOLD = 0.25

OUTPUT_DIR = "evaluation_output_v2"
# ======================================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_labeled_images(root_folders):
    """Walk each root folder recursively; ground truth is determined by
    an EXACT folder-name match ('damaged' or 'intact' as a whole path
    component), not a substring anywhere in the path - this avoids the
    dataset's own root folder name (which may itself contain the word
    "damaged") from mislabeling every image underneath it."""
    items = []
    for root in root_folders:
        for dirpath, _, filenames in os.walk(root):
            # relative path from this dataset's root, split into components
            rel = os.path.relpath(dirpath, root)
            parts = [p.lower() for p in rel.split(os.sep)]

            gt = None
            if "damaged" in parts or "damage" in parts:
                gt = "damaged"
            elif "intact" in parts:
                gt = "intact"
            else:
                continue  # this subfolder isn't a damaged/intact folder itself, skip

            for fname in filenames:
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                    continue
                full_path = os.path.join(dirpath, fname)
                items.append((full_path, gt, root))
    return items


def predict_condition(model, image_path, damage_id, package_id):
    results = model.predict(image_path, conf=0.05, verbose=False)
    r = results[0]
    damage_conf = 0.0
    package_conf = 0.0
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == damage_id and conf > damage_conf:
            damage_conf = conf
        if cls_id == package_id and conf > package_conf:
            package_conf = conf

    predicted = "damaged" if damage_conf >= DAMAGE_CONF_THRESHOLD else "intact"
    package_found = package_conf >= PACKAGE_CONF_THRESHOLD
    return predicted, damage_conf, package_conf, package_found


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    # find class ids by name (robust to ordering)
    name_to_id = {v: k for k, v in model.names.items()}
    damage_id = name_to_id.get("damage")
    package_id = name_to_id.get("package")
    if damage_id is None or package_id is None:
        print(f"Could not find 'damage'/'package' classes in model. Found: {model.names}")
        return

    # ---------------- Kaggle evaluation ----------------
    print("Finding labeled Kaggle images...")
    items = find_labeled_images(KAGGLE_FOLDERS)
    print(f"Found {len(items)} labeled images across {len(KAGGLE_FOLDERS)} folder(s).\n")

    rows = []
    tp = fp = tn = fn = 0
    no_package_count = 0

    for i, (img_path, gt, source_root) in enumerate(items, start=1):
        predicted, damage_conf, package_conf, package_found = predict_condition(
            model, img_path, damage_id, package_id
        )
        correct = (predicted == gt)

        if gt == "damaged" and predicted == "damaged":
            tp += 1
        elif gt == "damaged" and predicted == "intact":
            fn += 1
        elif gt == "intact" and predicted == "damaged":
            fp += 1
        elif gt == "intact" and predicted == "intact":
            tn += 1

        if not package_found:
            no_package_count += 1

        rows.append([img_path, source_root, gt, predicted, round(damage_conf, 3),
                     round(package_conf, 3), package_found, correct])

        if i % 100 == 0:
            print(f"  ...processed {i}/{len(items)}")

    # write CSV
    csv_path = os.path.join(OUTPUT_DIR, "kaggle_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "source_folder", "ground_truth", "predicted",
                          "damage_confidence", "package_confidence", "package_found", "correct"])
        writer.writerows(rows)

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n" + "=" * 60)
    print("KAGGLE EVALUATION RESULTS (data the model never trained on)")
    print("=" * 60)
    print(f"Total images evaluated: {total}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision (damaged): {precision:.3f}")
    print(f"Recall (damaged):    {recall:.3f}")
    print(f"F1 (damaged):        {f1:.3f}")
    print(f"Confusion -> TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}")
    print(f"Images where no package was even detected: {no_package_count}")
    print(f"Full per-image results: {csv_path}")

    # ---------------- Own photos (no ground truth, visual check only) ----------------
    if os.path.isdir(OWN_PHOTOS_FOLDER):
        print("\nRunning on your own photos (no ground truth - visual check only)...")
        own_out_dir = os.path.join(OUTPUT_DIR, "own_photos_annotated")
        os.makedirs(own_out_dir, exist_ok=True)

        own_files = [f for f in os.listdir(OWN_PHOTOS_FOLDER)
                     if os.path.splitext(f)[1].lower() in IMG_EXTS]

        for fname in own_files:
            img_path = os.path.join(OWN_PHOTOS_FOLDER, fname)
            results = model.predict(img_path, conf=0.05, verbose=False)
            results[0].save(filename=os.path.join(own_out_dir, fname))

        print(f"Saved {len(own_files)} annotated images to: {own_out_dir}")
        print("Open that folder and eyeball them - does the model catch obvious damage in your own photos?")
    else:
        print(f"\nOWN_PHOTOS_FOLDER not found at: {OWN_PHOTOS_FOLDER} (skipped)")

    print("\nDone.")


if __name__ == "__main__":
    main()