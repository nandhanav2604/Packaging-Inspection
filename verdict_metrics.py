"""
SYSTEM-LEVEL EVALUATION: runs the full readiness engine (not just the raw
detector) over the labelled Kaggle images and reports what actually matters
for shipping:

  For DAMAGED packages:
      caught      -> Repackage Required      (good)
      escalated   -> Manual Inspection       (safe - a human looks)
      MISSED      -> Ready for Shipment      (the costly error)
  For INTACT packages:
      correct     -> Ready for Shipment
      false alarm -> Repackage Required
      extra review-> Manual Inspection       (cost: human time, not a failure)

It reuses KAGGLE_FOLDERS and find_labeled_images from evaluate_model.py, so
there is nothing to configure except the model path below (or pass it as an
argument).

RUN (from the folder with evaluate_model.py and readiness_engine.py):
    python verdict_metrics.py                       # uses best_yolov8s.pt
    python verdict_metrics.py best_v1.pt            # compare another model

Takes a few minutes on CPU (barcode check is skipped - it does not affect
the verdict). Writes verdict_results_<model>.csv.
"""
import csv
import os
import sys
from collections import Counter, defaultdict

from evaluate_model import KAGGLE_FOLDERS, find_labeled_images
from readiness_engine import assess_package

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "best_yolov8s.pt"

READY, REPACK, MANUAL = "Ready for Shipment", "Repackage Required", "Manual Inspection"


def pct(n, d):
    return f"{100 * n / d:5.1f}%" if d else "  n/a"


def summarise(label, rows):
    dam = [r for r in rows if r["ground_truth"] == "damaged"]
    intact = [r for r in rows if r["ground_truth"] == "intact"]
    dc = Counter(r["verdict"] for r in dam)
    ic = Counter(r["verdict"] for r in intact)

    print(f"\n{label}   ({len(rows)} images: {len(dam)} damaged, {len(intact)} intact)")
    print("-" * 72)
    print(f"  DAMAGED packages (n={len(dam)})")
    print(f"    caught  (Repackage Required) : {dc[REPACK]:>4}  {pct(dc[REPACK], len(dam))}")
    print(f"    escalated (Manual Inspection): {dc[MANUAL]:>4}  {pct(dc[MANUAL], len(dam))}")
    print(f"    MISSED  (Ready for Shipment) : {dc[READY]:>4}  {pct(dc[READY], len(dam))}   <- costly error")
    print(f"    => damaged NOT shipped       : {dc[REPACK] + dc[MANUAL]:>4}  "
          f"{pct(dc[REPACK] + dc[MANUAL], len(dam))}")
    print(f"  INTACT packages (n={len(intact)})")
    print(f"    correct (Ready for Shipment) : {ic[READY]:>4}  {pct(ic[READY], len(intact))}")
    print(f"    false alarm (Repackage)      : {ic[REPACK]:>4}  {pct(ic[REPACK], len(intact))}")
    print(f"    extra review (Manual Insp.)  : {ic[MANUAL]:>4}  {pct(ic[MANUAL], len(intact))}")
    return {
        "label": label, "n": len(rows),
        "dam_missed": pct(dc[READY], len(dam)).strip(),
        "dam_not_shipped": pct(dc[REPACK] + dc[MANUAL], len(dam)).strip(),
        "intact_false_alarm": pct(ic[REPACK], len(intact)).strip(),
        "intact_manual": pct(ic[MANUAL], len(intact)).strip(),
    }


def main():
    items = find_labeled_images(KAGGLE_FOLDERS)
    print(f"Model: {MODEL_PATH}")
    print(f"Running full readiness engine on {len(items)} labelled images...\n")

    rows = []
    for i, (path, gt, root) in enumerate(items, start=1):
        res = assess_package(path, MODEL_PATH, check_barcode=False)
        rows.append({
            "image_path": path, "source_folder": root, "ground_truth": gt,
            "verdict": res["verdict"],
            "package_conf": res["package_confidence"],
            "damage_conf": res["damage_confidence"],
        })
        if i % 100 == 0:
            print(f"  ...{i}/{len(items)}")

    out_csv = f"verdict_results_{os.path.splitext(os.path.basename(MODEL_PATH))[0]}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source_folder"]].append(r)

    print("\n" + "=" * 72)
    print(f"SYSTEM-LEVEL VERDICT RESULTS  -  {MODEL_PATH}")
    print("=" * 72)
    summaries = []
    for src, rs in by_source.items():
        src_name = os.path.basename(os.path.normpath(src))
        summaries.append(summarise("SOURCE: " + src_name, rs))
    summaries.append(summarise("OVERALL", rows))

    print("\nCOMPACT TABLE (paste this back)")
    print(f"{'set':<45}{'n':>5}  {'dmg MISSED':>10}  {'dmg not shipped':>15}  "
          f"{'intact false alarm':>18}  {'intact->manual':>14}")
    for s in summaries:
        print(f"{s['label'][:44]:<45}{s['n']:>5}  {s['dam_missed']:>10}  {s['dam_not_shipped']:>15}  "
              f"{s['intact_false_alarm']:>18}  {s['intact_manual']:>14}")
    print(f"\nPer-image verdicts saved to: {out_csv}")


if __name__ == "__main__":
    main()