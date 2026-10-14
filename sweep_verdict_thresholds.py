"""
Picks the DAMAGE threshold for the readiness engine from data, using the
per-image confidences saved by verdict_metrics.py (verdict_results_*.csv).
No model is run here - it only re-applies the engine's decision rules to the
saved confidences, so it takes a second.

PREREQUISITE: re-run verdict_metrics.py AFTER the engine change that lets the
detector return weak boxes (conf=0.05); older CSVs only contain damage
confidences >= 0.25 and cannot be used for this.

RUN:
    python sweep_verdict_thresholds.py verdict_results_best_yolov8s.csv

HONEST EVALUATION: choosing a threshold on the same images you report on makes
the result look better than it really is. So the images are split 50/50 with
a fixed seed: the threshold is CHOSEN on half A and the number you should
quote is the one measured on half B.
"""
import csv
import random
import sys

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "verdict_results_best_yolov8s.csv"

# the engine's other rules (unchanged)
PACKAGE_NOT_FOUND = 0.25
PACKAGE_LOW_CONF = 0.60
THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]

# Selection rule: among thresholds on half A, take the one that stops the most
# damaged packages while wrongly sending no more than this share of INTACT
# packages to "Repackage Required".
MAX_INTACT_FALSE_ALARM = 0.20

READY, REPACK, MANUAL = "Ready", "Repackage", "Manual"


def verdict(pkg, dmg, thr):
    if pkg < PACKAGE_NOT_FOUND:
        return MANUAL
    if dmg >= thr:
        return REPACK
    if pkg < PACKAGE_LOW_CONF:
        return MANUAL
    return READY


def measure(rows, thr):
    dam = [r for r in rows if r["gt"] == "damaged"]
    intact = [r for r in rows if r["gt"] == "intact"]
    dv = [verdict(r["pkg"], r["dmg"], thr) for r in dam]
    iv = [verdict(r["pkg"], r["dmg"], thr) for r in intact]
    n_d, n_i = max(1, len(dam)), max(1, len(intact))
    return {
        "missed": dv.count(READY) / n_d,
        "stopped": (dv.count(REPACK) + dv.count(MANUAL)) / n_d,
        "false_alarm": iv.count(REPACK) / n_i,
        "intact_manual": iv.count(MANUAL) / n_i,
        "n": len(rows),
    }


def show(title, rows):
    print(f"\n{title}  ({len(rows)} images)")
    print(f"{'damage thr':>10}  {'dmg MISSED':>10}  {'dmg stopped':>11}  "
          f"{'intact false alarm':>18}  {'intact->manual':>14}")
    for t in THRESHOLDS:
        m = measure(rows, t)
        print(f"{t:>10.2f}  {100*m['missed']:>9.1f}%  {100*m['stopped']:>10.1f}%  "
              f"{100*m['false_alarm']:>17.1f}%  {100*m['intact_manual']:>13.1f}%")


def main():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows.append({"gt": r["ground_truth"], "pkg": float(r["package_conf"]),
                         "dmg": float(r["damage_conf"]), "src": r["source_folder"]})

    low = [r["dmg"] for r in rows if 0 < r["dmg"] < 0.25]
    print(f"Loaded {len(rows)} rows. Images with a damage confidence between 0 and "
          f"0.25: {len(low)}")
    if not low:
        print("\nWARNING: no damage confidences below 0.25 found. This CSV was "
              "produced with the old engine (detector dropped everything under "
              "0.25). Re-run verdict_metrics.py with the updated "
              "readiness_engine.py first, then run this again.")
        return

    rnd = random.Random(42)
    rnd.shuffle(rows)
    half = len(rows) // 2
    a, b = rows[:half], rows[half:]

    show("ALL IMAGES", rows)
    show("HALF A (used to choose the threshold)", a)

    best, best_stopped = None, -1
    for t in THRESHOLDS:
        m = measure(a, t)
        if m["false_alarm"] <= MAX_INTACT_FALSE_ALARM and m["stopped"] > best_stopped:
            best, best_stopped = t, m["stopped"]

    print("\n" + "=" * 70)
    if best is None:
        print("No threshold satisfies the false-alarm limit "
              f"({100*MAX_INTACT_FALSE_ALARM:.0f}%). Raise MAX_INTACT_FALSE_ALARM.")
        return
    mb, ma = measure(b, best), measure(a, best)
    print(f"CHOSEN damage threshold (on half A, intact false alarm <= "
          f"{100*MAX_INTACT_FALSE_ALARM:.0f}%): {best:.2f}")
    print(f"  half A : damaged missed {100*ma['missed']:.1f}%, stopped "
          f"{100*ma['stopped']:.1f}%, intact false alarm {100*ma['false_alarm']:.1f}%")
    print(f"  half B : damaged missed {100*mb['missed']:.1f}%, stopped "
          f"{100*mb['stopped']:.1f}%, intact false alarm {100*mb['false_alarm']:.1f}%"
          f", intact->manual {100*mb['intact_manual']:.1f}%   <-- quote THIS one")
    m20 = measure(b, 0.20)
    print(f"  (for comparison, the current 0.20 on half B: missed "
          f"{100*m20['missed']:.1f}%, false alarm {100*m20['false_alarm']:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()