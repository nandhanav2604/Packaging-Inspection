"""
DEPLOYMENT EVIDENCE: model size, CPU latency (PyTorch vs ONNX) and
end-to-end pipeline latency (detector + barcode check) on THIS machine.
Run it on your i3 laptop - that is your worst-case hardware, which is a
good deployment argument.

ONE-TIME INSTALL (ONNX export + runtime):
    pip install onnx onnxruntime

RUN (from the folder with merged_dataset/ and the weights):
    python benchmark_deploy.py                      # uses best_yolov8s.pt
    python benchmark_deploy.py best_v1.pt           # or another model

Close other heavy apps first (browser, Colab tabs) so timings are stable.
Takes ~3-6 minutes. Prints a table to paste back and writes
deploy_benchmark.json.
"""
import glob
import json
import os
import statistics
import sys
import time

from ultralytics import YOLO

MODEL_PT = sys.argv[1] if len(sys.argv) > 1 else "best_yolov8s.pt"
TEST_IMG_DIR = os.path.join("merged_dataset", "images", "test")
N_IMAGES = 25      # timed images
N_WARMUP = 3       # untimed warm-up runs (first call is always slow)
IMGSZ = 640


def mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def stats(xs):
    xs = sorted(xs)
    p95 = xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]
    return statistics.mean(xs), statistics.median(xs), p95


def bench_detector(weights, images):
    model = YOLO(weights, task="detect")
    for p in images[:N_WARMUP]:
        model.predict(p, imgsz=IMGSZ, device="cpu", verbose=False)
    wall, inf = [], []
    for p in images:
        t0 = time.perf_counter()
        r = model.predict(p, imgsz=IMGSZ, device="cpu", verbose=False)[0]
        wall.append((time.perf_counter() - t0) * 1000)
        inf.append(r.speed["inference"])
    return wall, inf


def main():
    imgs = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.*")))[:N_IMAGES + N_WARMUP]
    if len(imgs) < N_WARMUP + 5:
        print(f"Could not find enough test images in {TEST_IMG_DIR}. "
              "Run this from the folder that contains merged_dataset/.")
        sys.exit(1)
    timed = imgs[N_WARMUP:]
    warm_first = imgs[:N_WARMUP] + timed  # warm-up images first, then timed ones

    results = {"model": MODEL_PT, "images_timed": len(timed), "imgsz": IMGSZ}

    # ---- PyTorch ----
    print(f"Benchmarking PyTorch ({MODEL_PT}) on CPU...")
    wall, inf = bench_detector(MODEL_PT, warm_first)
    wall, inf = wall[N_WARMUP:], inf[N_WARMUP:]
    results["pt"] = {"size_mb": round(mb(MODEL_PT), 1),
                     "wall_ms_mean_median_p95": [round(x, 1) for x in stats(wall)],
                     "inference_only_ms_mean": round(statistics.mean(inf), 1)}

    # ---- ONNX ----
    onnx_path = os.path.splitext(MODEL_PT)[0] + ".onnx"
    if not os.path.exists(onnx_path):
        print("Exporting to ONNX...")
        YOLO(MODEL_PT).export(format="onnx", imgsz=IMGSZ, simplify=True)
    print(f"Benchmarking ONNX ({onnx_path}) on CPU...")
    wall_o, inf_o = bench_detector(onnx_path, warm_first)
    wall_o, inf_o = wall_o[N_WARMUP:], inf_o[N_WARMUP:]
    results["onnx"] = {"size_mb": round(mb(onnx_path), 1),
                       "wall_ms_mean_median_p95": [round(x, 1) for x in stats(wall_o)],
                       "inference_only_ms_mean": round(statistics.mean(inf_o), 1)}

    # ---- full pipeline (detector + barcode check) ----
    print("Benchmarking full pipeline (detector + barcode check)...")
    from readiness_engine import assess_package
    for p in imgs[:N_WARMUP]:
        assess_package(p, MODEL_PT)
    full = []
    for p in timed:
        t0 = time.perf_counter()
        assess_package(p, MODEL_PT)
        full.append((time.perf_counter() - t0) * 1000)
    results["full_pipeline_ms_mean_median_p95"] = [round(x, 1) for x in stats(full)]

    json.dump(results, open("deploy_benchmark.json", "w"), indent=2)

    pm, pmed, pp95 = results["pt"]["wall_ms_mean_median_p95"]
    om, omed, op95 = results["onnx"]["wall_ms_mean_median_p95"]
    fm, fmed, fp95 = results["full_pipeline_ms_mean_median_p95"]
    print("\n" + "=" * 66)
    print(f"DEPLOYMENT BENCHMARK  ({len(timed)} images, {IMGSZ}px, CPU, batch 1)")
    print("=" * 66)
    print(f"{'':<28}{'size MB':>8}{'mean ms':>10}{'median':>9}{'p95':>8}")
    print(f"{'PyTorch (.pt)':<28}{results['pt']['size_mb']:>8}{pm:>10}{pmed:>9}{pp95:>8}")
    print(f"{'ONNX (.onnx)':<28}{results['onnx']['size_mb']:>8}{om:>10}{omed:>9}{op95:>8}")
    print(f"{'Full pipeline (.pt + barcode)':<28}{'':>8}{fm:>10}{fmed:>9}{fp95:>8}")
    print(f"\nONNX speed-up vs PyTorch: {pm / om:.2f}x   |   ~{1000 / fm:.1f} packages/sec end to end")
    print("Saved: deploy_benchmark.json")


if __name__ == "__main__":
    main()