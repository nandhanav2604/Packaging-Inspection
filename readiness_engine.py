"""
READINESS ASSESSMENT ENGINE
Combines the trained YOLO model (package + damage detection) with a
barcode/label check to produce one of three verdicts:
    - Ready for Shipment
    - Repackage Required
    - Manual Inspection

BARCODE CHECK (3 tiers - see _check_barcode):
    "readable"             a decoder (pyzbar / zxing-cpp) read the code
    "detected_unreadable"  a barcode REGION was found but could not be decoded
                           (creased / torn / skewed / too small) - on a damaged
                           package this is itself a useful inspection signal
    "not_found"            nothing barcode-like in the image

REQUIREMENTS (one-time):
    pip install ultralytics pyzbar opencv-python zxing-cpp
    (OpenCV >= 4.8 is needed for the barcode-region locator; if your OpenCV
     is older the locator is skipped and the check still works, minus tier 2.)

USAGE:
    from readiness_engine import assess_package
    result = assess_package("some_photo.jpg", model_path="best_yolov8s.pt")
    print(result)

Or run directly on one image for a quick test:
    python readiness_engine.py path/to/image.jpg [path/to/model.pt]
"""
import sys
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import cv2

try:
    import zxingcpp
    _HAS_ZXING = True
except ImportError:
    _HAS_ZXING = False

DEFAULT_MODEL = "best_yolov8s.pt"

# ============================== THRESHOLDS ==============================
# These came directly from the evaluation work: damage detections were
# found to be near-binary in confidence (threshold sweep showed no
# meaningful middle ground), so a single cutoff is used for damage.
# Package confidence, by contrast, showed a real spectrum of certainty
# (0.35-0.91 across different photo conditions), so it gets two cutoffs:
# one for "no package at all" and one for "detected but ambiguous."
PACKAGE_NOT_FOUND_THRESHOLD = 0.25   # below this: treat as no package found
PACKAGE_LOW_CONFIDENCE_THRESHOLD = 0.60   # below this (but found): treat as an unclear/ambiguous photo
DAMAGE_CONF_THRESHOLD = 0.05         # chosen to favor catching real damage over avoiding false alarms

# The detector itself must be allowed to RETURN weak detections, otherwise the
# thresholds above can never apply.
MIN_DETECTION_CONF = 0.05
# ==========================================================================

_model_cache = {}


def _get_model(model_path):
    if model_path not in _model_cache:
        _model_cache[model_path] = YOLO(model_path)
    return _model_cache[model_path]


# ----------------------------- barcode helpers -----------------------------
def _try_decode(variants):
    """True if pyzbar (then zxing-cpp, which tolerates skew better) reads
    any code in any of the given image variants."""
    for v in variants:
        if decode(v):
            return True
    if _HAS_ZXING:
        for v in variants:
            if zxingcpp.read_barcodes(v):
                return True
    return False


def _variants(gray, color=None):
    """Classical pre-processing variants of one image/crop."""
    out = []
    if color is not None:
        out.append(color)
    out += [
        gray,
        cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
        cv2.equalizeHist(gray),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 31, 5),
        cv2.rotate(gray, cv2.ROTATE_180),
    ]
    return out


def _tiles(gray, grid, overlap=0.25, target_min_side=700):
    """Overlapping tiles, upscaled. A barcode that is tiny relative to the
    full photo gets decoded far more easily once it fills more of the frame."""
    h, w = gray.shape[:2]
    th, tw = min(h, int(h / grid * (1 + overlap))), min(w, int(w / grid * (1 + overlap)))
    step_y = max(1, (h - th) // max(1, grid - 1))
    step_x = max(1, (w - tw) // max(1, grid - 1))
    for gy in range(grid):
        for gx in range(grid):
            y0, x0 = min(gy * step_y, h - th), min(gx * step_x, w - tw)
            tile = gray[y0:y0 + th, x0:x0 + tw]
            scale = max(1.0, target_min_side / max(1, min(tile.shape[:2])))
            if scale > 1.0:
                tile = cv2.resize(tile, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            yield tile


def _locate_barcodes(img):
    """Barcode REGIONS (list of (x0, y0, x1, y1)) using OpenCV's built-in
    detector. Finds the bars without needing to decode them, so it still
    works on creased or partly torn labels. [] if unavailable / nothing found."""
    if not hasattr(cv2, "barcode_BarcodeDetector"):
        return []
    try:
        ok, pts = cv2.barcode_BarcodeDetector().detect(img)
    except Exception:
        return []
    boxes = []
    if ok and pts is not None:
        h, w = img.shape[:2]
        for quad in pts:
            xs, ys = quad[:, 0], quad[:, 1]
            boxes.append((max(0, int(xs.min())), max(0, int(ys.min())),
                          min(w, int(xs.max())), min(h, int(ys.max()))))
    return boxes


def _check_barcode(image_path):
    """Returns 'readable', 'detected_unreadable' or 'not_found'."""
    img = cv2.imread(image_path)
    if img is None:
        return "not_found"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tier 1a: whole-image decode with classical pre-processing variants
    if _try_decode(_variants(gray, color=img)):
        return "readable"

    # Tier 1b: tiled decode - helps when the code is small in a big photo
    for grid in (2, 3):
        if _try_decode(list(_tiles(gray, grid))):
            return "readable"

    # Tier 2: locate the barcode region, then retry decoding on a padded,
    # upscaled crop of just that region
    boxes = _locate_barcodes(img)
    h, w = gray.shape[:2]
    for (x0, y0, x1, y1) in boxes:
        pad_x, pad_y = int(0.3 * (x1 - x0)) + 10, int(0.3 * (y1 - y0)) + 10
        crop = gray[max(0, y0 - pad_y):min(h, y1 + pad_y),
                    max(0, x0 - pad_x):min(w, x1 + pad_x)]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(crop)
        if _try_decode(_variants(crop) + [clahe, cv2.GaussianBlur(crop, (0, 0), 1.0)]):
            return "readable"

    # A region was found but nothing could be decoded
    if boxes:
        return "detected_unreadable"
    return "not_found"


# ------------------------------- main entry -------------------------------
def assess_package(image_path, model_path=DEFAULT_MODEL, check_barcode=True):
    """Returns a dict with the verdict, the reasoning, and the raw signals
    that led to it - useful both for the frontend and for your report.
    check_barcode=False skips the (slower) barcode step, e.g. for bulk metrics."""
    model = _get_model(model_path)
    results = model.predict(image_path, conf=MIN_DETECTION_CONF, verbose=False)
    r = results[0]

    name_to_id = {v: k for k, v in model.names.items()}
    package_id = name_to_id.get("package")
    damage_id = name_to_id.get("damage")

    package_conf = 0.0
    damage_conf = 0.0
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == package_id and conf > package_conf:
            package_conf = conf
        if cls_id == damage_id and conf > damage_conf:
            damage_conf = conf

    barcode_status = _check_barcode(image_path) if check_barcode else "skipped"
    barcode_found = barcode_status == "readable"

    # ---------------- decision logic (unchanged) ----------------
    if package_conf < PACKAGE_NOT_FOUND_THRESHOLD:
        verdict = "Manual Inspection"
        reason = "No package could be clearly detected in the image."

    elif damage_conf >= DAMAGE_CONF_THRESHOLD:
        verdict = "Repackage Required"
        reason = f"Visible damage detected (confidence {damage_conf:.2f})."

    elif package_conf < PACKAGE_LOW_CONFIDENCE_THRESHOLD:
        verdict = "Manual Inspection"
        reason = (f"Package detected but with low confidence ({package_conf:.2f}) - "
                  "the image may be blurry, cluttered, or unclear, so the absence "
                  "of a damage detection cannot be fully trusted here.")

    else:
        verdict = "Ready for Shipment"
        reason = f"Package clearly detected (confidence {package_conf:.2f}), no damage found."
        if barcode_status == "detected_unreadable":
            reason += (" Note: a barcode/label was detected but could not be read - "
                       "it may be creased, torn or too small; verify it scans.")
        elif barcode_status == "not_found":
            reason += " Note: no barcode/label was detected - verify manually if one is expected."

    # Only draw detections that actually count toward the decision, so the
    # annotated image is not cluttered by the weak boxes kept for thresholding.
    shown = r
    try:
        keep = [i for i, b in enumerate(r.boxes)
                if (int(b.cls[0]) == damage_id and float(b.conf[0]) >= DAMAGE_CONF_THRESHOLD)
                or (int(b.cls[0]) == package_id and float(b.conf[0]) >= PACKAGE_NOT_FOUND_THRESHOLD)]
        shown = r[keep] if keep else r[:0]
    except Exception:
        shown = r

    return {
        "image": image_path,
        "verdict": verdict,
        "reason": reason,
        "package_confidence": round(package_conf, 3),
        "damage_confidence": round(damage_conf, 3),
        "barcode_found": barcode_found,          # True only if actually decoded
        "barcode_status": barcode_status,        # readable / detected_unreadable / not_found
        "annotated_result": shown,  # shown.plot() gives you the annotated image array if needed
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python readiness_engine.py path/to/image.jpg [path/to/model.pt]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    result = assess_package(image_path, model_path)
    print("\n" + "=" * 50)
    print(f"Image:      {result['image']}")
    print(f"VERDICT:    {result['verdict']}")
    print(f"Reason:     {result['reason']}")
    print(f"Package conf: {result['package_confidence']}")
    print(f"Damage conf:  {result['damage_confidence']}")
    print(f"Barcode:      {result['barcode_status']}")
    print("=" * 50)