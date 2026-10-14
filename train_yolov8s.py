"""
Training script for the final model (YOLOv8s), fine-tuned from COCO-pretrained weights.
Originally run in Google Colab on a T4 GPU (about 0.94 hours).

Expected layout (the dataset itself is NOT in this repo, see README):
    merged_dataset/
        data.yaml
        images/{train,val,test}/
        labels/{train,val,test}/

Run:
    pip install ultralytics
    python train_yolov8s.py
"""
from ultralytics import YOLO

DATA_YAML = "merged_dataset/data.yaml"   # in Colab: /content/merged_dataset/data.yaml
PROJECT = "training_run"                 # in Colab this pointed to a folder on Google Drive
RUN_NAME = "main_run_yolov8s"

# Pretrained COCO weights are downloaded automatically on first use.
model = YOLO("yolov8s.pt")

model.train(
    data=DATA_YAML,
    epochs=100,      # maximum number of passes over the training images
    patience=15,     # early stopping: stop if validation fitness does not improve for 15 epochs
    imgsz=640,       # images are resized to 640 x 640
    batch=16,        # images per weight update
    device=0,        # first GPU; use "cpu" if no GPU (much slower)
    project=PROJECT,
    name=RUN_NAME,
    verbose=True,
)

# Final check on the untouched test split (used once, after training).
best = YOLO(f"{PROJECT}/{RUN_NAME}/weights/best.pt")
metrics = best.val(data=DATA_YAML, split="test", imgsz=640)
print("precision, recall, mAP50, mAP50-95:",
      metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map)
