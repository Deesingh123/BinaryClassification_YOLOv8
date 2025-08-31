import os
import cv2
import glob
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "best.pt"   # Change to "yolo11s.pt" for pretrained
CLASS_NAMES = ['Hello', 'Love_you', 'No', 'Thank_you', 'Yes']
SOURCE_DIR = "Test_Data/images"
OUTPUT_DIR = "outputs"
RUNS_DIR = "runs/detect/predict"
CONF_THRES = 0.25
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

def run_on_images():
    # Collect image files
    image_files = glob.glob(os.path.join(SOURCE_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(SOURCE_DIR, "*.png")) + \
                  glob.glob(os.path.join(SOURCE_DIR, "*.jpeg"))

    if not image_files:
        print("❌ No images found in", SOURCE_DIR)
        return

    # Process only first 6 for demo
    image_files = image_files[:6]
    print(f"📷 Processing {len(image_files)} images...")

    for image_path in image_files:
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load {image_path}")
            continue

        img = cv2.resize(img, (640, 640))  # match training size
        results = model(img, conf=CONF_THRES)

        if results[0].boxes:
            cls_ids = results[0].boxes.cls
            confs = results[0].boxes.conf
            detected_classes = [CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else str(c) for c in cls_ids]

            print(f"✅ {os.path.basename(image_path)} → {detected_classes} ({[round(float(c),2) for c in confs]})")
        else:
            print(f"❌ No detections in {os.path.basename(image_path)}")

        # Save annotated output
        annotated = results[0].plot()
        save_path = os.path.join(OUTPUT_DIR, f"annotated_{os.path.basename(image_path)}")
        cv2.imwrite(save_path, annotated)

        # Save logs
        with open(os.path.join(RUNS_DIR, f"predict_{os.path.basename(image_path)}.txt"), "w") as f:
            f.write(f"Image: {image_path}\n")
            if results[0].boxes:
                f.write(f"Classes: {detected_classes}\n")
                f.write(f"Confidences: {[round(float(c), 3) for c in confs.tolist()]}\n")
            else:
                f.write("No detections.\n")

        print(f"💾 Results saved: {save_path}")

if __name__ == "__main__":
    run_on_images()
