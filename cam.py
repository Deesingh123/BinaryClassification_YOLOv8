import cv2
import time
import os
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "best.pt"   # Change to "yolo11s.pt" if using pretrained
CLASS_NAMES = ['Hello', 'Love_you', 'No', 'Thank_you', 'Yes']
OUTPUT_DIR = "outputs"
CONF_THRES = 0.25
IOU_THRES = 0.7
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

def run_on_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not detected.")
        return

    print("🎥 Press 'q' to quit | 's' to save frame")
    print(f"➡️  Expecting gestures: {', '.join(CLASS_NAMES)}")

    frame_count, start_time = 0, time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        # Resize for inference
        frame = cv2.resize(frame, (640, 640))

        # Run detection
        results = model(frame, conf=CONF_THRES, iou=IOU_THRES)

        annotated_frame = frame.copy()
        if results[0].boxes:
            confs = results[0].boxes.conf
            max_idx = confs.argmax()  # take top-1
            top_box = results[0].boxes[max_idx:max_idx+1]
            results[0].boxes = top_box

            cls_id = int(results[0].boxes.cls[0])
            conf = float(results[0].boxes.conf[0])
            detected_class = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

            print(f"✅ Detected: {detected_class} ({conf:.2f})")

            annotated_frame = results[0].plot()
        else:
            print("❌ No detections")

        # Show frame
        cv2.imshow("Sign Language Detection", annotated_frame)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_path = os.path.join(OUTPUT_DIR, f"frame_{int(time.time())}.png")
            cv2.imwrite(save_path, annotated_frame)
            print(f"💾 Frame saved: {save_path}")
        elif key == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # FPS calculation
    fps = frame_count / (time.time() - start_time)
    print(f"📊 Average FPS: {fps:.2f}")

if __name__ == "__main__":
    run_on_webcam()
