from ultralytics import YOLO
import cv2
import time

# ==============================
# CONFIGURATION
# ==============================

# Use a larger model for better accuracy OR your custom-trained model
# Replace with your own trained model path if available
MODEL_PATH = "yolov8m.pt"  # Can be: "yolov8n.pt", "yolov8l.pt", or "runs/detect/train/weights/best.pt"

# Minimum confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# ==============================
# INITIALIZE MODEL AND CAMERA
# ==============================

# Load the model
model = YOLO(MODEL_PATH)

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if camera is opened successfully
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Press 'Q' to quit.")
time.sleep(1)

# ==============================
# MAIN LOOP
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Run YOLOv8 detection on the frame
    results = model(frame, conf=CONFIDENCE_THRESHOLD)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Display detected class names with confidence
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[class_id]
        print(f"📌 Detected: {class_name} ({conf:.2f})")

    # Show the annotated frame
    cv2.imshow("🔍 YOLOv8 Real-Time Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()
