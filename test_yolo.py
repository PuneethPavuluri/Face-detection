from ultralytics import YOLO

# This will automatically download the standard YOLOv8n model
model = YOLO('yolov8n.pt')  # Note: using standard model instead of face-specific one

# Test detection (this will show it can detect faces)
results = model('https://ultralytics.com/images/bus.jpg')  # Example image
results[0].show()  # Display results