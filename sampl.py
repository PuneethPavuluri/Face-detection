import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
import cv2
import insightface
import ultralytics
from deep_sort_realtime.deepsort_tracker import DeepSort

print("[SUCCESS] All packages imported correctly!")
print(f"OpenCV Version: {cv2.__version__}")
print(f"InsightFace Version: {insightface.__version__}")
print(f"Ultralytics (YOLOv8) Version: {ultralytics.__version__}")

# Initialize DeepSort tracker (should work if no errors)
tracker = DeepSort(max_age=30)
print("[SUCCESS] DeepSort tracker initialized!")