from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
# import cv2
import os

# Get the number of connected webcams
num_webcams = int(input("Enter the number of connected webcams: "))

# Select the webcam to use
webcam_id = input("Enter the webcam ID (0 to {}): ".format(num_webcams - 1))
# model = YOLO("/Users/rishirajdatta7/best.pt")
model_path = os.path.join(os.path.expanduser("~"), "best.pt")
model = YOLO(model_path)
results = model.predict(source = webcam_id,show = True)

print(results)


