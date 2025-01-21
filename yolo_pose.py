from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("bus.jpg")

# Load a model
model = YOLO("yolov8n-pose.pt")  # load an official model

# Predict with the model
results = model.predict("bus.jpg")

# Display results
frame = results[0].plot()
plt.imshow(frame)
plt.show()