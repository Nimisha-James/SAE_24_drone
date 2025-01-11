import cv2
import numpy as np
import torch

model = torch.hub.load('yolov5', 'custom', path= r'C:\Users\nimis\OneDrive\Desktop\object\best.pt', source= 'local' ,force_reload=True )
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    results = model(frame)

    frame = results.render()[0]

    cv2.imshow('YOLOv5 Real-Time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()