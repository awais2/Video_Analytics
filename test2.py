from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import time
import torch
import os

# Creating a class for object detection which plots boxes and scores frames in addition to detecting an object
class YoloDetector():
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
        #self.model = YOLO("yolo-Weights/yolov8s.pt")
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', self.device)
    
    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))

        results = self.model(frame)

        # Extract predictions
        detections = results[0]  # Assume batch size of 1
        labels = detections.boxes.cls.cpu().numpy()  # Class labels
        cord = detections.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = detections.boxes.conf.cpu().numpy()  # Confidence scores

        return labels, np.hstack((cord, confidences.reshape(-1, 1)))
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame, height, width, confidence=0.3):
        labels, cord = results
        detections = []
        n = len(labels)

        for i in range(n):
            row = cord[i]
            if row[4] >= confidence:  # Confidence threshold
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                label = self.class_to_label(labels[i])

                # Only process detections for "person"
                if label == 'person':  # Replace 'person' with the correct label for humans if different
                    detections.append(([x1, y1, x2 - x1, y2 - y1], row[4], 'person'))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'{label} {row[4]:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame, detections

# Setting input video to webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initializing the detection and tracking classes
detector = YoloDetector()
object_tracker = DeepSort()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    start = time.perf_counter()

    results = detector.score_frame(img)
    img, detections = detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.5)

    tracks = object_tracker.update_tracks(detections, frame=img)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()

        bbox = ltrb
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(img, f"ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
        break

cap.release()
cv2.destroyAllWindows()
