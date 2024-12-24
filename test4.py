# Necessary imports
import cv2
import numpy as np
import sys
import glob
import time
import torch
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

# Creating a class for object detection which plots boxes and scores frames
class YoloDetector:
    def __init__(self, target_classes=None):
        # Using yolov5s for object detection
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', self.device)
        self.target_classes = target_classes if target_classes else ['person']

    def score_frame(self, frame):
        """
        Scores the frame for detected objects using the YOLO model.
        """
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))

        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, confidence=0.3):
        """
        Draws bounding boxes around detected objects and returns detections for tracking.
        """
        labels, cord = results
        detections = []
        n = len(labels)
        height, width = frame.shape[:2]

        for i in range(n):
            row = cord[i]
            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0] * width), int(row[1] * height), int(row[2] * width), int(row[3] * height)
                label = self.class_to_label(labels[i])

                if label in self.target_classes:
                    tlwh = np.asarray([x1, y1, int(x2 - x1), int(y2 - y1)], dtype=np.float32)
                    detections.append((tlwh, row[4].item(), label))
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {row[4]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame, detections


# Main pipeline for video processing
cap = cv2.VideoCapture("/home/awais/Downloads/video1.mp4")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = YoloDetector(target_classes=['person', 'car'])  # Include multiple classes if desired
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

object_tracker = DeepSort()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    start = time.perf_counter()

    results = detector.score_frame(img)
    img, detections = detector.plot_boxes(results, img, confidence=0.5)

    tracks = object_tracker.update_tracks(detections, frame=img)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        bbox = ltrb
        # Draw tracking info
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(img, f"ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    end = time.perf_counter()
    total_time = end - start
    fps = 1 / total_time

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
