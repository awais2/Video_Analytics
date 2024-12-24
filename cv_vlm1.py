# Necessary imports
import cv2
import numpy as np
import sys
import glob
import time
import torch
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import base64
import requests

# LLAVA API Endpoint
LLAVA_API_URL = "http://localhost:11434/api/generate"

# Function to interact with LLAVA API
def query_llava(full_image_b64, cropped_person_b64):
    payload = {
        "model": "llava",
        "prompt": "What is in this picture?",
        "images": [full_image_b64, cropped_person_b64]
    }
    response = requests.post(LLAVA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
# Creating a class for object detection which plots boxes and scores frames
class YoloV8Detector:
    def __init__(self, target_classes=None):
        # Using YOLOv8n for object detection
        self.model = YOLO('yolov8n.pt')  # Load YOLOv8 model
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', self.device)
        self.target_classes = target_classes if target_classes else ['person']

    def score_frame(self, frame):
        """
        Scores the frame for detected objects using the YOLO model.
        """
        self.model.to(self.device)
        results = self.model(frame, stream=True)  # Stream predictions to improve real-time performance
        return results

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, confidence=0.3):
        """
        Draws bounding boxes around detected objects and returns detections for tracking.
        """
        detections = []
        height, width = frame.shape[:2]

        for result in results:
            for box in result.boxes:
                score = box.conf.item()
                if score >= confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
                    label_id = int(box.cls.item())
                    label = self.class_to_label(label_id)

                    if label in self.target_classes:
                        tlwh = np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
                        detections.append((tlwh, score, label))
                        # Draw rectangle and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame, detections

detector = YoloV8Detector(target_classes=['person', 'car'])  # Include multiple classes if desired
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

object_tracker = DeepSort()
# Main pipeline: Person detection and LLAVA integration
# Load video
cap = cv2.VideoCapture("/home/awais/Downloads/video1.mp4")

while True:
    ret, img = cap.read()
    if not ret:
        break  # End of video

    # Person detection logic (use your existing code here)
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
        cv2.putText(img, f"ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Person description extraction with LLAVA
        x_min, y_min, x_max, y_max = map(int, bbox)
        detected_person = img[y_min:y_max, x_min:x_max]
        cv2.imshow('person', detected_person)
        # Base64 encoding for the image
        _, img_encoded = cv2.imencode('.jpg', detected_person)
        print(type(detected_person))
        base64_img = base64.b64encode(img_encoded).decode('utf-8')
        

        payload = {
            "model": "llava:7b-v1.6-vicuna-q4_K_M",
            "prompt": "Describe this person and their activity in one sentence.",
            "images": [base64_img]
        }

        print(payload)

        # LLAVA API call
        response = requests.post(LLAVA_API_URL, json=payload)
        #print("response: ", response.text)
        llava_response = response.json()
        description = llava_response.get('message', 'No response received')


    # Display processed video
    cv2.imshow('img', img)
    

    # Exit loop on 'q' key
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()