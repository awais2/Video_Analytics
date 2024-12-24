import base64
import requests
import cv2

# LLAVA API Endpoint
LLAVA_API_URL = "http://localhost:11434/api/generate"
img = cv2.imread('/home/awais/Downloads/football.jpeg') 

#Encode the image to Base64
_, img_encoded = cv2.imencode('.jpg', img)
#print(type(img))
base64_img = base64.b64encode(img_encoded).decode('utf-8')


payload = {
    "model": "llava:7b-v1.6-vicuna-q4_K_M",
    "prompt": "Describe this person and their activity in one sentence.",
    "images": [base64_img]
}

#print(payload)

# LLAVA API call
response = requests.post(LLAVA_API_URL, json=payload)
#print("response: ", response.text)
llava_response = response.json()
description = llava_response.get('message', 'No response received')
print(f"Response: {description}")