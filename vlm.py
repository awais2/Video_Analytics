#This sript is used to take the input Image and describe it.
import base64
import requests
import cv2
import json
import time
# LLAVA API Endpoint
LLAVA_API_URL = "http://localhost:11434/api/generate"

# Read and encode the image
img = cv2.imread('/home/awais/Downloads/football.jpeg') 

# Encode the image to Base64
_, img_encoded = cv2.imencode('.jpg', img)
base64_img = base64.b64encode(img_encoded).decode('utf-8')

# Payload for the API
payload = {
    "model": "llava:7b-v1.6-vicuna-q4_K_M",
    "prompt": "Describe this person and their activity in one sentence.",
    "images": [base64_img]
}

# Print payload without the full base64 image for readability
print({key: (value if key != "images" else "Base64 Image Truncated") for key, value in payload.items()})

# Initialize an empty sentence to combine the responses
full_sentence = ""

# Stream the response chunks and combine the "response" fields
try:
    response_stream = requests.post(LLAVA_API_URL, json=payload, stream=True)
    response_stream.raise_for_status()

    for chunk in response_stream.iter_lines(decode_unicode=True):
        if chunk:
            chunk_data = json.loads(chunk)  # Parse JSON from the chunk
            response_part = chunk_data.get("response", "")  # Get "response" field
            full_sentence += response_part  # Concatenate to form the sentence
            #print("CHUNK:", response_part)  # Print each piece for debugging
            #time.sleep(1)

    print("\nResponse:", full_sentence)

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")