import cv2
import numpy as np
import requests
import base64

RTSP_URL = "rtsp://192.168.3.214:7447/jdr9oLlBkjyl3gDm?"
SERVICE_URL = "http://localhost:8004/infer_mono"
API_KEY = "noesis_secret"

cap = cv2.VideoCapture(RTSP_URL)
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("No frame captured from stream.")
    exit(1)

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
h, w, c = rgb.shape
img_bytes = rgb.tobytes()
img_b64 = base64.b64encode(img_bytes).decode("ascii")

payload = {
    "view": {
        "img_b64": img_b64,
        "shape": [h, w, c],
        "cam_id": "test",
        "intrinsics": None
    }
}

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
}

resp = requests.post(SERVICE_URL, json=payload, headers=headers)
print(f"Status: {resp.status_code}")
if resp.ok:
    data = resp.json()
    depth = data.get("depth_z", [])
    conf = data.get("conf", [])
    mask = data.get("mask", [])
    depth_shape = (len(depth), len(depth[0]) if depth else 0)
    print(f"Depth shape: {depth_shape}")
    if depth:
        print("Depth sample:", depth[0][:5])
    if conf:
        print("Conf sample:", conf[0][:5])
    if mask:
        print("Mask sample:", mask[0][:5])
else:
    print("Error response:", resp.text)
