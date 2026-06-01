import cv2
import numpy as np
import os
import cv2
import numpy as np
import onnxruntime as ort

# Automatically calculate the absolute path to the backend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.onnx")

# Initialize the ONNX Runtime session using the secure absolute path
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Load the locally generated YOLOv8 ONNX model using OpenCV DNN
net = cv2.dnn.readNetFromONNX("yolov8n.onnx")

CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def predict_animals(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_h, img_w, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    
    rows = np.squeeze(outputs).T
    boxes, confidences, class_ids = [], [], []
    
    x_factor = img_w / 640
    y_factor = img_h / 640

    for row in rows:
        classes_scores = row[4:]
        max_score = np.amax(classes_scores)
        
        if max_score >= 0.25:
            class_id = np.argmax(classes_scores)
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            
            x1 = int((cx - w/2) * x_factor)
            y1 = int((cy - h/2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([x1, y1, width, height])
            confidences.append(float(max_score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detections.append({
                "box": [max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h)],
                "confidence": round(confidences[i], 2),
                "label": CLASSES[class_ids[i]]
            })
            
    return {"width": img_w, "height": img_h, "detections": detections}