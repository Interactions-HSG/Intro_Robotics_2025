import cv2
import os
import numpy as np

# --- Webcam setup ---
cap = cv2.VideoCapture(0)

# --- Face detection setup ---
cascade_file = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_file):
    # fallback: try to find it inside cv2 package
    base_path = os.path.dirname(cv2.__file__)
    cascade_file = os.path.join(base_path, "data", "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_file)

# --- Initial mode ---
mode = "original"

print("""
Controls:
[o] Original
[b] Blur
[e] Edges
[t] Threshold
[f] Face detection
[y] YOLO
[q] Quit
""")

# --- YOLO setup ---
yolo_cfg = "yolov4.cfg"
yolo_weights = "yolov4.weights"
yolo_names = "coco.names"
if os.path.exists(yolo_cfg) and os.path.exists(yolo_weights) and os.path.exists(yolo_names):
    net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
    with open(yolo_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
else:
    net = None
    classes = []
    output_layers = []
    print("YOLO files not found. YOLO mode will not work unless you provide yolov4.cfg, yolov4.weights, and coco.names in the current directory.")

def run_yolo(frame, net, output_layers, conf_threshold=0.5, nms_threshold=0.4):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        class_id = class_ids[i]
        if 'classes' in globals() and len(classes) > class_id:
            label = classes[class_id]
        else:
            label = str(class_id)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Process frame based on current mode ---
    if mode == "blur":
        output = cv2.GaussianBlur(frame, (35, 35), 0)
    elif mode == "edges":
        output = cv2.Canny(gray, 100, 200)
    elif mode == "threshold":
        _, output = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    elif mode == "face":
        output = frame.copy()
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    elif mode == "yolo":
        if net is not None:
            output = run_yolo(frame.copy(), net, output_layers)
        else:
            output = frame.copy()
            cv2.putText(output, "YOLO model not loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:  # original
        output = frame

    # --- Show the processed frame ---
    cv2.imshow("Vision in the Wild", output)

    # --- Key handling ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Quitting...")
        break
    elif key == ord("o"):
        mode = "original"
        print("Mode changed to: original")
    elif key == ord("b"):
        mode = "blur"
        print("Mode changed to: blur")
    elif key == ord("e"):
        mode = "edges"
        print("Mode changed to: edges")
    elif key == ord("t"):
        mode = "threshold"
        print("Mode changed to: threshold")
    elif key == ord("f"):
        mode = "face"
        print("Mode changed to: face detection")
    elif key == ord("y"):
        mode = "yolo"
        print("Mode changed to: YOLO")

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
