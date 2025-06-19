import cv2 as cv
import numpy as np
import os

# ------------------------ Face Detection Functions ------------------------

def detect_faces_dnnssd(frame):
    h, w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = f"{confidence:.2f}"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv.putText(frame, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

def detect_faces_haar(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

def detect_faces_lbp(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = lbp_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

# ------------------------ Choose Detection Method ------------------------

method = input("Which Face Detection method do you want to use? (Haar / LBP / DNN SSD): ").strip().lower()

if method == 'dnnssd':
    net = cv.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
elif method == 'haar':
    haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
elif method == 'lbp':
    lbp_cascade = cv.CascadeClassifier('lbpcascade_frontalface.xml')  # Ensure this file is downloaded
else:
    print("Invalid method selected. Defaulting to DNN SSD.")
    method = 'dnnssd'
    net = cv.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# ------------------------ Choose Input Type ------------------------

input_type = input("Do you want to use the webcam? (yes/no): ").strip().lower()

if input_type == 'yes':
    # ------------------- Webcam Mode -------------------
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        exit()

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        if method == 'haar':
            detect_faces_haar(frame)
        elif method == 'lbp':
            detect_faces_lbp(frame)
        else:
            detect_faces_dnnssd(frame)

        cv.imshow("Webcam Face Detection", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

else:
    # ------------------- Image Mode -------------------
    user_choice = input("Do you want to use the default image? (yes/no): ").strip().lower()
    if user_choice == 'no':
        custom_path = input("Enter the full path to your image: ").strip()
        if os.path.isfile(custom_path):
            img = cv.imread(custom_path)
        else:
            print("Invalid path. Falling back to default image.")
            img = cv.imread('Friends_face_detection.jpeg')
    else:
        img = cv.imread("Friends_face_detection.jpeg")

    if img is None:
        print("Failed to load image.")
        exit()

    if method == 'haar':
        detect_faces_haar(img)
    elif method == 'lbp':
        detect_faces_lbp(img)
    else:
        detect_faces_dnnssd(img)

    cv.imshow("Face Detection", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
