import cv2
import dlib
import numpy as np
import os
import csv
from datetime import datetime

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def get_face_descriptor(img_path):
    img = cv2.imread(img_path)
    dets = detector(img, 1)
    if len(dets) == 0:
        return None
    shape = sp(img, dets[0])
    return np.array(facerec.compute_face_descriptor(img, shape))

def load_dataset(dataset_path="dataset"):
    known_faces = {}
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        descriptors = []
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            desc = get_face_descriptor(img_path)
            if desc is not None:
                descriptors.append(desc)
        if descriptors:
            known_faces[person_name] = descriptors
    return known_faces

def recognize_face(face_descriptor, known_faces, threshold=0.6):
    name = "Unknown"
    min_dist = 1e9
    for person, descriptors in known_faces.items():
        for d in descriptors:
            dist = np.linalg.norm(face_descriptor - d)
            if dist < min_dist:
                min_dist = dist
                name = person
    return name if min_dist < threshold else "Unknown"

def mark_attendance(name, filename="attendance.csv"):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if os.path.exists(filename):
        with open(filename, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == name and row[1] == date_str:
                    return 

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])

known_faces = load_dataset("dataset")
print("โหลด dataset เสร็จแล้ว:", list(known_faces.keys()))

cap = cv2.VideoCapture(0)
print("กด q เพื่อออก")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector(frame, 1)
    for face in faces:
        shape = sp(frame, face)
        face_desc = np.array(facerec.compute_face_descriptor(frame, shape))
        name = recognize_face(face_desc, known_faces)

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        if name != "Unknown":
            mark_attendance(name)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
