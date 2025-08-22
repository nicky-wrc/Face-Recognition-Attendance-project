import cv2
import dlib
import numpy as np
import os
from datetime import datetime
import csv

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

known_faces = load_dataset("dataset")
print("โหลด dataset เสร็จแล้ว:", list(known_faces.keys()))

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

def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    with open("attendance.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, dt_string])

recorded_names = set()

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

        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"{name} {current_time}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if name != "Unknown" and name not in recorded_names:
            mark_attendance(name)
            recorded_names.add(name)
            print(f"[INFO] บันทึก {name} เวลา {current_time}")

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
