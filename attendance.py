import os
import csv
from datetime import datetime
from pathlib import Path
from time import monotonic

import cv2
import dlib
import numpy as np

# ---------- ตั้งค่าไฟล์โมเดล ----------
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACEREC_PATH   = "dlib_face_recognition_resnet_model_v1.dat"
DATASET_DIR    = "dataset"
ATTEND_CSV     = "attendance.csv"

# ---------- ตรวจไฟล์จำเป็น ----------
for need in [PREDICTOR_PATH, FACEREC_PATH]:
    if not Path(need).exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {Path(need).resolve()}")

if not Path(DATASET_DIR).exists():
    raise FileNotFoundError(f"ไม่พบโฟลเดอร์ dataset: {Path(DATASET_DIR).resolve()}")

# ---------- โหลดโมเดล dlib ----------
detector = dlib.get_frontal_face_detector()
sp       = dlib.shape_predictor(PREDICTOR_PATH)
facerec  = dlib.face_recognition_model_v1(FACEREC_PATH)

# ---------- ฟังก์ชันช่วย ----------
def get_face_descriptor(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        return None
    dets = detector(img, 1)
    if len(dets) == 0:
        return None
    shape = sp(img, dets[0])
    return np.array(facerec.compute_face_descriptor(img, shape))

def load_dataset(dataset_path=DATASET_DIR):
    known_faces = {}
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        descriptors = []
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
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

def mark_attendance(name, filename=ATTEND_CSV):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    new_file = not os.path.exists(filename)
    if new_file:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "date", "time"])

    # กันซ้ำวันเดียวกัน
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)[1:] if not new_file else []
        for row in rows:
            if row and row[0] == name and row[1] == date_str:
                return

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])

def draw_label_with_bg(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7,
                       color=(0,255,0), thickness=2):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    top_left     = (x, y - text_h - 6)
    bottom_right = (x + text_w + 6, y + 4)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y), font, scale, color, thickness)

# ---------- โหลด dataset ----------
known_faces = load_dataset(DATASET_DIR)
print("โหลด dataset เสร็จแล้ว:", list(known_faces.keys()))

# ---------- กันซ้ำในหน่วยความจำ ----------
seen_today = set()          # ชื่อที่เจอแล้วในวันนี้
last_seen_at = {}           # name -> monotonic() ล่าสุด
COOLDOWN_SEC = 10           # กันติ๊กซ้ำภายใน 10 วินาที

# ---------- เปิดกล้อง ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("เปิดกล้องไม่ได้ (VideoCapture(0) ล้มเหลว)")

print("กด q เพื่อออก")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector(frame, 1)
    now_s = monotonic()

    for face in faces:
        shape = sp(frame, face)
        face_desc = np.array(facerec.compute_face_descriptor(frame, shape))
        name = recognize_face(face_desc, known_faces)

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

        # ป้ายชื่อ + เวลา
        current_time = datetime.now().strftime("%H:%M:%S")
        label = f"{name} {current_time}"
        text_y = max(y1 - 10, 20)
        draw_label_with_bg(frame, label, (x1, text_y))

        # กันบันทึกซ้ำ (ต่อวัน + คูลดาวน์)
        if name != "Unknown":
            if (name not in seen_today) and \
               (name not in last_seen_at or now_s - last_seen_at[name] > COOLDOWN_SEC):
                mark_attendance(name)
                seen_today.add(name)
                last_seen_at[name] = now_s

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
