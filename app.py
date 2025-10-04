from flask import Flask, render_template, request, jsonify, send_from_directory
import os, csv
from datetime import datetime
from pathlib import Path

# ====== สร้าง Flask app ======
app = Flask(__name__)

print("🎯 Flask app created successfully!")

# ====== Config ======
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACEREC_PATH   = "dlib_face_recognition_resnet_model_v1.dat"
DATASET_DIR    = "dataset"
ATTEND_CSV     = "attendance.csv"
THRESHOLD      = 0.6

# ====== เตรียมโฟลเดอร์พื้นฐาน ======
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# ====== ตัวแปร global ======
detector = None
sp = None
rec = None
known_faces = {}   # dict[name] = {"descs": [128D...], "mean": 128D ndarray}

try:
    import cv2
    import dlib
    import numpy as np
    import threading
    encode_lock = threading.Lock()

    if os.path.exists(PREDICTOR_PATH) and os.path.exists(FACEREC_PATH):
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(PREDICTOR_PATH)
        rec = dlib.face_recognition_model_v1(FACEREC_PATH)
        print("✅ Dlib models loaded successfully")

        # ---------- Encoding dataset (แก้ไขใหม่) ----------
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        def _face_descriptors_from_image(bgr_img):
            """คืนค่า list ของ (rect, 128D descriptor) ที่ตรวจได้จากภาพ"""
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            dets = detector(rgb, 1)  # upsample=1 ให้ detect ได้แม่นขึ้น
            results = []
            for d in dets:
                shape = sp(rgb, d)
                desc = np.array(rec.compute_face_descriptor(rgb, shape), dtype=np.float32)
                results.append((d, desc))
            return results

        def _encode_file(path):
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return []
            return _face_descriptors_from_image(img)

        def load_dataset():
            """
            dataset/
              Krit/ img1.jpg img2.png ...
              Siriwan/ ...
            คืนค่า dict[name] = {"descs":[...], "mean": np.array(128)}
            """
            people = {}
            if not os.path.isdir(DATASET_DIR):
                return people

            for person in os.listdir(DATASET_DIR):
                pdir = os.path.join(DATASET_DIR, person)
                if not os.path.isdir(pdir):
                    continue

                descs = []
                for fn in os.listdir(pdir):
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in IMG_EXTS:
                        continue
                    fpath = os.path.join(pdir, fn)
                    try:
                        pairs = _encode_file(fpath)  # [(rect, desc), ...]
                        for _, desc in pairs:
                            descs.append(desc)
                    except Exception as e:
                        print(f"⚠️ Encode fail {fpath}: {e}")

                if len(descs) > 0:
                    mean = np.mean(descs, axis=0)
                    people[person] = {"descs": descs, "mean": mean}

            print(f"📁 Encoded {len(people)} people: {list(people.keys())}")
            return people

        known_faces = load_dataset()
    else:
        print("⚠️ Dlib model files not found - face recognition disabled")
        print(f"Missing: {PREDICTOR_PATH if not os.path.exists(PREDICTOR_PATH) else ''} {FACEREC_PATH if not os.path.exists(FACEREC_PATH) else ''}")

except ImportError as e:
    print(f"⚠️ Missing libraries: {e}")
    print("Face recognition will be disabled")

# ====== Helper ======
def mark_attendance(name, filename=ATTEND_CSV):
    """บันทึกการเข้าเรียนแบบกันซ้ำรายวัน"""
    try:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # สร้างไฟล์ใหม่ถ้าไม่มี
        if not os.path.exists(filename):
            with open(filename, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["name", "date", "time"])

        # อ่านข้อมูลเดิม
        existing_records = []
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_records = list(reader)

        # ข้ามหัวตาราง
        if existing_records and existing_records[0][0].lower() == "name":
            existing_records = existing_records[1:]

        # กันซ้ำรายวัน
        for record in existing_records:
            if len(record) >= 2 and record[0] == name and record[1] == date_str:
                print(f"ℹ️ {name} already recorded today")
                return False

        # เพิ่มรายการใหม่
        with open(filename, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([name, date_str, time_str])

        print(f"✅ Attendance recorded: {name} at {time_str}")
        return True

    except Exception as e:
        print(f"❌ Error in mark_attendance: {e}")
        return False

# ====== Routes ======

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/")
def index():
    """หน้าหลัก"""
    print("📄 Serving index page")
    data = []
    try:
        if os.path.exists(ATTEND_CSV):
            with open(ATTEND_CSV, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
                if rows and len(rows) > 0 and rows[0][0].lower() == "name":
                    rows = rows[1:]  # ข้ามหัวตาราง
                data = rows
        print(f"📊 Loaded {len(data)} attendance records")
    except Exception as e:
        print(f"❌ Error loading attendance data: {e}")

    return render_template("index.html", data=data)

@app.route("/test", methods=["GET"])
def test():
    """ทดสอบสถานะ server"""
    print("🧪 Test endpoint called")
    return jsonify({
        "status": "ok",
        "message": "Server is running!",
        "models_loaded": all([detector, sp, rec]),
        "known_faces": list(known_faces.keys()),
        "dataset_dir": DATASET_DIR,
        "attendance_file": ATTEND_CSV,
        "attendance_exists": os.path.exists(ATTEND_CSV),
        "dataset_exists": os.path.exists(DATASET_DIR)
    })

@app.route("/api/attendance", methods=["GET"])
def api_attendance():
    """API ดึงข้อมูลการเข้าเรียน"""
    print("📊 API attendance called")
    try:
        data = []
        if os.path.exists(ATTEND_CSV):
            with open(ATTEND_CSV, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
                if rows and len(rows) > 0 and rows[0][0].lower() == "name":
                    rows = rows[1:]  # ข้ามหัวตาราง
                data = rows

        print(f"📈 Returning {len(data)} attendance records")
        return jsonify({"ok": True, "rows": data})

    except Exception as e:
        print(f"❌ Error in api_attendance: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/frame", methods=["POST"])
def api_frame():
    """รับภาพจากกล้อง → ตรวจจับใบหน้า → ระบุชื่อ → บันทึกเข้า CSV"""
    print("📸 API frame called")
    try:
        if not all([detector, sp, rec]):
            return jsonify({"ok": False, "error": "Face recognition models not loaded"}), 500

        file = request.files.get("frame")
        if not file:
            return jsonify({"ok": False, "error": "no frame"}), 400

        # อ่าน blob เป็นภาพ
        import numpy as np, cv2
        data = np.frombuffer(file.read(), dtype=np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"ok": False, "error": "invalid image"}), 400

        # ย่อภาพเพื่อให้เร็วขึ้น (ไม่บังคับ)
        h, w = img.shape[:2]
        scale = 800 / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

        # ตรวจจับ + คำนวณ descriptor
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = detector(rgb, 1)
        detections = []

        if not dets:
            print("🙈 No faces detected in this frame")
            return jsonify({"ok": True, "detections": []})

        import numpy as np
        for d in dets:
            shape = sp(rgb, d)
            face_desc = np.array(rec.compute_face_descriptor(rgb, shape), dtype=np.float32)

            best_name = "Unknown"
            best_dist = 1e9

            # เทียบกับ centroid ของแต่ละคนก่อน (เร็ว)
            for name, bundle in known_faces.items():
                mean_desc = bundle["mean"]
                dist = np.linalg.norm(face_desc - mean_desc)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

            # Fine-tune กับทุกรูปของคนนั้นถ้าเข้าเค้า
            if best_name != "Unknown" and best_dist < 0.7:
                fine_best = 1e9
                for cand in known_faces[best_name]["descs"]:
                    d2 = np.linalg.norm(face_desc - cand)
                    if d2 < fine_best:
                        fine_best = d2
                best_dist = fine_best

            det_name = best_name if best_dist <= THRESHOLD else "Unknown"
            bbox = [int(d.left()), int(d.top()), int(d.right()), int(d.bottom())]
            detections.append({"name": det_name, "dist": round(float(best_dist), 3), "box": bbox})

            # บันทึกชื่อถ้ารู้จัก (กันซ้ำรายวันใน mark_attendance)
            if det_name != "Unknown":
                with encode_lock:
                    mark_attendance(det_name)

        print(f"🎯 Real detection: {len(detections)} face(s)")
        return jsonify({"ok": True, "detections": detections})

    except Exception as e:
        print(f"❌ Error in api_frame: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

# ====== Debug routes ======
@app.route("/debug/create-test-data")
def create_test_data():
    """สร้างข้อมูลทดสอบ"""
    try:
        test_data = [
           
        ]
        with open(ATTEND_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(test_data)
        return jsonify({"ok": True, "message": "Test data created"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ====== Error handlers ======
@app.errorhandler(404)
def not_found(error):
    print(f"❌ 404 Error: {request.url}")
    return jsonify({"ok": False, "error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    print(f"❌ 500 Error: {error}")
    return jsonify({"ok": False, "error": "Server error"}), 500

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    print("=" * 60)
    print("🚀 Starting Flask Application")
    print("=" * 60)
    print(f"📁 Dataset directory: {os.path.abspath(DATASET_DIR)}")
    print(f"📝 Attendance file: {os.path.abspath(ATTEND_CSV)}")
    print(f"🎯 Known faces: {list(known_faces.keys())}")
    print(f"🤖 Face recognition: {'✅ Enabled' if all([detector, sp, rec]) else '❌ Disabled'}")
    print("=" * 60)
    print("🌐 Server URLs:")
    print("   - Local: http://127.0.0.1:5000")
    print("   - Network: http://0.0.0.0:5000")
    print("   - Test: http://127.0.0.1:5000/test")
    print("   - Debug: http://127.0.0.1:5000/debug/create-test-data")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=debug, threaded=True)