# Face Recognition Attendance Project 🎥🧑‍💻

ระบบลงเวลาการเข้าเรียน/ทำงานด้วยการจดจำใบหน้า (Face Recognition Attendance System)  
โปรเจกต์นี้พัฒนาด้วย **Python, dlib, OpenCV และ Flask**  
สามารถตรวจจับใบหน้าแบบเรียลไทม์ บันทึกชื่อและเวลาเข้าเรียนไว้ในไฟล์ CSV และแสดงผลแบบ Dashboard ผ่านเว็บได้

---

## ⚙️ วิธีการติดตั้งและใช้งาน

### 1. Clone โปรเจกต์
```bash
git clone https://github.com/username/Face-Recognition-Attendance-project.git
cd Face-Recognition-Attendance-project
```

### 2. สร้าง Virtual Environment และติดตั้ง dependencies
python -m venv venv
venv\Scripts\activate   # สำหรับ Windows
source venv/bin/activate  # สำหรับ Mac/Linux

pip install -r requirements.txt


### 3. เตรียม Dataset

สร้างโฟลเดอร์ dataset

ภายใน dataset ให้สร้างโฟลเดอร์ย่อยตามชื่อจริง-นามสกุล เช่น
dataset/
├── Worachat/

│   ├── Worachat1.jpg

├── Krit/

│   ├── Krit1.jpg

└── Phuchit/

│    ├── Phuchit1.jpg

### รันพร้อมใช้งาน

# 🎯 Sprint ปัจจุบัน: สัปดาห์ที่ 1 - การสร้างรากฐาน (Foundation)
**สถานะ** : 🚧 In Progress
**เป้าหมายของสัปดาห์** : สร้างโครงสร้างหลักของโปรแกรม (Main Loop) ให้สามารถเปิดกล้องเว็บแคมและตรวจจับตัวบุคคลได้
**บทบาทสัปดาห์นี้** : 
Planner 🗺️ : พี
Coder 💻 : นิคกี้
Debugger 🕵️‍♀️ : ภู
**นิยามของคำว่า "เสร็จ" (Definition of Done)**
- ตัวโปรแกรมสามารถตรวจจับและแยกใบหน้าของผู้ใช้แต่ละคนออกได้อย่างแม่นยำ



