# Face Recognition Attendance Project 🎥🧑‍💻

Colab Test

https://colab.research.google.com/drive/1fchqs636mjrRhXy4D7V-IWOlRolXe1D_?usp=sharing&fbclid=IwY2xjawMnzqtleHRuA2FlbQIxMABicmlkETFXSWhybzI3Vm9mOUlrZXdoAR4WfltLijcuhopN65Exr30ig5YICDgSSATH5U8t17jY3LzjmSbHzDoIuIHl8g_aem_8Tsq_I5Mv2UBxMLYwIqjgA#scrollTo=XIawcFsnrLJ2

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

# 🎯 Sprint 1 : สัปดาห์ที่ 1 - การสร้างรากฐาน (Foundation) 

**เป้าหมายของสัปดาห์** : เตรียมเครื่องมือที่จำเป็น สร้างโครงสร้างหลักของโปรแกรม (Main Loop) ให้สามารถเปิดกล้องเว็บแคมและตรวจจับตัวบุคคลได้

**บทบาทสัปดาห์นี้** : 

Planner 🗺️ : พี

**ภารกิจหลัก** : กำหนด "ภาพรวม" และ "กฎเกณฑ์" ของสิ่งที่เราจะสร้างให้ชัดเจนที่สุด

**หน้าที่รับผิดชอบในสัปดาห์นี้** : หาข้อมูลและเลือกตัว model detection ว่าจะใช้ตัวใด พร้อมกับวางเป้าหมายของสัปดาห์นี่้ว่าควรจะทำถึงจุดใดเป็นขั้นต่ำ

Coder 💻 : นิคกี้

**ภารกิจหลัก** : เปลี่ยนแผนของพีให้กลายเป็นโค้ดที่ทำงานได้จริง

**หน้าที่รับผิดชอบในสัปดาห์นี้** : ทดลองเขียนโค้ดให้สามารถเปิดกล้องเว็บแคมได้และทดลองใช้ตัว model detection ที่พีเลือก

Debugger 🕵️‍♀️ : ภู

**ภารกิจหลัก** : ตรวจสอบคุณภาพและทำให้แน่ใจว่าโปรแกรมทำงานตรงตามแผนที่พีวางไว้

**หน้าที่รับผิดชอบในสัปดาห์นี้** : ทดลองใช้งานตัว model detection ว่ามีจุดด้อยหรือข้อจำกัดในการทำงานอย่างไรบ้าง

**นิยามของคำว่า "เสร็จ" (Definition of Done)**

- ตัวโปรแกรมสามารถเปิดกล้องและตรวจจับใบหน้าได้แบบเรียลไทม์

# 🎯 Sprint 2: สัปดาห์ที่ 2 - การสร้างรากฐาน (Foundation) 

**เป้าหมายของสัปดาห์** : เตรียมข้อมูลรูปภาพ (dataset) ที่จะใช้ในการทดลองตรวจจับ และทำการเขียนโค้ดโปรแกรมเพื่อทดลองตรวจจับใบหน้าตามชื่อและข้อมูลรูปภาพที่เตรียมไว้

**บทบาทสัปดาห์นี้** : 

Planner 🗺️ : พี

**ภารกิจหลัก** : กำหนด "ภาพรวม" และ "กฎเกณฑ์" ของสิ่งที่เราจะสร้างให้ชัดเจนที่สุด

**หน้าที่รับผิดชอบในสัปดาห์นี้** : กำหนดรูปแบบของข้อมูลรูปภาพ (dataset) ที่จะใช้ในการทดลองตรวจจับว่าจะใช้รูปแบบใด กี่รูปบ้างในการตรวจจับให้ผลลัพธ์ออกมามีประสิทธิภาพที่สุด

Coder 💻 : นิคกี้

**ภารกิจหลัก** : เปลี่ยนแผนของพีให้กลายเป็นโค้ดที่ทำงานได้จริง

**หน้าที่รับผิดชอบในสัปดาห์นี้** : ทดลองเขียนโค้ดให้สามารถตตรวจจับใบหน้าและแสดงชื่อของบุคคลได้ตามข้อมูลรูปภาพ (dataset) ที่เตรียมไว้

Debugger 🕵️‍♀️ : ภู

**ภารกิจหลัก** : ตรวจสอบคุณภาพและทำให้แน่ใจว่าโปรแกรมทำงานตรงตามแผนที่พีวางไว้

**หน้าที่รับผิดชอบในสัปดาห์นี้** : ทดลองใช้งานตัว model detection จากข้อมูลรูปภาพ (dataset) ที่เตรียมไว้ ว่ามีจุดด้อยหรือข้อจำกัดอย่างไรบ้าง

**นิยามของคำว่า "เสร็จ" (Definition of Done)**

- ตัวโปรแกรมสามารถตรวจจับและแยกใบหน้าของผู้ใช้แต่ละคนออกพร้อมแสดงชื่อได้อย่างแม่นยำ


## 🗓️ Week 3:

### เป้าหมาย
- เขียนแก้ไฟล์หลักที่ต้องใช้จริงๆ ให้สมบูรณ์
- บันทึกการเข้าชั้นเรียนลงไฟล์ `attendance.csv`
- เขียนโค้ดแสดง dashbord

### การแบ่งหน้าที่
- **พี (Coder)**:
  - เขียน html ในไฟล์ index.html
  - เขียน app.py เพื่อดึงข้อมูลจากไฟล์ attendance.csv ไปแสดงหน้าเว็บ
  - เขียนโค้ดไฟล์ attendance.py เป็นเวอร์ชันสมบูรณ์แบบ
- **นิกกี้ (Planner)**:
  - กำหนดแผนการว่าเราจำเอาข้อมูลที่ได้ไปแสดงให้คนอื่นเห็นได้ยังไงจึงเสนอให้เขียนเว็บ dashbord
  - เขียนโครงสร้างโฟล์เดอร์โปรเจคขึ้นมาใหม่
    ```
    project/
    ├─ dataset/
      ├─Worachat
        └─Worachat1.jpg
      ├─Krit
        └─Krit1.jpg
      ├─Phuchit
        └─Phuchit1.jpg
    ├─ templates/
      └─index.html
    ├─ venv/
    ├─app.py
    ├─attendance.csv
    ├─attendance.py
    ├─camera.py
    ├─dlib_face_recognition_resnet_model_v1.dat
    ├─face_detection.py
    ├─face_recognition.py
    ├─landmarks.py
    ├─main.py
    ├─requirements.txt
    ├─shape_predictor_68_face_landmarks.dat
    └─utils.py
    ```
- **ภู (Debugger)**:
  - ทดสอบระบบโค้ด html และ app.py ว่าใช้งานได้มั้ย
  - ทดสอบ เปิดกล้อง แล้วสังเกตว่ารายชื่อถูกบันทึกไว้ที่ attendance.csv มั้ย
  - แก้บั๊ก เมื่อแสดงรายชื่อซ้ำมากกว่า 1 รอบ
