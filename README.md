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

สามารถติดตามในส่วนของ Sprint 1-3 ได้ที่ : https://colab.research.google.com/drive/1fchqs636mjrRhXy4D7V-IWOlRolXe1D_?usp=sharing
