# 🎯 Face Recognition Attendance System (Flask)

A **Smart Attendance System** that uses **face recognition** to automatically mark attendance and prevent proxy attendance.  
The system captures a face using a webcam, matches it with registered students, records attendance **only once per day**, and also detects **facial sentiment (emotion)**.

---

## 🚀 Features

- 👤 Student registration using face image
- 📸 Real-time face recognition via webcam
- ❌ Prevents proxy / duplicate attendance
- 📅 One attendance per student per day
- 😊 Facial sentiment detection (Happy, Sad, Angry, Neutral, etc.)
- 🌗 Dark Mode / Light Mode UI
- 📊 Live attendance dashboard
- 💾 Data stored locally in JSON files
- ⚡ Fast & lightweight (no heavy ML models)

---

## 🛠️ Tech Stack

### Backend
- Python 3
- Flask
- NumPy
- Pillow (PIL)

### Frontend
- HTML5
- CSS3
- JavaScript
- Webcam API

### Data Storage
- JSON files (No database required)

---

## 📁 Project Structure

face-recognition-attendance-system/
│
├── app.py # Flask backend
├── students.json # Registered students data
├── attendance_history.json # Attendance logs
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── templates/
│ └── index.html # Frontend UI
│
├── static/
│ ├── css/
│ │ └── style.css # UI styling
│ └── js/
│ └── script.js # Frontend logic
│
└── .gitignore




---

## ⚙️ How the System Works (Simple Explanation)

1. **Student Registration**
   - Student enters ID and Name
   - Face image is captured
   - Face features are extracted and saved

2. **Attendance Process**
   - Webcam captures live frame
   - Face is detected from center of image
   - Extracted face features are compared
   - If match is above strict threshold → attendance marked

3. **Anti-Proxy Logic**
   - Same student cannot mark attendance twice in one day
   - Strict similarity threshold avoids fake matches

4. **Sentiment Detection**
   - Brightness, contrast, symmetry, and edges are analyzed
   - Emotion like Happy / Sad / Angry is detected
   - Stored with attendance record

---

## 🧠 Face Recognition Logic (Important)

- Uses **histogram-based face feature extraction**
- Compares faces using **Chi-Square similarity**
- Very **strict similarity threshold** to avoid proxy
- No external AI models → works offline

---

## ▶️ How to Run the Project

### 1️⃣ Install Python
Download Python from:

https://www.python.org/downloads/


✔ Make sure **“Add Python to PATH”** is checked

---

### 2️⃣ Install Dependencies
Open terminal in project folder and run:
```bash
pip install -r requirements.txt

### Run the Application
python app.py

### Open in Browser
http://127.0.0.1:5000


📊 API Endpoints

| Endpoint                      | Method | Description            |
| ----------------------------- | ------ | ---------------------- |
| `/api/register_student`       | POST   | Register a new student |
| `/api/process_frame`          | POST   | Process webcam frame   |
| `/api/get_students`           | GET    | Get all students       |
| `/api/get_today_attendance`   | GET    | Today's attendance     |
| `/api/clear_today_attendance` | POST   | Clear today’s records  |
| `/api/clear_all_students`     | POST   | Remove all students    |


👨‍💻 Author

Vikrant Jadhav
Face Recognition Attendance System Project
