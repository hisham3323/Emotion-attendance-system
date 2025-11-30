# Emotion-attendance-system
An emotion-based facial recognition attendance system using deep learning.

This project is an AI-powered attendance system that uses real-time **facial emotion recognition** and **face identification** to monitor and record student attendance. It integrates computer vision, deep learning, and a user-friendly web interface.

---

##  Features

- 🎭 Emotion recognition using a trained deep learning model
- 👤 Face recognition & student database
- 📸 Real-time webcam capture
- 📊 Auto attendance logging to CSV
- 🌐 Web interface (Flask) for emotion visualization and monitoring
- 🧠 CPU-based real-time processing

---

## Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- face-recognition
- Flask (web server)
- SQLite (student face database)
- HTML/CSS/JS (frontend)

---

##  Project Structure

```bash
emotion-attendance/
│
├── emotion.py              # Main application logic
├── requirements.txt        # Project dependencies
├── face_model.h5           # Trained emotion recognition model
├── face_recognition.db     # SQLite DB with known faces
├── monitordb.py / .ipynb   # Monitor DB entries visually
├── templates/              # Flask HTML templates
├── static/                 # CSS and JS assets
├── *.csv                   # Auto-generated attendance logs
└── context.md              # Additional context & notes
```

---
## How to Run:
1. **Clone the repository:**
   ```bash
   git clone https://github.com/hisham3323/Emotion-attendance-system.git
   cd Emotion-attendance-system
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python emotion.py
   ```
   > Make sure your webcam/front facing camera is connected and functional.
---
#License
This project is licensed under the APACHE License. See the LICENSE file for details.
#Contributing
Pull requests and feedback are welcome! If you'd like to improve the model or frontend, feel free to fork and submit a PR.

Author
Mohammad Hisham
github.com/hisham3323


