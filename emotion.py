# ---------------------------- IMPORTS ---------------------------------------
from flask import Flask, render_template, request, jsonify, Response
import cv2, base64, numpy as np, os, pickle, sqlite3, csv, smtplib
from datetime import datetime
from email.message import EmailMessage

import face_recognition
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
# ---------------------------------------------------------------------------

app = Flask(__name__)

# ---------------- Global CONFIG --------------------------------------------
#   tweak these two for FPS vs. accuracy trade-off
SCALE             = 0.50   # 0.5 ⇒ 640×360 if camera is 1280×720
PROCESS_EVERY_N   = 2      # use 1 for max accuracy / more GPU load
# ---------------------------------------------------------------------------

# ---------------- Models & Constants ---------------------------------------
model        = load_model('face_model.h5', compile=False)
class_names  = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')

DB_PATH      = 'face_recognition.db'
CSV_DIR      = os.path.abspath(".")
CSV_HEADERS  = ['name', 'age', 'email', 'timestamp']

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")         # e.g. your_email@gmail.com
SMTP_PASS = os.getenv("SMTP_PASS")         # app-password / mail password

video_capture    = None
known_encodings  = {}
present_users    = set()
all_user_names   = set()
attendance_saved = False
# ---------------------------------------------------------------------------


# ---------------- Utility helpers ------------------------------------------
def init_camera():
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)

def init_db():
    """Create tables if they don’t yet exist."""
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users(
                name TEXT PRIMARY KEY,
                age  INTEGER,
                email TEXT,
                face_encoding BLOB
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS attendance(
                name TEXT,
                timestamp TEXT
            )""")
        conn.commit()

def load_users_from_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        rows = c.execute("SELECT name, face_encoding FROM users").fetchall()
        return {n: pickle.loads(enc) for n, enc in rows}

def write_csv(present):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(CSV_DIR, f"attendance_{ts}.csv")

    with sqlite3.connect(DB_PATH) as conn, open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f);  writer.writerow(CSV_HEADERS)
        c = conn.cursor()
        for name in present:
            age, email = c.execute(
                "SELECT age, email FROM users WHERE name=?", (name,)
            ).fetchone()
            writer.writerow([name, age, email, ts])

def send_email(to_addr: str, subj: str, body: str):
    if not (SMTP_USER and SMTP_PASS):
        print("[WARN] SMTP credentials not configured – e-mails skipped.")
        return
    msg = EmailMessage();  msg["Subject"]=subj; msg["From"]=SMTP_USER; msg["To"]=to_addr
    msg.set_content(body)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls(); server.login(SMTP_USER, SMTP_PASS); server.send_message(msg)

def notify_present_users(present):
    ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        for name in present:
            row = c.execute("SELECT email FROM users WHERE name=?", (name,)).fetchone()
            if row and row[0]:
                send_email(
                    row[0], "Attendance Recorded",
                    f"Hello {name},\nYour attendance was recorded at {ts_human}.\n\nRegards,\nAI Attendance System"
                )
# ---------------------------------------------------------------------------


# ------------------------- ROUTES ------------------------------------------
@app.route("/")
def dashboard():
    return render_template("index.html")


# --------------- Emotion & Attendance System ---------------
@app.route("/predict", methods=["POST"])
def predict():
    """Receives an image, performs emotion prediction, and returns the result."""
    try:
        data_url = request.json["image"]
        encoded = data_url.split(",")[1]
        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR
        )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        emotion = "No Face"
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = cv2.resize(gray[y : y + h, x : x + w], (48, 48))
            face = img_to_array(face)[None, ...]
            emotion = class_names[np.argmax(model.predict(face))]

        return jsonify({"emotion": emotion})
    except Exception as e:
        return jsonify({"error": str(e)})


def video_frames():
    """Generator that yields JPEG frames for video streaming."""
    global present_users, all_user_names, attendance_saved
    init_camera()

    present_users.clear()
    encodes = load_users_from_db()
    all_user_names = set(encodes.keys())
    attendance_saved = False

    frame_id = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize and convert to RGB for face recognition
        rgb_small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
        rgb_small = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB)

        # Process every Nth frame to save resources
        if frame_id % PROCESS_EVERY_N == 0:
            # Detect faces and recognize them
            locs = face_recognition.face_locations(rgb_small, model="hog")
            encs = face_recognition.face_encodings(rgb_small, locs)
            matches = []

            for enc in encs:
                name = "Unknown"
                # Compare against known faces
                distances = face_recognition.face_distance(list(encodes.values()), enc)
                if len(distances) > 0:
                    best_match_idx = np.argmin(distances)
                    if distances[best_match_idx] < 0.4:
                        name = list(encodes.keys())[best_match_idx]
                        present_users.add(name)
                matches.append(name)

            # Draw bounding boxes and names on the original frame
            for (top, right, bottom, left), name in zip(locs, matches):
                top, right, bottom, left = (
                    int(v / SCALE) for v in (top, right, bottom, left)
                )
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        # Encode frame as JPEG and yield it
        _, buf = cv2.imencode(".jpg", frame)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        frame_id += 1


@app.route("/attendance")
def attendance_page():
    """Serves the main attendance monitoring page."""
    return render_template("attendance.html")

@app.route("/video_feed")
def video_feed():
    return Response(video_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stop_attendance", methods=["POST"])
def stop_attendance():
    global attendance_saved
    if attendance_saved:
        return jsonify({"message": "Attendance already saved."})

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            for name in present_users:
                c.execute("INSERT INTO attendance VALUES(?,?)", (name, ts))
            conn.commit()

        write_csv(present_users)
        notify_present_users(present_users)

        attendance_saved = True
        return jsonify({"message": "Attendance saved successfully (DB + CSV) and e-mails sent."})
    except Exception as e:
        return jsonify({"error": str(e)})


# --------------- Registration ---------------------
@app.route("/register_user", methods=["POST"])
def register_user():
    name  = request.form.get("name", "").strip()
    age   = request.form.get("age", "").strip()
    email = request.form.get("email", "").strip()

    if not (name and age and email):
        return jsonify({"error": "Name, age and e-mail are all required."})

    init_camera()
    success, frame = video_capture.read()
    if not success:
        return jsonify({"error": "Could not access camera."})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(rgb)
    if not enc:
        return jsonify({"error": "No face detected, try again."})

    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""INSERT OR REPLACE INTO users
                         VALUES (?,?,?,?)""",
                      (name, int(age), email, pickle.dumps(enc[0])))
            conn.commit()
        return jsonify({"message": f"{name} registered successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)})


# ---------------- MAIN ----------------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
