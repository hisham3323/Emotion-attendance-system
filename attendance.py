from flask import Flask, render_template, Response, jsonify, request
import cv2
import face_recognition
import sqlite3
import numpy as np
import pickle
import csv
import os
from datetime import datetime

app = Flask(__name__)
db_path = 'face_recognition.db'
present_users = set()
all_user_names = set()
video_capture = None
attendance_date = ""
last_saved_csv = ""

def load_users_from_db():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT name, face_encoding FROM users")
    users = {}
    for row in c.fetchall():
        name, encoding_blob = row
        encoding = pickle.loads(encoding_blob)
        users[name] = encoding
    conn.close()
    return users

known_users = load_users_from_db()
all_user_names = set(known_users.keys())

def gen_frames():
    global video_capture, present_users

    video_capture = cv2.VideoCapture(0)
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            name = "Unknown"
            best_match = None
            lowest_distance = 0.4

            for user_name, user_encoding in known_users.items():
                distance = face_recognition.face_distance([user_encoding], encoding)[0]
                if distance < lowest_distance:
                    lowest_distance = distance
                    best_match = user_name

            if best_match:
                name = best_match
                present_users.add(name)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_attendance', methods=['POST'])
def stop_attendance():
    global last_saved_csv, attendance_date
    attendance_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    desktop = os.path.expanduser("~/Desktop")
    filename = os.path.join(desktop, f"attendance_{attendance_date}.csv")
    last_saved_csv = filename

    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Status", "Timestamp"])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for user in all_user_names:
                status = "Present" if user in present_users else "Absent"
                writer.writerow([user, status, timestamp])

        return jsonify({"message": "Attendance saved!", "file": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
