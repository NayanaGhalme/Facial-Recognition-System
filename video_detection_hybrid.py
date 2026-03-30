import os
import sys
import sqlite3
import cv2
import pickle
import torch
import numpy as np
import smtplib
from datetime import datetime
from PIL import Image
from numpy.linalg import norm
from email.mime.text import MIMEText
import insightface
from facenet_pytorch import InceptionResnetV1

# --------------------------------------------------
# GET CURRENT USER ID FROM FLASK
# --------------------------------------------------

current_user_id = int(sys.argv[1])

# --------------------------------------------------
# EMAIL FUNCTION
# --------------------------------------------------

def send_email_alert(to_email, person_name, location, detection_time):

    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASS")

    subject = "🚨 Missing Person Detected!"

    body = f"""
🚨 ALERT: Missing Person Detected

Name: {person_name}
📍 Location: {location}
🕒 Time: {detection_time}

Please log in to the system for more details.
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("📩 Email sent successfully.")
    except Exception as e:
        print("❌ Email failed:", e)

# --------------------------------------------------
# MODEL SETUP
# --------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
arcface = insightface.app.FaceAnalysis(providers=providers)
arcface.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))

# --------------------------------------------------
# LOAD LOST PERSONS FOR CURRENT USER
# --------------------------------------------------

conn = sqlite3.connect("app.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT id, embedding FROM lost_persons
    WHERE is_active = 1 AND user_id = ?
""", (current_user_id,))

lost_persons = cursor.fetchall()
conn.close()

if not lost_persons:
    print("No lost persons registered for this user.")
    exit()

print(f"Loaded {len(lost_persons)} lost persons.")
print("✅ System Ready.")

# --------------------------------------------------
# VIDEO SOURCE
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(BASE_DIR, "dataset", "video", "gate_test_video7.mp4")

video_filename = os.path.basename(video_path).lower()

if "gate" in video_filename:
    LOCATION_NAME = "VPKBIET College Entrance"
elif "parking" in video_filename:
    LOCATION_NAME = "Parking Area - ABC College"
elif "library" in video_filename:
    LOCATION_NAME = "Library Entrance - ABC College"
else:
    LOCATION_NAME = "Unknown Location"

print("📍 Location Set To:", LOCATION_NAME)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error opening video.")
    exit()

log_dir = os.path.join("static", "logs")
os.makedirs(log_dir, exist_ok=True)

# --------------------------------------------------
# EMBEDDING FUNCTION
# --------------------------------------------------

def get_facenet_embedding(face_crop_bgr):

    face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb).resize((160, 160))

    face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float().to(device)
    face_tensor = (face_tensor - 127.5) / 128.0

    with torch.no_grad():
        emb = facenet(face_tensor.unsqueeze(0)).cpu().numpy()[0]

    return emb / (norm(emb) + 1e-6)

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

print("🎬 Processing started... Press 'q' to stop.")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    faces = arcface.get(frame)

    for face in faces:

        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        arc_emb = face.embedding / (norm(face.embedding) + 1e-6)
        fn_emb = get_facenet_embedding(face_crop)

        hybrid_emb = np.concatenate([fn_emb, arc_emb])
        hybrid_emb /= (norm(hybrid_emb) + 1e-6)

        best_match_id = None
        max_score = 0

        for person_id, emb_blob in lost_persons:
            db_emb = pickle.loads(emb_blob)
            score = np.dot(hybrid_emb, db_emb.flatten())

            if score > max_score:
                max_score = score
                best_match_id = person_id

        threshold = 0.35
        color = (0, 0, 255)

        if max_score > threshold and best_match_id is not None:

            color = (0, 255, 0)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"found_{best_match_id}_{timestamp}.jpg"
            image_path = os.path.join(log_dir, img_name)

            cv2.imwrite(image_path, frame)

            print(">>> INSERTING INTO DATABASE <<<")

            conn = sqlite3.connect("app.db")
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO found_persons
                (lost_person_id, confidence, image_path, timestamp, location, is_notified)
                VALUES (?, ?, ?, datetime('now'), ?, 0)
            """, (best_match_id, float(max_score), image_path, LOCATION_NAME))

            conn.commit()

            print(f"Match found! Lost Person ID: {best_match_id}")

            cursor.execute("""
                SELECT u.email, lp.name
                FROM users u
                JOIN lost_persons lp ON lp.user_id = u.id
                WHERE lp.id = ?
            """, (best_match_id,))

            result = cursor.fetchone()

            if result:
                email, person_name = result

                cursor.execute("""
                    SELECT id FROM found_persons
                    WHERE lost_person_id = ? AND is_notified = 0
                    ORDER BY id DESC LIMIT 1
                """, (best_match_id,))

                row = cursor.fetchone()

                if row:
                    found_id = row[0]

                    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    send_email_alert(
                        email,
                        person_name,
                        LOCATION_NAME,
                        detection_time
                    )

                    cursor.execute("""
                        UPDATE found_persons
                        SET is_notified = 1
                        WHERE id = ?
                    """, (found_id,))

                    conn.commit()

            conn.close()

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                    f"ID:{best_match_id} ({max_score:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2)

    cv2.imshow("M-Secure Hybrid Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🏁 Finished.")