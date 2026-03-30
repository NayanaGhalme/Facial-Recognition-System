from database import init_db
from flask import Flask, render_template, request, redirect, url_for, session
import os
import subprocess
import subprocess
import sys
import json
import smtplib
from email.mime.text import MIMEText
from flask import redirect, url_for
import cv2
import pickle
import numpy as np
from numpy.linalg import norm
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
import insightface
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

app = Flask(__name__)
app.secret_key = os.urandom(24)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
init_db()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MODEL INITIALIZATION
# ---------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

arcface = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
arcface.prepare(ctx_id=-1)
class User(UserMixin):
    def __init__(self, id, name, email):
        self.id = id
        self.name = name
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user:
        return User(user[0], user[1], user[2])
    return None

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = generate_password_hash(request.form["password"])

        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO users (name, email, phone, password_hash)
                VALUES (?, ?, ?, ?)
            """, (name, email, phone, password))
            conn.commit()
        except:
            return "Email already registered"

        conn.close()
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("app.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, password_hash FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            login_user(User(user[0], user[1], user[2]))
            return redirect(url_for("dashboard"))

        else:
            return render_template("login.html", error="Invalid email or password")  

    # VERY IMPORTANT (for GET request)
    return render_template("login.html")  
    

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", name=current_user.name)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/register_lost", methods=["POST"])
@login_required
def register_lost():

    name = request.form["name"]
    age = request.form["age"]
    file = request.files.get("photo")

    if not file:
        return "No file uploaded"

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    img = cv2.imread(save_path)
    faces = arcface.get(img)

    if len(faces) == 0:
        return "No face detected"

    face = faces[0]
    arc_emb = face.embedding / (norm(face.embedding) + 1e-6)

    x1, y1, x2, y2 = face.bbox.astype(int)
    face_crop = img[y1:y2, x1:x2]

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb).resize((160, 160))
    face_tensor = torch.tensor(np.array(face_pil)).permute(2,0,1).float().to(device)
    face_tensor = (face_tensor - 127.5) / 128.0

    with torch.no_grad():
        fn_emb = facenet(face_tensor.unsqueeze(0)).cpu().numpy()[0]

    fn_emb /= (norm(fn_emb) + 1e-6)

    hybrid_emb = np.concatenate([fn_emb, arc_emb])
    hybrid_emb /= (norm(hybrid_emb) + 1e-6)

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO lost_persons (user_id, name, age, photo_path, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, (
        current_user.id,
        name,
        age,
        save_path,
        pickle.dumps(hybrid_emb)
    ))

    conn.commit()
    conn.close()

    return "Lost person registered successfully"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["POST"])
@login_required
def upload():

    print("Upload route triggered")   

    file = request.files.get("photo")
    if not file or file.filename == "":
        return "❌ No file selected"

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    print("Image saved at:", save_path)  

    # Read image
    img = cv2.imread(save_path)

    faces = arcface.get(img)

    print("Faces detected:", len(faces))  

    if len(faces) == 0:
        return "❌ No face detected in uploaded image"

    face = faces[0]
    arc_emb = face.embedding / (norm(face.embedding) + 1e-6)

    # FaceNet embedding
    x1, y1, x2, y2 = face.bbox.astype(int)
    face_crop = img[y1:y2, x1:x2]

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb).resize((160, 160))
    face_tensor = torch.tensor(np.array(face_pil)).permute(2,0,1).float().to(device)
    face_tensor = (face_tensor - 127.5) / 128.0

    with torch.no_grad():
        fn_emb = facenet(face_tensor.unsqueeze(0)).cpu().numpy()[0]

    fn_emb /= (norm(fn_emb) + 1e-6)

    hybrid_emb = np.concatenate([fn_emb, arc_emb])
    hybrid_emb /= (norm(hybrid_emb) + 1e-6)

    # Save active target
    with open("active_target.pkl", "wb") as f:
        pickle.dump(hybrid_emb, f)

    print("Active target saved successfully")  

    return "✅ Target Registered Successfully"



@app.route("/start_detection", methods=["POST"])
@login_required
def start_detection():

    user_id = current_user.id
    script_path = os.path.join(os.getcwd(), "video_detection_hybrid.py")

    python_path = sys.executable   

    try:
        subprocess.Popen(
            ["cmd", "/k", python_path, script_path, str(user_id)],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        print("Detection launched")
    except Exception as e:
        print("Error:", e)

    return redirect(url_for("results"))

@app.route("/results")
@login_required
def results():

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT fp.id, fp.confidence, fp.image_path, fp.timestamp, lp.name
        FROM found_persons fp
        JOIN lost_persons lp ON fp.lost_person_id = lp.id
        ORDER BY fp.timestamp DESC
    """)

    detections = cursor.fetchall()
    conn.close()

    return render_template("results.html", detections=detections)



@app.route("/api/detections")
@login_required
def api_detections():

    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        lp.name,
        fp.confidence,
        fp.image_path,
        fp.timestamp
    FROM found_persons fp
    JOIN lost_persons lp
        ON fp.lost_person_id = lp.id
    WHERE lp.user_id = ?
    ORDER BY fp.timestamp DESC
""", (current_user.id,))

    rows = cursor.fetchall()
    conn.close()

    detections = []

    for name, confidence, image_path, timestamp in rows:

        filename = os.path.basename(image_path)

        detections.append({
            "name": name,
            "confidence": round(confidence, 2),
            "time": timestamp,
            "image_url": f"logs/{filename}"
        })

    return {"detections": detections}



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
