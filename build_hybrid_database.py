import os
import pickle
import cv2
import numpy as np
from PIL import Image
from numpy.linalg import norm
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import insightface

# --------------------------------------------------
# 1️⃣ SETUP
# --------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# FaceNet
mtcnn = MTCNN(image_size=160, margin=30, device=device)
facenet_model = InceptionResnetV1(
    pretrained='vggface2'
).eval().to(device)

# ArcFace
arcface = insightface.app.FaceAnalysis()
arcface.prepare(ctx_id=0)

print("✅ FaceNet + ArcFace loaded")

# --------------------------------------------------
# 2️⃣ DATASET PATH
# --------------------------------------------------
dataset_path = r"C:\Users\jadha\Downloads\Facial_Recognition_System\dataset\lost"

   # same as your old project

face_database = {}

# --------------------------------------------------
# 3️⃣ HYBRID EMBEDDING FUNCTION
# --------------------------------------------------
def get_hybrid_embedding(img_pil):

    # ---------- FaceNet ----------
    face = mtcnn(img_pil)
    if face is None:
        return None

    face = face.float().unsqueeze(0).to(device)

    facenet_emb = facenet_model(face).detach().cpu().numpy()[0]
    facenet_emb = facenet_emb / (norm(facenet_emb) + 1e-6)

    # ---------- ArcFace ----------
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    faces = arcface.get(img_cv)

    if len(faces) == 0:
        return None

    arcface_emb = faces[0].embedding
    arcface_emb = arcface_emb / (norm(arcface_emb) + 1e-6)

    # ---------- Hybrid ----------
    hybrid_emb = np.concatenate([facenet_emb, arcface_emb])

    return hybrid_emb


# --------------------------------------------------
# 4️⃣ BUILD DATABASE
# --------------------------------------------------
for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    embeddings = []

    for img_name in os.listdir(person_folder):

        img_path = os.path.join(person_folder, img_name)

        try:
            img = Image.open(img_path).convert("RGB")

            emb = get_hybrid_embedding(img)

            if emb is not None:
                embeddings.append(emb)

        except:
            continue

    if len(embeddings) > 0:
        avg_emb = np.mean(embeddings, axis=0)
        avg_emb = avg_emb / (norm(avg_emb) + 1e-6)

        face_database[person_name] = avg_emb
        print(f"Added: {person_name}")

# --------------------------------------------------
# 5️⃣ SAVE DATABASE
# --------------------------------------------------
with open("face_database_hybrid.pkl", "wb") as f:
    pickle.dump(face_database, f)

print("✅ Hybrid database saved")