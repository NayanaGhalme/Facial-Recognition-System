import os
import pickle
import numpy as np
from PIL import Image
from numpy.linalg import norm
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# model setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

project_path = "dataset/lost"

face_database = {}

for person in os.listdir(project_path):
    person_path = os.path.join(project_path, person)

    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = Image.open(img_path).convert("RGB")
        face = mtcnn(img)

        if face is None:
            continue

        face = face.unsqueeze(0).to(device)
        emb = model(face).detach().cpu().numpy()[0]
        emb = emb / norm(emb)

        embeddings.append(emb)

    if embeddings:
        face_database[person] = np.mean(embeddings, axis=0)

with open("face_database.pkl", "wb") as f:
    pickle.dump(face_database, f)

print("✅ New FaceNet database created!")
