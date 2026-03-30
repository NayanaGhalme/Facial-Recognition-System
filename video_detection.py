import cv2
import pickle
import numpy as np
from PIL import Image
from numpy.linalg import norm
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# --------------------------------------------------
# 1️⃣ SETUP
# --------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    image_size=160,
    margin=30,
    keep_all=True,
    min_face_size=40,
    device=device
)

model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --------------------------------------------------
# 2️⃣ LOAD DATABASE
# --------------------------------------------------
with open("face_database.pkl", "rb") as f:
    face_database = pickle.load(f)

print("✅ Database Loaded")


# --------------------------------------------------
# 3️⃣ HELPER FUNCTIONS
# --------------------------------------------------
def get_detection_and_embedding(img_pil):
    """
    Rotation-aware detection (helps horizontal faces)
    """
    angles = [0, 90, 270, 180]

    for angle in angles:

        rotated_img = img_pil.rotate(angle, expand=True) if angle != 0 else img_pil

        boxes, probs = mtcnn.detect(rotated_img)

        if boxes is not None:

            face_tensor = mtcnn(rotated_img)

            if face_tensor is not None:

                best_idx = np.argmax(probs)

                emb = model(
                    face_tensor[best_idx].unsqueeze(0).to(device)
                ).detach().cpu().numpy()[0]

                emb = emb / (norm(emb) + 1e-6)

                return boxes[best_idx], emb, angle

    return None, None, 0


def find_best_match(embedding, database, threshold=0.45):
    best_match, best_score = None, -1.0

    for name, db_emb in database.items():
        score = np.dot(
            embedding,
            db_emb / (norm(db_emb) + 1e-6)
        )

        if score > best_score:
            best_score = score
            best_match = name

    if best_score > threshold:
        return best_match, best_score

    return None, best_score


# --------------------------------------------------
# 4️⃣ VIDEO LOOP
# --------------------------------------------------
cap = cv2.VideoCapture("dataset/video/test_video2.mp4")
cv2.namedWindow("Missing Person Detection", cv2.WINDOW_NORMAL)

# 🔥 SMOOTH TRACKING VARIABLES
last_box = None
alpha = 0.7   # higher = smoother movement

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    box, embedding, angle = get_detection_and_embedding(img_pil)

    if box is not None:

        match_name, score = find_best_match(embedding, face_database)

        x1, y1, x2, y2 = map(int, box)

        # --------------------------------------------------
        # ⭐ SMOOTH BOX (THIS FIXES JUMPING)
        # --------------------------------------------------
        if last_box is not None:
            lx1, ly1, lx2, ly2 = last_box

            x1 = int(alpha * lx1 + (1 - alpha) * x1)
            y1 = int(alpha * ly1 + (1 - alpha) * y1)
            x2 = int(alpha * lx2 + (1 - alpha) * x2)
            y2 = int(alpha * ly2 + (1 - alpha) * y2)

        last_box = (x1, y1, x2, y2)

        # --------------------------------------------------
        # DRAW BOX
        # --------------------------------------------------
        color = (0, 0, 255) if match_name else (0, 255, 0)
        label = f"{match_name} ({score:.2f})" if match_name else "Unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

    cv2.imshow("Missing Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# --------------------------------------------------
# 5️⃣ CLEANUP
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()


