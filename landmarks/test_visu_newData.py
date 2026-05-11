import cv2
import torch
import torch.nn as nn
import numpy as np
import joblib
import mediapipe as mp

# =========================================================
# DEVICE
# =========================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# =========================================================
# LOAD CHECKPOINT
# =========================================================
checkpoint = torch.load("mlp_best.pt", map_location=device)

input_dim = checkpoint["input_dim"]
n_classes = checkpoint["num_classes"]

print("Input shape:", input_dim)
print("Classes:", n_classes)

# =========================================================
# LOAD SCALER + LABELS
# =========================================================
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# =========================================================
# MODEL
# =========================================================
class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)

# =========================================================
# LOAD WEIGHTS
# =========================================================
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Model loaded successfully")

# =========================================================
# MEDIAPIPE
# =========================================================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# =========================================================
# CAMERA
# =========================================================
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

print("Camera started")
print("Press Q to quit")

# =========================================================
# MAIN LOOP
# =========================================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    text = "No face"

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        coords = []

        for lm in landmarks:
            coords.extend([lm.x, lm.y])

        features = np.array(coords).reshape(1, -1)

        # =================================================
        # SCALE
        # =================================================
        features = scaler.transform(features)

        x = torch.tensor(features, dtype=torch.float32).to(device)

        # =================================================
        # PREDICT
        # =================================================
        with torch.no_grad():

            outputs = model(x)

            probs = torch.softmax(outputs, dim=1)

            pred = torch.argmax(probs, dim=1).item()

            confidence = torch.max(probs).item()

            emotion = label_encoder.inverse_transform([pred])[0]

            text = f"{emotion} ({confidence:.2f})"

        # =================================================
        # DRAW LANDMARKS
        # =================================================
        h, w, _ = frame.shape

        for lm in landmarks:

            x_lm = int(lm.x * w)
            y_lm = int(lm.y * h)

            cv2.circle(frame, (x_lm, y_lm), 1, (0, 255, 0), -1)

    # =====================================================
    # DISPLAY
    # =====================================================
    cv2.putText(
        frame,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Emotion Recognition", frame)

    # =====================================================
    # QUIT
    # =====================================================
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================================================
# CLEANUP
# =========================================================
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
