"""
Emotion Recognition - Webcam en temps réel
Utilise le modèle CNN entraîné + OpenCV + Haar Cascade (détection visage)
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import time
import sys
import os

# Importer le modèle depuis train.py
sys.path.append(os.path.dirname(__file__))
from CNN.CNN import EmotionCNN


# ─────────────────────────────────────────────
# CONFIG WEBCAM
# ─────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_COLORS = {
    "angry":    (0, 0, 255),       # Rouge
    "disgust":  (0, 140, 255),     # Orange
    "fear":     (128, 0, 128),     # Violet
    "happy":    (0, 255, 0),       # Vert
    "neutral":  (200, 200, 200),   # Gris
    "sad":      (255, 100, 0),     # Bleu
    "surprise": (0, 255, 255),     # Cyan
}
EMOTION_EMOJIS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲"
}

MODEL_PATH = "./models/best_model.pth"
IMG_SIZE = 48


# ─────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ─────────────────────────────────────────────
def load_model(model_path: str, device: str):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get("classes", EMOTIONS)
    num_classes = len(classes)

    model = EmotionCNN(num_classes=num_classes, dropout=0.0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f" Modèle chargé: {model_path}")
    print(f"   Classes: {classes}")
    print(f"   Accuracy val: {checkpoint.get('val_acc', 'N/A'):.4f}")
    return model, classes


# ─────────────────────────────────────────────
# PRÉTRAITEMENT FRAME
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def preprocess_face(face_roi: np.ndarray) -> torch.Tensor:
    """Convertit une ROI visage (BGR) en tensor normalisé."""
    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    tensor = transform(face_rgb).unsqueeze(0)
    return tensor


# ─────────────────────────────────────────────
# DESSIN UI
# ─────────────────────────────────────────────
def draw_emotion_bar(frame, x, y, w, probabilities, classes, emotion, color):
    """Affiche un panneau avec les probabilités pour chaque émotion."""
    bar_x = x + w + 10
    bar_y = y
    bar_h = 20
    bar_gap = 5
    bar_max_w = 150
    panel_w = 230
    panel_h = len(classes) * (bar_h + bar_gap) + 20

    # Fond semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_x - 5, bar_y - 5),
                  (bar_x + panel_w, bar_y + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for i, (cls, prob) in enumerate(zip(classes, probabilities)):
        cy = bar_y + i * (bar_h + bar_gap) + 10
        bar_fill = int(prob * bar_max_w)
        bar_color = EMOTION_COLORS.get(cls, (150, 150, 150))

        # Barre de fond
        cv2.rectangle(frame, (bar_x, cy), (bar_x + bar_max_w, cy + bar_h),
                      (60, 60, 60), -1)
        # Barre remplie
        if bar_fill > 0:
            cv2.rectangle(frame, (bar_x, cy), (bar_x + bar_fill, cy + bar_h),
                          bar_color, -1)
        # Label
        label = f"{cls}: {prob:.0%}"
        cv2.putText(frame, label, (bar_x + bar_max_w + 5, cy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)


def draw_face_box(frame, x, y, w, h, emotion, confidence, color):
    """Dessine un rectangle stylisé autour du visage."""
    thickness = 2
    corner_len = 20

    # Coins stylisés
    for cx, cy in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
        dx = corner_len if cx == x else -corner_len
        dy = corner_len if cy == y else -corner_len
        cv2.line(frame, (cx, cy), (cx + dx, cy), color, thickness + 1)
        cv2.line(frame, (cx, cy), (cx, cy + dy), color, thickness + 1)

    # Rectangle principal
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)

    # Label émotion
    label = f"{emotion.upper()} {confidence:.0%}"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    cv2.rectangle(frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)


def draw_fps_stats(frame, fps, face_count, dominant_emotion):
    """Affiche les stats en haut à gauche."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (280, 85), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)
    cv2.putText(frame, f"Visages: {face_count}", (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 100), 2)
    if dominant_emotion:
        cv2.putText(frame, f"Dominant: {dominant_emotion}", (12, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)


# ─────────────────────────────────────────────
# BOUCLE PRINCIPALE WEBCAM
# ─────────────────────────────────────────────
def run_webcam(model, classes, device, camera_id=0):
    # Haar Cascade pour détection visage
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(f" Impossible d'ouvrir la caméra {camera_id}")

    print(f"\n Webcam démarrée (caméra {camera_id})")
    print("   Appuyez sur 'q' pour quitter | 's' pour screenshot")
    print()

    fps_counter = 0
    fps_display = 0.0
    fps_timer = time.time()
    emotion_history = []  # pour statistiques session

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Miroir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détection visages
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE
        )

        dominant_emotion = None

        for (x, y, w, h) in faces:
            # Marge autour du visage
            margin = int(0.1 * min(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            # Inférence
            with torch.no_grad():
                tensor = preprocess_face(face_roi).to(device)
                output = model(tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]

            emotion_idx = np.argmax(probs)
            emotion = classes[emotion_idx]
            confidence = probs[emotion_idx]
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))

            dominant_emotion = emotion
            emotion_history.append(emotion)

            # Dessin
            draw_face_box(frame, x1, y1, x2-x1, y2-y1, emotion, confidence, color)
            draw_emotion_bar(frame, x1, y1, x2-x1, list(probs), classes, emotion, color)

        # FPS
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()

        draw_fps_stats(frame, fps_display, len(faces), dominant_emotion)

        # Aide
        cv2.putText(frame, "Q: Quitter | S: Screenshot", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow("Emotion Recognition - CNN", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = f"screenshot_{ts}.jpg"
            cv2.imwrite(path, frame)
            print(f" Screenshot sauvegardé: {path}")

    cap.release()
    cv2.destroyAllWindows()

    # Stats de session
    if emotion_history:
        print("\n STATISTIQUES DE SESSION:")
        from collections import Counter
        counts = Counter(emotion_history)
        total = sum(counts.values())
        for emotion, count in counts.most_common():
            print(f"  {emotion:10s}: {count:4d} détections ({count/total:.1%})")
        print(f"  TOTAL       : {total} détections")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Emotion Recognition - Webcam")
    parser.add_argument("--model", default=MODEL_PATH, help="Chemin vers best_model.pth")
    parser.add_argument("--camera", type=int, default=0, help="ID caméra (défaut: 0)")
    parser.add_argument("--device", default="auto", help="cpu / cuda / auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(" Emotion Recognition - Webcam Temps Réel")
    print("=" * 50)
    print(f"   Device: {device.upper()}")

    model, classes = load_model(args.model, device)
    run_webcam(model, classes, device, camera_id=args.camera)
