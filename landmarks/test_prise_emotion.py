import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

#face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

#camera mac À CHANGER POUR WINDOWS JE PENSE

print("Camera started. Press keys to label emotions:")
print("h=happy s=sad a=angry n=neutral f=fear d=disgust u=surprise q=quit")

data = []

#mapping du label
label_map = {
    ord('h'): "happy",
    ord('s'): "sad",
    ord('a'): "angry",
    ord('n'): "neutral",
    ord('f'): "fear",
    ord('d'): "disgust",
    ord('u'): "surprise"
}

#main
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # draw face (optional)
            for lm in face_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Dataset Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    # quit
    if key == ord('q'):
        break

    # save sample
    if key in label_map and results.multi_face_landmarks:

        label = label_map[key]

        face_landmarks = results.multi_face_landmarks[0]

        sample = []

        for lm in face_landmarks.landmark:
            sample.append(lm.x)
            sample.append(lm.y)

        sample.append(label)

        data.append(sample)

        print(f"Saved sample: {label} | total={len(data)}")

#sauvegarde csv

cap.release()
cv2.destroyAllWindows()

if len(data) > 0:
    columns = [f"x{i}" if i % 2 == 0 else f"y{i}" for i in range(len(data[0]) - 1)]
    columns.append("emotion")

    df = pd.DataFrame(data, columns=columns)
    df.to_csv("custom_emotion_dataset.csv", index=False)

    print("Saved dataset: custom_emotion_dataset.csv")
else:
    print("No data collected.")
