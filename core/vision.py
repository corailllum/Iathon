import cv2

class Vision:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def process(self, frame):

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        face_crop = None

        for (x, y, w, h) in faces:

            cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(rgb, "Calm",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            face_crop = rgb[y:y+h, x:x+w]
            break

        return rgb, face_crop