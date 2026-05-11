import cv2

class Camera:

    def __init__(self, device_index: int = 0):
        self.cap = None
        self.device_index = device_index

    def start(self):
        self.cap = cv2.VideoCapture(self.device_index)

    def set_device_index(self, device_index: int):
        self.device_index = device_index

    def read(self):
        if self.cap is None:
            return None, None

        ret, frame = self.cap.read()
        if not ret:
            return None, None

        return frame, True

    def get_info(self):
        if self.cap is None:
            return {
                "device": f"Caméra {self.device_index}",
                "fps": None,
                "width": None,
                "height": None,
                "backend": None,
                "is_opened": False,
            }

        try:
            backend = self.cap.getBackendName()
        except Exception:
            backend = None

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        def _norm_number(x):
            try:
                if x is None:
                    return None
                xf = float(x)
                if xf <= 0:
                    return None
                if abs(xf - round(xf)) < 1e-6:
                    return int(round(xf))
                return round(xf, 2)
            except Exception:
                return None

        return {
            "device": f"Caméra {self.device_index}",
            "fps": _norm_number(fps),
            "width": _norm_number(width),
            "height": _norm_number(height),
            "backend": backend,
            "is_opened": bool(self.cap.isOpened()),
        }

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    @staticmethod
    def detect_available_devices(max_devices: int = 10, start_index: int = 0):
        """
        Retourne la liste des indices de caméras détectées.
        Méthode simple basée sur l'ouverture de `cv2.VideoCapture`.
        """
        available = []

        for i in range(start_index, start_index + max_devices):
            cap = None
            try:
                # Sur Windows, CAP_DSHOW est souvent plus rapide/stable.
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # type: ignore[attr-defined]
                except Exception:
                    cap = cv2.VideoCapture(i)

                if cap is not None and cap.isOpened():
                    available.append(i)
            except Exception:
                pass
            finally:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass

        return available
