import importlib.util
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

_DEFAULT_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

_EMOTION_COLORS_BGR = {
    "angry": (0, 0, 255),
    "disgust": (0, 140, 255),
    "fear": (128, 0, 128),
    "happy": (0, 255, 0),
    "neutral": (200, 200, 200),
    "sad": (255, 100, 0),
    "surprise": (0, 255, 255),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


_FACE_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def _mean(seq) -> Optional[float]:
    if not seq:
        return None
    return float(sum(seq) / len(seq))


def _device_label(device: Any) -> str:
    if device is None:
        return ""
    s = str(device)
    return "GPU (CUDA)" if s.startswith("cuda") else "CPU"


class Vision:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self._metrics_lock = threading.Lock()
        self._process_times: deque[float] = deque(maxlen=45)
        self._infer_times: deque[float] = deque(maxlen=30)
        self._last_infer_ms: Optional[float] = None
        self._last_confidence: Optional[float] = None
        self._last_emotion: Optional[str] = None
        self._last_face_seen = False
        self._cnn_val_acc: Optional[float] = None
        self._cnn_offline_report: Optional[dict] = None

        self._torch: Optional[Any] = None
        self._torch_F: Optional[Any] = None

        self._cnn_lock = threading.Lock()
        self._cnn_ready = False
        self._cnn_failed = False
        self._cnn_error_msg = ""
        self._cnn_model = None
        self._cnn_classes: Optional[List[str]] = None
        self._cnn_device = None
        self._cnn_weights_path = _project_root() / "models" / "CNN" / "models" / "best_model.pth"
        self._cnn_transform: Optional[Any] = None

        self._tl_lock = threading.Lock()
        self._tl_ready = False
        self._tl_failed = False
        self._tl_error_msg = ""
        self._tl_model = None
        self._tl_classes: Optional[List[str]] = None
        self._tl_device = None
        self._tl_weights_path = _project_root() / "models" / "transfer_learning" / "best_model.pth"
        self._tl_transform: Optional[Any] = None

        self._lm_lock = threading.Lock()
        self._lm_ready = False
        self._lm_failed = False
        self._lm_error_msg = ""
        self._lm_face_mesh = None
        self._lm_model = None
        self._lm_scaler = None
        self._lm_label_encoder = None
        self._lm_device: Optional[str] = None
        self._lm_weights_path = _project_root() / "models" / "landmarks" / "mlp_best.pt"
        self._lm_scaler_path = _project_root() / "models" / "landmarks" / "scaler.joblib"
        self._lm_le_path = _project_root() / "models" / "landmarks" / "label_encoder.joblib"
        self._lm_task_model_path = _project_root() / "models" / "landmarks" / "face_landmarker.task"
        self._lm_use_tasks_api = False
        self._lm_landmarker = None
        self._lm_ts_ms = 0
        self._lm_input_dim: Optional[int] = None

        self._try_load_offline_cnn_report()

    def _try_load_offline_cnn_report(self) -> None:
        if self._cnn_offline_report is not None:
            return
        path = _project_root() / "models" / "CNN" / "results" / "results.json"
        try:
            with open(path, encoding="utf-8") as f:
                self._cnn_offline_report = json.load(f)
        except Exception:
            self._cnn_offline_report = {}

    def _import_torch_if_needed(self) -> bool:
        if self._torch is not None:
            return True
        try:
            import torch
            import torch.nn.functional as torch_nn_functional

            self._torch = torch
            self._torch_F = torch_nn_functional
            return True
        except ImportError:
            return False

    def _torch_load_file(self, path: Path, device: str) -> Any:
        torch = self._torch
        try:
            return torch.load(str(path), map_location=device, weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location=device)

    def get_live_metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            proc = list(self._process_times)
            inf = list(self._infer_times)
            avg_proc = _mean(proc)
            avg_inf = _mean(inf)
            last_inf = self._last_infer_ms
            last_conf = self._last_confidence
            last_emo = self._last_emotion
            face = self._last_face_seen
            val_acc = self._cnn_val_acc
            offline = dict(self._cnn_offline_report) if isinstance(self._cnn_offline_report, dict) else {}
            cnn_device = self._cnn_device
            cnn_ready = self._cnn_ready
            tl_device = self._tl_device
            tl_ready = self._tl_ready

        fps_pipe = None
        if avg_proc and avg_proc > 0:
            fps_pipe = 1000.0 / avg_proc

        macro_f1 = None
        weighted_f1 = None
        offline_acc = offline.get("best_val_accuracy")
        rep = offline.get("classification_report")
        if isinstance(rep, dict):
            ma = rep.get("macro avg")
            wa = rep.get("weighted avg")
            if isinstance(ma, dict) and "f1-score" in ma:
                macro_f1 = float(ma["f1-score"])
            if isinstance(wa, dict) and "f1-score" in wa:
                weighted_f1 = float(wa["f1-score"])

        return {
            "avg_process_ms": avg_proc,
            "fps_pipeline": fps_pipe,
            "avg_infer_ms": avg_inf,
            "last_infer_ms": last_inf,
            "last_confidence": last_conf,
            "last_emotion": last_emo,
            "last_face_seen": face,
            "cnn_val_accuracy": val_acc,
            "cnn_offline_val_accuracy": float(offline_acc) if offline_acc is not None else None,
            "cnn_macro_f1": macro_f1,
            "cnn_weighted_f1": weighted_f1,
            "cnn_inference_device": _device_label(cnn_device) if cnn_ready else None,
            "transfer_inference_device": _device_label(tl_device) if tl_ready else None,
        }

    def _record_process_ms(self, ms: float) -> None:
        with self._metrics_lock:
            self._process_times.append(ms)

    def _record_infer_ms(self, ms: float) -> None:
        with self._metrics_lock:
            self._infer_times.append(ms)
            self._last_infer_ms = ms

    def _record_prediction(self, emotion: str, confidence: float) -> None:
        with self._metrics_lock:
            self._last_face_seen = True
            self._last_emotion = emotion
            self._last_confidence = confidence

    def _clear_live_prediction(self) -> None:
        with self._metrics_lock:
            self._last_face_seen = False
            self._last_emotion = None
            self._last_confidence = None

    def _ensure_cnn(self) -> bool:
        with self._cnn_lock:
            if self._cnn_ready:
                return True
            if self._cnn_failed:
                return False

            if not self._import_torch_if_needed():
                self._cnn_failed = True
                self._cnn_error_msg = "Installez PyTorch: pip install torch torchvision"
                return False

            torch = self._torch
            from torchvision import transforms

            if self._cnn_transform is None:
                self._cnn_transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Grayscale(num_output_channels=1),
                        transforms.Resize((48, 48)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5], std=[0.5]),
                    ]
                )

            if not self._cnn_weights_path.is_file():
                self._cnn_failed = True
                self._cnn_error_msg = f"Modèle absent: {self._cnn_weights_path}"
                return False

            try:
                cnn_file = _project_root() / "models" / "CNN" / "CNN.py"
                spec = importlib.util.spec_from_file_location("iathon_emotion_cnn", cnn_file)
                if spec is None or spec.loader is None:
                    raise RuntimeError("spec CNN introuvable")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                EmotionCNN = mod.EmotionCNN
                default_classes = list(getattr(mod, "EMOTIONS", _DEFAULT_EMOTIONS))

                device = "cuda" if torch.cuda.is_available() else "cpu"
                checkpoint = self._torch_load_file(self._cnn_weights_path, device)
                classes = checkpoint.get("classes", default_classes)
                if not isinstance(classes, (list, tuple)):
                    classes = list(default_classes)
                else:
                    classes = list(classes)

                num_classes = len(classes)
                model = EmotionCNN(num_classes=num_classes, dropout=0.0)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)
                model.eval()

                self._cnn_model = model
                self._cnn_classes = classes
                self._cnn_device = device
                va = checkpoint.get("val_acc")
                self._cnn_val_acc = float(va) if va is not None else None
                self._cnn_ready = True
                return True
            except Exception as e:
                self._cnn_failed = True
                self._cnn_error_msg = str(e)
                return False

    def _ensure_transfer(self) -> bool:
        with self._tl_lock:
            if self._tl_ready:
                return True
            if self._tl_failed:
                return False

            if not self._import_torch_if_needed():
                self._tl_failed = True
                self._tl_error_msg = "Installez PyTorch: pip install torch torchvision"
                return False

            torch = self._torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms

            if self._tl_transform is None:
                self._tl_transform = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )

            if not self._tl_weights_path.is_file():
                self._tl_failed = True
                self._tl_error_msg = f"Modèle absent: {self._tl_weights_path}"
                return False

            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                raw = self._torch_load_file(self._tl_weights_path, device)

                if isinstance(raw, dict) and "model_state_dict" in raw:
                    state = raw["model_state_dict"]
                elif isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
                    state = raw["state_dict"]
                elif isinstance(raw, dict) and "fc.weight" in raw:
                    state = raw
                else:
                    state = raw

                if not isinstance(state, dict):
                    raise RuntimeError("Format de poids ResNet non reconnu")

                fc_w = state.get("fc.weight")
                if fc_w is None:
                    raise RuntimeError("Poids fc introuvables dans le state_dict")
                n_cls = int(fc_w.shape[0])

                model = models.resnet50(weights=None)
                model.fc = nn.Linear(model.fc.in_features, n_cls)
                model.load_state_dict(state, strict=True)
                model.to(device)
                model.eval()

                if n_cls <= len(_DEFAULT_EMOTIONS):
                    classes = list(_DEFAULT_EMOTIONS[:n_cls])
                else:
                    classes = [f"class_{i}" for i in range(n_cls)]

                self._tl_model = model
                self._tl_classes = classes
                self._tl_device = device
                self._tl_ready = True
                return True
            except Exception as e:
                self._tl_failed = True
                self._tl_error_msg = str(e)
                return False

    def _download_face_landmarker_model_if_needed(self) -> None:
        if self._lm_task_model_path.is_file():
            return
        self._lm_task_model_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        try:
            urllib.request.urlretrieve(_FACE_LANDMARKER_TASK_URL, self._lm_task_model_path)
        except Exception as e:
            raise RuntimeError(
                f"Téléchargement du modèle Face Landmarker impossible. "
                f"Enregistrez le fichier .task manuellement sous :\n{self._lm_task_model_path}\n"
                f"URL : {_FACE_LANDMARKER_TASK_URL}\n({e})"
            ) from e

    def _ensure_landmarks(self) -> bool:
        with self._lm_lock:
            if self._lm_ready:
                return True
            if self._lm_failed:
                return False

            try:
                import joblib
            except ImportError:
                self._lm_failed = True
                self._lm_error_msg = "Installez joblib: pip install joblib"
                return False

            try:
                import mediapipe as mp
            except ImportError:
                self._lm_failed = True
                self._lm_error_msg = "Installez mediapipe: pip install mediapipe"
                return False

            if not self._import_torch_if_needed():
                self._lm_failed = True
                self._lm_error_msg = "Installez PyTorch pour le mode Landmarks"
                return False

            if not self._lm_weights_path.is_file():
                self._lm_failed = True
                self._lm_error_msg = f"Modèle absent: {self._lm_weights_path}"
                return False
            if not self._lm_scaler_path.is_file():
                self._lm_failed = True
                self._lm_error_msg = f"Scaler absent: {self._lm_scaler_path}"
                return False
            if not self._lm_le_path.is_file():
                self._lm_failed = True
                self._lm_error_msg = f"Label encoder absent: {self._lm_le_path}"
                return False

            torch = self._torch
            import torch.nn as nn

            try:
                if torch.cuda.is_available():
                    device = "cuda"
                elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

                ckpt = self._torch_load_file(self._lm_weights_path, device)
                if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
                    raise RuntimeError("Checkpoint Landmarks invalide (mlp_best.pt)")

                input_dim = int(ckpt["input_dim"])
                num_classes = int(ckpt["num_classes"])

                scaler = joblib.load(self._lm_scaler_path)
                label_encoder = joblib.load(self._lm_le_path)

                class _LandmarksMLP(nn.Module):
                    def __init__(self, in_dim: int, n_cls: int):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(in_dim, 256),
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(256, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, n_cls),
                        )

                    def forward(self, x: Any) -> Any:
                        return self.net(x)

                model = _LandmarksMLP(input_dim, num_classes).to(device)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()

                self._lm_model = model
                self._lm_scaler = scaler
                self._lm_label_encoder = label_encoder
                self._lm_device = device
                self._lm_input_dim = input_dim

                self._lm_use_tasks_api = not hasattr(mp, "solutions")
                if self._lm_use_tasks_api:
                    self._download_face_landmarker_model_if_needed()
                    from mediapipe.tasks.python import vision as mp_tasks_vision
                    from mediapipe.tasks.python.core import base_options as mp_tasks_base

                    opts = mp_tasks_vision.FaceLandmarkerOptions(
                        base_options=mp_tasks_base.BaseOptions(
                            model_asset_path=str(self._lm_task_model_path)
                        ),
                        running_mode=mp_tasks_vision.RunningMode.VIDEO,
                        num_faces=1,
                        min_face_detection_confidence=0.5,
                        min_face_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self._lm_landmarker = mp_tasks_vision.FaceLandmarker.create_from_options(opts)
                    self._lm_face_mesh = None
                    self._lm_ts_ms = 0
                else:
                    self._lm_landmarker = None
                    self._lm_face_mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )

                self._lm_ready = True
                return True
            except Exception as e:
                self._lm_failed = True
                self._lm_error_msg = str(e)
                return False

    def _tensor_from_face_bgr_cnn(self, face_bgr: np.ndarray) -> Any:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return self._cnn_transform(face_rgb).unsqueeze(0)

    def _tensor_from_face_bgr_transfer(self, face_bgr: np.ndarray) -> Any:
        from PIL import Image

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(face_rgb)
        return self._tl_transform(pil).unsqueeze(0)

    def _draw_emotion_face(
        self,
        rgb: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        emotion: str,
        confidence: float,
    ) -> np.ndarray:
        color = _EMOTION_COLORS_BGR.get(emotion, (255, 255, 255))
        color_rgb = (int(color[2]), int(color[1]), int(color[0]))
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color_rgb, 2)
        label = f"{emotion} {confidence:.0%}"
        cv2.putText(
            rgb,
            label,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color_rgb,
            2,
            cv2.LINE_AA,
        )
        return rgb

    def process(self, frame, ai_model: str = "Landmarks"):
        t_start = time.perf_counter()
        try:
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            face_crop = None

            if ai_model == "CNN":
                if not self._ensure_cnn():
                    err = self._cnn_error_msg or "CNN indisponible"
                    cv2.putText(
                        rgb,
                        f"CNN: {err[:80]}",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 80, 80),
                        1,
                        cv2.LINE_AA,
                    )
                    self._clear_live_prediction()
                    return rgb, None

                torch = self._torch
                F = self._torch_F

                for (x, y, w, h) in faces:
                    margin = int(0.1 * min(w, h))
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    y2 = min(frame.shape[0], y + h + margin)

                    face_roi_bgr = frame[y1:y2, x1:x2]
                    if face_roi_bgr.size == 0:
                        continue

                    t_inf0 = time.perf_counter()
                    with torch.no_grad():
                        tensor = self._tensor_from_face_bgr_cnn(face_roi_bgr).to(self._cnn_device)
                        output = self._cnn_model(tensor)
                        probs = F.softmax(output, dim=1).cpu().numpy()[0]
                    t_inf1 = time.perf_counter()
                    self._record_infer_ms((t_inf1 - t_inf0) * 1000.0)

                    idx = int(np.argmax(probs))
                    emotion = self._cnn_classes[idx]
                    confidence = float(probs[idx])
                    self._record_prediction(emotion, confidence)
                    self._draw_emotion_face(rgb, x1, y1, x2, y2, emotion, confidence)

                    face_crop = rgb[y1:y2, x1:x2]
                    break

                if face_crop is None:
                    self._clear_live_prediction()

                return rgb, face_crop

            if ai_model == "ResNet50":
                if not self._ensure_transfer():
                    err = self._tl_error_msg or "ResNet50 indisponible"
                    cv2.putText(
                        rgb,
                        f"TL: {err[:80]}",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 80, 80),
                        1,
                        cv2.LINE_AA,
                    )
                    self._clear_live_prediction()
                    return rgb, None

                torch = self._torch
                F = self._torch_F

                for (x, y, w, h) in faces:
                    margin = int(0.1 * min(w, h))
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    y2 = min(frame.shape[0], y + h + margin)

                    face_roi_bgr = frame[y1:y2, x1:x2]
                    if face_roi_bgr.size == 0:
                        continue

                    t_inf0 = time.perf_counter()
                    with torch.no_grad():
                        tensor = self._tensor_from_face_bgr_transfer(face_roi_bgr).to(self._tl_device)
                        output = self._tl_model(tensor)
                        probs = F.softmax(output, dim=1).cpu().numpy()[0]
                    t_inf1 = time.perf_counter()
                    self._record_infer_ms((t_inf1 - t_inf0) * 1000.0)

                    idx = int(np.argmax(probs))
                    emotion = self._tl_classes[idx]
                    confidence = float(probs[idx])
                    self._record_prediction(emotion, confidence)
                    self._draw_emotion_face(rgb, x1, y1, x2, y2, emotion, confidence)

                    face_crop = rgb[y1:y2, x1:x2]
                    break

                if face_crop is None:
                    self._clear_live_prediction()

                return rgb, face_crop

            if ai_model == "Landmarks":
                if not self._ensure_landmarks():
                    err = self._lm_error_msg or "Landmarks indisponible"
                    cv2.putText(
                        rgb,
                        f"LM: {err[:80]}",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 80, 80),
                        1,
                        cv2.LINE_AA,
                    )
                    self._clear_live_prediction()
                    return rgb, None

                torch = self._torch
                F = self._torch_F

                h, w = rgb.shape[:2]

                if self._lm_use_tasks_api:
                    import mediapipe as mp

                    rgb_c = np.ascontiguousarray(rgb)
                    self._lm_ts_ms = int(time.perf_counter() * 1000)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_c)
                    det = self._lm_landmarker.detect_for_video(mp_image, self._lm_ts_ms)
                    if not det.face_landmarks:
                        self._clear_live_prediction()
                        return rgb, None
                    lm_iter = det.face_landmarks[0]
                else:
                    mp_results = self._lm_face_mesh.process(rgb)
                    if not mp_results.multi_face_landmarks:
                        self._clear_live_prediction()
                        return rgb, None
                    lm_iter = mp_results.multi_face_landmarks[0].landmark

                coords: List[float] = []
                xs: List[float] = []
                ys: List[float] = []
                for lm in lm_iter:
                    fx = float(lm.x)
                    fy = float(lm.y)
                    coords.extend((fx, fy))
                    xs.append(fx * w)
                    ys.append(fy * h)

                features = np.asarray(coords, dtype=np.float64).reshape(1, -1)
                if features.shape[1] != int(self._lm_input_dim):
                    self._clear_live_prediction()
                    return rgb, None

                try:
                    features_s = self._lm_scaler.transform(features)
                except Exception:
                    self._clear_live_prediction()
                    return rgb, None

                t_inf0 = time.perf_counter()
                with torch.no_grad():
                    x = torch.from_numpy(features_s.astype(np.float32)).to(self._lm_device)
                    out = self._lm_model(x)
                    probs = F.softmax(out, dim=1).cpu().numpy()[0]
                t_inf1 = time.perf_counter()
                self._record_infer_ms((t_inf1 - t_inf0) * 1000.0)

                idx = int(np.argmax(probs))
                confidence = float(probs[idx])
                raw_label = self._lm_label_encoder.inverse_transform([idx])[0]
                emotion = str(raw_label).strip().lower()

                self._record_prediction(emotion, confidence)

                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))
                span = max(x2 - x1, y2 - y1, 8)
                pad = max(4, int(0.06 * span))
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w - 1, x2 + pad)
                y2 = min(h - 1, y2 + pad)

                self._draw_emotion_face(rgb, x1, y1, x2, y2, emotion, confidence)
                face_crop = rgb[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                return rgb, face_crop

            self._clear_live_prediction()
            return rgb, None
        finally:
            self._record_process_ms((time.perf_counter() - t_start) * 1000.0)
