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
        """Lecture thread-safe pour l’UI (performances temps réel + stats offline CNN)."""
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
        """
        Traite une frame BGR OpenCV.
        Retourne (frame_rgb_annotée, crop_visage_rgb_ou_None).
        """
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

            self._clear_live_prediction()

            for (x, y, w, h) in faces:
                cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    rgb,
                    "Calm",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                face_crop = rgb[y : y + h, x : x + w]
                break

            return rgb, face_crop
        finally:
            self._record_process_ms((time.perf_counter() - t_start) * 1000.0)
