import customtkinter as ctk
import random
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps, ImageSequence, ImageTk

from core.camera import Camera
from core.vision import Vision

_AI_MODEL_OPTIONS = ("CNN", "Landmarks", "ResNet50")

_NEG_EMOTION_KEYS = frozenset({"angry", "disgust", "fear", "sad", "surprise"})
_NEG_EMOTION_FR = {
    "angry": "la colère",
    "disgust": "le dégoût",
    "fear": "la peur",
    "sad": "la tristesse",
    "surprise": "la surprise",
}
_NEG_ALERT_ENTER_CONF = 0.58
_NEG_ALERT_EXIT_CONF = 0.42

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_NEG_OVERLAY_MIN_SECONDS = 5.0


def _list_negative_feedback_gifs() -> list[Path]:
    if not _ASSETS_DIR.is_dir():
        return []
    out: list[Path] = []
    try:
        for p in _ASSETS_DIR.iterdir():
            if p.is_file() and p.suffix.lower() == ".gif":
                out.append(p)
    except OSError:
        return []
    out.sort(key=lambda x: x.name.lower())
    return out


def _neg_decode_gif_contained_frames(
    src: Path, w: int, h: int, frame_start: int, frame_stop_exclusive: Optional[int]
) -> list:
    out: list = []
    try:
        with Image.open(src) as im:
            for i, frame in enumerate(ImageSequence.Iterator(im)):
                if i < frame_start:
                    continue
                if frame_stop_exclusive is not None and i >= frame_stop_exclusive:
                    break
                rgba = frame.convert("RGBA").copy()
                rgba = ImageOps.contain(rgba, (w, h), method=Image.Resampling.LANCZOS)
                out.append(ImageTk.PhotoImage(rgba))
    except Exception:
        return []
    return out


_THEME = {
    "accent": "#1f8ceb",
    "danger": "#e74c3c",
    "success": "#2ecc71",
    "muted": ("#5c5f66", "#9aa0a8"),
    "bg": ("#f5f6f7", "#0f1115"),
    "surface": ("#ffffff", "#151922"),
    "card": ("#ffffff", "#131827"),
    "card_raised": ("#fbfbfc", "#171d2e"),
    "border": ("#e3e3e3", "#2a2f3a"),
    "divider": ("#e8e8e8", "#2a2f3a"),
}


def _theme_solid(key: str) -> str:
    val = _THEME[key]
    if isinstance(val, tuple):
        return val[1] if ctk.get_appearance_mode() == "Dark" else val[0]
    return str(val)


class EmotionApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("EmotiDetect")
        self.geometry("1200x750")
        self.minsize(1100, 700)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.camera = Camera()
        self.vision = Vision()

        self.running = False
        self._desired_running = False
        self._ui_frame_count = 0
        self._ai_model_choice = _AI_MODEL_OPTIONS[0]
        self._is_fullscreen_runtime = False
        self._prev_geometry = None
        self._runtime_output_imgtk = None
        self._camera_indices = []
        self._starting_stream = False
        self._stream_loading_layers = []
        self._neg_emotion_alert_hold = False
        self._neg_cartoon_frames: list = []
        self._neg_cartoon_size_key: Optional[tuple] = None
        self._neg_cartoon_job = None
        self._neg_cartoon_i = 0
        self._neg_feedback_active = False
        self._neg_last_present: Optional[tuple] = None
        self._neg_canvas_mode: Optional[str] = None
        self._neg_configure_job = None
        self._neg_canvas_gif_item_id = None
        self._neg_cartoon_source_path: Optional[Path] = None
        self._neg_overlay_hide_not_before: Optional[float] = None
        self._neg_overlay_min_duration_job = None
        self._neg_overlay_first_frame_committed = False
        self._neg_last_face_present_for_banner = False
        self._neg_cartoon_preload_job = None
        self._neg_cartoon_preload_pending = False

        self.build_ui()

    def build_ui(self):

        self.configure(fg_color=_THEME["bg"])

        self.topbar = ctk.CTkFrame(
            self,
            height=70,
            corner_radius=0,
            fg_color=_THEME["surface"],
            border_width=1,
            border_color=_THEME["border"],
        )
        self.topbar.pack(fill="x")

        self.app_title = ctk.CTkLabel(
            self.topbar,
            text="EmotiDetect",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.app_title.pack(side="left", padx=16, pady=18)

        self.play_btn = ctk.CTkButton(
            self.topbar,
            text="▶",
            width=44,
            height=32,
            command=self.on_play_clicked,
        )
        self.play_btn.pack(side="right", padx=(0, 10), pady=18)

        self.tabs = ctk.CTkTabview(
            self,
            fg_color="transparent",
            segmented_button_fg_color=_THEME["surface"],
            segmented_button_selected_color=_THEME["accent"],
            segmented_button_unselected_color=_THEME["surface"],
            segmented_button_selected_hover_color=_THEME["accent"],
        )
        self.tabs.pack(fill="both", expand=True, padx=15, pady=15)

        self.tab1 = self.tabs.add("Runtime")
        self.tab2 = self.tabs.add("Configuration")

        self.tab1.grid_columnconfigure(0, weight=3)
        self.tab1.grid_columnconfigure(1, weight=1)
        self.tab1.grid_rowconfigure(0, weight=1)

        self.runtime_output_frame = ctk.CTkFrame(
            self.tab1,
            corner_radius=15,
            fg_color=_THEME["card"],
            border_width=1,
            border_color=_THEME["border"],
        )
        self.runtime_output_frame.grid(row=0, column=0, sticky="nsew", padx=(15, 10), pady=15)
        self.runtime_output_frame.grid_rowconfigure(0, weight=1)
        self.runtime_output_frame.grid_columnconfigure(0, weight=1)

        self.runtime_output = ctk.CTkLabel(
            self.runtime_output_frame,
            text="",
            corner_radius=13,
            justify="center",
        )
        self.runtime_output.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.runtime_neg_canvas = tk.Canvas(
            self.runtime_output_frame,
            highlightthickness=0,
            bd=0,
            bg=_theme_solid("surface"),
        )
        self.runtime_neg_canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.runtime_neg_canvas.grid_remove()
        self.runtime_output_frame.bind("<Configure>", self._on_runtime_output_frame_configure, add="+")

        self.runtime_side = ctk.CTkFrame(
            self.tab1,
            corner_radius=15,
            fg_color=_THEME["card"],
            border_width=1,
            border_color=_THEME["border"],
        )
        self.runtime_side.grid(row=0, column=1, sticky="nsew", padx=(10, 15), pady=15)
        self.runtime_side.grid_columnconfigure(0, weight=1)

        self.runtime_title = ctk.CTkLabel(
            self.runtime_side,
            text="Runtime",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.runtime_title.grid(row=0, column=0, sticky="w", padx=15, pady=(15, 6))

        self.runtime_desc = ctk.CTkLabel(
            self.runtime_side,
            text="Renvoyer du contenu approprié à l'émotion détectée.",
            justify="left",
            text_color=_THEME["muted"],
        )
        self.runtime_desc.grid(row=1, column=0, sticky="w", padx=15, pady=(0, 10))

        self.runtime_selected = ctk.CTkLabel(self.runtime_side, text="Vidéo sélectionnée : —")
        self.runtime_selected.grid(row=2, column=0, sticky="w", padx=15, pady=(0, 12))

        self.fullscreen_btn = ctk.CTkButton(
            self.runtime_side,
            text="Plein écran",
            command=self.open_fullscreen,
        )
        self.fullscreen_btn.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 12))

        self.face_title = ctk.CTkLabel(
            self.runtime_side,
            text="Visage détecté",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.face_title.grid(row=4, column=0, sticky="w", padx=15, pady=(0, 8))

        self.face_preview_host = ctk.CTkFrame(self.runtime_side, fg_color="transparent")
        self.face_preview_host.grid(row=5, column=0, sticky="ew", padx=15, pady=(0, 15))
        self.face_preview_host.grid_columnconfigure(0, weight=1)
        self.face_preview_host.grid_rowconfigure(0, weight=1, minsize=200)

        self.small_face = ctk.CTkLabel(
            self.face_preview_host, text="—", width=280, height=200, corner_radius=15
        )
        self.small_face.grid(row=0, column=0, sticky="nsew")

        self.msg1 = ctk.CTkLabel(
            self.tab1,
            text="La caméra est actuellement éteinte",
            text_color=_THEME["danger"],
        )
        self.msg1.grid(row=1, column=0, columnspan=2, sticky="w", padx=15, pady=(0, 12))

        self.tab2.grid_columnconfigure(0, weight=5)
        self.tab2.grid_columnconfigure(1, weight=3)
        self.tab2.grid_rowconfigure(0, weight=1)

        _muted = _THEME["muted"]
        _card = _THEME["card_raised"]
        _card_border = _THEME["border"]
        _accent = _THEME["accent"]

        self.config_left = ctk.CTkFrame(self.tab2, fg_color="transparent")
        self.config_left.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=12)
        self.config_left.grid_rowconfigure(1, weight=1)
        self.config_left.grid_columnconfigure(0, weight=1)

        preview_hdr = ctk.CTkFrame(self.config_left, fg_color="transparent")
        preview_hdr.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        preview_hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            preview_hdr,
            text="Aperçu en direct",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.preview_subtitle = ctk.CTkLabel(
            preview_hdr,
            text="Flux traité en 640×480",
            font=ctk.CTkFont(size=12),
            text_color=_muted,
            anchor="e",
        )
        self.preview_subtitle.grid(row=0, column=1, sticky="e")

        self.live_view_frame = ctk.CTkFrame(
            self.config_left,
            corner_radius=16,
            border_width=1,
            border_color=_card_border,
            fg_color=_THEME["surface"],
        )
        self.live_view_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.live_view_frame.grid_rowconfigure(0, weight=1)
        self.live_view_frame.grid_columnconfigure(0, weight=1)

        self.live_view = ctk.CTkLabel(
            self.live_view_frame,
            text="",
            corner_radius=12,
        )
        self.live_view.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.config_sidebar = ctk.CTkScrollableFrame(
            self.tab2,
            fg_color=_THEME["card"],
            corner_radius=16,
            border_width=1,
            border_color=_card_border,
            scrollbar_button_color=("#bababa", "#404040"),
            scrollbar_button_hover_color=("#a0a0a0", "#505050"),
        )
        self.config_sidebar.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=12)
        self.config_sidebar.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.config_sidebar,
            text="Configuration",
            font=ctk.CTkFont(size=20, weight="bold"),
            anchor="w",
        ).pack(anchor="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(
            self.config_sidebar,
            text="Source vidéo, performances et métriques des modèles",
            font=ctk.CTkFont(size=11),
            text_color=_muted,
            anchor="w",
            wraplength=300,
        ).pack(anchor="w", padx=12, pady=(2, 12))

        def _section_title(parent, label):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=(4, 10))
            ctk.CTkFrame(row, width=4, height=20, corner_radius=2, fg_color=_accent).pack(side="left", padx=(0, 10), pady=2)
            ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=15, weight="bold"), anchor="w").pack(side="left")

        def _divider(parent):
            ctk.CTkFrame(parent, height=1, fg_color=_THEME["divider"]).pack(fill="x", padx=12, pady=(4, 8))

        self.card_cam = ctk.CTkFrame(
            self.config_sidebar,
            corner_radius=12,
            fg_color=_card,
            border_width=1,
            border_color=_card_border,
        )
        self.card_cam.pack(fill="x", padx=8, pady=(0, 10))
        _section_title(self.card_cam, "Caméra")
        ctk.CTkLabel(
            self.card_cam,
            text="Choix du périphérique",
            font=ctk.CTkFont(size=12),
            text_color=_muted,
            anchor="w",
        ).pack(anchor="w", padx=14, pady=(0, 4))
        self.cam_selector = ctk.CTkComboBox(
            self.card_cam,
            values=["Chargement..."],
            state="disabled",
            height=32,
            command=self.on_camera_selected,
        )
        self.cam_selector.pack(fill="x", padx=14, pady=(0, 8))
        _divider(self.card_cam)

        _spec_font = ctk.CTkFont(size=12)
        self.cam_info_device = ctk.CTkLabel(
            self.card_cam, text="Périphérique  —", font=_spec_font, anchor="w", justify="left"
        )
        self.cam_info_device.pack(anchor="w", padx=14, pady=3)
        self.cam_info_fps = ctk.CTkLabel(self.card_cam, text="FPS signal  —", font=_spec_font, anchor="w")
        self.cam_info_fps.pack(anchor="w", padx=14, pady=3)
        self.cam_info_res = ctk.CTkLabel(self.card_cam, text="Résolution  —", font=_spec_font, anchor="w")
        self.cam_info_res.pack(anchor="w", padx=14, pady=3)
        self.cam_info_backend = ctk.CTkLabel(self.card_cam, text="Backend  —", font=_spec_font, anchor="w")
        self.cam_info_backend.pack(anchor="w", padx=14, pady=3)
        self.cam_info_status = ctk.CTkLabel(
            self.card_cam,
            text="Statut  —",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        )
        self.cam_info_status.pack(anchor="w", padx=14, pady=(6, 14))

        self.card_perf = ctk.CTkFrame(
            self.config_sidebar,
            corner_radius=12,
            fg_color=_card,
            border_width=1,
            border_color=_card_border,
        )
        self.card_perf.pack(fill="x", padx=8, pady=(0, 10))
        _section_title(self.card_perf, "Performances")
        ctk.CTkLabel(
            self.card_perf,
            text="Temps réel (pipeline + inférence)",
            font=ctk.CTkFont(size=12),
            text_color=_muted,
            anchor="w",
        ).pack(anchor="w", padx=14, pady=(0, 6))

        self.perf_model_active = ctk.CTkLabel(
            self.card_perf,
            text="Modèle actif  —",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        self.perf_model_active.pack(anchor="w", padx=14, pady=(0, 6))

        _wrap = 280
        self.perf_throughput = ctk.CTkLabel(
            self.card_perf,
            text="Débit : —",
            justify="left",
            anchor="w",
            font=_spec_font,
            text_color=_muted,
            wraplength=_wrap,
        )
        self.perf_throughput.pack(anchor="w", padx=14, pady=2)
        self.perf_cnn_infer = ctk.CTkLabel(
            self.card_perf,
            text="Inférence modèle : —",
            justify="left",
            anchor="w",
            font=_spec_font,
            text_color=_muted,
            wraplength=_wrap,
        )
        self.perf_cnn_infer.pack(anchor="w", padx=14, pady=(2, 14))

        self.card_runtime = ctk.CTkFrame(
            self.config_sidebar,
            corner_radius=12,
            fg_color=_card,
            border_width=1,
            border_color=_card_border,
        )
        self.card_runtime.pack(fill="x", padx=8, pady=(0, 10))
        _section_title(self.card_runtime, "Réglages")
        ctk.CTkLabel(
            self.card_runtime,
            text="Modèle d'IA",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).pack(anchor="w", padx=14, pady=(2, 4))

        ctk.CTkLabel(
            self.card_runtime,
            text="CNN : models/CNN/models/best_model.pth · ResNet50 : models/transfer_learning/best_model.pth · Landmarks : mlp_best.pt + MediaPipe (FaceMesh ou face_landmarker.task)",
            justify="left",
            text_color=_muted,
            wraplength=300,
        ).pack(anchor="w", padx=14, pady=(0, 8))

        self._ai_model_menu = ctk.CTkOptionMenu(
            self.card_runtime,
            values=list(_AI_MODEL_OPTIONS),
            command=self.on_ai_model_selected,
            width=260,
        )
        self._ai_model_menu.set(self._ai_model_choice)
        self._ai_model_menu.pack(fill="x", padx=14, pady=(0, 14))

        self.msg2 = ctk.CTkLabel(
            self.tab2,
            text="La caméra est actuellement éteinte",
            text_color=_THEME["danger"],
            font=ctk.CTkFont(size=12),
        )
        self.msg2.grid(row=1, column=0, columnspan=2, sticky="w", padx=20, pady=(0, 10))

        self.update_warning()
        self.refresh_camera_list()
        self.update_camera_info()

    def on_play_clicked(self):
        self._desired_running = not self.running
        self.toggle_camera()

    def toggle_camera(self):

        if self._desired_running:
            self._starting_stream = True
            self.show_preview_loading("Démarrage de la caméra…")
            threading.Thread(target=self._start_camera_worker, daemon=True).start()
        else:
            self.running = False
            self.camera.release()
            self._starting_stream = False
            self.hide_preview_loading()
            self._reset_runtime_neg_emotion_banner()

        self.update_warning()
        self.update_camera_info()

    def refresh_camera_list(self, max_devices: int = 10):
        indices = Camera.detect_available_devices(max_devices=max_devices)
        self._camera_indices = indices

        if not indices:
            self.cam_selector.configure(values=["Aucune caméra détectée"], state="disabled")
            self.cam_selector.set("Aucune caméra détectée")
            return

        values = [f"Caméra {i}" for i in indices]
        self.cam_selector.configure(values=values, state="readonly")

        current_index = getattr(self.camera, "device_index", None)
        selected = f"Caméra {current_index}" if current_index in indices else values[0]
        self.cam_selector.set(selected)
        self.on_camera_selected(selected)

    def on_camera_selected(self, value: str):
        if not value:
            return

        if not value.startswith("Caméra "):
            return

        try:
            index = int(value.split("Caméra ", 1)[1].strip())
        except Exception:
            return

        if index == getattr(self.camera, "device_index", None):
            self.update_camera_info()
            return

        self.set_active_camera(index)

    def set_active_camera(self, index: int):
        was_streaming = self.toggle.get() == 1

        if self.running:
            self.running = False
        self.camera.release()

        self.camera.set_device_index(index)
        self.update_camera_info()

        if was_streaming:
            self.camera.start()
            self.running = True
            threading.Thread(target=self.update_frame, daemon=True).start()

        self.update_warning()
        self.update_camera_info()

    def update_frame(self):

        while self.running:

            frame, ok = self.camera.read()
            if not ok:
                continue

            frame, face = self.vision.process(frame, self._ai_model_choice)

            self.update_ui(frame, face)

    def show_preview_loading(self, message: str = "Chargement…"):
        self.hide_preview_loading()

        targets = (
            (self.live_view_frame, (10, 10), False),
            (self.face_preview_host, (0, 0), True),
        )

        for parent, (padx, pady), compact in targets:
            overlay = ctk.CTkFrame(
                parent,
                fg_color=_THEME["surface"],
                corner_radius=8 if compact else 12,
                border_width=1,
                border_color=_THEME["border"],
            )
            overlay.grid(row=0, column=0, sticky="nsew", padx=padx, pady=pady)
            try:
                overlay.lift()
            except Exception:
                pass
            overlay.grid_rowconfigure(2, weight=1)
            overlay.grid_columnconfigure(0, weight=1)

            title_font = ctk.CTkFont(size=11, weight="bold") if compact else ctk.CTkFont(size=14, weight="bold")
            sub_font = ctk.CTkFont(size=10) if compact else None

            title_lbl = ctk.CTkLabel(
                overlay,
                text=message,
                font=title_font,
            )
            if compact:
                title_lbl.configure(wraplength=240)
            title_lbl.grid(row=0, column=0, sticky="w", padx=10 if compact else 14, pady=(10 if compact else 14, 4))

            if not compact:
                ctk.CTkLabel(
                    overlay,
                    text="Merci de patienter…",
                    text_color=_THEME["muted"],
                ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))
            else:
                ctk.CTkLabel(
                    overlay,
                    text="Patientez…",
                    font=sub_font,
                    text_color=_THEME["muted"],
                ).grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))

            if compact:
                bar = ctk.CTkProgressBar(overlay, height=6)
            else:
                bar = ctk.CTkProgressBar(overlay)
            bar.grid(
                row=3,
                column=0,
                sticky="ew",
                padx=10 if compact else 14,
                pady=(0, 10 if compact else 14),
            )
            try:
                bar.configure(mode="indeterminate")
            except Exception:
                pass
            try:
                bar.start()
            except Exception:
                pass

            self._stream_loading_layers.append((overlay, bar))

    def hide_preview_loading(self):
        layers = list(self._stream_loading_layers)
        self._stream_loading_layers.clear()

        for overlay, bar in layers:
            if bar is not None:
                try:
                    bar.stop()
                except Exception:
                    pass
            if overlay is not None:
                try:
                    if overlay.winfo_exists():
                        overlay.destroy()
                except Exception:
                    pass

    def _start_camera_worker(self):
        try:
            self.camera.start()
            ok = True
        except Exception:
            ok = False

        def _finish():
            if ok and self._desired_running:
                self.running = True
                threading.Thread(target=self.update_frame, daemon=True).start()
            else:
                self.running = False
                try:
                    self.camera.release()
                except Exception:
                    pass
                self._desired_running = False

            if not ok:
                self._starting_stream = False
                self.hide_preview_loading()
            self.update_warning()
            self.update_camera_info()

        try:
            self.after(0, _finish)
        except Exception:
            _finish()

    def update_ui(self, frame, face):
        try:
            frame_copy = frame.copy()
        except Exception:
            frame_copy = frame
        if face is not None:
            try:
                face_copy = face.copy()
            except Exception:
                face_copy = face
        else:
            face_copy = None

        def _apply():
            self._update_ui_main_thread(frame_copy, face_copy)

        try:
            self.after(0, _apply)
        except Exception:
            self._update_ui_main_thread(frame_copy, face_copy)

    def _update_ui_main_thread(self, frame, face):

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(img)

        if self._starting_stream:
            self._starting_stream = False
            self.hide_preview_loading()

        self.live_view.configure(image=imgtk, text="")
        self.live_view.image = imgtk

        if face is not None:
            face_img = Image.fromarray(face)
            face_img = face_img.resize((300, 200))
            face_tk = ImageTk.PhotoImage(face_img)

            self.small_face.configure(image=face_tk, text="")
            self.small_face.image = face_tk
        else:
            self.small_face.configure(text="—", image=None)

        self._update_runtime_neg_emotion_banner(face is not None)

        self._ui_frame_count += 1
        if self._ui_frame_count % 15 == 0:
            self.update_camera_info()

    def update_warning(self):

        if self.running:
            self.msg1.configure(text="")
            self.msg2.configure(text="")
            self.play_btn.configure(text="⏸")
        else:
            self.msg1.configure(text="La caméra est actuellement éteinte")
            self.msg2.configure(text="La caméra est actuellement éteinte")
            self.play_btn.configure(text="▶")
            self._reset_runtime_neg_emotion_banner()

    def _on_runtime_output_frame_configure(self, _evt=None):
        if not self._neg_feedback_active or self._neg_last_present is None:
            return
        if self._neg_configure_job is not None:
            try:
                self.after_cancel(self._neg_configure_job)
            except Exception:
                pass
        self._neg_configure_job = self.after(120, self._neg_configure_refresh)

    def _neg_configure_refresh(self):
        self._neg_configure_job = None
        self._neg_refresh_after_geometry_change()

    def _neg_refresh_after_geometry_change(self):
        if not (self._neg_feedback_active and self._neg_last_present):
            return
        k, e, c = self._neg_last_present
        self._present_negative_feedback(k, e, c, force_reload=True)

    def _neg_canvas_center(self) -> tuple:
        self.update_idletasks()
        w = max(self.runtime_neg_canvas.winfo_width(), 4)
        h = max(self.runtime_neg_canvas.winfo_height(), 4)
        return (w // 2, h // 2)

    def _neg_canvas_show_gif_index(self, idx: int, force_new_item: bool = False) -> None:
        if not self._neg_cartoon_frames:
            return
        img = self._neg_cartoon_frames[idx % len(self._neg_cartoon_frames)]
        cx, cy = self._neg_canvas_center()
        if force_new_item:
            self.runtime_neg_canvas.delete("all")
            self._neg_canvas_gif_item_id = None
        if self._neg_canvas_gif_item_id is None:
            self._neg_canvas_gif_item_id = self.runtime_neg_canvas.create_image(
                cx, cy, anchor="center", image=img
            )
        else:
            self.runtime_neg_canvas.coords(self._neg_canvas_gif_item_id, cx, cy)
            self.runtime_neg_canvas.itemconfig(self._neg_canvas_gif_item_id, image=img)
        self.runtime_neg_canvas._gif_ref = img

    def _neg_canvas_draw_text_fallback(self, text: str, wrap_px: int) -> None:
        self.runtime_neg_canvas.delete("all")
        self._neg_canvas_gif_item_id = None
        cx, cy = self._neg_canvas_center()
        self.runtime_neg_canvas.create_text(
            cx,
            cy,
            text=text,
            fill=str(_THEME["danger"]),
            font=("Segoe UI", 16),
            width=wrap_px,
            justify="center",
        )

    def _stop_neg_cartoon_animation(self):
        if self._neg_cartoon_job is not None:
            try:
                self.after_cancel(self._neg_cartoon_job)
            except Exception:
                pass
            self._neg_cartoon_job = None
        self._neg_cartoon_i = 0

    def _hide_runtime_neg_overlay(self):
        if self._neg_configure_job is not None:
            try:
                self.after_cancel(self._neg_configure_job)
            except Exception:
                pass
            self._neg_configure_job = None
        self._neg_cancel_cartoon_preload()
        self._stop_neg_cartoon_animation()
        self._neg_cartoon_frames = []
        self._neg_cartoon_size_key = None
        self._neg_last_present = None
        self._neg_canvas_mode = None
        self._neg_cartoon_source_path = None
        self._neg_overlay_hide_not_before = None
        self._neg_cancel_overlay_min_timer()
        self._neg_overlay_first_frame_committed = False
        self._neg_last_face_present_for_banner = False
        try:
            self.runtime_neg_canvas.delete("all")
            self._neg_canvas_gif_item_id = None
            self.runtime_neg_canvas.grid_remove()
        except Exception:
            pass

    def _neg_cancel_overlay_min_timer(self) -> None:
        if self._neg_overlay_min_duration_job is not None:
            try:
                self.after_cancel(self._neg_overlay_min_duration_job)
            except Exception:
                pass
            self._neg_overlay_min_duration_job = None

    def _neg_arm_overlay_min_timer_after_first_frame(self) -> None:
        if not self._neg_feedback_active or self._neg_overlay_first_frame_committed:
            return
        self._neg_overlay_first_frame_committed = True
        self._neg_cancel_overlay_min_timer()
        self._neg_overlay_hide_not_before = time.monotonic() + float(_NEG_OVERLAY_MIN_SECONDS)
        ms = int(round(_NEG_OVERLAY_MIN_SECONDS * 1000))

        def _ping():
            self._neg_overlay_min_duration_job = None
            try:
                self._update_runtime_neg_emotion_banner(self._neg_last_face_present_for_banner)
            except Exception:
                pass

        self._neg_overlay_min_duration_job = self.after(ms, _ping)

    def _neg_cancel_cartoon_preload(self) -> None:
        if self._neg_cartoon_preload_job is not None:
            try:
                self.after_cancel(self._neg_cartoon_preload_job)
            except Exception:
                pass
            self._neg_cartoon_preload_job = None
        self._neg_cartoon_preload_pending = False

    def _neg_finish_cartoon_preload(self, src: Path, w: int, h: int, key: tuple) -> None:
        self._neg_cartoon_preload_job = None
        if not self._neg_feedback_active or self._neg_canvas_mode != "gif":
            self._neg_cartoon_preload_pending = False
            return
        if self._neg_cartoon_source_path != src or self._neg_cartoon_size_key != key:
            self._neg_cartoon_preload_pending = False
            return
        rest = _neg_decode_gif_contained_frames(src, w, h, 1, None)
        self._neg_cartoon_preload_pending = False
        if not rest:
            return
        try:
            first = self._neg_cartoon_frames[0]
        except (IndexError, TypeError):
            return
        self._neg_cartoon_frames = [first] + rest

    def _neg_overlay_inner_size(self) -> tuple:
        try:
            self.update_idletasks()
            fw = self.runtime_output_frame.winfo_width()
            fh = self.runtime_output_frame.winfo_height()
            edge = 0 if self._is_fullscreen_runtime else 16
            if fw <= 50 or fh <= 50:
                return (760, 560)
            w = max(fw - edge, 480)
            h = max(fh - edge, 320)
            return (int(w), int(h))
        except Exception:
            return (760, 560)

    def _reload_neg_cartoon_frames_for_overlay(self) -> bool:
        src = self._neg_cartoon_source_path
        if src is None or not src.is_file():
            return False
        w, h = self._neg_overlay_inner_size()
        key = (w, h)
        if self._neg_cartoon_frames and self._neg_cartoon_size_key == key:
            return True
        self._neg_cancel_cartoon_preload()
        try:
            first = _neg_decode_gif_contained_frames(src, w, h, 0, 1)
            if not first:
                self._neg_cartoon_frames = []
                self._neg_cartoon_size_key = None
                return False
            self._neg_cartoon_frames = first
            self._neg_cartoon_size_key = key
            self._neg_cartoon_preload_pending = True
            self._neg_cartoon_preload_job = self.after(
                0, lambda s=src, ww=w, hh=h, k=key: self._neg_finish_cartoon_preload(s, ww, hh, k)
            )
            return True
        except Exception:
            self._neg_cartoon_frames = []
            self._neg_cartoon_size_key = None
            self._neg_cartoon_preload_pending = False
            return False

    def _tick_neg_cartoon(self):
        if not self._neg_cartoon_frames or not self._neg_feedback_active:
            self._neg_cartoon_job = None
            return
        try:
            self._neg_cartoon_i = (self._neg_cartoon_i + 1) % len(self._neg_cartoon_frames)
            self._neg_canvas_show_gif_index(self._neg_cartoon_i)
        except Exception:
            self._neg_cartoon_job = None
            return
        self._neg_cartoon_job = self.after(90, self._tick_neg_cartoon)

    def _present_negative_feedback(self, key: str, em, cf_f: float, force_reload: bool = False):
        if self._neg_cartoon_source_path is None:
            gifs = _list_negative_feedback_gifs()
            if gifs:
                self._neg_cartoon_source_path = random.choice(gifs)
        self._neg_last_present = (key, em, cf_f)
        if force_reload:
            self._neg_cancel_cartoon_preload()
            self._neg_cartoon_frames = []
            self._neg_cartoon_size_key = None
            self._stop_neg_cartoon_animation()
            try:
                self.runtime_neg_canvas.delete("all")
                self._neg_canvas_gif_item_id = None
            except Exception:
                pass

        pad = 0 if self._is_fullscreen_runtime else 8
        self.runtime_neg_canvas.grid(row=0, column=0, sticky="nsew", padx=pad, pady=pad)
        try:
            self.runtime_neg_canvas.lift()
        except Exception:
            pass

        if self._reload_neg_cartoon_frames_for_overlay():
            self._neg_canvas_mode = "gif"
            try:
                self.runtime_neg_canvas.configure(bg=_theme_solid("surface"))
            except Exception:
                pass
            self._stop_neg_cartoon_animation()
            self._neg_cartoon_i = 0
            self._neg_canvas_show_gif_index(0, force_new_item=True)
            self._neg_cartoon_job = self.after(90, self._tick_neg_cartoon)
            self._neg_arm_overlay_min_timer_after_first_frame()
            return

        self._neg_canvas_mode = "text"
        try:
            self.runtime_neg_canvas.configure(bg=_theme_solid("card"))
        except Exception:
            pass
        self._stop_neg_cartoon_animation()
        w, _h = self._neg_overlay_inner_size()
        fr = _NEG_EMOTION_FR.get(key, str(em))
        msg = (
            f"Le visage semble exprimer {fr} (≈ {cf_f:.0%}). "
            f"Adaptez le contenu proposé avec bienveillance."
        )
        self._neg_canvas_draw_text_fallback(msg, max(int(w * 0.85), 320))
        self._neg_arm_overlay_min_timer_after_first_frame()

    def _reset_runtime_neg_emotion_banner(self):
        self._neg_emotion_alert_hold = False
        self._neg_feedback_active = False
        self._hide_runtime_neg_overlay()

    def _update_runtime_neg_emotion_banner(self, face_present: bool):
        self._neg_last_face_present_for_banner = face_present
        m = self.vision.get_live_metrics()
        seen = bool(m.get("last_face_seen"))
        em = m.get("last_emotion")
        cf = m.get("last_confidence")
        key = str(em).strip().lower() if em is not None else ""
        is_neg = key in _NEG_EMOTION_KEYS
        try:
            cf_f = float(cf) if cf is not None else 0.0
        except (TypeError, ValueError):
            cf_f = 0.0

        base_ok = bool(face_present and seen and is_neg and em is not None and cf is not None)

        if self._neg_emotion_alert_hold:
            if not base_ok or cf_f < _NEG_ALERT_EXIT_CONF:
                self._neg_emotion_alert_hold = False
        else:
            if base_ok and cf_f >= _NEG_ALERT_ENTER_CONF:
                self._neg_emotion_alert_hold = True

        show = bool(self._neg_emotion_alert_hold and base_ok)
        if show:
            if not self._neg_feedback_active:
                self._neg_feedback_active = True
                self._neg_overlay_hide_not_before = None
                self._neg_overlay_first_frame_committed = False
                self._neg_cancel_overlay_min_timer()
                self._present_negative_feedback(key, em, cf_f)
        else:
            if self._neg_feedback_active and not show:
                hb = self._neg_overlay_hide_not_before
                if hb is not None and time.monotonic() >= hb:
                    self._neg_feedback_active = False
                    self._hide_runtime_neg_overlay()

    def on_ai_model_selected(self, value: str):
        if value in _AI_MODEL_OPTIONS:
            self._ai_model_choice = value
        try:
            self.update_camera_info()
        except Exception:
            pass

    def open_fullscreen(self):
        try:
            if not self._is_fullscreen_runtime:
                self._enter_runtime_fullscreen()
            else:
                self._exit_runtime_fullscreen()
        except Exception:
            if not self._is_fullscreen_runtime:
                self._enter_runtime_fullscreen()
            else:
                self._exit_runtime_fullscreen()

    def _enter_runtime_fullscreen(self):
        self._is_fullscreen_runtime = True
        try:
            self._prev_geometry = self.geometry()
        except Exception:
            self._prev_geometry = None

        try:
            self.attributes("-fullscreen", True)
        except Exception:
            try:
                self.state("zoomed")
            except Exception:
                pass

        try:
            self.tabs.set("Runtime")
        except Exception:
            pass

        try:
            self.runtime_side.grid_remove()
        except Exception:
            pass

        try:
            self.msg1.grid_remove()
        except Exception:
            pass

        self.runtime_output_frame.grid_configure(
            row=0,
            column=0,
            columnspan=2,
            sticky="nsew",
            padx=0,
            pady=0,
        )

        try:
            self.runtime_output_frame.configure(corner_radius=0)
            self.runtime_output.configure(corner_radius=0)
        except Exception:
            pass

        try:
            self.bind("<Escape>", self._exit_runtime_fullscreen)
        except Exception:
            pass

        self.runtime_output.grid_configure(padx=0, pady=0)
        self.runtime_neg_canvas.grid_configure(padx=0, pady=0)
        self.after(80, self._neg_refresh_after_geometry_change)

    def _exit_runtime_fullscreen(self, _evt=None):
        self._is_fullscreen_runtime = False

        try:
            self.attributes("-fullscreen", False)
        except Exception:
            pass

        if self._prev_geometry:
            try:
                self.geometry(self._prev_geometry)
            except Exception:
                pass

        try:
            self.unbind("<Escape>")
        except Exception:
            pass

        try:
            self.runtime_side.grid()
        except Exception:
            pass

        try:
            self.msg1.grid()
        except Exception:
            pass

        self.runtime_output_frame.grid_configure(
            row=0,
            column=0,
            columnspan=1,
            sticky="nsew",
            padx=(15, 10),
            pady=15,
        )

        try:
            self.runtime_output_frame.configure(corner_radius=15)
            self.runtime_output.configure(corner_radius=13)
        except Exception:
            pass

        self.runtime_output.grid_configure(padx=8, pady=8)
        self.runtime_neg_canvas.grid_configure(padx=8, pady=8)
        self.after(80, self._neg_refresh_after_geometry_change)

    def set_runtime_frame(self, frame_rgb):
        try:
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(img)
        except Exception:
            return

        self._runtime_output_imgtk = imgtk
        self.runtime_output.configure(image=imgtk, text="")
        self.runtime_output.image = imgtk

    def update_camera_info(self):
        info = self.camera.get_info()

        device = info.get("device") or "—"
        fps = info.get("fps")
        width = info.get("width")
        height = info.get("height")
        backend = info.get("backend") or "—"
        opened = info.get("is_opened")

        self.cam_info_device.configure(text=f"Périphérique  ·  {device}")
        self.cam_info_fps.configure(text=f"FPS signal  ·  {fps if fps is not None else '—'}")
        if width is not None and height is not None:
            self.cam_info_res.configure(text=f"Résolution  ·  {width}×{height}")
            try:
                self.preview_subtitle.configure(
                    text=f"Analyse 640×480  ·  capteur {int(width)}×{int(height)}"
                )
            except Exception:
                self.preview_subtitle.configure(text="Flux traité en 640×480")
        else:
            self.cam_info_res.configure(text="Résolution  ·  —")
            self.preview_subtitle.configure(text="Flux traité en 640×480")
        self.cam_info_backend.configure(text=f"Backend  ·  {backend}")
        if opened and self.running:
            self.cam_info_status.configure(
                text="Statut  ·  Actif",
                text_color=_THEME["success"],
            )
        else:
            self.cam_info_status.configure(
                text="Statut  ·  Inactif",
                text_color=_THEME["muted"],
            )

        m = self.vision.get_live_metrics()
        choice = self._ai_model_choice

        self.perf_model_active.configure(text=f"Modèle actif  ·  {choice}")

        fps_p = m.get("fps_pipeline")
        avg_p = m.get("avg_process_ms")
        if fps_p is not None and avg_p is not None:
            self.perf_throughput.configure(
                text=f"Débit : {fps_p:.1f} FPS · latence moy. {avg_p:.1f} ms"
            )
        elif avg_p is not None:
            self.perf_throughput.configure(text=f"Latence moy. traitement : {avg_p:.1f} ms")
        else:
            self.perf_throughput.configure(text="Débit : — (démarrez la caméra)")

        if choice == "CNN":
            ai = m.get("avg_infer_ms")
            li = m.get("last_infer_ms")
            p_ai = f"{ai:.1f} ms" if ai is not None else "—"
            p_li = f"{li:.1f} ms" if li is not None else "—"
            self.perf_cnn_infer.configure(text=f"Inférence CNN : moy. {p_ai} · dernière {p_li}")
        elif choice == "ResNet50":
            ai = m.get("avg_infer_ms")
            li = m.get("last_infer_ms")
            p_ai = f"{ai:.1f} ms" if ai is not None else "—"
            p_li = f"{li:.1f} ms" if li is not None else "—"
            self.perf_cnn_infer.configure(
                text=f"Inférence ResNet50 : moy. {p_ai} · dernière {p_li}"
            )
        elif choice == "Landmarks":
            ai = m.get("avg_infer_ms")
            li = m.get("last_infer_ms")
            p_ai = f"{ai:.1f} ms" if ai is not None else "—"
            p_li = f"{li:.1f} ms" if li is not None else "—"
            self.perf_cnn_infer.configure(
                text=f"Inférence Landmarks : moy. {p_ai} · dernière {p_li}"
            )
        else:
            self.perf_cnn_infer.configure(text="Inférence modèle : —")

    def on_close(self):
        if self._neg_configure_job is not None:
            try:
                self.after_cancel(self._neg_configure_job)
            except Exception:
                pass
            self._neg_configure_job = None
        self._stop_neg_cartoon_animation()
        self.running = False
        self.camera.release()
        self.destroy()