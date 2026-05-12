import customtkinter as ctk
import threading
from PIL import Image, ImageTk

from core.camera import Camera
from core.vision import Vision

_AI_MODEL_OPTIONS = ("CNN", "Landmarks", "ResNet50")


class EmotionApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("App")
        self.geometry("1200x750")
        self.minsize(1100, 700)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.camera = Camera()
        self.vision = Vision()

        self.running = False
        self._ui_frame_count = 0
        self._small_preview_enabled = True
        self._ai_model_choice = _AI_MODEL_OPTIONS[0]
        self._settings_menu = None
        self._fullscreen_win = None
        self._fullscreen_label = None
        self._runtime_output_imgtk = None
        self._camera_indices = []

        self.build_ui()

    def build_ui(self):

        self.topbar = ctk.CTkFrame(self, height=70, corner_radius=0)
        self.topbar.pack(fill="x")

        self.settings_btn = ctk.CTkButton(
            self.topbar,
            text="⚙",
            width=44,
            height=32,
            command=self.open_runtime_settings,
        )
        self.settings_btn.pack(side="right", padx=(0, 10), pady=18)

        self.play_btn = ctk.CTkButton(
            self.topbar,
            text="▶",
            width=44,
            height=32,
            command=self.on_play_clicked,
        )
        self.play_btn.pack(side="right", padx=(0, 10), pady=18)

        self.toggle = ctk.CTkSwitch(
            self.topbar,
            text="Caméra",
            command=self.toggle_camera
        )
        self.toggle.pack(side="right", padx=(0, 25), pady=22)

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=15, pady=15)

        self.tab1 = self.tabs.add("Runtime")
        self.tab2 = self.tabs.add("Configuration caméra")

        self.tab1.grid_columnconfigure(0, weight=3)
        self.tab1.grid_columnconfigure(1, weight=1)
        self.tab1.grid_rowconfigure(0, weight=1)

        self.runtime_output_frame = ctk.CTkFrame(
            self.tab1,
            corner_radius=15,
            border_width=2,
            border_color="black",
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

        self.runtime_side = ctk.CTkFrame(
            self.tab1,
            corner_radius=15,
            border_width=2,
            border_color="black",
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
            text_color="#cfcfcf",
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

        self.small_face = ctk.CTkLabel(self.runtime_side, text="—", width=280, height=200, corner_radius=15)
        self.small_face.grid(row=5, column=0, sticky="ew", padx=15, pady=(0, 15))

        self.msg1 = ctk.CTkLabel(self.tab1, text="La caméra est actuellement éteinte", text_color="red")
        self.msg1.grid(row=1, column=0, columnspan=2, sticky="w", padx=15, pady=(0, 12))

        self.tab2.grid_columnconfigure(0, weight=5)
        self.tab2.grid_columnconfigure(1, weight=3)
        self.tab2.grid_rowconfigure(0, weight=1)

        _muted = ("#5c5f66", "#9aa0a8")
        _card = ("gray92", "gray22")
        _card_border = ("#dcdcdc", "#3d3d3d")
        _accent = "#1f8ceb"

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
            fg_color=("gray90", "gray19"),
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
            fg_color=("gray96", "gray16"),
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
            text="Réglages",
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
            ctk.CTkFrame(parent, height=1, fg_color=("#e4e4e4", "#383838")).pack(fill="x", padx=12, pady=(4, 8))

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
        self.perf_cnn_infer.pack(anchor="w", padx=14, pady=2)
        self.perf_last_pred = ctk.CTkLabel(
            self.card_perf,
            text="Dernière prédiction : —",
            justify="left",
            anchor="w",
            font=_spec_font,
            text_color=_muted,
            wraplength=_wrap,
        )
        self.perf_last_pred.pack(anchor="w", padx=14, pady=(2, 14))

        self.msg2 = ctk.CTkLabel(
            self.tab2,
            text="La caméra est actuellement éteinte",
            text_color="#e74c3c",
            font=ctk.CTkFont(size=12),
        )
        self.msg2.grid(row=1, column=0, columnspan=2, sticky="w", padx=20, pady=(0, 10))

        self.update_warning()
        self.refresh_camera_list()
        self.update_camera_info()
        self.apply_small_preview_visibility()

    def on_play_clicked(self):
        # Synchronise le bouton Play/Pause avec le switch existant.
        if self.running:
            self.toggle.deselect()
        else:
            self.toggle.select()
        self.toggle_camera()

    def toggle_camera(self):

        if self.toggle.get() == 1:
            self.camera.start()
            self.running = True
            threading.Thread(target=self.update_frame, daemon=True).start()
        else:
            self.running = False
            self.camera.release()

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

        # Stop proprement le flux actuel avant de changer la source.
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

    def update_ui(self, frame, face):

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(img)

        self.live_view.configure(image=imgtk, text="")
        self.live_view.image = imgtk

        if face is not None:
            face_img = Image.fromarray(face)
            face_img = face_img.resize((300, 200))
            face_tk = ImageTk.PhotoImage(face_img)

            if self._small_preview_enabled:
                self.small_face.configure(image=face_tk, text="")
                self.small_face.image = face_tk
        else:
            if self._small_preview_enabled:
                self.small_face.configure(text="—", image=None)

        self._ui_frame_count += 1
        if self._ui_frame_count % 15 == 0:
            self.update_camera_info()

    def update_warning(self):

        if self.toggle.get() == 1:
            self.msg1.configure(text="")
            self.msg2.configure(text="")
            self.play_btn.configure(text="⏸")
        else:
            self.msg1.configure(text="La caméra est actuellement éteinte")
            self.msg2.configure(text="La caméra est actuellement éteinte")
            self.play_btn.configure(text="▶")

    def apply_small_preview_visibility(self):
        if self._small_preview_enabled:
            try:
                self.small_face.grid()
            except Exception:
                pass
        else:
            try:
                self.small_face.grid_remove()
            except Exception:
                pass

    def open_runtime_settings(self):
        if self._settings_menu is not None and self._settings_menu.winfo_exists():
            self._settings_menu.focus()
            return

        win = ctk.CTkToplevel(self)
        win.title("Runtime - Settings")
        win.geometry("380x300")
        win.resizable(False, False)
        try:
            win.transient(self)
            win.grab_set()
        except Exception:
            pass

        title = ctk.CTkLabel(win, text="Options runtime", font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(anchor="w", padx=15, pady=(15, 10))

        self._small_preview_switch = ctk.CTkSwitch(
            win,
            text="Afficher la petite fenêtre de retour caméra",
            command=self.on_small_preview_toggled,
        )
        if self._small_preview_enabled:
            self._small_preview_switch.select()
        else:
            self._small_preview_switch.deselect()
        self._small_preview_switch.pack(anchor="w", padx=15, pady=10)

        model_title = ctk.CTkLabel(win, text="Modèle d'IA", font=ctk.CTkFont(size=14, weight="bold"))
        model_title.pack(anchor="w", padx=15, pady=(8, 4))

        model_hint = ctk.CTkLabel(
            win,
            text="CNN : models/CNN/models/best_model.pth · ResNet50 : models/transfer_learning/best_model.pth",
            justify="left",
            text_color="#a0a0a0",
            wraplength=340,
        )
        model_hint.pack(anchor="w", padx=15, pady=(0, 6))

        self._ai_model_menu = ctk.CTkOptionMenu(
            win,
            values=list(_AI_MODEL_OPTIONS),
            command=self.on_ai_model_selected,
            width=280,
        )
        self._ai_model_menu.set(self._ai_model_choice)
        self._ai_model_menu.pack(anchor="w", padx=15, pady=(0, 10))

        close_btn = ctk.CTkButton(win, text="Fermer", width=100, command=win.destroy)
        close_btn.pack(anchor="e", padx=15, pady=(10, 15))

        self._settings_menu = win

    def on_small_preview_toggled(self):
        try:
            self._small_preview_enabled = bool(self._small_preview_switch.get())
        except Exception:
            self._small_preview_enabled = True
        self.apply_small_preview_visibility()

    def on_ai_model_selected(self, value: str):
        if value in _AI_MODEL_OPTIONS:
            self._ai_model_choice = value
        try:
            self.update_camera_info()
        except Exception:
            pass

    # ================= FULLSCREEN =================
    def open_fullscreen(self):
        if self._fullscreen_win is not None and self._fullscreen_win.winfo_exists():
            self._fullscreen_win.focus()
            return

        win = ctk.CTkToplevel(self)
        win.title("Fullscreen")
        try:
            win.attributes("-fullscreen", True)
        except Exception:
            try:
                win.state("zoomed")
            except Exception:
                pass

        lbl = ctk.CTkLabel(win, text="")
        lbl.pack(fill="both", expand=True)

        def _close(_evt=None):
            try:
                win.destroy()
            except Exception:
                pass

        win.bind("<Escape>", _close)
        win.bind("<Button-1>", _close)

        self._fullscreen_win = win
        self._fullscreen_label = lbl

        try:
            current = getattr(self.runtime_output, "image", None)
            if current is not None:
                self._fullscreen_label.configure(image=current, text="")
                self._fullscreen_label.image = current
            else:
                self._fullscreen_label.configure(
                    text="",
                    justify="center",
                )
        except Exception:
            pass

        def _on_destroy(_evt=None):
            self._fullscreen_win = None
            self._fullscreen_label = None

        win.bind("<Destroy>", _on_destroy)

    def set_runtime_frame(self, frame_rgb):
        """
        Prévu pour la vidéo choisie par le modèle IA.
        `frame_rgb` doit être une image numpy en RGB.
        """
        try:
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(img)
        except Exception:
            return

        self._runtime_output_imgtk = imgtk
        self.runtime_output.configure(image=imgtk, text="")
        self.runtime_output.image = imgtk

        if self._fullscreen_label is not None and self._fullscreen_win is not None:
            try:
                self._fullscreen_label.configure(image=imgtk, text="")
                self._fullscreen_label.image = imgtk
            except Exception:
                pass

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
                text_color="#2ecc71",
            )
        else:
            self.cam_info_status.configure(
                text="Statut  ·  Inactif",
                text_color="#95a5a6",
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
            em = m.get("last_emotion")
            cf = m.get("last_confidence")
            if m.get("last_face_seen") and em is not None and cf is not None:
                self.perf_last_pred.configure(text=f"Dernière prédiction : {em} ({cf:.0%})")
            else:
                self.perf_last_pred.configure(text="Dernière prédiction : — (pas de visage)")
        elif choice == "ResNet50":
            ai = m.get("avg_infer_ms")
            li = m.get("last_infer_ms")
            p_ai = f"{ai:.1f} ms" if ai is not None else "—"
            p_li = f"{li:.1f} ms" if li is not None else "—"
            self.perf_cnn_infer.configure(
                text=f"Inférence ResNet50 : moy. {p_ai} · dernière {p_li}"
            )
            em = m.get("last_emotion")
            cf = m.get("last_confidence")
            if m.get("last_face_seen") and em is not None and cf is not None:
                self.perf_last_pred.configure(text=f"Dernière prédiction : {em} ({cf:.0%})")
            else:
                self.perf_last_pred.configure(text="Dernière prédiction : — (pas de visage)")
        else:
            self.perf_cnn_infer.configure(text="Inférence modèle : — (mode Landmarks)")
            self.perf_last_pred.configure(text="Confiance modèle : —")

    def on_close(self):
        self.running = False
        self.camera.release()
        self.destroy()