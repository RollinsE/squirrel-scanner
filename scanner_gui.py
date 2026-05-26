import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

os.environ["WANDB_DISABLED"] = "true"
os.environ["YOLO_AUTO_UPDATE"] = "false"
sys.modules["wandb"] = None

import torch

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
if hasattr(torch.serialization, "load"):
    torch.serialization.load = _patched_torch_load

from src.logger import setup_logger
from src.scan_det_any import scan_videos_with_champion_detector


def resource_path(relative_path: str) -> str:
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


def infer_detector_type_from_path(detector_path: str) -> str:
    name = Path(detector_path).name.lower()
    suffix = Path(detector_path).suffix.lower()
    if "fasterrcnn" in name or suffix == ".pth":
        return "FasterRCNN"
    if "rtdetr" in name or "rt-detr" in name:
        return "RT-DETR"
    if "yolo" in name or "yolov" in name or suffix == ".pt":
        return "YOLOv8"
    return "Auto"


class TextHandler:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, msg: str):
        self.text_widget.after(0, self._append, msg)

    def flush(self):
        pass

    def _append(self, msg: str):
        self.text_widget.insert(tk.END, msg)
        self.text_widget.see(tk.END)


class SimpleGuiLogger:
    def __init__(self, text_widget):
        self.stream = TextHandler(text_widget)

    def emit(self, level: str, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stream.write(f"{ts} | {level.upper()} | {msg}\n")

    def info(self, msg):
        self.emit("INFO", msg)

    def warning(self, msg):
        self.emit("WARN", msg)

    def error(self, msg):
        self.emit("ERROR", msg)

    def debug(self, msg):
        self.emit("DEBUG", msg)

    def exception(self, msg):
        self.emit("ERROR", msg)


class SimpleFileLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, level: str, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} | {level.upper()} | {msg}\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)

    def info(self, msg):
        self.emit("INFO", msg)

    def warning(self, msg):
        self.emit("WARN", msg)

    def error(self, msg):
        self.emit("ERROR", msg)

    def debug(self, msg):
        self.emit("DEBUG", msg)

    def exception(self, msg):
        self.emit("ERROR", msg)


class MultiLogger:
    def __init__(self, run_logger, gui_logger, simple_file_logger):
        self.run_logger = run_logger
        self.gui_logger = gui_logger
        self.simple_file_logger = simple_file_logger

    def _emit(self, method: str, msg: str):
        for logger in (self.run_logger, self.gui_logger, self.simple_file_logger):
            try:
                getattr(logger, method)(msg)
            except Exception:
                pass

    def info(self, msg):
        self._emit("info", msg)

    def warning(self, msg):
        self._emit("warning", msg)

    def error(self, msg):
        self._emit("error", msg)

    def debug(self, msg):
        self._emit("debug", msg)

    def exception(self, msg):
        self._emit("exception", msg)


class ScannerGUI:
    DETECTOR_TYPES = ("Auto", "YOLOv8", "YOLOv11", "YOLOv26", "RT-DETR", "FasterRCNN")

    def __init__(self, root):
        self.root = root
        self.root.title("Red Squirrel Scanner")
        self.root.geometry("1000x680")
        self.root.minsize(900, 580)

        self.detector_path = tk.StringVar()
        self.detector_type = tk.StringVar(value="Auto")
        self.resolved_detector = tk.StringVar(value="Resolved detector: not selected")

        self.video_dir = tk.StringVar()
        self.output_run_dir = tk.StringVar()
        self.raw_dataset = tk.StringVar()

        self.target_class = tk.StringVar(value="red")
        self.conf = tk.StringVar(value="0.6")
        self.iou = tk.StringVar(value="0.5")
        self.frame_skip = tk.StringVar(value="2")
        self.scan_imgsz = tk.StringVar(value="512")
        self.save_every_hit = tk.BooleanVar(value=False)

        self.rescue_on_negative = tk.BooleanVar(value=False)
        self.rescue_conf = tk.StringVar(value="0.35")
        self.rescue_frame_skip = tk.StringVar(value="1")
        self.rescue_imgsz = tk.StringVar(value="960")

        self.min_box_area_ratio = tk.StringVar(value="0.0")
        self.max_box_area_ratio = tk.StringVar(value="0.40")
        self.max_box_aspect_ratio = tk.StringVar(value="10.0")

        self.edge_margin_ratio = tk.StringVar(value="0.05")
        self.blur_threshold = tk.StringVar(value="40")
        self.require_confirmation_for_suspicious = tk.BooleanVar(value=False)
        self.confirm_window_sec = tk.StringVar(value="1.5")
        self.confirm_iou = tk.StringVar(value="0.30")

        self.scan_thread = None
        self.cancel_event = threading.Event()
        self.raw_dataset_entry = None
        self.raw_dataset_button = None
        self._applying_defaults = False
        self.advanced_visible = tk.BooleanVar(value=False)

        self._build_ui()
        self.detector_type.trace_add("write", lambda *_: self._on_detector_type_change())
        self._update_detector_type_ui()
        self._apply_detector_defaults(force=True)

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.main_pane = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        self.main_pane.grid(row=0, column=0, sticky="nsew")

        top_container = ttk.Frame(self.main_pane, padding=8)
        top_container.columnconfigure(0, weight=1)
        top_container.rowconfigure(1, weight=1)

        actions = ttk.Frame(top_container)
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        actions.columnconfigure(5, weight=1)

        self.run_btn = ttk.Button(actions, text="Start Scan", command=self.start_scan)
        self.run_btn.grid(row=0, column=0, padx=(0, 6), sticky="w")

        self.cancel_btn = ttk.Button(actions, text="Cancel Scan", command=self.cancel_scan, state="disabled")
        self.cancel_btn.grid(row=0, column=1, padx=(0, 6), sticky="w")

        ttk.Button(actions, text="Clear Log", command=self.clear_log).grid(row=0, column=2, padx=(0, 6), sticky="w")
        ttk.Button(actions, text="Apply detector defaults", command=lambda: self._apply_detector_defaults(force=True)).grid(
            row=0, column=3, padx=(0, 12), sticky="w"
        )

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(actions, textvariable=self.status_var).grid(row=0, column=5, sticky="e")

        notebook = ttk.Notebook(top_container)
        notebook.grid(row=1, column=0, sticky="nsew")

        paths_tab = ttk.Frame(notebook, padding=10)
        paths_tab.columnconfigure(1, weight=1)
        notebook.add(paths_tab, text="Paths")

        scan_tab = ttk.Frame(notebook, padding=10)
        scan_tab.columnconfigure(0, weight=1)
        scan_tab.columnconfigure(1, weight=1)
        notebook.add(scan_tab, text="Scan")

        help_tab = ttk.Frame(notebook, padding=10)
        help_tab.columnconfigure(0, weight=1)
        help_tab.rowconfigure(0, weight=1)
        notebook.add(help_tab, text="Help")

        about_tab = ttk.Frame(notebook, padding=10)
        about_tab.columnconfigure(0, weight=1)
        about_tab.rowconfigure(0, weight=1)
        notebook.add(about_tab, text="About")

        self._build_paths_tab(paths_tab)
        self._build_scan_tab(scan_tab)
        self._build_help_tab(help_tab)
        self._build_about_tab(about_tab)

        log_container = ttk.Frame(self.main_pane, padding=(8, 0, 8, 8))
        log_container.columnconfigure(0, weight=1)
        log_container.rowconfigure(1, weight=1)

        ttk.Label(log_container, text="Log").grid(row=0, column=0, sticky="w", pady=(0, 4))

        text_frame = ttk.Frame(log_container)
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(text_frame, wrap="word", height=18)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=yscroll.set)

        self.gui_log = SimpleGuiLogger(self.log_text)

        self.main_pane.add(top_container, weight=0)
        self.main_pane.add(log_container, weight=1)
        self.root.after(100, self._set_initial_sash)
        self.root.after(500, self._set_initial_sash)

    def _build_help_tab(self, parent):
        container = ttk.Frame(parent)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        text = tk.Text(container, wrap="word", height=18, padx=8, pady=8)
        text.grid(row=0, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(container, orient="vertical", command=text.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=yscroll.set)

        text.tag_configure("title", font=("TkDefaultFont", 11, "bold"), spacing1=4, spacing3=10)
        text.tag_configure("heading", font=("TkDefaultFont", 10, "bold"), spacing1=8, spacing3=2)
        text.tag_configure("body", spacing1=0, spacing3=8)

        sections = [
            ("Detector file", "Select the model weights file you want to scan with. Use .pt for YOLO or RT-DETR and .pth for FasterRCNN."),
            ("Detector type", "Auto tries to infer the detector type from the file name. Choose the type manually if Auto gets it wrong or if you want to be explicit about which model family is being used."),
            ("Raw dataset folder", "Only needed for FasterRCNN. The scanner uses data.yaml in that folder to read class names so it can tell which detections are red squirrels, martens, and other classes."),
            ("Target class", "The class the scan is looking for. In normal use this will usually be set to red. A video is marked positive only when the chosen class is detected and accepted by the scan rules."),
            ("Confidence", "Minimum confidence score required for a detection to be considered. Higher values reduce false detections but can miss real squirrels. Lower values find more candidates but may increase false positives."),
            ("IoU", "Intersection over Union threshold used during detector post-processing. It affects how overlapping boxes are handled on each frame. In most cases 0.5 is a sensible default."),
            ("Frame skip", "The scanner checks every Nth frame. Higher values are faster but can miss brief appearances. Lower values are slower but more sensitive."),
            ("Scan image size", "The image size used for model inference. Larger values can improve detection quality, especially for small animals, but increase scan time."),
            ("Save every hit", "If off, the scanner saves the first accepted hit for a video and then moves on to the next video. If on, it can save multiple accepted hits from the same video."),
            ("Run rescue pass on negatives", "If enabled, videos with no accepted detections from the primary pass are scanned again using the rescue settings below. This can recover missed squirrels but increases scan time."),
            ("Rescue confidence / Rescue frame skip / Rescue image size", "These settings are used only during the rescue pass. Lower rescue confidence and lower rescue frame skip make the rescue pass more sensitive, while a larger rescue image size may help with small or difficult detections at the cost of speed."),
            ("Min box area ratio", "Rejects detections that are too small relative to the full frame. Useful for removing tiny specks or noise, but setting it too high can remove distant squirrels."),
            ("Max box area ratio", "Rejects detections that are too large relative to the full frame. This can help filter out close foreground blobs, hands, or large obstructions near the camera."),
            ("Max box aspect ratio", "Rejects boxes that are extremely stretched or unusually thin. This can help suppress implausible shapes that do not look like an animal."),
            ("Edge margin ratio", "Defines how much of the outer edge of the frame is treated as an edge zone. Detections too close to the border can be treated more cautiously because edge-of-frame blobs and obstructions are a common failure mode."),
            ("Blur threshold", "Used to identify very blurry detections. Higher values are stricter and will treat more blurry detections as unreliable."),
            ("Require confirmation for suspicious hits", "If enabled, detections that look risky, such as edge-near or blurry hits, must be seen again before they are accepted."),
            ("Confirm window (sec)", "How long the scanner will wait for a second matching suspicious detection. Smaller values are stricter."),
            ("Confirm IoU", "How much the second suspicious box must overlap the first suspicious box to count as the same object. Higher values are stricter and require the boxes to line up more closely."),
        ]

        text.insert("end", "Help\n", "title")
        for heading, body in sections:
            text.insert("end", heading + "\n", "heading")
            text.insert("end", body + "\n\n", "body")

        text.configure(state="disabled")

    def _build_about_tab(self, parent):
        about_text = (
            "Created by Rollins Edeh for the National Trust Brownsea Island Countryside Team to support the review of wildlife survey footage for red squirrel monitoring."
        )
        text = tk.Text(parent, wrap="word", height=18)
        text.grid(row=0, column=0, sticky="nsew")
        text.insert("1.0", about_text)
        text.configure(state="disabled")

    def _set_initial_sash(self):
        try:
            h = max(580, self.root.winfo_height())
            target = min(220, int(h * 0.35))
            self.main_pane.sashpos(0, target)
        except Exception:
            pass

    def _set_initial_sash(self):
        try:
            self.root.update_idletasks()
            h = max(580, self.root.winfo_height())
            target = min(340, max(300, int(h * 0.46)))
            self.main_pane.sashpos(0, target)
        except Exception:
            pass

    def _path_row(self, parent, row, label, variable, browse_cmd, pad_y=5):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=pad_y, padx=(0, 8))
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=pad_y)
        btn = ttk.Button(parent, text="Browse", command=browse_cmd, width=10)
        btn.grid(row=row, column=2, sticky="e", padx=(8, 0), pady=pad_y)
        parent.columnconfigure(1, weight=1)
        return entry, btn

    def _build_paths_tab(self, parent):
        self._path_row(parent, 0, "Detector file", self.detector_path, self.pick_detector_file)
        ttk.Label(parent, text="Detector type").grid(row=1, column=0, sticky="w", pady=5, padx=(0, 8))
        combo = ttk.Combobox(parent, textvariable=self.detector_type, values=self.DETECTOR_TYPES, state="readonly", width=14)
        combo.grid(row=1, column=1, sticky="w", pady=5)
        ttk.Label(parent, textvariable=self.resolved_detector, foreground="#555555").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )

        self._path_row(parent, 3, "Video folder", self.video_dir, self.pick_video_dir)
        self._path_row(parent, 4, "Output folder", self.output_run_dir, self.pick_output_dir)
        self.raw_dataset_entry, self.raw_dataset_button = self._path_row(
            parent, 5, "Raw dataset folder (FasterRCNN only)", self.raw_dataset, self.pick_raw_dataset_dir
        )

        ttk.Label(
            parent,
            text="Tip: the top pane starts compact on purpose. Drag the divider only if you need more room for the controls.",
            foreground="#555555",
            wraplength=820,
            justify="left",
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(10, 0))

    def _build_scan_tab(self, parent):
        primary = ttk.LabelFrame(parent, text="Primary pass", padding=10)
        primary.grid(row=0, column=0, sticky="nsew", columnspan=2, pady=(0, 8))
        primary.columnconfigure(1, weight=1)
        primary.columnconfigure(3, weight=1)

        ttk.Label(primary, text="Target class").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(primary, textvariable=self.target_class, width=14).grid(row=0, column=1, sticky="w", pady=4, padx=(0, 16))
        ttk.Label(primary, text="Confidence").grid(row=0, column=2, sticky="w", pady=4)
        ttk.Entry(primary, textvariable=self.conf, width=14).grid(row=0, column=3, sticky="w", pady=4)

        ttk.Label(primary, text="IoU").grid(row=1, column=0, sticky="w", pady=4)
        self.iou_entry = ttk.Entry(primary, textvariable=self.iou, width=14)
        self.iou_entry.grid(row=1, column=1, sticky="w", pady=4, padx=(0, 16))
        ttk.Label(primary, text="Frame skip").grid(row=1, column=2, sticky="w", pady=4)
        ttk.Entry(primary, textvariable=self.frame_skip, width=14).grid(row=1, column=3, sticky="w", pady=4)

        ttk.Label(primary, text="Scan image size").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(primary, textvariable=self.scan_imgsz, width=14).grid(row=2, column=1, sticky="w", pady=4, padx=(0, 16))
        ttk.Checkbutton(primary, text="Save every hit", variable=self.save_every_hit).grid(row=2, column=2, columnspan=2, sticky="w", pady=4)

        self.save_every_hit.trace_add("write", lambda *_: self._update_iou_ui())
        self._update_iou_ui()

        adv_toggle = ttk.Frame(parent)
        adv_toggle.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        adv_toggle.columnconfigure(1, weight=1)
        self.advanced_button = ttk.Button(adv_toggle, text="Advanced ▸", command=self._toggle_advanced)
        self.advanced_button.grid(row=0, column=0, sticky="w")
        ttk.Label(
            adv_toggle,
            text="Rescue pass, box filters, and suspicious-hit filters",
            foreground="#666666",
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.advanced_frame = ttk.Frame(parent)
        self.advanced_frame.columnconfigure(0, weight=1)
        self.advanced_frame.columnconfigure(1, weight=1)

        rescue = ttk.LabelFrame(self.advanced_frame, text="Rescue pass", padding=10)
        rescue.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        rescue.columnconfigure(1, weight=1)

        ttk.Checkbutton(rescue, text="Run rescue pass on negatives", variable=self.rescue_on_negative).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        ttk.Label(rescue, text="Rescue confidence").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(rescue, textvariable=self.rescue_conf, width=12).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(rescue, text="Rescue frame skip").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(rescue, textvariable=self.rescue_frame_skip, width=12).grid(row=2, column=1, sticky="w", pady=4)
        ttk.Label(rescue, text="Rescue image size").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(rescue, textvariable=self.rescue_imgsz, width=12).grid(row=3, column=1, sticky="w", pady=4)

        boxf = ttk.LabelFrame(self.advanced_frame, text="Box filters", padding=10)
        boxf.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        boxf.columnconfigure(1, weight=1)

        ttk.Label(boxf, text="Min box area ratio").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(boxf, textvariable=self.min_box_area_ratio, width=12).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(boxf, text="Max box area ratio").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(boxf, textvariable=self.max_box_area_ratio, width=12).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(boxf, text="Max box aspect ratio").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(boxf, textvariable=self.max_box_aspect_ratio, width=12).grid(row=2, column=1, sticky="w", pady=4)

        susp = ttk.LabelFrame(self.advanced_frame, text="Suspicious-hit filters", padding=10)
        susp.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 6))
        susp.columnconfigure(1, weight=1)
        susp.columnconfigure(3, weight=1)

        ttk.Label(susp, text="Edge margin ratio").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(susp, textvariable=self.edge_margin_ratio, width=12).grid(row=0, column=1, sticky="w", pady=4, padx=(0, 16))
        ttk.Label(susp, text="Blur threshold").grid(row=0, column=2, sticky="w", pady=4)
        ttk.Entry(susp, textvariable=self.blur_threshold, width=12).grid(row=0, column=3, sticky="w", pady=4)

        ttk.Checkbutton(
            susp,
            text="Require confirmation for suspicious hits",
            variable=self.require_confirmation_for_suspicious,
            command=self._update_confirmation_ui,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 4))

        ttk.Label(susp, text="Confirm window (sec)").grid(row=2, column=0, sticky="w", pady=4)
        self.confirm_window_entry = ttk.Entry(susp, textvariable=self.confirm_window_sec, width=12)
        self.confirm_window_entry.grid(row=2, column=1, sticky="w", pady=4, padx=(0, 16))
        ttk.Label(susp, text="Confirm IoU").grid(row=2, column=2, sticky="w", pady=4)
        self.confirm_iou_entry = ttk.Entry(susp, textvariable=self.confirm_iou, width=12)
        self.confirm_iou_entry.grid(row=2, column=3, sticky="w", pady=4)

        self._update_confirmation_ui()

    def _toggle_advanced(self):
        if self.advanced_visible.get():
            self.advanced_frame.grid_forget()
            self.advanced_visible.set(False)
            self.advanced_button.configure(text="Advanced ▸")
        else:
            self.advanced_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
            self.advanced_visible.set(True)
            self.advanced_button.configure(text="Advanced ▾")

    def _update_confirmation_ui(self):
        state = "normal" if self.require_confirmation_for_suspicious.get() else "disabled"
        self.confirm_window_entry.configure(state=state)
        self.confirm_iou_entry.configure(state=state)

    def _update_iou_ui(self):
        # Kept active and editable; not tied to Save every hit.
        self.iou_entry.configure(state="normal")

    def _on_detector_type_change(self):
        self._update_detector_type_ui()
        if not self._applying_defaults:
            self._apply_detector_defaults(force=True)

    def _update_detector_type_ui(self):
        selected = self.detector_type.get().strip()
        requires_dataset = selected == "FasterRCNN"
        state = "normal" if requires_dataset else "disabled"
        if self.raw_dataset_entry is not None:
            self.raw_dataset_entry.configure(state=state)
        if self.raw_dataset_button is not None:
            self.raw_dataset_button.configure(state=state)
        self._update_resolved_detector_label()

    def _update_resolved_detector_label(self):
        detector_path = self.detector_path.get().strip()
        selected = self.detector_type.get().strip()
        if not detector_path:
            self.resolved_detector.set("Resolved detector: not selected")
            return

        resolved = infer_detector_type_from_path(detector_path) if selected == "Auto" else selected
        self.resolved_detector.set(f"Resolved detector: {resolved} ({Path(detector_path).name})")

    def _get_effective_detector_type(self) -> str:
        detector_path = self.detector_path.get().strip()
        selected = self.detector_type.get().strip() or "Auto"
        if selected == "Auto":
            inferred = infer_detector_type_from_path(detector_path) if detector_path else "Auto"
            return inferred if inferred != "Auto" else "YOLOv8"
        return selected

    def _apply_detector_defaults(self, force=False):
        resolved = self._get_effective_detector_type()
        self._applying_defaults = True
        try:
            self.target_class.set("red")
            self.conf.set("0.7")
            self.iou.set("0.5")
            self.scan_imgsz.set("640")
            self.save_every_hit.set(False)

            if resolved == "RT-DETR":
                self.frame_skip.set("10")
            else:
                self.frame_skip.set("5")

            self.rescue_on_negative.set(False)
            self.rescue_conf.set("0.35")
            self.rescue_frame_skip.set("1")
            self.rescue_imgsz.set("960")

            self.min_box_area_ratio.set("0.0")
            self.max_box_area_ratio.set("0.40")
            self.max_box_aspect_ratio.set("10.0")

            self.edge_margin_ratio.set("0.05")
            self.blur_threshold.set("40")
            self.require_confirmation_for_suspicious.set(False)
            self.confirm_window_sec.set("1.5")
            self.confirm_iou.set("0.30")
            self._update_confirmation_ui()
            self._update_iou_ui()
        finally:
            self._applying_defaults = False

    def pick_detector_file(self):
        initial_dir = os.path.dirname(self.detector_path.get()) if self.detector_path.get() else None
        path = filedialog.askopenfilename(
            title="Select detector weights file",
            initialdir=initial_dir,
            filetypes=[
                ("Model weights", "*.pt *.pth"),
                ("PyTorch model", "*.pt *.pth"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.detector_path.set(path)
            self._update_resolved_detector_label()
            self._apply_detector_defaults(force=True)

    def pick_video_dir(self):
        initial_dir = self.video_dir.get() if self.video_dir.get() else None
        path = filedialog.askdirectory(title="Select video folder", initialdir=initial_dir)
        if path:
            self.video_dir.set(path)

    def pick_output_dir(self):
        initial_dir = self.output_run_dir.get() if self.output_run_dir.get() else None
        path = filedialog.askdirectory(title="Select output folder", initialdir=initial_dir)
        if path:
            self.output_run_dir.set(path)

    def pick_raw_dataset_dir(self):
        initial_dir = self.raw_dataset.get() if self.raw_dataset.get() else None
        path = filedialog.askdirectory(title="Select raw dataset folder", initialdir=initial_dir)
        if path:
            self.raw_dataset.set(path)

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def validate_inputs(self):
        detector_path = self.detector_path.get().strip()
        detector_type = self.detector_type.get().strip() or "Auto"
        video_dir = self.video_dir.get().strip()
        output_run_dir = self.output_run_dir.get().strip()
        raw_dataset = self.raw_dataset.get().strip() or None

        if not detector_path or not os.path.isfile(detector_path):
            raise ValueError("Please select a valid detector file.")
        if Path(detector_path).suffix.lower() not in (".pt", ".pth"):
            raise ValueError("Detector file must be a .pt or .pth file.")

        resolved_type = infer_detector_type_from_path(detector_path) if detector_type == "Auto" else detector_type

        if not video_dir or not os.path.isdir(video_dir):
            raise ValueError("Please select a valid video folder.")
        if not output_run_dir:
            raise ValueError("Please select a valid output folder.")

        conf = float(self.conf.get().strip())
        iou = float(self.iou.get().strip())
        frame_skip = int(self.frame_skip.get().strip())
        target_class = self.target_class.get().strip()
        scan_imgsz = int(self.scan_imgsz.get().strip())

        rescue_on_negative = bool(self.rescue_on_negative.get())
        rescue_conf = float(self.rescue_conf.get().strip()) if rescue_on_negative else None
        rescue_frame_skip = int(self.rescue_frame_skip.get().strip()) if rescue_on_negative else None
        rescue_imgsz = int(self.rescue_imgsz.get().strip()) if rescue_on_negative else None

        min_box_area_ratio = float(self.min_box_area_ratio.get().strip())
        max_box_area_ratio = float(self.max_box_area_ratio.get().strip())
        max_box_aspect_ratio = float(self.max_box_aspect_ratio.get().strip())

        edge_margin_ratio = float(self.edge_margin_ratio.get().strip())
        blur_threshold = float(self.blur_threshold.get().strip())
        require_confirmation_for_suspicious = bool(self.require_confirmation_for_suspicious.get())
        confirm_window_sec = float(self.confirm_window_sec.get().strip())
        confirm_iou = float(self.confirm_iou.get().strip())

        if not (0.0 <= conf <= 1.0):
            raise ValueError("Confidence threshold must be between 0 and 1.")
        if not (0.0 <= iou <= 1.0):
            raise ValueError("IoU threshold must be between 0 and 1.")
        if frame_skip < 1:
            raise ValueError("Frame skip must be at least 1.")
        if scan_imgsz < 32:
            raise ValueError("Scan image size must be at least 32.")
        if not target_class:
            raise ValueError("Target class cannot be empty.")

        if rescue_on_negative:
            if rescue_conf is None or not (0.0 <= rescue_conf <= 1.0):
                raise ValueError("Rescue confidence must be between 0 and 1.")
            if rescue_frame_skip is None or rescue_frame_skip < 1:
                raise ValueError("Rescue frame skip must be at least 1.")
            if rescue_imgsz is None or rescue_imgsz < 32:
                raise ValueError("Rescue image size must be at least 32.")

        if min_box_area_ratio < 0.0:
            raise ValueError("Min box area ratio must be >= 0.")
        if max_box_area_ratio <= 0.0:
            raise ValueError("Max box area ratio must be > 0.")
        if min_box_area_ratio > max_box_area_ratio:
            raise ValueError("Min box area ratio cannot be greater than max box area ratio.")
        if max_box_aspect_ratio <= 0.0:
            raise ValueError("Max box aspect ratio must be > 0.")
        if not (0.0 <= edge_margin_ratio <= 0.5):
            raise ValueError("Edge margin ratio must be between 0 and 0.5.")
        if blur_threshold < 0.0:
            raise ValueError("Blur threshold must be >= 0.")
        if confirm_window_sec <= 0.0:
            raise ValueError("Confirm window must be > 0.")
        if not (0.0 <= confirm_iou <= 1.0):
            raise ValueError("Confirm IoU must be between 0 and 1.")

        if resolved_type == "FasterRCNN":
            if not raw_dataset or not os.path.isdir(raw_dataset):
                raise ValueError("FasterRCNN scanning needs the raw dataset folder so class names can be read from data.yaml.")
            if not os.path.isfile(os.path.join(raw_dataset, "data.yaml")):
                raise ValueError("The selected raw dataset folder must contain data.yaml")

        detector_type_map = {
            "Auto": "auto",
            "YOLOv8": "yolo",
            "YOLOv11": "yolo",
            "YOLOv26": "yolo",
            "RT-DETR": "rtdetr",
            "FasterRCNN": "fasterrcnn",
        }

        return {
            "detector_path": detector_path,
            "detector_type": detector_type_map[detector_type],
            "resolved_type": resolved_type,
            "video_dir": video_dir,
            "output_run_dir": output_run_dir,
            "raw_dataset": raw_dataset,
            "target_class": target_class,
            "conf": conf,
            "iou": iou,
            "frame_skip": frame_skip,
            "scan_imgsz": scan_imgsz,
            "save_every_hit": bool(self.save_every_hit.get()),
            "rescue_on_negative": rescue_on_negative,
            "rescue_conf": rescue_conf,
            "rescue_frame_skip": rescue_frame_skip,
            "rescue_imgsz": rescue_imgsz,
            "min_box_area_ratio": min_box_area_ratio,
            "max_box_area_ratio": max_box_area_ratio,
            "max_box_aspect_ratio": max_box_aspect_ratio,
            "edge_margin_ratio": edge_margin_ratio,
            "blur_threshold": blur_threshold,
            "require_confirmation_for_suspicious": require_confirmation_for_suspicious,
            "confirm_window_sec": confirm_window_sec,
            "confirm_iou": confirm_iou,
        }

    def start_scan(self):
        try:
            cfg = self.validate_inputs()
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self.clear_log()
        self.cancel_event.clear()
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.status_var.set("Scanning...")

        self.scan_thread = threading.Thread(target=self._run_scan_thread, args=(cfg,), daemon=True)
        self.scan_thread.start()

    def cancel_scan(self):
        if self.scan_thread is None or not self.scan_thread.is_alive():
            return
        if self.cancel_event.is_set():
            return

        self.cancel_event.set()
        self.cancel_btn.config(state="disabled")
        self.status_var.set("Cancelling...")
        self.gui_log.warning("cancel requested")

    def _run_scan_thread(self, cfg):
        try:
            os.makedirs(cfg["output_run_dir"], exist_ok=True)
            results_dir = os.path.join(cfg["output_run_dir"], "artifacts", "scan")
            os.makedirs(results_dir, exist_ok=True)

            run_dir = cfg["output_run_dir"]
            run_id = os.path.basename(os.path.normpath(run_dir)) or "scan"
            run_logger = setup_logger(run_id, run_dir, log_level="INFO")
            simple_file_logger = SimpleFileLogger(os.path.join(results_dir, "scan.log"))
            LOG = MultiLogger(run_logger, self.gui_log, simple_file_logger)

            LOG.info(
                f"[START] component=ScanGUI detector_file={Path(cfg['detector_path']).name} "
                f"detector_type={cfg['resolved_type']} video_dir={cfg['video_dir']} output_run_dir={cfg['output_run_dir']}"
            )
            LOG.info(f"log_export={os.path.join(results_dir, 'scan.log')}")

            report = scan_videos_with_champion_detector(
                detector_run_dir=None,
                detector_path=cfg["detector_path"],
                detector_type=cfg["detector_type"],
                video_dir=cfg["video_dir"],
                output_run_dir=cfg["output_run_dir"],
                target_class=cfg["target_class"],
                conf=cfg["conf"],
                iou=cfg["iou"],
                frame_skip=cfg["frame_skip"],
                save_every_hit=cfg["save_every_hit"],
                raw_dataset=cfg["raw_dataset"],
                imgsz=cfg["scan_imgsz"],
                rescue_on_negative=cfg["rescue_on_negative"],
                rescue_conf=cfg["rescue_conf"],
                rescue_frame_skip=cfg["rescue_frame_skip"],
                rescue_imgsz=cfg["rescue_imgsz"],
                min_box_area_ratio=cfg["min_box_area_ratio"],
                max_box_area_ratio=cfg["max_box_area_ratio"],
                max_box_aspect_ratio=cfg["max_box_aspect_ratio"],
                edge_margin_ratio=cfg["edge_margin_ratio"],
                blur_threshold=cfg["blur_threshold"],
                require_confirmation_for_suspicious=cfg["require_confirmation_for_suspicious"],
                confirm_window_sec=cfg["confirm_window_sec"],
                confirm_iou=cfg["confirm_iou"],
                cancel_event=self.cancel_event,
                LOG=LOG,
            )

            total = report.get("total_snapshots", 0)
            vids = report.get("videos_found", 0)
            completed = len(report.get("videos", []))
            rescued = sum(1 for v in report.get("videos", []) if v.get("scan_pass") == "rescue")

            if report.get("cancelled"):
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Scan cancelled",
                        f"Scan cancelled.\n\n"
                        f"Videos found: {vids}\n"
                        f"Videos completed: {completed}\n"
                        f"Snapshots saved: {total}\n"
                        f"Cancelled at: {report.get('cancelled_at_video')} ({report.get('cancelled_at_pass')})\n\n"
                        f"Results: {results_dir}"
                    ),
                )
            else:
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Scan complete",
                        f"Finished scanning.\n\n"
                        f"Videos found: {vids}\n"
                        f"Videos completed: {completed}\n"
                        f"Snapshots saved: {total}\n"
                        f"Rescue-pass hits: {rescued}\n\n"
                        f"Results: {results_dir}"
                    ),
                )

        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            self.gui_log.error(err)
            self.root.after(0, lambda: messagebox.showerror("Scan failed", str(e)))
        finally:
            self.root.after(0, self._finish_ui)

    def _finish_ui(self):
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.status_var.set("Ready")
        self.scan_thread = None


def main():
    root = tk.Tk()

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SquirrelScanner")

    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass

    icon_set = False

    try:
        ico_path = resource_path("squirrel.ico")
        if os.path.exists(ico_path):
            root.iconbitmap(default=ico_path)
            icon_set = True
    except Exception as e:
        print(f"Failed to load squirrel.ico: {e}")

    if not icon_set:
        try:
            png_path = resource_path("squirrel.png")
            if os.path.exists(png_path):
                icon_img = tk.PhotoImage(file=png_path)
                root.iconphoto(True, icon_img)
                root._icon_img = icon_img
        except Exception as e:
            print(f"Failed to load squirrel.png: {e}")

    ScannerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
