"""
gui.py
======
Scopo: interfaccia grafica Tkinter per l'analisi video di crisi epilettiche.
       Versione per informatici — finestra unica con tutte le funzionalità.

Layout:
  ┌─────────────────────────────────────────────────────┐
  │  [Carica Video]  path/al/video.mp4                  │
  ├──────────────────────────┬──────────────────────────┤
  │                          │  Parametri               │
  │   Preview primo frame    │  Min durata: [40]s       │
  │   (con ROI disegnabile)  │  Skip inizio: [35]s      │
  │                          │  Conf. window: [12]s     │
  │                          │  Conf. ratio: [0.85]     │
  │                          │  Confirm frames: [12]    │
  │                          ├──────────────────────────┤
  │                          │  [Seleziona ROI]         │
  │                          │  [Avvia Analisi]         │
  ├──────────────────────────┴──────────────────────────┤
  │  ████████████░░░░░░░░  65%   t=150s                 │
  ├─────────────────────────────────────────────────────┤
  │  Log:                                               │
  │  > Modello caricato                                 │
  │  > *** ONSET @ 37.2s ***                            │
  │  > *** OFFSET @ 86.8s ***                           │
  ├─────────────────────────────────────────────────────┤
  │  RISULTATI:  1 crisi rilevata                       │
  │  Onset: 37.2s (f.1116)  Offset: 86.8s  Durata: 49s │
  │  [Salva CSV]  [Apri cartella output]                │
  └─────────────────────────────────────────────────────┘

Utilizzo:
    python src/gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import cv2
import sys
import os
from pathlib import Path
from PIL import Image, ImageTk
import torch
import numpy as np
from collections import deque
from datetime import datetime
import csv
import time

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from model      import CNNLSTM
from transforms import eval_transforms

def get_resource_path(relative_path):
    """
    Restituisce il path assoluto a una risorsa, compatibile sia con
    l'esecuzione normale da terminale che con l'eseguibile PyInstaller.

    PyInstaller estrae i file in una cartella temporanea (_MEIPASS)
    durante l'esecuzione — questa funzione trova quella cartella
    automaticamente. In modalità normale usa la cartella del progetto.
    """
    if hasattr(sys, '_MEIPASS'):
        # modalità eseguibile PyInstaller
        base_path = Path(sys._MEIPASS)
    else:
        # modalità sviluppo — cartella root del progetto
        base_path = Path(__file__).parent.parent

    return str(base_path / relative_path)

# Costanti
CROP_SIZE          = 210
DEFAULT_CHECKPOINT = get_resource_path(
    'runs/20260501_160105/best_model.pt'
)
DEFAULT_WEIGHTS = get_resource_path(
    'model_weights/mobilenet_v3_small_imagenet.pth'
)
PREVIEW_W          = 480
PREVIEW_H          = 360

class Tooltip:
    """
    Tooltip che appare al passaggio del mouse su un widget.
    Mostra un testo descrittivo in una piccola finestra gialla.
    """

    def __init__(self, widget, text):
        self.widget  = widget
        self.text    = text
        self.tooltip = None
        widget.bind('<Enter>', self._show)
        widget.bind('<Leave>', self._hide)

    def _show(self, event=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)   # niente bordi finestra
        self.tooltip.wm_geometry(f"+{x}+{y}")

        lbl = tk.Label(
            self.tooltip,
            text       = self.text,
            justify    = 'left',
            background = '#FFFACD',   # giallo chiaro
            foreground = '#333333',
            relief     = 'solid',
            borderwidth= 1,
            wraplength = 280,         # va a capo dopo 280px
            font       = ('', 9),
            padx       = 6,
            pady       = 4,
        )
        lbl.pack()

    def _hide(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

# Classe principale GUI

class EpilepsyGUI:

    def __init__(self, root):
        self.root       = root
        self.root.title("Epilepsy Seizure Detector — v1.0")
        self.root.resizable(True, True)
        self.root.minsize(900, 700)

        # stato interno
        self.video_path    = None
        self.cap           = None
        self.first_frame   = None
        self.roi           = None           # (x, y, w, h) coordinate originali
        self.model         = None
        self.events        = []
        self.events_filtered = []
        self.msg_queue     = queue.Queue()  # comunicazione thread → GUI
        self.running       = False

        # variabili Tkinter per i parametri
        self.var_checkpoint    = tk.StringVar(value=DEFAULT_CHECKPOINT)
        self.var_weights       = tk.StringVar(value=DEFAULT_WEIGHTS)
        self.var_threshold     = tk.DoubleVar(value=0.874)
        self.var_min_duration  = tk.DoubleVar(value=40.0)
        self.var_skip_seconds  = tk.DoubleVar(value=35.0)
        self.var_conf_window   = tk.DoubleVar(value=12.0)
        self.var_conf_ratio    = tk.DoubleVar(value=0.85)
        self.var_confirm       = tk.IntVar(value=12)
        self.var_frame_step    = tk.IntVar(value=6)
        self.var_seq_len       = tk.IntVar(value=20)
        self.var_smooth_window = tk.IntVar(value=10)
        self.var_progress      = tk.DoubleVar(value=0.0)
        self.var_progress_lbl  = tk.StringVar(value="")

        self._build_ui()
        self._start_queue_polling()

    # Costruzione UI

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 4}

        # Barra superiore: carica video
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill='x', **pad)

        ttk.Button(top_frame, text="📂  Carica Video",
                   command=self._load_video).pack(side='left')
        self.lbl_video = ttk.Label(top_frame,
                                   text="Nessun video caricato",
                                   foreground='gray')
        self.lbl_video.pack(side='left', padx=10)

        ttk.Button(top_frame, text="⚙  Checkpoint",
                   command=self._load_checkpoint).pack(side='right')

        ttk.Separator(self.root, orient='horizontal').pack(fill='x')

        # Corpo centrale: preview + parametri
        body = ttk.Frame(self.root)
        body.pack(fill='both', expand=True, **pad)

        # colonna sinistra — preview
        left = ttk.LabelFrame(body, text="Preview frame")
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))

        self.canvas = tk.Canvas(left, width=PREVIEW_W, height=PREVIEW_H,
                                bg='#1a1a1a', cursor='crosshair')
        self.canvas.pack(fill='both', expand=True, padx=4, pady=4)
        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_press)
        self.canvas.bind("<B1-Motion>",       self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)

        self.lbl_roi = ttk.Label(left, text="ROI: non selezionata",
                                 foreground='gray')
        self.lbl_roi.pack(pady=2)

        # colonna destra — parametri + bottoni
        right = ttk.Frame(body)
        right.pack(side='right', fill='y', padx=(6, 0))

        # parametri principali
        params_frame = ttk.LabelFrame(right, text="Parametri analisi")
        params_frame.pack(fill='x', pady=(0, 6))

        # parametri principali
        self._param_row(params_frame, "Min durata crisi (s):",
            self.var_min_duration, 0,
            "Durata minima in secondi perché una crisi venga considerata reale.\n"
            "Tutti gli eventi più brevi vengono scartati automaticamente.\n"
            "Le crisi epilettiche nei topi durano in genere 45-80 secondi.\n"
            "➜ Abbassare se le crisi del topo sono più brevi del solito.")

        self._param_row(params_frame, "Skip inizio video (s):",
            self.var_skip_seconds, 1,
            "Secondi iniziali del video da ignorare durante l'analisi.\n"
            "Serve ad evitare falsi allarmi: nei primi secondi il topo\n"
            "si muove normalmente ma il modello può confondersi.\n"
            "➜ Aumentare se compaiono falsi allarmi all'inizio del video.")

        self._param_row(params_frame, "Finestra confidenza (s):",
            self.var_conf_window, 2,
            "Finestra temporale (in secondi) usata per confermare una crisi.\n"
            "Il modello richiede che per tutta questa finestra i segnali\n"
            "di crisi siano presenti in modo continuativo, non isolato.\n"
            "➜ Aumentare per essere più severi nel riconoscimento.")

        self._param_row(params_frame, "Ratio confidenza [0-1]:",
            self.var_conf_ratio, 3,
            "Percentuale minima di segnali positivi nella finestra di\n"
            "confidenza per considerare l'evento come crisi reale.\n"
            "Esempio: 0.85 significa che almeno l'85% dei segnali nella\n"
            "finestra devono indicare crisi.\n"
            "➜ Avvicinare a 1.0 per ridurre i falsi allarmi.")

        self._param_row(params_frame, "Confirm frames:",
            self.var_confirm, 4,
            "Numero di rilevamenti consecutivi necessari prima di\n"
            "dichiarare l'inizio di una crisi.\n"
            "Ogni 'frame' corrisponde a circa 0.2 secondi di video.\n"
            "Esempio: 12 frames = ~2.4 secondi consecutivi di segnale.\n"
            "➜ Aumentare per evitare falsi allarmi da movimenti bruschi.")

        self._param_row(params_frame, "Threshold:",
            self.var_threshold, 5,
            "Soglia di probabilità oltre la quale il modello considera\n"
            "un momento come 'crisi'. Calcolata automaticamente durante\n"
            "l'addestramento — modificare con cautela.\n"
            "➜ Abbassare per rilevare più crisi (ma più falsi allarmi).\n"
            "➜ Alzare per avere meno falsi allarmi (ma rischio di perdere crisi).")

        # parametri avanzati (collassabili)
        adv_frame = ttk.LabelFrame(right, text="Parametri avanzati")
        adv_frame.pack(fill='x', pady=(0, 6))

        # parametri avanzati
        self._param_row(adv_frame, "Frame step (subsampling):",
            self.var_frame_step, 0,
            "Quanti fotogrammi saltare tra un'analisi e la successiva.\n"
            "Con un video a 30fps e step=6, si analizzano 5 fotogrammi\n"
            "al secondo. Valori più alti = analisi più veloce ma meno\n"
            "precisa. Non modificare salvo necessità di velocità.")

        self._param_row(adv_frame, "Seq length:",
            self.var_seq_len, 1,
            "Quanti fotogrammi consecutivi il modello analizza insieme\n"
            "per ogni predizione. Un valore più alto dà più contesto\n"
            "temporale ma rallenta l'analisi su PC senza scheda grafica.\n"
            "Non modificare salvo indicazione tecnica.")

        self._param_row(adv_frame, "Smooth window:",
            self.var_smooth_window, 2,
            "Dimensione della 'media mobile' applicata ai risultati.\n"
            "Smussa le fluttuazioni rapide evitando che un singolo\n"
            "fotogramma anomalo influenzi il risultato finale.\n"
            "Non modificare salvo indicazione tecnica.")

        # bottoni azione
        btn_frame = ttk.Frame(right)
        btn_frame.pack(fill='x', pady=4)

        ttk.Button(btn_frame, text="🖱  Seleziona ROI (finestra separata)",
                   command=self._select_roi_window).pack(fill='x', pady=2)

        self.btn_start = ttk.Button(btn_frame, text="▶  Avvia Analisi",
                                    command=self._start_analysis,
                                    state='disabled')
        self.btn_start.pack(fill='x', pady=2)

        self.btn_stop = ttk.Button(btn_frame, text="⏹  Interrompi",
                                   command=self._stop_analysis,
                                   state='disabled')
        self.btn_stop.pack(fill='x', pady=2)

        ttk.Separator(self.root, orient='horizontal').pack(fill='x')

        # Barra di avanzamento
        prog_frame = ttk.Frame(self.root)
        prog_frame.pack(fill='x', padx=8, pady=4)

        self.progressbar = ttk.Progressbar(
            prog_frame, variable=self.var_progress,
            maximum=100, length=400
        )
        self.progressbar.pack(side='left', fill='x', expand=True)
        ttk.Label(prog_frame,
                  textvariable=self.var_progress_lbl,
                  width=30).pack(side='left', padx=8)

        # Log testuale
        log_frame = ttk.LabelFrame(self.root, text="Log analisi")
        log_frame.pack(fill='both', expand=False, padx=8, pady=4)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=8, state='disabled',
            font=('Courier', 10), bg='#1e1e1e', fg='#d4d4d4'
        )
        self.log_text.pack(fill='both', expand=True, padx=4, pady=4)

        # tag colori per il log
        self.log_text.tag_config('onset',  foreground='#f44747')
        self.log_text.tag_config('offset', foreground='#4ec9b0')
        self.log_text.tag_config('info',   foreground='#9cdcfe')
        self.log_text.tag_config('ok',     foreground='#b5cea8')
        self.log_text.tag_config('warn',   foreground='#ce9178')

        # Risultati
        res_frame = ttk.LabelFrame(self.root, text="Risultati")
        res_frame.pack(fill='x', padx=8, pady=4)

        self.lbl_results = ttk.Label(res_frame,
                                     text="Nessuna analisi eseguita.",
                                     foreground='gray', font=('', 10))
        self.lbl_results.pack(side='left', padx=8, pady=6)

        btn_res = ttk.Frame(res_frame)
        btn_res.pack(side='right', padx=8)

        self.btn_save_csv = ttk.Button(btn_res, text="💾  Salva CSV",
                                       command=self._save_csv,
                                       state='disabled')
        self.btn_save_csv.pack(side='left', padx=4)

        self.btn_open_dir = ttk.Button(btn_res, text="📁  Apri cartella",
                                       command=self._open_output_dir,
                                       state='disabled')
        self.btn_open_dir.pack(side='left', padx=4)

    def _param_row(self, parent, label, variable, row, help_text=None):
        """
        Crea una riga label + entry + icona informativa (se help_text fornito).
        """
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky='w', padx=6, pady=2
        )
        ttk.Entry(parent, textvariable=variable, width=8).grid(
            row=row, column=1, sticky='e', padx=6, pady=2
        )
        if help_text:
            info_lbl = tk.Label(
                parent, text='ℹ', foreground='#1a73e8',
                cursor='question_arrow', font=('', 10)
            )
            info_lbl.grid(row=row, column=2, padx=(2, 4), pady=2)
            Tooltip(info_lbl, help_text)

    # Caricamento video e modello

    def _load_video(self):
        path = filedialog.askopenfilename(
            title="Seleziona video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Tutti", "*.*")]
        )
        if not path:
            return

        self.video_path = path
        self.lbl_video.config(text=Path(path).name, foreground='black')

        # leggi primo frame
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Errore", "Impossibile leggere il video.")
            return

        self.first_frame = frame
        self.roi         = None
        self.lbl_roi.config(text="ROI: non selezionata", foreground='gray')
        self._show_preview(frame)
        self._log(f"Video caricato: {Path(path).name}", 'info')

        # carica modello in background
        self._load_model_async()

    def _load_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Seleziona checkpoint",
            filetypes=[("PyTorch", "*.pt *.pth"), ("Tutti", "*.*")]
        )
        if path:
            self.var_checkpoint.set(path)
            self.model = None
            self._log(f"Checkpoint cambiato: {Path(path).name}", 'info')
            if self.video_path:
                self._load_model_async()

    def _load_model_async(self):
        """Carica il modello in un thread separato per non bloccare la GUI."""
        self._log("Caricamento modello...", 'info')
        self.btn_start.config(state='disabled')

        def _load():
            try:
                checkpoint = torch.load(
                    self.var_checkpoint.get(),
                    map_location='cpu', weights_only=False
                )
                model = CNNLSTM(
                    freeze_layers = checkpoint['args'].get('freeze_layers', 8),
                    weights_path  = self.var_weights.get(),
                )
                model.load_state_dict(checkpoint['model_state'])
                model.eval()
                self.model = model
                self.msg_queue.put(('model_ready', None))
            except Exception as e:
                self.msg_queue.put(('error', f"Errore caricamento modello: {e}"))

        threading.Thread(target=_load, daemon=True).start()

    # Preview e selezione ROI

    def _show_preview(self, frame_bgr, roi=None):
        """
        Mostra il frame sul canvas adattando la scala.
        Se roi è fornita, disegna il rettangolo verde.
        """
        h, w  = frame_bgr.shape[:2]
        scale = min(PREVIEW_W / w, PREVIEW_H / h)
        nw, nh = int(w * scale), int(h * scale)

        self._preview_scale = scale
        self._preview_offset = ((PREVIEW_W - nw) // 2,
                                (PREVIEW_H - nh) // 2)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img       = Image.fromarray(frame_rgb).resize((nw, nh))

        # disegna ROI se presente
        if roi:
            rx, ry, rw, rh = roi
            # converti a coordinate canvas
            cx  = int(rx * scale) + self._preview_offset[0]
            cy  = int(ry * scale) + self._preview_offset[1]
            cx2 = cx + int(rw * scale)
            cy2 = cy + int(rh * scale)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            # draw opera su coordinate relative all'immagine ridimensionata
            rx_s  = int(rx * scale)
            ry_s  = int(ry * scale)
            rx2_s = rx_s + int(rw * scale)
            ry2_s = ry_s + int(rh * scale)
            draw.rectangle([rx_s, ry_s, rx2_s, ry2_s],
                           outline=(0, 220, 0), width=2)

        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
            self._preview_offset[0], self._preview_offset[1],
            anchor='nw', image=self._tk_img
        )

    # variabili per drag sul canvas
    _drag_start = None
    _drag_rect  = None
    _preview_scale  = 1.0
    _preview_offset = (0, 0)

    def _canvas_to_video(self, cx, cy):
        """Converte coordinate canvas → coordinate video originali."""
        vx = (cx - self._preview_offset[0]) / self._preview_scale
        vy = (cy - self._preview_offset[1]) / self._preview_scale
        return int(vx), int(vy)

    def _on_mouse_press(self, event):
        self._drag_start = (event.x, event.y)
        if self._drag_rect:
            self.canvas.delete(self._drag_rect)

    def _on_mouse_drag(self, event):
        if self._drag_start:
            if self._drag_rect:
                self.canvas.delete(self._drag_rect)
            self._drag_rect = self.canvas.create_rectangle(
                self._drag_start[0], self._drag_start[1],
                event.x, event.y,
                outline='#00dc00', width=2
            )
            # mostra dimensioni reali
            vx1, vy1 = self._canvas_to_video(*self._drag_start)
            vx2, vy2 = self._canvas_to_video(event.x, event.y)
            w = abs(vx2 - vx1)
            h = abs(vy2 - vy1)
            self.lbl_roi.config(
                text=f"ROI: {w}×{h} px (trascina per ridefinire)",
                foreground='#007700'
            )

    def _on_mouse_release(self, event):
        if self._drag_start:
            vx1, vy1 = self._canvas_to_video(*self._drag_start)
            vx2, vy2 = self._canvas_to_video(event.x, event.y)

            x = min(vx1, vx2)
            y = min(vy1, vy2)
            w = abs(vx2 - vx1)
            h = abs(vy2 - vy1)

            if w > 20 and h > 20 and self.first_frame is not None:
                self.roi = (x, y, w, h)
                self.lbl_roi.config(
                    text=f"ROI: x={x} y={y} w={w} h={h}",
                    foreground='green'
                )
                self._show_preview(self.first_frame, roi=self.roi)
                self._log(f"ROI selezionata: x={x} y={y} w={w} h={h}", 'ok')
                self._update_start_button()
            self._drag_start = None

    def _select_roi_window(self):
        """
        Apre la selezione ROI in una finestra OpenCV separata —
        più comoda per video ad alta risoluzione.
        """
        if self.first_frame is None:
            messagebox.showwarning("Attenzione", "Carica prima un video.")
            return

        from analyze_video import ROISelector
        selector = ROISelector(self.first_frame)
        roi      = selector.select()

        if roi:
            self.roi = roi
            x, y, w, h = roi
            self.lbl_roi.config(
                text=f"ROI: x={x} y={y} w={w} h={h}",
                foreground='green'
            )
            self._show_preview(self.first_frame, roi=self.roi)
            self._log(f"ROI selezionata (finestra): x={x} y={y} "
                      f"w={w} h={h}", 'ok')
            self._update_start_button()

    def _update_start_button(self):
        if self.video_path and self.roi and self.model:
            self.btn_start.config(state='normal')

    # Analisi

    def _start_analysis(self):
        if not self.video_path or not self.roi or not self.model:
            messagebox.showwarning("Attenzione",
                                "Carica video e seleziona ROI prima.")
            return

        # Reset completo dello stato — fondamentale per riproducibilità
        self.running          = True
        self.events           = []
        self.events_filtered  = []

        # Svuota la queue da eventuali messaggi residui del run precedente
        while not self.msg_queue.empty():
            try:
                self.msg_queue.get_nowait()
            except queue.Empty:
                break

        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.btn_save_csv.config(state='disabled')
        self.btn_open_dir.config(state='disabled')
        self.var_progress.set(0)
        self.var_progress_lbl.set("")
        self.lbl_results.config(text="Analisi in corso...", foreground='gray')

        # Avvia inferenza in thread separato
        thread = threading.Thread(target=self._run_inference_thread,
                                daemon=True)
        thread.start()

    def _stop_analysis(self):
        self.running = False
        self._log("Analisi interrotta dall'utente.", 'warn')
        self.btn_stop.config(state='disabled')
        self.btn_start.config(state='normal')

    def _run_inference_thread(self):
        """
        Esegue l'inferenza in un thread separato per non bloccare la GUI.
        Comunica con il thread principale tramite msg_queue.
        """
        try:
            cap = cv2.VideoCapture(self.video_path)
            roi = self.roi
            x, y, w, h    = roi
            orig_fps      = cap.get(cv2.CAP_PROP_FPS)
            total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_step    = self.var_frame_step.get()
            seq_len       = self.var_seq_len.get()
            smooth_window = self.var_smooth_window.get()
            threshold     = self.var_threshold.get()
            skip_sec      = self.var_skip_seconds.get()
            conf_window   = self.var_conf_window.get()
            conf_ratio    = self.var_conf_ratio.get()
            confirm_fr    = self.var_confirm.get()
            min_duration  = self.var_min_duration.get()
            target_fps    = orig_fps / frame_step
            inference_step = 2

            conf_window_frames = max(1, int(conf_window * target_fps
                                            / inference_step))

            frame_buffer  = deque(maxlen=seq_len)
            prob_buffer   = deque(maxlen=smooth_window)
            conf_buffer   = deque(maxlen=conf_window_frames)

            crisis_state  = False
            confirm_count = 0
            frame_idx     = 0
            sampled_count = 0
            last_prob     = 0.0
            events        = []

            # Watchdog per sospensione PC
            READ_TIMEOUT   = 30  # secondi
            last_read_time = time.time()

            self.msg_queue.put(('log',
                f"Inizio analisi — {total_frames} frame "
                f"({total_frames/orig_fps:.1f}s)", 'info'))

            while True:
                if not self.running:
                    break

                # Watchdog: controlla se il thread è bloccato da troppo tempo
                # (es. PC in sospensione) prima di chiamare cap.read()
                if time.time() - last_read_time > READ_TIMEOUT:
                    self.msg_queue.put(('log',
                        'Timeout lettura video (PC in sospensione?), '
                        'analisi interrotta.', 'warn'))
                    break

                ret, frame_bgr = cap.read()

                if not ret:
                    break

                # Aggiorna il watchdog solo dopo una lettura riuscita
                last_read_time = time.time()

                if frame_idx % frame_step == 0:
                    crop = frame_bgr[y:y+h, x:x+w]
                    if crop.size == 0:
                        frame_idx += 1
                        continue

                    crop_r   = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                    crop_rgb = cv2.cvtColor(crop_r, cv2.COLOR_BGR2RGB)
                    tensor   = eval_transforms(Image.fromarray(crop_rgb))
                    frame_buffer.append(tensor)
                    sampled_count += 1

                    if (len(frame_buffer) == seq_len and
                            sampled_count % inference_step == 0):
                        seq = torch.stack(
                            list(frame_buffer), dim=0
                        ).unsqueeze(0)
                        with torch.no_grad():
                            logit = self.model(seq).squeeze()
                            prob  = torch.sigmoid(logit).item()
                        prob_buffer.append(prob)
                        last_prob = float(np.median(list(prob_buffer)))

                    conf_buffer.append(last_prob)

                    # progress ogni 100 frame campionati
                    if sampled_count % 100 == 0:
                        pct      = 100 * frame_idx / total_frames
                        time_sec = frame_idx / orig_fps
                        stato    = "CRISI" if crisis_state else "normale"
                        self.msg_queue.put(('progress',
                            (pct, f"{pct:.0f}%  t={time_sec:.0f}s  "
                                f"P={last_prob:.3f}  {stato}")))

                    time_sec = frame_idx / orig_fps

                    # skip zone
                    if time_sec < skip_sec:
                        frame_idx += 1
                        continue

                    # confidenza
                    if len(conf_buffer) >= conf_window_frames // 2:
                        cr = sum(1 for p in conf_buffer
                                if p >= threshold) / len(conf_buffer)
                    else:
                        cr = 0.0

                    if not crisis_state:
                        if (last_prob >= threshold and cr >= conf_ratio):
                            confirm_count += 1
                            if confirm_count >= confirm_fr:
                                crisis_state  = True
                                confirm_count = 0
                                events.append({
                                    'type'    : 'onset',
                                    'frame'   : frame_idx,
                                    'time_sec': round(time_sec, 2),
                                })
                                self.msg_queue.put((
                                    'log',
                                    f"*** ONSET  @ {time_sec:.1f}s "
                                    f"(frame {frame_idx})",
                                    'onset'
                                ))
                        else:
                            confirm_count = 0
                    else:
                        if last_prob < threshold:
                            confirm_count += 1
                            if confirm_count >= confirm_fr:
                                crisis_state  = False
                                confirm_count = 0
                                events.append({
                                    'type'    : 'offset',
                                    'frame'   : frame_idx,
                                    'time_sec': round(time_sec, 2),
                                })
                                self.msg_queue.put((
                                    'log',
                                    f"*** OFFSET @ {time_sec:.1f}s "
                                    f"(frame {frame_idx})",
                                    'offset'
                                ))
                        else:
                            confirm_count = 0

                frame_idx += 1

            cap.release()

            # filtra eventi
            filtered = self._filter_events(events, min_duration)
            self.msg_queue.put(('done', (events, filtered)))

        except Exception as e:
            self.msg_queue.put(('error', str(e)))

    def _filter_events(self, events, min_duration_sec):
        onsets  = [e for e in events if e['type'] == 'onset']
        offsets = [e for e in events if e['type'] == 'offset']
        filtered = []
        for i, onset in enumerate(onsets):
            if i >= len(offsets):
                continue
            offset   = offsets[i]
            duration = offset['time_sec'] - onset['time_sec']
            if duration >= min_duration_sec:
                filtered.append(onset)
                filtered.append(offset)
        return filtered

    # Salva CSV

    def _save_csv(self):
        if not self.events_filtered and not self.events:
            messagebox.showinfo("Info", "Nessun risultato da salvare.")
            return

        default_name = Path(self.video_path).stem + '_risultati.csv'
        default_dir  = Path(self.video_path).parent

        path = filedialog.asksaveasfilename(
            title="Salva CSV",
            initialdir=str(default_dir),
            initialfile=default_name,
            defaultextension='.csv',
            filetypes=[("CSV", "*.csv"), ("Tutti", "*.*")]
        )
        if not path:
            return

        events_to_save = self.events_filtered if self.events_filtered \
                         else self.events
        onsets  = [e for e in events_to_save if e['type'] == 'onset']
        offsets = [e for e in events_to_save if e['type'] == 'offset']
        x, y, w, h = self.roi

        rows = []
        for i, onset in enumerate(onsets):
            if i < len(offsets):
                off    = offsets[i]
                durata = round(off['time_sec'] - onset['time_sec'], 2)
            else:
                off    = {'frame': None, 'time_sec': None}
                durata = None

            rows.append({
                'video'        : Path(self.video_path).name,
                'roi_x': x, 'roi_y': y, 'roi_w': w, 'roi_h': h,
                'onset_frame'  : onset['frame'],
                'onset_sec'    : onset['time_sec'],
                'offset_frame' : off['frame'],
                'offset_sec'   : off['time_sec'],
                'durata_sec'   : durata,
                'data_analisi' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })

        if not rows:
            rows.append({
                'video': Path(self.video_path).name,
                'roi_x': x, 'roi_y': y, 'roi_w': w, 'roi_h': h,
                'onset_frame': None, 'onset_sec': None,
                'offset_frame': None, 'offset_sec': None,
                'durata_sec': None,
                'data_analisi': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })

        fieldnames = ['video', 'roi_x', 'roi_y', 'roi_w', 'roi_h',
                      'onset_frame', 'onset_sec', 'offset_frame',
                      'offset_sec', 'durata_sec', 'data_analisi']

        file_exists = Path(path).exists()
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

        self._log(f"CSV salvato in: {path}", 'ok')
        messagebox.showinfo("Salvato", f"Risultati salvati in:\n{path}")

    def _open_output_dir(self):
        if self.video_path:
            folder = str(Path(self.video_path).parent)
            if sys.platform == 'darwin':
                os.system(f'open "{folder}"')
            elif sys.platform == 'win32':
                os.startfile(folder)
            else:
                os.system(f'xdg-open "{folder}"')

    # Log e queue polling

    def _log(self, msg, tag='info'):
        self.log_text.config(state='normal')
        self.log_text.insert('end', f"> {msg}\n", tag)
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def _start_queue_polling(self):
        """Controlla la coda messaggi ogni 100ms per aggiornare la GUI."""
        self._poll_queue()

    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                msg_type = msg[0]

                if msg_type == 'log':
                    _, text, tag = msg
                    self._log(text, tag)

                elif msg_type == 'progress':
                    _, (pct, label) = msg
                    self.var_progress.set(pct)
                    self.var_progress_lbl.set(label)

                elif msg_type == 'model_ready':
                    self._log("Modello caricato.", 'ok')
                    self._update_start_button()

                elif msg_type == 'done':
                    _, (events, filtered) = msg
                    self.events          = events
                    self.events_filtered = filtered
                    self.running         = False
                    self.var_progress.set(100)
                    self.var_progress_lbl.set("Completato")
                    self.btn_start.config(state='normal')
                    self.btn_stop.config(state='disabled')
                    self.btn_save_csv.config(state='normal')
                    self.btn_open_dir.config(state='normal')
                    self._show_results(filtered if filtered else events)

                elif msg_type == 'error':
                    _, err = msg
                    self._log(f"ERRORE: {err}", 'warn')
                    self.running = False
                    self.btn_start.config(state='normal')
                    self.btn_stop.config(state='disabled')
                    messagebox.showerror("Errore", err)

        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)

    def _show_results(self, events):
        onsets  = [e for e in events if e['type'] == 'onset']
        offsets = [e for e in events if e['type'] == 'offset']

        if not onsets:
            self.lbl_results.config(
                text="Nessuna crisi rilevata.",
                foreground='gray'
            )
            return

        lines = [f"{len(onsets)} crisi rilevata/e:"]
        for i, onset in enumerate(onsets):
            if i < len(offsets):
                off = offsets[i]
                dur = round(off['time_sec'] - onset['time_sec'], 1)
                lines.append(
                    f"  Crisi {i+1}: "
                    f"onset={onset['time_sec']}s (f.{onset['frame']})  "
                    f"offset={off['time_sec']}s  "
                    f"durata={dur}s"
                )
            else:
                lines.append(
                    f"  Crisi {i+1}: "
                    f"onset={onset['time_sec']}s — offset non rilevato"
                )

        self.lbl_results.config(
            text="\n".join(lines),
            foreground='#006600'
        )
        self._log(f"Analisi completata — {len(onsets)} crisi rilevata/e.", 'ok')


# Entry point

if __name__ == '__main__':
    root = tk.Tk()
    app  = EpilepsyGUI(root)
    root.mainloop()