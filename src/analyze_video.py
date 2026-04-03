"""
analyze_video.py
================
Scopo: interfaccia per informatici che permette di analizzare un video
       con più topi, selezionare manualmente il topo di interesse tramite
       click-and-drag, eseguire l'inferenza CNN+LSTM e salvare i timestamp
       di inizio e fine crisi in un file CSV.

Ottimizzazioni CPU rispetto alla versione precedente:
  - seq_len ridotto da 60 a 20 (291ms vs 680ms per inferenza)
  - frame_step aumentato da 3 a 6 (5fps invece di 10fps)
  - inference_step=2: inferenza ogni 2 frame campionati invece di ogni 1
    (la finestra scorre ma il modello gira ogni 0.4s invece di ogni 0.2s)
  - confirm_frames aumentato a 8: evita onset/offset multipli da fluttuazioni
    brevi — richiede 8 frame consecutivi (~1.6s) per confermare un evento
  - Risultato: ~2.5 minuti per video da 4 minuti su CPU

Utilizzo:
    python src/analyze_video.py --video path/al/video.mp4
"""

import cv2
import csv
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import deque
from PIL import Image
from datetime import datetime

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from model      import CNNLSTM
from transforms import eval_transforms

CROP_SIZE          = 210
DEFAULT_CHECKPOINT = 'runs/20260327_101301/best_model.pt'
DEFAULT_WEIGHTS    = 'model_weights/mobilenet_v3_small_imagenet.pth'


# ── Argomenti CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Analisi video epilessia — selezione manuale del topo'
    )
    p.add_argument('--video',
                   type=str, required=True)
    p.add_argument('--checkpoint',
                   type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument('--weights_path',
                   type=str, default=DEFAULT_WEIGHTS)
    p.add_argument('--output_csv',
                   type=str, default=None)
    p.add_argument('--threshold',
                   type=float, default=0.874)
    p.add_argument('--smooth_window',
                   type=int, default=10)
    p.add_argument('--seq_len',
                   type=int, default=20,
                   help='Lunghezza sequenza (default=20, ottimizzato CPU)')
    p.add_argument('--frame_step',
                   type=int, default=6,
                   help='Subsampling: 1 frame ogni N (default=6, 30fps→5fps)')
    p.add_argument('--inference_step',
                   type=int, default=2,
                   help='Inferenza ogni N frame campionati (default=2)')
    p.add_argument('--confirm_frames',
                   type=int, default=8,
                   help='Frame campionati consecutivi per confermare onset/offset')
    p.add_argument('--min_duration_sec',
               type=float, default=40.0,
               help='Durata minima in secondi per considerare una crisi reale '
                    '(default=40.0 — scarta eventi più brevi come falsi positivi)')
    p.add_argument('--min_gap_sec',
               type=float, default=10.0,
               help='Gap minimo in secondi tra due crisi consecutive '
                    '(default=10.0 — unisce eventi troppo vicini)')
    p.add_argument('--skip_seconds',
               type=float, default=0.0,
               help='Ignora le predizioni nei primi N secondi '
                    '(default=0 — disabilitato)')
    p.add_argument('--confidence_window_sec',
                type=float, default=8.0,
                help='Secondi di probabilità alta sostenuta richiesti '
                        'per confermare onset (default=8s). '
                        'Con 5fps = 40 frame campionati di cui almeno '
                        'confidence_ratio devono superare il threshold.')
    p.add_argument('--confidence_ratio',
                type=float, default=0.75,
                help='Frazione minima di frame nella finestra di confidenza '
                        'che devono superare il threshold (default=0.75)')
    p.add_argument('--threshold_high',
                type=float, default=None,
                help='Threshold più alto per i primi skip_seconds secondi '
                        '(default=None — usa threshold normale). '
                        'Es: 0.95 rende il modello molto più conservativo '
                        'nella fase iniziale del video.')
    return p.parse_args()


# ── Selezione ROI ──────────────────────────────────────────────────────────────

class ROISelector:
    """
    Selezione interattiva del riquadro sul primo frame con click-and-drag.
    Tasti: Invio/Spazio=conferma | R=ridisegna | Q/Esc=esci
    """

    def __init__(self, frame):
        # ridimensiona il frame per la visualizzazione se troppo grande
        h, w     = frame.shape[:2]
        max_dim  = 1000
        scale    = min(max_dim / w, max_dim / h, 1.0)
        self.scale      = scale
        self.disp_frame = cv2.resize(frame, (int(w*scale), int(h*scale))) \
                          if scale < 1.0 else frame.copy()
        self.orig_frame = frame.copy()
        self.display    = self.disp_frame.copy()
        self.start_pt   = None
        self.end_pt     = None
        self.drawing    = False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_pt = (x, y)
            self.end_pt   = (x, y)
            self.drawing  = True

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_pt  = (x, y)
            self.display = self.disp_frame.copy()
            cv2.rectangle(self.display, self.start_pt,
                          self.end_pt, (0, 200, 0), 2)
            w = abs(self.end_pt[0] - self.start_pt[0])
            h = abs(self.end_pt[1] - self.start_pt[1])
            # mostra dimensioni reali (non scalate)
            real_w = int(w / self.scale)
            real_h = int(h / self.scale)
            cv2.putText(self.display,
                        f'{real_w}x{real_h} pixel reali',
                        (min(self.start_pt[0], self.end_pt[0]),
                         min(self.start_pt[1], self.end_pt[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.end_pt  = (x, y)
            self.drawing = False
            self.display = self.disp_frame.copy()
            cv2.rectangle(self.display, self.start_pt,
                          self.end_pt, (0, 200, 0), 2)

    def select(self):
        """
        Apre la finestra e attende la selezione.
        Restituisce (x, y, w, h) nelle coordinate ORIGINALI del video.
        """
        win = 'Seleziona il topo  |  Invio=conferma   R=ridisegna   Q=esci'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        # istruzioni sul frame
        overlay = self.disp_frame.copy()
        cv2.rectangle(overlay, (0, 0),
                      (overlay.shape[1], 45), (0, 0, 0), -1)
        cv2.putText(overlay,
                    'Clicca e trascina per selezionare il topo',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(overlay,
                    'Invio = conferma   |   R = ridisegna   |   Q = esci',
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)
        self.disp_frame = overlay
        self.display    = overlay.copy()

        cv2.setMouseCallback(win, self._mouse_callback)

        while True:
            cv2.imshow(win, self.display)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32) and self.start_pt and self.end_pt:
                # converti coordinate display → coordinate originali
                x1 = int(min(self.start_pt[0], self.end_pt[0]) / self.scale)
                y1 = int(min(self.start_pt[1], self.end_pt[1]) / self.scale)
                x2 = int(max(self.start_pt[0], self.end_pt[0]) / self.scale)
                y2 = int(max(self.start_pt[1], self.end_pt[1]) / self.scale)
                w  = x2 - x1
                h  = y2 - y1
                if w > 20 and h > 20:
                    cv2.destroyWindow(win)
                    return (x1, y1, w, h)
                else:
                    print("Riquadro troppo piccolo — ridisegna.")

            elif key == ord('r'):
                self.start_pt = None
                self.end_pt   = None
                self.display  = self.disp_frame.copy()

            elif key in (ord('q'), 27):
                cv2.destroyWindow(win)
                return None


# ── Inferenza ottimizzata ──────────────────────────────────────────────────────

def run_inference_cpu(cap, roi, model, args):
    """
    Inferenza ottimizzata per CPU con tre livelli di post-processing
    per ridurre i falsi positivi:

    Approccio 1 — skip_seconds:
      Ignora completamente le predizioni nei primi N secondi.
      Utile quando il pre-ictal è sempre almeno N secondi.

    Approccio 2 — threshold_high:
      Usa un threshold più alto nella fase iniziale (primi skip_seconds).
      Più flessibile dell'approccio 1 — non ignora ma richiede
      maggiore confidenza prima che la crisi sia iniziata.

    Approccio 3 — finestra di confidenza:
      Per confermare un onset richiede che almeno confidence_ratio
      dei frame in una finestra di confidence_window_sec secondi
      superino il threshold. Evita che picchi isolati triggerino
      onset — richiede probabilità alta e SOSTENUTA nel tempo.
    """
    x, y, w, h    = roi
    orig_fps      = cap.get(cv2.CAP_PROP_FPS)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_fps    = orig_fps / args.frame_step

    # dimensione finestra di confidenza in frame campionati
    conf_window_frames = max(1, int(args.confidence_window_sec * target_fps
                                    / args.inference_step))

    frame_buffer   = deque(maxlen=args.seq_len)
    prob_buffer    = deque(maxlen=args.smooth_window)
    # finestra di confidenza: tiene le ultime N probabilità smoothed
    conf_buffer    = deque(maxlen=conf_window_frames)

    all_probs      = []
    all_frames     = []
    events         = []

    crisis_state   = False
    confirm_count  = 0
    frame_idx      = 0
    sampled_count  = 0
    last_prob      = 0.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # threshold adattivo: più alto nei primi skip_seconds se richiesto
    threshold_initial = args.threshold_high if args.threshold_high \
                        else args.threshold

    print(f"\nAnalisi in corso...")
    print(f"  Video        : {total_frames} frame ({total_frames/orig_fps:.1f}s)")
    print(f"  Campion.     : ogni {args.frame_step} frame ({target_fps:.1f}fps)")
    print(f"  Seq len      : {args.seq_len} frame ({args.seq_len/target_fps:.1f}s)")
    print(f"  Inferenza    : ogni {args.inference_step} frame campionati")
    print(f"  Conferma     : {args.confirm_frames} frame consecutivi")
    if args.skip_seconds > 0:
        print(f"  Skip iniziale: {args.skip_seconds}s")
    if args.threshold_high:
        print(f"  Threshold    : {threshold_initial:.3f} (primi "
              f"{args.skip_seconds}s) → {args.threshold:.3f} (resto)")
    print(f"  Confidenza   : {args.confidence_ratio*100:.0f}% su "
          f"{args.confidence_window_sec}s ({conf_window_frames} campioni)\n")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % args.frame_step == 0:
            crop = frame_bgr[y:y+h, x:x+w]
            if crop.size == 0:
                frame_idx += 1
                continue

            crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
            crop_rgb     = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            tensor       = eval_transforms(Image.fromarray(crop_rgb))
            frame_buffer.append(tensor)
            sampled_count += 1

            if (len(frame_buffer) == args.seq_len and
                    sampled_count % args.inference_step == 0):
                sequence = torch.stack(
                    list(frame_buffer), dim=0
                ).unsqueeze(0)
                with torch.no_grad():
                    logit = model(sequence).squeeze()
                    prob  = torch.sigmoid(logit).item()
                prob_buffer.append(prob)
                last_prob = float(np.median(list(prob_buffer)))

            # aggiorna finestra di confidenza
            conf_buffer.append(last_prob)

            all_probs.append(last_prob)
            all_frames.append(frame_idx)

            if sampled_count % 200 == 0:
                pct      = 100 * frame_idx / total_frames
                time_sec = frame_idx / orig_fps
                stato    = "CRISI" if crisis_state else "normale"
                print(f"  {pct:>5.1f}% | t={time_sec:>6.1f}s | "
                      f"P={last_prob:.3f} | {stato}")

            time_sec = frame_idx / orig_fps

            # ── Approccio 1+2: skip e threshold adattivo ───────────────────
            # nei primi skip_seconds usiamo threshold_initial (più alto)
            # oppure ignoriamo completamente (se threshold_high non impostato)
            in_skip_zone = time_sec < args.skip_seconds
            current_threshold = threshold_initial if in_skip_zone \
                                else args.threshold

            if in_skip_zone and not args.threshold_high:
                # approccio 1 puro: ignora completamente
                frame_idx += 1
                continue

            # ── Approccio 3: finestra di confidenza ────────────────────────
            # calcola la frazione di frame nella finestra sopra threshold
            if len(conf_buffer) >= conf_window_frames // 2:
                conf_ratio = sum(1 for p in conf_buffer
                                 if p >= current_threshold) / len(conf_buffer)
            else:
                conf_ratio = 0.0

            # onset confermato solo se:
            # 1. prob corrente sopra threshold
            # 2. conf_ratio sopra confidence_ratio (pattern sostenuto)
            # 3. confirm_frames consecutivi già accumulati
            if not crisis_state:
                if (last_prob >= current_threshold and
                        conf_ratio >= args.confidence_ratio):
                    confirm_count += 1
                    if confirm_count >= args.confirm_frames:
                        crisis_state  = True
                        confirm_count = 0
                        events.append({
                            'type'    : 'onset',
                            'frame'   : frame_idx,
                            'time_sec': round(time_sec, 2),
                        })
                        print(f"\n  *** ONSET  @ {time_sec:.1f}s "
                              f"(frame {frame_idx}) | "
                              f"conf={conf_ratio:.2f} ***\n")
                else:
                    confirm_count = 0
            else:
                if last_prob < args.threshold:
                    confirm_count += 1
                    if confirm_count >= args.confirm_frames:
                        crisis_state  = False
                        confirm_count = 0
                        events.append({
                            'type'    : 'offset',
                            'frame'   : frame_idx,
                            'time_sec': round(time_sec, 2),
                        })
                        print(f"\n  *** OFFSET @ {time_sec:.1f}s "
                              f"(frame {frame_idx}) ***\n")
                else:
                    confirm_count = 0

        frame_idx += 1

    return all_probs, all_frames, events, orig_fps

def filter_events(events, min_duration_sec, min_gap_sec):
    """
    Filtra gli eventi rilevati scartando le crisi più brevi di
    min_duration_sec. Il parametro min_gap_sec è mantenuto per
    compatibilità CLI ma il merge è disabilitato — con confirm_frames
    alto i falsi positivi brevi vengono già gestiti correttamente
    e il merge causa unioni indesiderate a catena.

    Crisi reali: 45-80s → soglia 40s è conservativa e corretta.
    Falsi positivi tipici: 2-20s → vengono tutti scartati.
    """
    onsets  = [e for e in events if e['type'] == 'onset']
    offsets = [e for e in events if e['type'] == 'offset']

    filtered = []
    for i, onset in enumerate(onsets):
        if i >= len(offsets):
            # onset senza offset — fine video durante crisi
            # lo teniamo solo se la crisi è iniziata da abbastanza tempo
            print(f"  [senza offset] onset @ {onset['time_sec']}s "
                  f"— fine video durante crisi")
            continue

        offset   = offsets[i]
        duration = offset['time_sec'] - onset['time_sec']

        if duration >= min_duration_sec:
            filtered.append(onset)
            filtered.append(offset)
            print(f"  [tenuta] crisi {onset['time_sec']}s → "
                  f"{offset['time_sec']}s (durata={duration:.1f}s)")
        else:
            print(f"  [scartata] crisi {onset['time_sec']}s → "
                  f"{offset['time_sec']}s (durata={duration:.1f}s "
                  f"< {min_duration_sec}s)")

    return filtered

# ── Salva CSV ──────────────────────────────────────────────────────────────────

def save_csv(video_path, events, roi, output_csv):
    """
    Salva i risultati in CSV — una riga per crisi rilevata.
    Il file viene aperto in append così più analisi si accumulano.
    """
    onsets  = [e for e in events if e['type'] == 'onset']
    offsets = [e for e in events if e['type'] == 'offset']
    x, y, w, h = roi

    rows = []
    for i, onset in enumerate(onsets):
        if i < len(offsets):
            off    = offsets[i]
            durata = round(off['time_sec'] - onset['time_sec'], 2)
        else:
            off    = {'frame': None, 'time_sec': None}
            durata = None

        rows.append({
            'video'        : Path(video_path).name,
            'roi_x'        : x,
            'roi_y'        : y,
            'roi_w'        : w,
            'roi_h'        : h,
            'onset_frame'  : onset['frame'],
            'onset_sec'    : onset['time_sec'],
            'offset_frame' : off['frame'],
            'offset_sec'   : off['time_sec'],
            'durata_sec'   : durata,
            'data_analisi' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })

    if not rows:
        rows.append({
            'video'        : Path(video_path).name,
            'roi_x': x, 'roi_y': y, 'roi_w': w, 'roi_h': h,
            'onset_frame'  : None,
            'onset_sec'    : None,
            'offset_frame' : None,
            'offset_sec'   : None,
            'durata_sec'   : None,
            'data_analisi' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })

    fieldnames = ['video', 'roi_x', 'roi_y', 'roi_w', 'roi_h',
                  'onset_frame', 'onset_sec', 'offset_frame',
                  'offset_sec', 'durata_sec', 'data_analisi']

    file_exists = Path(output_csv).exists()
    with open(output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV salvato in: {output_csv}")
    for row in rows:
        if row['onset_sec'] is not None:
            print(f"  Crisi: {row['onset_sec']}s (f.{row['onset_frame']}) → "
                  f"{row['offset_sec']}s (f.{row['offset_frame']}) | "
                  f"durata={row['durata_sec']}s")
        else:
            print("  Nessuna crisi rilevata")


# ── Salva plot ─────────────────────────────────────────────────────────────────

def save_plot(all_frames, all_probs, events, orig_fps,
              threshold, video_path, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    time_axis = [f / orig_fps for f in all_frames]
    fig, ax   = plt.subplots(figsize=(14, 4))

    ax.plot(time_axis, all_probs, color='#E8944A', lw=1.2,
            label='P(crisi) smoothed')
    ax.fill_between(time_axis, all_probs, alpha=0.2, color='#E8944A')
    ax.axhline(y=threshold, color='red', linestyle='--',
               lw=1.5, label=f'Threshold={threshold:.3f}')

    for e in events:
        c = 'red' if e['type'] == 'onset' else 'green'
        l = 'Onset' if e['type'] == 'onset' else 'Offset'
        ax.axvline(x=e['time_sec'], color=c, linestyle=':', lw=1.5)
        ax.text(e['time_sec'] + 0.5, 1.02, l, color=c,
                fontsize=9, transform=ax.get_xaxis_transform())

    ax.set_xlabel('Tempo (secondi)')
    ax.set_ylabel('P(crisi)')
    ax.set_title(f'{Path(video_path).name}')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plot_path = Path(output_dir) / f"{Path(video_path).stem}_analisi.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot salvato in: {plot_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not Path(args.video).exists():
        print(f"ERRORE: video non trovato: {args.video}")
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        print(f"ERRORE: checkpoint non trovato: {args.checkpoint}")
        sys.exit(1)

    video_dir  = Path(args.video).parent
    output_csv = args.output_csv or str(video_dir / 'risultati_crisi.csv')

    # ── Carica modello ─────────────────────────────────────────────────────────
    print("Caricamento modello...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=False)
    model = CNNLSTM(
        freeze_layers = checkpoint['args'].get('freeze_layers', 8),
        weights_path  = args.weights_path,
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("Modello caricato.\n")

    # ── Leggi primo frame ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERRORE: impossibile aprire: {args.video}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        print("ERRORE: impossibile leggere il primo frame")
        sys.exit(1)

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_s  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / orig_fps
    print(f"Video: {Path(args.video).name}")
    print(f"Durata: {total_s:.1f}s | FPS: {orig_fps:.1f}")
    print(f"Tempo stimato analisi: "
          f"~{total_s * 0.65:.0f}s su CPU\n")   # stima empirica

    # ── Selezione ROI ──────────────────────────────────────────────────────────
    selector = ROISelector(first_frame)
    roi      = selector.select()

    if roi is None:
        print("Selezione annullata.")
        cap.release()
        sys.exit(0)

    # ── Anteprima ritaglio ─────────────────────────────────────────────────────
    x, y, w, h = roi
    preview    = cv2.resize(first_frame[y:y+h, x:x+w],
                            (CROP_SIZE, CROP_SIZE))
    cv2.imshow('Anteprima ritaglio — premi un tasto per avviare', preview)
    print(f"ROI: x={x} y={y} w={w} h={h}")
    print("Premi un tasto per avviare l'analisi...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ── Inferenza ──────────────────────────────────────────────────────────────
    import time
    t_start = time.time()

    all_probs, all_frames, events, fps = run_inference_cpu(
        cap, roi, model, args
    )
    cap.release()

    elapsed = time.time() - t_start
    print(f"\nAnalisi completata in {elapsed:.0f}s "
          f"({elapsed/total_s:.2f}x tempo reale)")


    # ── Filtra eventi ──────────────────────────────────────────────────────────
    print("\nFiltro eventi...")
    events_filtered = filter_events(
        events,
        min_duration_sec = args.min_duration_sec,
        min_gap_sec      = args.min_gap_sec,
    )
    print(f"Crisi dopo filtro: {len([e for e in events_filtered if e['type']=='onset'])}"
        f" (erano {len([e for e in events if e['type']=='onset'])})")
    
    # ── Report ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    onsets  = [e for e in events if e['type'] == 'onset']
    offsets = [e for e in events if e['type'] == 'offset']
    print(f"Crisi rilevate: {len(onsets)}")
    for i, onset in enumerate(onsets):
        print(f"\n  Crisi {i+1}:")
        print(f"    Onset  → {onset['time_sec']}s (frame {onset['frame']})")
        if i < len(offsets):
            off = offsets[i]
            print(f"    Offset → {off['time_sec']}s (frame {off['frame']})")
            print(f"    Durata → "
                  f"{round(off['time_sec']-onset['time_sec'],2)}s")
    print("=" * 50)

    # ── Salva output ───────────────────────────────────────────────────────────
    save_csv(args.video, events_filtered, roi, output_csv)
    save_plot(all_frames, all_probs, events_filtered, fps,
          args.threshold, args.video, video_dir)


if __name__ == '__main__':
    main()