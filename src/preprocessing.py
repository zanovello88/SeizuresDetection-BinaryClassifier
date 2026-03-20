"""
preprocessing.py
================
Scopo: estrarre frame dai video grezzi e costruire un manifest JSON
       che associa ogni frame estratto alla sua label binaria (0=non crisi, 1=crisi).

Motivazioni delle scelte:
  - Subsampling a 10fps (1 frame ogni 3): frame consecutivi a 30fps sono quasi
    identici; ridurre di 3x diminuisce il carico computazionale senza perdere
    la dinamica dei movimenti convulsivi nei topi.
  - Risoluzione 210x210 mantenuta invariata: il dataset è già ritagliato sul topo,
    qualsiasi resize introdurrebbe perdita di dettaglio o distorsione inutile.
  - Il manifest JSON centralizza tutte le informazioni (path, label, indici)
    così gli step successivi (dataset, training) non toccano mai i video originali.
  - I frame vengono salvati come JPEG q=95 per bilanciare qualità e spazio su disco.

Struttura CSV (separatore = ';'):
  clip_name  → nome del file video
  topo       → nome del topo
  fps        → frame rate (costante a 30.0)
  f_inizio   → frame di inizio crisi
  f_fine     → frame di fine crisi
  f_tot      → durata crisi in frame (= f_fine - f_inizio, usato solo come check)

Struttura dell'output:
  data/frames/<clip_name_stem>/frame_<XXXXX>.jpg
  data/manifest.json
"""

import cv2
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List

#Configurazione
ORIGINAL_FPS = 30.0
TARGET_FPS   = 10.0
FRAME_STEP   = int(ORIGINAL_FPS / TARGET_FPS)   # = 3

logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s | %(levelname)s | %(message)s',
    datefmt = '%H:%M:%S',
)
log = logging.getLogger(__name__)


#Funzioni core

def extract_frames(video_path: Path, output_dir: Path,
                   frame_step: int = FRAME_STEP) -> int:
    """
    Legge il video e salva 1 frame ogni `frame_step` come JPEG.
    Il nome file usa l'indice del frame ORIGINALE (scala 30fps) così
    la mappatura con f_inizio/f_fine del CSV rimane diretta e non ambigua.
    Restituisce il numero totale di frame letti dal video.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            out_path = output_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_idx += 1

    cap.release()
    log.info(f"  {video_path.name}: {frame_idx} frame originali → "
             f"{frame_idx // frame_step} frame estratti")
    return frame_idx


def build_frame_labels(total_frames: int,
                       onset_frame:  int,
                       offset_frame: int,
                       frame_step:   int = FRAME_STEP) -> Dict[int, int]:
    """
    Restituisce {indice_frame_originale: label} per tutti i frame estratti.
    Label 1 se il frame cade in [f_inizio, f_fine] (estremi inclusi), 0 altrimenti.
    Lavorare sull'indice originale (30fps) garantisce coerenza con il CSV.
    """
    return {
        f: (1 if onset_frame <= f <= offset_frame else 0)
        for f in range(0, total_frames, frame_step)
    }


def process_single_video(video_path: Path,
                         frames_root: Path,
                         row: pd.Series) -> List[dict]:
    """
    Pipeline completa per un singolo video:
      1. Estrae i frame con subsampling
      2. Costruisce le label frame-level dal CSV
      3. Verifica la coerenza tra f_tot del CSV e i frame effettivamente etichettati
      4. Restituisce lista di record per il manifest
    """
    clip_name  = row['clip_name']
    mouse_name = row['topo']
    onset      = int(row['f_inizio'])
    offset     = int(row['f_fine'])
    f_tot_csv  = int(row['f_tot'])          # durata dichiarata nel CSV
    output_dir = frames_root / Path(clip_name).stem

    total_frames = extract_frames(video_path, output_dir)
    labels       = build_frame_labels(total_frames, onset, offset)

    #Sanity check 
    #Verifichiamo che f_inizio e f_fine siano dentro i limiti del video.
    f_tot_csv = int(row['f_tot'])
    if onset < 0 or offset > f_tot_csv:
        log.warning(f"  [{clip_name}] f_inizio={onset} o f_fine={offset} "
                    f"fuori dai limiti del video ({f_tot_csv} frame)")
    elif abs(f_tot_csv - total_frames) > 5:
        #Controlla che i frame totali dichiarati nel CSV corrispondano
        #a quelli effettivamente letti da OpenCV (tolleranza 5 frame)
        log.warning(f"  [{clip_name}] f_tot CSV={f_tot_csv} ma "
                    f"OpenCV ha letto {total_frames} frame")
    else:
        log.info(f"  [{clip_name}] sanity check OK "
                f"(crisi: {onset}→{offset}, durata {(offset-onset)/30:.1f}s)")

    records = []
    for orig_idx, label in sorted(labels.items()):
        frame_path = output_dir / f"frame_{orig_idx:05d}.jpg"
        if frame_path.exists():
            records.append({
                'clip_name'          : clip_name,
                'mouse_name'         : mouse_name,
                'frame_path'         : str(frame_path),
                'original_frame_idx' : orig_idx,
                'sampled_frame_idx'  : orig_idx // FRAME_STEP,
                'label'              : label,
                'onset_frame'        : onset,
                'offset_frame'       : offset,
            })
    return records


def run_preprocessing(csv_path:        Path,
                      videos_root:     Path,
                      frames_root:     Path,
                      output_manifest: Path) -> None:
    """
    Esegue il preprocessing su tutti i video del CSV e salva il manifest JSON.
    Stampa un report finale con distribuzione classi e pos_weight suggerito
    per BCEWithLogitsLoss — quel numero va annotato perché serve nel training.
    """
    # sep=';' perché il CSV usa punto e virgola come separatore
    df      = pd.read_csv(csv_path, sep=';')
    all_rec = []
    missing = []

    log.info(f"Video nel CSV: {len(df)}")

    for _, row in df.iterrows():
        video_path = videos_root / row['clip_name']
        if not video_path.exists():
            log.warning(f"Video non trovato, skippato: {video_path}")
            missing.append(row['clip_name'])
            continue

        log.info(f"Processing → {row['clip_name']}")
        records = process_single_video(video_path, frames_root, row)
        all_rec.extend(records)

    #Salva manifest
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, 'w') as f:
        json.dump(all_rec, f, indent=2)

    #Report finale
    total    = len(all_rec)
    n_crisis = sum(r['label'] == 1 for r in all_rec)
    n_normal = total - n_crisis
    ratio    = n_normal / n_crisis if n_crisis > 0 else float('inf')

    log.info("=" * 55)
    log.info("PREPROCESSING COMPLETATO")
    log.info(f"  Video processati : {len(df) - len(missing)}/{len(df)}")
    log.info(f"  Frame totali     : {total:,}")
    log.info(f"  Label 0 (normal) : {n_normal:,}  ({100*n_normal/total:.1f}%)")
    log.info(f"  Label 1 (crisis) : {n_crisis:,}  ({100*n_crisis/total:.1f}%)")
    log.info(f"  Rapporto 0/1     : {ratio:.2f}")
    log.info(f"  pos_weight       : {ratio:.4f}  ← salva questo valore")
    if missing:
        log.warning(f"  Video mancanti   : {missing}")
    log.info("=" * 55)
    log.info(f"Manifest salvato in: {output_manifest}")


if __name__ == '__main__':
    run_preprocessing(
        csv_path        = Path('data/mappa_labels.csv'),
        videos_root     = Path('data/dataset_tagliato'),
        frames_root     = Path('data/frames'),
        output_manifest = Path('data/manifest.json'),
    )