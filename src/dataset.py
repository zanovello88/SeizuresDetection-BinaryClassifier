"""
Scopo: trasformare il manifest JSON in un dataset PyTorch pronto per il training
       del modello CNN+LSTM. Il cuore del modulo è la sliding window che converte
       la sequenza di frame di ogni video in campioni (sequenza, label) sovrapposti.

Motivazioni delle scelte:
  - Sliding window con T=30 frame (3s a 10fps): abbastanza lunga da catturare
    la dinamica temporale della crisi, abbastanza corta da stare in memoria GPU.
  - Stride=15 (overlap 50%): massimizza il numero di campioni mantenendo
    diversità tra sequenze adiacenti. Con stride=1 si avrebbe troppa ridondanza.
  - Label di sequenza = label del frame centrale: più stabile rispetto alla
    maggioranza o all'ultimo frame, e allineata con la predizione frame-level
    che vogliamo fare in inference.
  - Split per video (non per sequenza): evita il data leakage. Sequenze dello
    stesso video devono stare tutte nello stesso split.
  - I frame vengono caricati lazy (al momento del __getitem__) per non saturare
    la RAM del cluster con 66k immagini pre-caricate.
"""

import json
import random
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List, Dict, Tuple

#Costanti 
SEQ_LEN    = 30     # T: numero di frame per sequenza (3 secondi a 10fps)
STRIDE     = 15     # passo della sliding window (overlap 50%)
TRAIN_FRAC = 0.70   # 70% dei video per training
VAL_FRAC   = 0.15   # 15% per validation
TEST_FRAC  = 0.15   # 15% per test  (train+val+test = 1.0)
SEED       = 42


#Costruzione delle sequenze 

def build_sequences(manifest: List[dict],
                    seq_len:  int = SEQ_LEN,
                    stride:   int = STRIDE) -> List[dict]:
    """
    Raggruppa i frame del manifest per video e applica la sliding window.
    Ogni sequenza è un dict con:
      - 'frame_paths' : lista di seq_len path
      - 'labels'      : lista di seq_len label (0/1)
      - 'seq_label'   : label del frame centrale (usata come target della sequenza)
      - 'video_name'  : nome del video di provenienza
      - 'start_idx'   : indice sampled del primo frame nella sequenza
    """
    # raggruppa frame per video, ordinati per indice campionato
    video_frames: Dict[str, List[dict]] = defaultdict(list)
    for rec in manifest:
        video_frames[rec['clip_name']].append(rec)

    for v in video_frames:
        video_frames[v].sort(key=lambda x: x['sampled_frame_idx'])

    sequences = []
    for video_name, frames in video_frames.items():
        n = len(frames)
        for start in range(0, n - seq_len + 1, stride):
            window       = frames[start : start + seq_len]
            frame_paths  = [w['frame_path'] for w in window]
            labels       = [w['label']      for w in window]
            center_label = labels[seq_len // 2]   # label del frame centrale

            sequences.append({
                'frame_paths' : frame_paths,
                'labels'      : labels,
                'seq_label'   : center_label,
                'video_name'  : video_name,
                'start_idx'   : frames[start]['sampled_frame_idx'],
            })

    return sequences


def split_sequences(sequences:   List[dict],
                    train_frac:  float = TRAIN_FRAC,
                    val_frac:    float = VAL_FRAC,
                    seed:        int   = SEED) -> Tuple[List, List, List]:
    """
    Divide le sequenze in train/val/test splittando PER VIDEO (non per sequenza).
    In questo modo non ci sono sequenze dello stesso video in split diversi,
    evitando completamente il data leakage.
    """
    # lista unica dei video, shuffle riproducibile
    video_names = list({s['video_name'] for s in sequences})
    random.seed(seed)
    random.shuffle(video_names)

    n          = len(video_names)
    n_train    = int(n * train_frac)
    n_val      = int(n * val_frac)

    train_vids = set(video_names[:n_train])
    val_vids   = set(video_names[n_train : n_train + n_val])
    test_vids  = set(video_names[n_train + n_val:])

    train_seq  = [s for s in sequences if s['video_name'] in train_vids]
    val_seq    = [s for s in sequences if s['video_name'] in val_vids]
    test_seq   = [s for s in sequences if s['video_name'] in test_vids]

    return train_seq, val_seq, test_seq


#Dataset PyTorch

class EpilepsyDataset(Dataset):
    """
    Dataset PyTorch per sequenze di frame con label binaria.

    Ogni campione restituisce:
      - frames : Tensor [T, C, H, W]  float32, normalizzato
      - label  : Tensor scalare        float32 (0.0 o 1.0)

    Il caricamento dei frame è lazy: le immagini vengono lette da disco
    solo al momento del __getitem__, non durante l'inizializzazione.
    Questo mantiene basso il consumo di RAM anche su dataset grandi.
    """

    def __init__(self, sequences: List[dict], transform=None):
        self.sequences = sequences
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq   = self.sequences[idx]
        frame_tensors = []

        for path in seq['frame_paths']:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frame_tensors.append(img)

        # stack → [T, C, H, W]
        frames = torch.stack(frame_tensors, dim=0)
        label  = torch.tensor(seq['seq_label'], dtype=torch.float32)

        return frames, label


#Factory function

def build_dataloaders(manifest_path: Path,
                      train_transform,
                      eval_transform,
                      batch_size:  int = 8,
                      num_workers: int = 4,
                      seq_len:     int = SEQ_LEN,
                      stride:      int = STRIDE) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Funzione di alto livello che:
      1. Carica il manifest
      2. Costruisce le sequenze con sliding window
      3. Splitta per video
      4. Crea i tre DataLoader (train, val, test)

    batch_size=8 è un punto di partenza conservativo per GPU con 16GB VRAM
    con frame 210x210 e sequenze da 30 frame. Aumenta a 16 se la VRAM lo permette.
    num_workers=4 è adatto per il cluster; aumenta fino a num_cpu_cores//2.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    sequences              = build_sequences(manifest, seq_len, stride)
    train_seq, val_seq, test_seq = split_sequences(sequences)

    train_ds = EpilepsyDataset(train_seq, transform=train_transform)
    val_ds   = EpilepsyDataset(val_seq,   transform=eval_transform)
    test_ds  = EpilepsyDataset(test_seq,  transform=eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,          # shuffle solo in training
        num_workers = num_workers,
        pin_memory  = True,          # accelera il trasferimento CPU→GPU
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    return train_loader, val_loader, test_loader