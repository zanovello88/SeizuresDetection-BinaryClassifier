"""
Scopo: verifica la correttezza del dataset PRIMA di avviare il training.
Controlla la distribuzione delle sequenze per split, il bilanciamento delle
classi in ogni split e la forma dei tensori in uscita dal DataLoader.
Eseguilo sempre dopo aver sistemato dataset.py e prima di train.py — costa
pochi secondi e ti salva da lunghe sessioni di debugging sul cluster.
"""

import json
import sys
from pathlib import Path

#Aggiunge src/ al path in modo che i moduli del progetto abbiano priorità
SRC_DIR = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

from dataset import build_sequences, split_sequences, EpilepsyDataset

#Import diretto dal file per evitare conflitti con librerie di sistema
import importlib.util, types

def _load_module(name: str, filepath: Path):
    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

transforms_mod  = _load_module('transforms', SRC_DIR / 'transforms.py')
train_transforms = transforms_mod.train_transforms
eval_transforms  = transforms_mod.eval_transforms

MANIFEST_PATH = Path('data/manifest.json')

#Carica e processa
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

sequences            = build_sequences(manifest)
train_seq, val_seq, test_seq = split_sequences(sequences)

#Report split
def report_split(name: str, seqs: list):
    n      = len(seqs)
    n_pos  = sum(s['seq_label'] == 1 for s in seqs)
    n_neg  = n - n_pos
    videos = {s['video_name'] for s in seqs}
    print(f"\n{name}")
    print(f"  Sequenze  : {n:,}")
    print(f"  Video     : {len(videos)}")
    print(f"  Label 1   : {n_pos:,}  ({100*n_pos/n:.1f}%)")
    print(f"  Label 0   : {n_neg:,}  ({100*n_neg/n:.1f}%)")

print("=" * 45)
print("DISTRIBUZIONE SPLIT")
report_split("TRAIN", train_seq)
report_split("VAL",   val_seq)
report_split("TEST",  test_seq)
print("=" * 45)

#Verifica forma tensori 
print("\nVerifica forma tensori (primo campione di train)...")
ds            = EpilepsyDataset(train_seq[:1], transform=eval_transforms)
frames, label = ds[0]
print(f"  frames shape : {frames.shape}")   # atteso: [30, 3, 210, 210]
print(f"  label        : {label}")
print(f"  dtype frames : {frames.dtype}")
print(f"  min / max    : {frames.min():.3f} / {frames.max():.3f}")
print()
if list(frames.shape) == [30, 3, 210, 210]:
    print("OK  frames.shape corretto → procedi con il modello")
else:
    print("ATTENZIONE  shape inattesa — controlla SEQ_LEN e IMG_SIZE")

"""
francescozanovello@Mac Tesi % python3 tools/inspect_dataset.py
=============================================
DISTRIBUZIONE SPLIT

TRAIN
  Sequenze  : 2,974
  Video     : 70
  Label 1   : 2,160  (72.6%)
  Label 0   : 814  (27.4%)

VAL
  Sequenze  : 750
  Video     : 15
  Label 1   : 574  (76.5%)
  Label 0   : 176  (23.5%)

TEST
  Sequenze  : 574
  Video     : 15
  Label 1   : 402  (70.0%)
  Label 0   : 172  (30.0%)
=============================================

Verifica forma tensori (primo campione di train)...
  frames shape : torch.Size([30, 3, 210, 210])
  label        : 0.0
  dtype frames : torch.float32
  min / max    : -2.118 / 2.640

OK  frames.shape corretto → procedi con il modello
"""