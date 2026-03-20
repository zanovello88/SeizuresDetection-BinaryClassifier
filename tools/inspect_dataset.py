"""
Scopo: verifica la correttezza del dataset PRIMA di avviare il training.
Controlla la distribuzione delle sequenze per split, il bilanciamento delle
classi in ogni split e la forma dei tensori in uscita dal DataLoader.
Da eseguire sempre dopo dataset.py e prima di train.py — costa pochi secondi
e ti salva da debugging lungo sul cluster.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset    import build_sequences, split_sequences, EpilepsyDataset
from transforms import train_transforms, eval_transforms

MANIFEST_PATH = Path('data/manifest.json')

#Carica e processa
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

sequences = build_sequences(manifest)
train_seq, val_seq, test_seq = split_sequences(sequences)

#Report split
def report_split(name: str, seqs: list):
    n       = len(seqs)
    n_pos   = sum(s['seq_label'] == 1 for s in seqs)
    n_neg   = n - n_pos
    videos  = {s['video_name'] for s in seqs}
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
ds     = EpilepsyDataset(train_seq[:1], transform=eval_transforms)
frames, label = ds[0]
print(f"  frames shape : {frames.shape}")   # atteso: [30, 3, 210, 210]
print(f"  label        : {label}")
print(f"  dtype frames : {frames.dtype}")
print(f"  min/max      : {frames.min():.3f} / {frames.max():.3f}")
print("\nSe frames.shape == [30, 3, 210, 210] → tutto corretto, procedi con train.py")