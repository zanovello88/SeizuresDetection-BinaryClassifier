# Epilepsy Seizure Detection in Mice — CNN+LSTM Classifier

Tesi magistrale in Ingegneria Informatica (specializzazione Intelligenza Artificiale)
Università degli Studi di Ferrara

## Obiettivo

Classificatore binario per il riconoscimento automatico di episodi di crisi
epilettiche in topi da laboratorio, tramite analisi video con un modello CNN+LSTM.
Il modello analizza sequenze di frame, estrae feature spaziali (CNN) e modella
la dinamica temporale (LSTM), producendo per ogni sequenza una probabilità
di crisi e identificando automaticamente onset e offset dell'evento.

## Dataset

- 101 video (~90s ciascuno, 210×210px, 30fps)
- Ogni video contiene: 10s pre-ictal + fase ictal + 10s post-ictal
- Annotazioni in `mappa_labels.csv` (onset/offset frame per ogni video)
- Distribuzione classi: ~70% crisi, ~30% non-crisi (dopo subsampling a 10fps)

## Architettura
```
Input [B, T, C, H, W]
  → MobileNetV3-Small (pre-trained ImageNet, parzialmente frozen)
  → Proiezione lineare (576 → 256)
  → LSTM (2 layer, hidden=256, dropout=0.3)
  → FC classifier (256 → 64 → 1)
  → Output: P(crisi) per sequenza
```

- **CNN**: MobileNetV3-Small — scelto per il basso numero di parametri
  (2.1M totali) che riduce il rischio di overfitting con dataset piccoli
- **LSTM**: 2 layer con hidden size 256 — cattura dipendenze temporali
  su finestre di 3 secondi (30 frame a 10fps)
- **Loss**: BCEWithLogitsLoss con pos_weight=0.4265 per bilanciamento classi

## Struttura del progetto
```
├── data/
│   ├── mappa_labels.csv         # annotazioni 
│   ├── manifest.json            # generato da preprocessing.py
│   └── frames/                  # generato da preprocessing.py (non in git)
├── src/
│   ├── creazione_dataset.py     # ritaglio video sul topo
│   ├── preprocessing.py         # estrazione frame + costruzione manifest
│   ├── transforms.py            # augmentation e normalizzazione
│   ├── dataset.py               # Dataset PyTorch + sliding window
│   ├── model.py                 # architettura CNN+LSTM
│   └── train.py                 # training loop
├── tools/
│   ├── inspect_manifest.py      # verifica distribuzione classi
│   └── inspect_dataset.py       # verifica split e shape tensori
├── runs/                        # output training (non in git)
├── requirements.txt
└── README.md
```

## Installazione
```bash
git clone <url-repo>
cd Tesi
pip install -r requirements.txt
```

## Pipeline di esecuzione

### 1. Preprocessing (Mac locale)
```bash
python src/preprocessing.py
python tools/inspect_manifest.py
python tools/inspect_dataset.py
```

### 2. Training (cluster SLURM)
```bash
# trasferimento dati
rsync -avz --progress data/frames/ utente@cluster:/path/progetto/data/frames/
rsync -avz data/manifest.json utente@cluster:/path/progetto/data/

# submit job
sbatch jobs/train_job.sh
```

### 3. Monitoraggio
```bash
squeue -u $USER                          # stato del job
tail -f runs/<run_id>/train.log          # log in tempo reale
```

## Parametri principali

| Parametro | Valore | Motivazione |
|---|---|---|
| Frame rate | 10fps | Subsampling da 30fps, riduce ridondanza |
| Seq length | 30 frame | 3 secondi, cattura dinamica convulsiva |
| Stride | 15 frame | Overlap 50%, massimizza campioni |
| Batch size | 8 | Conservativo per GPU 16GB con frame 210×210 |
| Learning rate | 1e-4 | Standard per fine-tuning |
| pos_weight | 0.4265 | Calcolato dalla distribuzione reale del dataset |
| Early stopping | patience=10 | Evita overfitting senza fissare epoche |

## Metriche di valutazione

- **Frame-level**: accuracy, precision, recall, F1, ROC-AUC
- **Event-level**: detection delay (onset), overlap predizione/ground truth

## Requisiti

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (per training su GPU)
- Vedi `requirements.txt` per la lista completa

## Autore

Francesco Zanovello
Corso di Laurea Magistrale in Ingegneria Informatica
Università degli Studi di Ferrara