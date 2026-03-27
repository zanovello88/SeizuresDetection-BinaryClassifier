# Epilepsy Seizure Detection in Mice — CNN+LSTM Classifier

Tesi magistrale in Ingegneria Informatica (specializzazione Intelligenza Artificiale)
Università degli Studi di Ferrara

## Obiettivo

Classificatore binario per il riconoscimento automatico di episodi di crisi
epilettiche in topi da laboratorio, tramite analisi video con un modello CNN+LSTM.
Il modello analizza sequenze di frame, estrae feature spaziali (CNN) e modella
la dinamica temporale (LSTM), producendo per ogni sequenza una probabilità
di crisi e identificando automaticamente onset e offset dell'evento.

## Risultati ottenuti

| Metrica | Baseline | Modello finale | + Smoothing |
|---|---|---|---|
| F1-score | 0.613 | 0.765 | **0.833** |
| Recall | 0.491 | 0.687 | **0.843** |
| Precision | 0.755 | 0.862 | 0.823 |
| ROC-AUC | 0.585 | 0.695 | 0.653 |
| Crisi non rilevate | 7/15 | 0/15 | **0/15** |
| Overlap medio | 0.455 | 0.669 | **0.831** |
| Detection delay mediano | 0.00s | 0.50s | 0.00s |

## Dataset

- 101 video (~90s ciascuno, 210×210px, 30fps)
- Ogni video contiene: 10s pre-ictal + fase ictal + 10s post-ictal
- Annotazioni in `mappa_labels.csv` (onset/offset frame per ogni video)
- Distribuzione classi dopo subsampling a 10fps: ~70% crisi, ~30% non-crisi
- Split train/val/test: 70/15/15 video (split per video, no data leakage)

## Architettura
```
Input [B, T, C, H, W]
  → MobileNetV3-Small (pre-trained ImageNet, freeze_layers=8)
  → Proiezione lineare (576 → 256)
  → LSTM (2 layer, hidden=256, dropout=0.3)
  → FC classifier (256 → 64 → 1)
  → Sigmoid → P(crisi) per sequenza
```

- **CNN**: MobileNetV3-Small — 2.1M parametri totali, scelto per il basso
  numero di parametri che riduce il rischio di overfitting con dataset piccoli
- **LSTM**: 2 layer con hidden size 256 — cattura dipendenze temporali
  su finestre di 6 secondi (60 frame a 10fps)
- **Loss**: BCEWithLogitsLoss con pos_weight=3.0 per bilanciamento classi
- **Post-processing**: smoothing con mediana mobile (window=10) per
  ridurre i falsi positivi e aumentare la copertura della crisi

## Struttura del progetto
```
├── data/
│   ├── mappa_labels.csv              # annotazioni
│   ├── manifest.json                 # generato da preprocessing.py
│   └── frames/                       # generato da preprocessing.py (non in git)
│
├── src/
│   ├── creazione_dataset.py          # ritaglio video sul topo
│   ├── preprocessing.py             # estrazione frame + costruzione manifest
│   ├── transforms.py                # augmentation e normalizzazione
│   ├── dataset.py                   # Dataset PyTorch + sliding window
│   ├── model.py                     # architettura CNN+LSTM
│   ├── train.py                     # training loop
│   ├── evaluate.py                  # valutazione + metriche + plot
│   └── inference.py                 # inferenza real-time su video
│
├── tools/
│   ├── inspect_manifest.py          # verifica distribuzione classi
│   ├── inspect_dataset.py           # verifica split e shape tensori
│   └── plot_thesis.py               # grafici comparativi per la tesi
│
├── jobs/
│   └── train_job.sh                 # script SLURM per il cluster
│
├── model_weights/
│   └── mobilenet_v3_small_imagenet.pth   # pesi pre-addestrati (non in git)
│
├── runs/                            # output training (non in git)
│   └── <run_id>/
│       ├── best_model.pt
│       ├── history.json
│       ├── train.log
│       └── eval_results*.json
│
├── thesis_plots/                    # grafici per la tesi (non in git)
├── inference_results/               # output inferenza (non in git)
├── requirements.txt
└── README.md
```

## Installazione
```bash
git clone <url-repo>
cd Tesi
pip install -r requirements.txt
```

Per il cluster universitario (SLURM + CUDA 12.2):
```bash
module purge
module load cuda/12.2
module load python/3.11.6-gcc-11.3.1-6nwylkz
python -m venv venv_tesi
source venv_tesi/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
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
rsync -avz data/frames/ utente@cluster:~/tesi/data/frames/
rsync -avz data/manifest.json utente@cluster:~/tesi/data/
rsync -avz model_weights/ utente@cluster:~/tesi/model_weights/

# submit job
sbatch jobs/train_job.sh

# monitoraggio
squeue -u $USER
tail -f epilepsy-train-<JOBID>.log
```

### 3. Valutazione (cluster)
```bash
python src/evaluate.py \
  --checkpoint runs/<run_id>/best_model.pt \
  --manifest   data/manifest.json
```

### 4. Inferenza real-time (cluster)
```bash
python src/inference.py \
  --video      data/dataset_tagliato/<video>.mp4 \
  --checkpoint runs/<run_id>/best_model.pt \
  --threshold  0.874 \
  --gt_onset   300 \
  --gt_offset  <frame_fine> \
  --save_video
```

### 5. Grafici per la tesi (Mac locale)
```bash
python tools/plot_thesis.py
```

## Parametri del modello finale

| Parametro | Valore | Motivazione |
|---|---|---|
| Frame rate analisi | 10fps | Subsampling da 30fps, riduce ridondanza |
| Seq length | 60 frame | 6 secondi, cattura dinamica convulsiva |
| Stride | 15 frame | Overlap 75%, massimizza sequenze |
| Batch size | 16 | Bilanciamento memoria/gradients |
| Learning rate | 3e-5 | Fine-tuning conservativo |
| pos_weight | 3.0 | Penalizza falsi negativi (recall critico) |
| freeze_layers | 8 | Sblocca layer CNN per adattamento dominio |
| Smoothing | median, w=10 | Riduce picchi isolati, aumenta overlap |
| Threshold | 0.874 | Ottimizzato con Youden's J sulla ROC |

## Cluster universitario

- **Nodo GPU**: gnode01
- **GPU**: NVIDIA H100 NVL
- **Scheduler**: SLURM
- **Moduli**: `cuda/12.2`, `python/3.11.6-gcc-11.3.1-6nwylkz`
- **Tempo per run**: ~45 minuti (80 epoche max, early stopping ~30 epoche)

## Possibili miglioramenti futuri

- **Attention mechanisms**: aggiungere self-attention sull'output LSTM
  per pesare differentemente i timestep più informativi
- **Vision Transformer**: sostituire MobileNetV3 con ViT pre-addestrato
  su dataset medici (es. BioViL)
- **3D CNN**: convoluzione spazio-temporale per catturare pattern
  di movimento direttamente invece di separare CNN e LSTM
- **Self-supervised pre-training**: pre-addestrare la CNN su tutti i
  frame del dataset (inclusi quelli non annotati) prima del fine-tuning
- **Dataset aumentato**: raccogliere più video per ridurre overfitting
  e migliorare la generalizzazione tra topi diversi

## Autore

Francesco Zanovello
Corso di Laurea Magistrale in Ingegneria Informatica
Università degli Studi di Ferrara
