"""
Scopo: eseguire il training completo del modello CNN+LSTM sul cluster.
       Questo è l'unico file da lanciare sul cluster — legge tutto dalla
       configurazione e salva checkpoint e log in modo strutturato.

Motivazioni delle scelte:
  - BCEWithLogitsLoss con pos_weight=0.4265: compensa lo sbilanciamento
    classi (70% crisi) penalizzando di più gli errori sulla classe minoritaria.
  - AdamW come ottimizzatore: variante di Adam con weight decay corretto,
    standard per fine-tuning di reti pre-addestrate.
  - ReduceLROnPlateau: riduce il learning rate quando la val_loss smette
    di migliorare — più robusto di uno scheduler fisso con dataset piccoli.
  - Early stopping su val_loss con patience=10: evita overfitting senza
    dover fissare il numero di epoche a priori.
  - Gradient clipping (max_norm=1.0): stabilizza il training dell'LSTM
    che è soggetto a exploding gradients su sequenze lunghe.
  - Salvataggio del best model per val_loss: il checkpoint migliore è
    quello con la loss di validazione più bassa, non l'ultimo.
  - Logging su file + console: sul cluster non hai una sessione interattiva,
    quindi tutto deve essere scritto su file per poterlo leggere dopo.
"""

import json
import logging
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime

import sys
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from dataset    import build_sequences, split_sequences, EpilepsyDataset, build_dataloaders
from model      import CNNLSTM, count_parameters
from transforms import train_transforms, eval_transforms

# Argomenti da riga di comando 
# Tutti i parametri sono configurabili da CLI così puoi lanciare esperimenti
# diversi sul cluster senza modificare il codice.

def parse_args():
    p = argparse.ArgumentParser(description='Training CNN+LSTM epilessia')
    p.add_argument('--manifest',     type=str, default='data/manifest.json')
    p.add_argument('--output_dir',   type=str, default='runs')
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--pos_weight',   type=float, default=0.4265)
    p.add_argument('--patience',     type=int,   default=10)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--seq_len',      type=int,   default=30)
    p.add_argument('--stride',       type=int,   default=15)
    p.add_argument('--freeze_layers',type=int,   default=10)
    p.add_argument('--seed',         type=int,   default=42)
    return p.parse_args()


# Setup logging 

def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Configura logging su file e console simultaneamente.
    Sul cluster leggi il file .log dopo il job — non perdi nulla.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / 'train.log'

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%H:%M:%S')

    # handler file
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # handler console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# Funzioni di training e validazione 

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    """
    Esegue una singola epoca di training.
    Restituisce loss media e accuracy sul training set.
    Il gradient clipping è applicato dopo ogni backward pass per
    prevenire exploding gradients nell'LSTM.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)   # [B, T, C, H, W]
        labels = labels.to(device, non_blocking=True)   # [B]

        optimizer.zero_grad()
        logits = model(frames).squeeze(1)               # [B]
        loss   = criterion(logits, labels)
        loss.backward()

        # gradient clipping — fondamentale per LSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item() * frames.size(0)
        preds       = (torch.sigmoid(logits) >= 0.5).float()
        correct    += (preds == labels).sum().item()
        total      += frames.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Valuta il modello su un loader (val o test).
    Restituisce loss, accuracy e tutte le predizioni/label
    per il calcolo di metriche più ricche in evaluate.py.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs  = []
    all_labels = []

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(frames).squeeze(1)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * frames.size(0)
        probs       = torch.sigmoid(logits)
        preds       = (probs >= 0.5).float()
        correct    += (preds == labels).sum().item()
        total      += frames.size(0)

        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_probs, all_labels


# Main

def main():
    args   = parse_args()

    # cartella di output con timestamp — ogni run ha la sua cartella
    run_id     = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / run_id
    log        = setup_logging(output_dir)

    # riproducibilità
    torch.manual_seed(args.seed)

    # device — su cluster con GPU sarà automaticamente 'cuda'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")
    log.info(f"Run ID: {run_id}")
    log.info(f"Args: {vars(args)}")

    # DataLoader 
    log.info("Caricamento dataset...")
    train_loader, val_loader, test_loader = build_dataloaders(
        manifest_path   = Path(args.manifest),
        train_transform = train_transforms,
        eval_transform  = eval_transforms,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        seq_len         = args.seq_len,
        stride          = args.stride,
    )
    log.info(f"Train: {len(train_loader.dataset):,} seq | "
             f"Val: {len(val_loader.dataset):,} seq | "
             f"Test: {len(test_loader.dataset):,} seq")

    # Modello 
    model = CNNLSTM(freeze_layers=args.freeze_layers).to(device)
    log.info("Architettura:")
    count_parameters(model)

    # Loss, ottimizzatore, scheduler 
    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    # Training loop 
    best_val_loss  = float('inf')
    epochs_no_imp  = 0
    history        = []

    log.info("Inizio training...")
    log.info(f"{'Epoca':>5} | {'TrainLoss':>9} | {'TrainAcc':>8} | "
             f"{'ValLoss':>7} | {'ValAcc':>6} | {'LR':>8}")
    log.info("-" * 62)

    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_probs, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        log.info(f"{epoch:>5} | {train_loss:>9.4f} | {train_acc:>7.3f}% | "
                 f"{val_loss:>7.4f} | {val_acc:>5.3f}% | {current_lr:>8.2e}")

        history.append({
            'epoch'     : epoch,
            'train_loss': train_loss,
            'train_acc' : train_acc,
            'val_loss'  : val_loss,
            'val_acc'   : val_acc,
            'lr'        : current_lr,
        })

        # Salva best model 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_imp = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss'   : val_loss,
                'val_acc'    : val_acc,
                'args'       : vars(args),
            }, output_dir / 'best_model.pt')
            log.info(f"        ✓ best model salvato (val_loss={val_loss:.4f})")

        # Early stopping 
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= args.patience:
                log.info(f"Early stopping alla epoca {epoch} "
                         f"(nessun miglioramento per {args.patience} epoche)")
                break

    # Salva history 
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    log.info(f"\nTraining completato. Best val_loss: {best_val_loss:.4f}")
    log.info(f"Checkpoint salvato in: {output_dir / 'best_model.pt'}")
    log.info(f"History salvata in:    {output_dir / 'history.json'}")


if __name__ == '__main__':
    main()
