"""
Scopo: definire l'architettura CNN+LSTM per la classificazione binaria
       di sequenze di frame (crisi / non crisi).

Architettura:
  1. CNN backbone (MobileNetV3-Small, pre-addestrato su ImageNet):
     - Scelta rispetto a ResNet: MobileNetV3-Small ha 2.5M parametri vs 11M
       di ResNet-18, fondamentale con solo 100 video per evitare overfitting.
     - Il layer di classificazione finale viene rimosso; usiamo solo il feature
       extractor che produce un vettore da 576 dimensioni per frame.
     - I pesi vengono congelati nei primi strati (feature di basso livello
       universali) e fine-tunati negli ultimi (feature di alto livello
       domain-specific).

  2. Proiezione lineare (576 → 256):
     - Riduce la dimensionalità prima dell'LSTM per diminuire il numero
       di parametri e stabilizzare il training.

  3. LSTM (2 layer, hidden=256, dropout=0.3 tra i layer):
     - 2 layer: sufficiente per catturare dipendenze temporali a medio raggio
       (3 secondi di sequenza). Più layer non portano benefici con sequenze corte.
     - hidden=256: bilanciamento tra capacità espressiva e rischio overfitting.
     - Dropout tra layer LSTM: regolarizzazione standard per sequenze temporali.

  4. Classificatore finale (FC + Dropout + Sigmoid implicita):
     - Output singolo scalare: probabilità P(crisi) per la sequenza.
     - Il Sigmoid non è nel modello ma nella loss (BCEWithLogitsLoss) per
       stabilità numerica — comportamento standard in PyTorch.

Flusso dati:
  Input [B, T, C, H, W]
    → CNN applicata frame per frame → [B, T, 576]
    → Proiezione lineare            → [B, T, 256]
    → LSTM                          → [B, T, 256]
    → Ultimo hidden state           → [B, 256]
    → FC + Dropout                  → [B, 1]
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class CNNEncoder(nn.Module):
    """
    Wrapper attorno a MobileNetV3-Small che:
      - carica i pesi ImageNet pre-addestrati
      - rimuove il classifier finale
      - congela i primi N layer (feature universali)
      - espone il metodo forward che restituisce [B, 576]

    Il congelamento parziale è una strategia di fine-tuning standard:
    i primi layer della CNN imparano feature generiche (bordi, texture)
    che sono utili per qualsiasi dominio; congelarli riduce il numero di
    parametri ottimizzabili e accelera la convergenza.
    """

    def __init__(self, freeze_layers: int = 10):
        super().__init__()

        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # rimuove il classifier: teniamo solo il feature extractor
        # output shape dopo avgpool: [B, 576, 1, 1] → flatten → [B, 576]
        self.features  = backbone.features
        self.avgpool   = backbone.avgpool
        self.feat_dim  = 576

        # congela i primi `freeze_layers` layer del feature extractor
        children = list(self.features.children())
        for layer in children[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → output: [B, 576]"""
        x = self.features(x)           # [B, 576, 7, 7] circa
        x = self.avgpool(x)            # [B, 576, 1, 1]
        x = torch.flatten(x, 1)        # [B, 576]
        return x


class CNNLSTM(nn.Module):
    """
    Modello completo CNN+LSTM per classificazione binaria di sequenze.

    Parametri:
      cnn_feat_dim  : dimensione output CNN (576 per MobileNetV3-Small)
      proj_dim      : dimensione dopo proiezione lineare (default 256)
      lstm_hidden   : dimensione hidden state LSTM (default 256)
      lstm_layers   : numero di layer LSTM (default 2)
      lstm_dropout  : dropout tra layer LSTM (default 0.3)
      fc_dropout    : dropout prima del classificatore finale (default 0.5)
    """

    def __init__(self,
                 cnn_feat_dim : int   = 576,
                 proj_dim     : int   = 256,
                 lstm_hidden  : int   = 256,
                 lstm_layers  : int   = 2,
                 lstm_dropout : float = 0.3,
                 fc_dropout   : float = 0.5,
                 freeze_layers: int   = 10):
        super().__init__()

        #1. CNN encoder
        self.cnn = CNNEncoder(freeze_layers=freeze_layers)

        #2. Proiezione lineare CNN → LSTM
        self.projection = nn.Sequential(
            nn.Linear(cnn_feat_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        #3. LSTM 
        self.lstm = nn.LSTM(
            input_size   = proj_dim,
            hidden_size  = lstm_hidden,
            num_layers   = lstm_layers,
            batch_first  = True,       # input: [B, T, features]
            dropout      = lstm_dropout if lstm_layers > 1 else 0.0,
        )

        #4. Classificatore finale
        self.classifier = nn.Sequential(
            nn.Dropout(p=fc_dropout),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),          # output scalare — sigmoid nella loss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W]
        output: [B, 1]  (logit, non probabilità — sigmoid applicata dalla loss)
        """
        B, T, C, H, W = x.shape

        # applica CNN a ogni frame indipendentemente
        # reshape: [B*T, C, H, W] → CNN → [B*T, 576]
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)                    # [B*T, 576]

        # proiezione
        x = self.projection(x)             # [B*T, 256]

        # torna a [B, T, 256] per l'LSTM
        x = x.view(B, T, -1)

        # LSTM — prendiamo solo l'output dell'ultimo timestep
        # output: [B, T, hidden], (h_n, c_n)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]             # [B, hidden] ultimo timestep

        # classificatore
        x = self.classifier(x)             # [B, 1]
        return x


#Utility: conta parametri 

def count_parameters(model: nn.Module) -> None:
    """Stampa il numero di parametri totali e trainabili del modello."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"  Parametri totali     : {total:,}")
    print(f"  Parametri trainabili : {trainable:,}")
    print(f"  Parametri congelati  : {frozen:,}")


#Test rapido del modello

if __name__ == '__main__':
    """
    Eseguendo direttamente model.py si verifica che:
      - il modello si costruisce senza errori
      - il forward pass produce la shape corretta
      - il numero di parametri è ragionevole
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model = CNNLSTM().to(device)

    print("Architettura — parametri:")
    count_parameters(model)

    #forward pass con batch fittizio [B=2, T=30, C=3, H=210, W=210]
    dummy = torch.randn(2, 30, 3, 210, 210).to(device)
    with torch.no_grad():
        out = model(dummy)

    print(f"\nForward pass OK")
    print(f"  Input shape  : {dummy.shape}")
    print(f"  Output shape : {out.shape}")    # atteso: [2, 1]
    print(f"  Output values: {out.squeeze().tolist()}")