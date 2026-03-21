"""
Scopo: definire le trasformazioni applicate ai frame prima di entrare nella CNN.

Motivazioni delle scelte:
  - Nessun resize: i video sono già 210x210 e ritagliati sul topo.
    Qualsiasi resize introdurrebbe interpolazione inutile o perdita di area.
  - ImageNet mean/std per la normalizzazione: usiamo un backbone pre-addestrato
    su ImageNet (MobileNetV3), quindi dobbiamo rispettare la sua distribuzione
    di input attesa. Anche se i video di topi sono fuori dominio, il transfer
    learning funziona comunque perché le feature di basso livello (bordi,
    texture) sono universali.
  - Augmentation conservativa in training:
      · ColorJitter: simula variazioni reali di illuminazione tra sessioni
      · RandomAffine (piccola): simula micro-movimenti della telecamera
      · NO flip orizzontale: i movimenti convulsivi possono avere asimmetria
        laterale diagnosticamente rilevante
      · NO rotazioni grandi: il topo è sempre orientato in modo simile nel frame
  - eval_transforms: solo ToTensor + Normalize, deterministico e riproducibile.
"""

from torchvision import transforms

#Training: augmentation leggera e domain-aware 
train_transforms = transforms.Compose([
    transforms.ColorJitter(
        brightness = 0.2,
        contrast   = 0.2,
        saturation = 0.1,
    ),
    transforms.RandomAffine(
        degrees   = 3,
        translate = (0.02, 0.02),
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225],
    ),
])

#Validation / Test: nessuna augmentation
eval_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225],
    ),
])