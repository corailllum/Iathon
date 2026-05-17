#  Emotion Recognition CNN

Reconnaissance d'émotions faciales en temps réel avec un CNN PyTorch.

## Émotions reconnues
`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

##  Installation

```bash
pip install -r requirements.txt
```

Pour GPU (recommandé) :
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---



---

##  Entraînement

```bash
python train.py
```

Ce script va :
1. Charger et augmenter les données
2. Entraîner le CNN (~50 epochs)
3. Sauvegarder le meilleur modèle → `models/best_model.pth`
4. Générer les métriques → `results/metrics_analysis.png`
5. Afficher l'émotion la mieux et la moins bien reconnue

---

##  Webcam temps réel

```bash
python webcam.py
```

Options :
```bash
python webcam.py --model ./models/best_model.pth --camera 0 --device auto
```

Contrôles :
- **Q** → Quitter
- **S** → Prendre un screenshot

---

##  Métriques générées

| Fichier | Description |
|---------|-------------|
| `results/metrics_analysis.png` | Courbes loss/acc, confusion matrix, F1 par émotion |
| `results/results.json` | Résultats complets en JSON |
| `models/best_model.pth` | Poids du meilleur modèle |

---

##  Architecture CNN

```
Input (1×48×48)
    ↓
Block 1: Conv 32 + Conv 32 + MaxPool + Dropout
    ↓
Block 2: Conv 64 + Conv 64 + MaxPool + Dropout
    ↓
Block 3: Conv 128 + Conv 128 + MaxPool + Dropout
    ↓
Block 4: Conv 256 + Conv 256 + MaxPool + Dropout
    ↓
Global Average Pooling
    ↓
FC 256 → FC 512 → FC 256 → FC 7
    ↓
Softmax → Émotion
```