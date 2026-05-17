# IAthon — Reconnaissance d’émotions

Application de bureau pour la détection d’émotions faciales en temps réel, développée dans le cadre de l’**IAthon** (Université du Québec à Chicoutimi / École Louvain en Hainaut). Le projet vise à proposer un retour adapté lorsque des émotions perçues comme difficiles sont détectées chez l’enfant.

## Équipe

- Charlotte Chanudet  
- Mahaut Galice  
- Amal Ouedraogo  
- Colin Pouliart  

## Fonctionnalités

- Interface **CustomTkinter** (thème sombre, onglets **Runtime** et **Configuration**)
- Flux webcam avec bouton **Play / Pause**
- Trois modèles au choix :
  - **CNN** — réseau convolutif entraîné sur visages 48×48
  - **Landmarks** — MediaPipe (points du visage) + MLP
  - **ResNet50** — transfer learning (ImageNet → émotions)
- Détection des émotions **angry**, **disgust**, **fear**, **sad**, **surprise** avec seuils de confiance (hystérésis)
- Affichage d’un **GIF** plein écran dans la zone Runtime (fichier choisi au hasard dans `assets/`)
- Durée minimale d’affichage après la **première frame** visible (réglable via `_NEG_OVERLAY_MIN_SECONDS` dans `ui.py`)
- Mode **plein écran** (Échap pour quitter)
- Métriques de performance (débit, temps d’inférence) dans Configuration

## Prérequis

- **Python 3.10+** (3.12 testé)
- Webcam
- **GPU CUDA** optionnel (accélère PyTorch)

## Installation

```bash
cd Iathon
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install customtkinter
```

### Modèle Landmarks (MediaPipe)

Au premier lancement avec **Landmarks**, le fichier `models/landmarks/face_landmarker.task` est utilisé s’il est présent ; sinon il peut être téléchargé automatiquement par l’application.

Fichiers attendus pour **Landmarks** :

| Fichier | Emplacement |
|---------|-------------|
| `mlp_best.pt` | `models/landmarks/` |
| `scaler.joblib` | `models/landmarks/` |
| `label_encoder.joblib` | `models/landmarks/` |
| `face_landmarker.task` | `models/landmarks/` (optionnel si téléchargement auto) |

### Autres modèles

| Modèle | Poids |
|--------|--------|
| CNN | `models/CNN/models/best_model.pth` |
| ResNet50 | `models/transfer_learning/best_model.pth` |

## Lancement

```bash
python main.py
```

1. Onglet **Configuration** : choisir la caméra et le modèle d’IA.  
2. Cliquer sur **▶** pour démarrer le flux.  
3. Onglet **Runtime** : aperçu principal ; en cas d’émotion négative suffisamment confiante, un GIF s’affiche par-dessus la zone vidéo.

## GIF d’alerte (`assets/`)

Placez un ou plusieurs fichiers **`.gif`** dans le dossier `assets/`. À chaque alerte, un fichier est tiré au hasard. Le GIF conserve son **ratio** et occupe le maximum d’espace dans la zone (bandes si nécessaire).

## Structure du projet

```
Iathon/
├── main.py              # Point d’entrée
├── ui.py                # Interface CustomTkinter
├── requirements.txt
├── assets/              # GIFs d’alerte
├── core/
│   ├── camera.py        # Capture OpenCV
│   └── vision.py        # Détection visage + inférence
└── models/
    ├── CNN/             # Entraînement et poids CNN
    ├── landmarks/       # MLP + MediaPipe + scripts d’entraînement
    └── transfer_learning/  # ResNet50 + app Streamlit (hors UI principale)
```

## Émotions et seuils

Émotions prises en compte pour l’alerte : **angry**, **disgust**, **fear**, **sad**, **surprise**.

Seuils par défaut dans `ui.py` :

- Entrée alerte : **58 %** de confiance (`_NEG_ALERT_ENTER_CONF`)
- Sortie (hystérésis) : **42 %** (`_NEG_ALERT_EXIT_CONF`)

## Entraînement des modèles (optionnel)

Les scripts d’entraînement et la documentation détaillée se trouvent dans chaque sous-dossier de `models/` :

- `models/CNN/` — CNN custom  
- `models/landmarks/` — `train_mlp_landmarks.py`, jeux de données CSV  
- `models/transfer_learning/` — fine-tuning ResNet50, `TRANSFER_LEARNING.md`, application Streamlit `app.py`

## Dépannage

| Problème | Piste |
|----------|--------|
| Pas de caméra listée | Vérifier les permissions ; sur Windows, les indices 0–9 sont testés avec le backend DirectShow |
| `mediapipe` / Landmarks | Installer `mediapipe>=0.10.9` ; vérifier `face_landmarker.task` |
| Modèle CNN / ResNet absent | Vérifier les chemins des fichiers `.pth` ci-dessus |
| Interface lente | Réduire la résolution côté caméra ou utiliser un GPU pour PyTorch |

## Licence et crédits

Projet académique IAthon. Les GIF placés dans `assets/` doivent respecter les droits d’utilisation de leurs auteurs.
