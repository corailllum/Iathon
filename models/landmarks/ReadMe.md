# Détection d’émotions faciales à partir de landmarks

Cette partie du projet permet de détecter des émotions faciales en temps réel à partir des landmarks du visage extraits avec MediaPipe, puis classifiés par un réseau de neurones MLP entraîné avec PyTorch.

## Objectif du projet

L’objectif est de construire un système complet de reconnaissance d’émotions basé uniquement sur les points clés du visage (landmarks), sans image brute. On suit donc les étapes suivantes : 

1. Capture des landmarks via caméra (MediaPipe FaceMesh)
2. Construction d’un dataset personnalisé
3. Augmentation des données
4. Entraînement d’un MLP (PyTorch)
5. Détection en temps réel via webcam pour test

## Structure du projet

landmarks/
├── prise_emotion_dataset_custom.py  
├── custom_emotion_dataset.csv  
├── custom_dataset_augmented.csv  
├── augmentation_custom_dataset.py  
├── train_mlp_landmarks.py  
├── visualisation_mlp_custom.py  
├── mlp_best.pt  
├── scaler.joblib  
├── label_encoder.joblib  
├── requirements.txt  

## Installation

Créer un environnement Python puis installer les dépendances :

pip install -r requirements.txt

ou manuellement :

pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib joblib opencv-python mediapipe

## Génération du dataset

### Capture des données

Le script prise_emotion_dataset_custom.py permet de capturer des landmarks via webcam.

Chaque touche correspond à une émotion :

- h = happy  
- s = sad  
- a = angry  
- n = neutral  
- f = fear  
- d = disgust  
- u = surprise  

Les landmarks (x, y) sont enregistrés dans :

custom_emotion_dataset.csv

## Augmentation des données

Le script augmentation_custom_dataset.py permet d’augmenter le dataset en appliquant :

- jitter (bruit)
- scale (zoom)
- shift (translation)
- rotation

Le dataset augmenté est sauvegardé dans :

custom_dataset_augmented.csv

## Entraînement du modèle

Le script train_mlp_landmarks.py entraîne un MLP sur les landmarks.

### Architecture du modèle

- Entrée : landmarks du visage (x, y)
- Couche 1 : 256 neurones + BatchNorm + Dropout
- Couche 2 : 128 neurones + BatchNorm + Dropout
- Couche 3 : 64 neurones
- Sortie : nombre d’émotions

### Fichiers générés

- mlp_best.pt (modèle entraîné)
- scaler.joblib (normalisation des données)
- label_encoder.joblib (encodage des émotions)

## Détection en temps réel

Le script visualisation_mlp_custom.py permet de :

- capturer la webcam
- extraire les landmarks avec MediaPipe
- normaliser les données
- prédire l’émotion avec le MLP
- afficher le résultat en direct
