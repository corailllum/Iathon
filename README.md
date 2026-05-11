# IAthon
## Equipe 1 : 
- **Charlotte Chanudet**
- **Mahaut Galice**
- **AMAL OUEDRAOGO**
- **Colin POULIART**

# Explication du projet
Ce projet fais partie du IAthon organisée par l'iuniversité du quebec a Chicoutimi et du ecole louvain en hainaut. Nous avonsn choisi de réalisée le projet de reconnaissance des émotions chez les enfants. Ce README regroupe les information sur les données utilisée et modele d'ia mis en place, ainsi qu'un mode d'emploie pour les differents fichier.


# Technologies utilisées
- Python 3.10+
- PyTorch (torch, torchvision)
- Streamlit (interface web)
- OpenCV, Pillow (traitement d'image)
- streamlit-webrtc (webcam temps réel)

# Structure du projet
- `transfer_learning/` : Code principal, modèle, requirements, documentation
- `train.py` : Script d'entraînement
- `webcam.py` : Script de détection via webcam (hors Streamlit)


# Explication des modèles
Le modèle utilisé est un réseau de neurones convolutif (CNN) de type **ResNet50** :

- **ResNet50** est un modèle profond composé de 50 couches, conçu pour l’analyse d’images. Il utilise des blocs résiduels qui facilitent l’entraînement de réseaux très profonds en permettant le passage direct de l’information entre les couches.
- **Pré-entraînement** : Le modèle est d’abord entraîné sur ImageNet (14 millions d’images, 1000 classes) pour apprendre des caractéristiques générales des images.
- **Fine-tuning** : On adapte ensuite la dernière couche du modèle pour prédire 7 émotions. Seules les dernières couches (Layer4 + fully connected) sont réentraînées sur notre jeu de données d’émotions, les autres couches restent figées.
- **Entrée** : Une image couleur (RGB) de taille 224x224 pixels, normalisée selon les statistiques d’ImageNet.
- **Sortie** : Un vecteur de 7 scores (logits), un par émotion. On applique une fonction softmax pour obtenir les probabilités, puis on retient l’émotion avec la probabilité la plus élevée.
- **Utilisation** : Le modèle est utilisé via Streamlit pour la détection en temps réel (webcam) ou sur image uploadée.

**Résumé du pipeline :**
1. L’image est redimensionnée et normalisée.
2. Elle passe dans le ResNet50 (toutes les couches convolutives extraites, puis la couche finale adaptée).
3. Le modèle prédit la probabilité de chaque émotion.
4. L’émotion dominante est affichée à l’utilisateur.

# Mode d'emploi

