# IAthon
### Équipe 1 :
- **Charlotte Chanudet**
- **Mahaut Galice**
- **Amal Ouedraogo**
- **Colin Pouliart**

## Explication du projet
Ce projet fait partie de l'IAthon organisé par l'Université du Québec à Chicoutimi et l'École Louvain en Hainaut. Nous avons choisi de réaliser un projet de reconnaissance des émotions chez les enfants. Ce README regroupe les informations sur les données utilisées et les modèles d'IA mis en place, ainsi qu'un mode d'emploi pour les différents fichiers.

## Technologies utilisées
- Python 3.10+
- PyTorch (torch, torchvision)
- Streamlit (interface web)
- OpenCV, Pillow (traitement d'image)
- streamlit-webrtc (webcam temps réel)
- Tkinter

## Structure du projet
- `CNN/` : Code et sauvegarde du modèle CNN
- `transfer_learning/` : Code et sauvegarde du modèle ResNet50
- `landmarks/` : Code, sauvegarde et données du modèle landmarks
- `interface/` : Code principal et front de l'application

Chacun des dossiers possède un README qui détaille le fonctionnement des modèles ou du code présent dans le dossier.

## Données utilisées

### Structure du dataset

Lien du dataset :

```
data/
│── angry/
│── disgust/
│── fear/
│── happy/
│── neutral/
│── sad/
│── surprise/
```

Cela ne concerne pas le dossier `landmarks/` qui utilise son propre dataset fait main.


