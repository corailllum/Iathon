# PHASE 3: STREAMLIT DEMO - WEBCAM EN DIRECT
# À exécuter en local: streamlit run app.py

import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# CONFIGURATION STREAMLIT


st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("😊 Emotion Recognition with Webcam")
st.markdown("Real-time emotion detection using ResNet50")

# DÉFINIR LES ÉMOTIONS ET COLORS


emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
emotion_colors = {
    "angry": "🔴",
    "disgust": "🟣",
    "fear": "🟠",
    "happy": "🟢",
    "neutral": "⚪",
    "sad": "🔵",
    "surprise": "⭐"
}

emotion_emoji = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲"
}

# ==========================================
# CHARGER LE MODÈLE (mise en cache)
# ==========================================

@st.cache_resource
def load_model(model_path):
    """Charge le modèle une seule fois (mise en cache)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crée le modèle
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(emotions))
    
    # Charge les poids
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

# ==========================================
# PRÉPARER L'IMAGE POUR LE MODÈLE
# ==========================================

def prepare_image(image, device):
    """Convertit une image PIL en tenseur prêt pour le modèle"""
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Convertis en RGB si nécessaire
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Applique les transformations
    img_tensor = transform(image).unsqueeze(0)  # Ajoute dimension batch
    img_tensor = img_tensor.to(device)
    
    return img_tensor

# ==========================================
# FAIRE UNE PRÉDICTION
# ==========================================

def predict_emotion(image, model, device):
    """Prédit l'émotion à partir d'une image"""
    
    img_tensor = prepare_image(image, device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Récupère la prédiction
    predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_emotion = emotions[predicted_class]
    confidence = probabilities[0, predicted_class].item() * 100
    
    # Récupère toutes les probabilités
    all_probs = {
        emotion: prob.item() * 100 
        for emotion, prob in zip(emotions, probabilities[0])
    }
    
    return predicted_emotion, confidence, all_probs

# ==========================================
# SIDEBAR - CONFIGURATION
# ==========================================

st.sidebar.header("⚙️ Configuration")

model_source = st.sidebar.radio(
    "Où est ton modèle ?",
    ["Local (sur ton PC)", "Kaggle (lien)"],
    help="Choisis où charger le modèle"
)

if model_source == "Local (sur ton PC)":
    st.sidebar.info(" Mets le fichier `best_model.pth` dans le même dossier que ce script")
    model_path = "best_model.pth"
else:
    st.sidebar.info(" Entre le chemin du modèle Kaggle")
    model_path = st.sidebar.text_input("Chemin du modèle", "best_model.pth")

# Vérifie que le modèle existe
import os
if not os.path.exists(model_path):
    st.sidebar.error(f" Modèle non trouvé: {model_path}")
    st.stop()

st.sidebar.success(f" Modèle chargé: {model_path}")

# Charge le modèle
try:
    model, device = load_model(model_path)
    st.sidebar.success(f" Modèle sur {device}")
except Exception as e:
    st.sidebar.error(f" Erreur lors du chargement: {e}")
    st.stop()

# ==========================================
# ONGLETS PRINCIPAUX
# ==========================================

tab1, tab2, tab3 = st.tabs(["📹 Webcam Live", "📸 Upload Image", "ℹ️ Infos"])

# ==========================================
# TAB 1: WEBCAM EN DIRECT
# ==========================================

with tab1:
    st.subheader("Webcam en temps réel")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Option 1: Avec streamlit_webrtc (si installé)
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
            
            st.markdown("**Option 1: Flux en direct (meilleur)**")
            
            webrtc_ctx = webrtc_streamer(
                key="emotion-recognition",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            if webrtc_ctx.state.playing:
                st.info("✓ Webcam activée")
            
        except ImportError:
            st.warning(" streamlit-webrtc non installé. Utilise l'option upload ci-dessous.")
    
    with col2:
        st.markdown("**Statistiques**")
        st.metric("Modèle", "ResNet50")
        st.metric("Émotions", len(emotions))
        st.metric("Device", str(device))
    
    st.markdown("---")
    st.markdown("""
    **Instructions:**
    1. Clique sur "Start" pour activer ta webcam
    2. Pointe ta face à la caméra
    3. Le modèle prédit l'émotion en temps réel
    4. Clique "Stop" pour arrêter
    """)

# ==========================================
# TAB 2: UPLOAD D'IMAGE
# ==========================================

with tab2:
    st.subheader("Télécharge une image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Option 1: Upload depuis ton PC**")
        uploaded_file = st.file_uploader(
            "Choisis une image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Charge l'image
            image = Image.open(uploaded_file)
            
            # Affiche l'image originale
            st.image(image, caption="Image uploadée", use_column_width=True)
            
            # Prédiction
            predicted_emotion, confidence, all_probs = predict_emotion(image, model, device)
            
            # Affiche la prédiction principale
            st.success(f"### {emotion_emoji[predicted_emotion]} {predicted_emotion.upper()}")
            st.metric("Confiance", f"{confidence:.1f}%")
            
            # Affiche toutes les probabilités
            st.markdown("**Détail des prédictions:**")
            for emotion in emotions:
                prob = all_probs[emotion]
                # Barre de progression
                st.write(f"{emotion_emoji[emotion]} {emotion:12s} {prob:5.1f}%")
                st.progress(prob / 100)
    
    with col2:
        st.markdown("**Option 2: Prendre une photo (webcam)**")
        
        camera_image = st.camera_input(
            "Prends une photo",
            label_visibility="collapsed"
        )
        
        if camera_image is not None:
            image = Image.open(camera_image)
            
            # Affiche l'image capturée
            st.image(image, caption="Photo capturée", use_column_width=True)
            
            # Prédiction
            predicted_emotion, confidence, all_probs = predict_emotion(image, model, device)
            
            # Affiche la prédiction
            st.success(f"### {emotion_emoji[predicted_emotion]} {predicted_emotion.upper()}")
            st.metric("Confiance", f"{confidence:.1f}%")
            
            # Détail
            st.markdown("**Détail des prédictions:**")
            for emotion in emotions:
                prob = all_probs[emotion]
                st.write(f"{emotion_emoji[emotion]} {emotion:12s} {prob:5.1f}%")
                st.progress(prob / 100)

# ==========================================
# TAB 3: INFORMATIONS
# ==========================================

with tab3:
    st.subheader("À propos du modèle")
    
    st.markdown("""
    ### Architecture
    - **Modèle de base**: ResNet50
    - **Pré-entraînement**: ImageNet (14M images)
    - **Fine-tuning**: 7 émotions FER Kaggle
    - **Accuracy**: 68.71%
    
    ### Émotions reconnues
    """)
    
    cols = st.columns(4)
    for i, emotion in enumerate(emotions):
        with cols[i % 4]:
            st.write(f"{emotion_emoji[emotion]} {emotion.capitalize()}")
    
    st.markdown("""
    ### Transfer Learning
    - Couches figées: Conv1 → Layer3
    - Couches entraînées: Layer4 + FC
    - Learning rate: 0.0001
    - Epochs: 15
    
    ### Données d'entraînement
    - Train: 34,844 images
    - Val: 7,467 images
    - Test: 7,468 images
    - **Total: 49,779 images**
    
    ### Limitations
    - Accuracy ~69% (difficile à améliorer sans données plus variées)
    - Mieux avec des visages clairs et bien éclairés
    - Sensible à l'angle et l'expression partielle
    - Peut confondre: angry/disgust, fear/sad, happy/neutral
    
    ### Comment ça marche
    1. L'image est redimensionnée à 224×224
    2. Normalisée avec la moyenne/std ImageNet
    3. Passe par ResNet50 (50 couches)
    4. Sortie: 7 scores (logits)
    5. Softmax: convertis en probabilités
    6. Argmax: prend la classe avec le score max
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Installation (en local)
    
    ```bash
    # 1. Clone ou crée le dossier du projet
    mkdir emotion-recognition
    cd emotion-recognition
    
    # 2. Crée un environnement virtuel
    python -m venv venv
    source venv/bin/activate  # Sur Windows: venv\\Scripts\\activate
    
    # 3. Installe les dépendances
    pip install -r requirements.txt
    
    # 4. Exécute l'app
    streamlit run app.py
    ```
    
    ### Dépendances (requirements.txt)
    ```
    streamlit==1.28.1
    torch==2.0.0
    torchvision==0.15.1
    Pillow==10.0.0
    numpy==1.24.0
    opencv-python==4.8.0
    streamlit-webrtc==0.47.0
    ```
    """)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with using PyTorch & Streamlit</p>
    <p>Phase 3: Emotion Recognition Hackathon</p>
</div>
""", unsafe_allow_html=True)