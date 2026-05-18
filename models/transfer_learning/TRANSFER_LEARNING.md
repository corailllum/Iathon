
# Emotion Recognition

Modèle ResNet50 fine-tuné pour reconnaître 7 émotions.

**Accuracy**: 68.71%

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```python
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# Charge le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = nn.Linear(2048, 7)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Prédit
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

image = Image.open("image.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = transform(image).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(img)
    emotion = emotions[output.argmax(1).item()]

print(f"Emotion: {emotion}")
```


