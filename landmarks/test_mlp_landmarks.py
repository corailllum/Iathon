import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

#device à changer pour Cuda si pas sous mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# chargement des datas
df = pd.read_csv("dataset_augmented.csv")

X = df.drop(columns=["emotion"]).values
y = df["emotion"].values

print("Input shape:", X.shape)

#label encoding
le = LabelEncoder()
y = le.fit_transform(y)

#split train et test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#tensor 

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

input_dim = X_train.shape[1]
num_classes = len(np.unique(y))

print("Features:", input_dim)
print("Classes:", num_classes)

#modèle
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(input_dim, num_classes).to(device)

# optimisation et loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

#boucle de train
EPOCHS = 80

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()
    out = model(X_train)

    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()

    # eval
    model.eval()
    with torch.no_grad():
        val_out = model(X_test)
        val_loss = criterion(val_out, y_test)

    # save best model
    if val_loss.item() < best_loss:
        best_loss = val_loss.item()

        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes
        }, "mlp_best.pt")

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | train={loss.item():.4f} val={val_loss.item():.4f}")

# evaluation finale
model.load_state_dict(torch.load("mlp_best.pt")["model_state_dict"])
model.eval()

with torch.no_grad():
    preds = model(X_test).argmax(dim=1).cpu().numpy()

print("\nClassification report:")
print(classification_report(y_test.cpu(), preds))

# SAUVEGARDE
joblib.dump(scaler, "scaler.joblib")
joblib.dump(le, "label_encoder.joblib")

print("\nSaved:")
print("- mlp_best.pt")
print("- scaler.joblib")
print("- label_encoder.joblib")
