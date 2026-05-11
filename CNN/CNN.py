"""
Emotion Recognition CNN - Training Script
Dataset: Facial Emotion Recognition Dataset (Kaggle)
Emotions: angry, disgust, fear, happy, neutral, sad, surprise
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import json
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    # Utilisation du dossier d'images dans archive(1)/processed_data
    "data_dir": "./archive(1)/processed_data",
    "img_size": 48,
    "batch_size": 64,
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.5,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "./models",
    "results_dir": "./results",
    "early_stopping_patience": 10,
    "test_split": 0.2,  # fraction du test split
    "random_seed": 42,
}

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ─────────────────────────────────────────────
# DATA TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return train_transform, val_transform



from torch.utils.data import Subset, random_split

def get_dataloaders():
    train_transform, val_transform = get_transforms()

    # On charge tout le dataset (toutes les images dans chaque dossier d'émotion)
    full_dataset = datasets.ImageFolder(CONFIG["data_dir"], transform=train_transform)

    # Split train/test (stratifié par classe)
    from collections import defaultdict
    import random
    random.seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    # Regrouper les indices par classe
    class_indices = defaultdict(list)
    for idx, target in enumerate(full_dataset.targets):
        class_indices[target].append(idx)

    train_indices, test_indices = [], []
    for indices in class_indices.values():
        n_total = len(indices)
        n_test = int(n_total * CONFIG["test_split"])
        shuffled = indices.copy()
        random.shuffle(shuffled)
        test_indices.extend(shuffled[:n_test])
        train_indices.extend(shuffled[n_test:])

    # Créer les sous-ensembles
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(datasets.ImageFolder(CONFIG["data_dir"], transform=val_transform), test_indices)

    # Weighted sampler pour gérer le déséquilibre de classes
    train_targets = [full_dataset.targets[i] for i in train_indices]
    class_counts = np.array([
        train_targets.count(i) for i in range(len(full_dataset.classes))
    ])
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[t] for t in train_targets])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    print(f"✅ Dataset chargé:")
    print(f"   Train: {len(train_indices)} images")
    print(f"   Test:  {len(test_indices)} images")
    print(f"   Classes: {full_dataset.classes}")
    print(f"   Distribution train: {dict(zip(full_dataset.classes, class_counts))}")

    return train_loader, test_loader, full_dataset.classes


# ─────────────────────────────────────────────
# CNN ARCHITECTURE
# ─────────────────────────────────────────────
class EmotionCNN(nn.Module):
    """
    CNN profond pour la reconnaissance d'émotions faciales.
    Architecture: 4 blocs conv avec BatchNorm + Residual connections
    """
    def __init__(self, num_classes=7, dropout=0.5):
        super(EmotionCNN, self).__init__()

        # Bloc 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )

        # Bloc 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )

        # Bloc 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        # Bloc 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        # Global Average Pooling + Classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def train(model, train_loader, test_loader, classes):
    device = CONFIG["device"]
    model = model.to(device)

    # Loss avec class weights pour déséquilibre
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6
    )
    early_stopping = EarlyStopping(patience=CONFIG["early_stopping_patience"])

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    print(f"\n🚀 Entraînement sur {device.upper()}")
    print(f"   Modèle: {sum(p.numel() for p in model.parameters()):,} paramètres")
    print("=" * 70)

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:3d}/{CONFIG['num_epochs']}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": classes,
                "config": CONFIG,
            }, os.path.join(CONFIG["save_dir"], "best_model.pth"))
            print(f"   💾 Meilleur modèle sauvegardé (acc={best_acc:.4f})")

        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"\n⏹️  Early stopping à l'epoch {epoch}")
            break

    print(f"\n✅ Entraînement terminé. Meilleure accuracy: {best_acc:.4f}")
    return history, best_acc


# ─────────────────────────────────────────────
# MÉTRIQUES & VISUALISATIONS
# ─────────────────────────────────────────────
def analyze_metrics(model, test_loader, classes, device, history):
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    # Charger le meilleur modèle
    checkpoint = torch.load(
        os.path.join(CONFIG["save_dir"], "best_model.pth"),
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    _, _, all_preds, all_labels = evaluate(model, test_loader, criterion, device)

    # ── 1. Rapport de classification ─────────────
    report = classification_report(
        all_labels, all_preds,
        target_names=classes,
        output_dict=True
    )
    print("\n" + "=" * 70)
    print("📊 RAPPORT DE CLASSIFICATION")
    print("=" * 70)
    print(classification_report(all_labels, all_preds, target_names=classes))

    # ── 2. Émotion la mieux reconnue ─────────────
    per_class_acc = {}
    cm = confusion_matrix(all_labels, all_preds)
    for i, cls in enumerate(classes):
        correct = cm[i, i]
        total = cm[i].sum()
        per_class_acc[cls] = correct / total if total > 0 else 0

    best_emotion = max(per_class_acc, key=per_class_acc.get)
    worst_emotion = min(per_class_acc, key=per_class_acc.get)

    print("\n🏆 ANALYSE PAR ÉMOTION:")
    for emotion, acc in sorted(per_class_acc.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {emotion:10s} [{bar}] {acc:.1%}")

    print(f"\n  ✅ Meilleure reconnaissance: {best_emotion} ({per_class_acc[best_emotion]:.1%})")
    print(f"  ⚠️  Moins bien reconnue:     {worst_emotion} ({per_class_acc[worst_emotion]:.1%})")

    # ── 3. Figures ────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Analyse CNN - Reconnaissance d'Émotions", fontsize=16, fontweight="bold")

    # (a) Courbe d'apprentissage - Loss
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train Loss", color="#2196F3")
    ax.plot(history["val_loss"], label="Val Loss", color="#F44336")
    ax.set_title("Courbe de Perte")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Courbe d'apprentissage - Accuracy
    ax = axes[0, 1]
    ax.plot(history["train_acc"], label="Train Acc", color="#4CAF50")
    ax.plot(history["val_acc"], label="Val Acc", color="#FF9800")
    ax.set_title("Courbe d'Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Accuracy par émotion (barres)
    ax = axes[0, 2]
    emotions = list(per_class_acc.keys())
    accs = [per_class_acc[e] for e in emotions]
    colors = ["#4CAF50" if e == best_emotion else "#F44336" if e == worst_emotion else "#2196F3" for e in emotions]
    bars = ax.bar(emotions, accs, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_title("Accuracy par Émotion")
    ax.set_xlabel("Émotion")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=30)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # (d) Matrice de confusion
    ax = axes[1, 0]
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Matrice de Confusion (normalisée)")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    # (e) F1-Score par émotion
    ax = axes[1, 1]
    f1_scores = {cls: report[cls]["f1-score"] for cls in classes}
    ax.barh(list(f1_scores.keys()), list(f1_scores.values()),
            color="#9C27B0", edgecolor="white")
    ax.set_title("F1-Score par Émotion")
    ax.set_xlabel("F1-Score")
    ax.set_xlim(0, 1.1)
    for i, (cls, f1) in enumerate(f1_scores.items()):
        ax.text(f1 + 0.01, i, f"{f1:.2f}", va="center", fontsize=9)

    # (f) Distribution des prédictions
    ax = axes[1, 2]
    pred_counts = [all_preds.count(i) for i in range(len(classes))]
    real_counts = [all_labels.count(i) for i in range(len(classes))]
    x = np.arange(len(classes))
    width = 0.35
    ax.bar(x - width/2, real_counts, width, label="Réel", color="#00BCD4", alpha=0.8)
    ax.bar(x + width/2, pred_counts, width, label="Prédit", color="#FF5722", alpha=0.8)
    ax.set_title("Distribution Réel vs Prédit")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(CONFIG["results_dir"], "metrics_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n📈 Graphiques sauvegardés: {plot_path}")

    # Sauvegarde JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "best_val_accuracy": max(history["val_acc"]),
        "per_class_accuracy": per_class_acc,
        "best_emotion": best_emotion,
        "worst_emotion": worst_emotion,
        "classification_report": report,
        "config": CONFIG,
    }
    with open(os.path.join(CONFIG["results_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return per_class_acc, best_emotion


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🎭 Emotion Recognition CNN")
    print("=" * 70)

    train_loader, test_loader, classes = get_dataloaders()

    model = EmotionCNN(num_classes=len(classes), dropout=CONFIG["dropout"])

    history, best_acc = train(model, train_loader, test_loader, classes)

    analyze_metrics(model, test_loader, classes, CONFIG["device"], history)

    print("\n✅ Pipeline terminé!")
    print(f"   Modèle: {CONFIG['save_dir']}/best_model.pth")
    print(f"   Résultats: {CONFIG['results_dir']}/")
