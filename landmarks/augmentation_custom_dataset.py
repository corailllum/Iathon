import pandas as pd
import numpy as np

#chargement des données
df = pd.read_csv("custom_emotion_dataset.csv")

X = df.drop(columns=["filename","emotion"]).astype(np.float32).values
y = df["emotion"].values

#reshape
def to_coords(sample):
    return sample.reshape(-1, 2)

def to_flat(coords):
    return coords.reshape(-1)

#augmentation
def jitter(coords, sigma=0.005):
    noise = np.random.normal(0, sigma, coords.shape)
    return coords + noise

def scale(coords, factor_range=(0.95, 1.05)):
    factor = np.random.uniform(*factor_range)
    return coords * factor

def shift(coords, max_shift=0.02):
    shift = np.random.uniform(-max_shift, max_shift, (1, 2))
    return coords + shift

def rotate(coords, angle_range=(-10, 10)):
    angle = np.radians(np.random.uniform(*angle_range))
    rot = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    return coords @ rot.T

#pipeline
def augment(coords):
    if np.random.rand() < 0.5:
        coords = jitter(coords)

    if np.random.rand() < 0.5:
        coords = scale(coords)

    if np.random.rand() < 0.5:
        coords = shift(coords)

    if np.random.rand() < 0.3:
        coords = rotate(coords)

    return coords

#construction du dataset augmenté
augmented_X = []
augmented_y = []

AUGMENT_FACTOR = 3  # x3 dataset

for i in range(len(X)):
    coords = to_coords(X[i])
    label = y[i]

    # original
    augmented_X.append(X[i])
    augmented_y.append(label)

    # augmented
    for _ in range(AUGMENT_FACTOR):
        aug = augment(coords.copy())
        augmented_X.append(to_flat(aug))
        augmented_y.append(label)

# sauvegarde
df_aug = pd.DataFrame(augmented_X)
df_aug["emotion"] = augmented_y

df_aug.to_csv("dataset_augmented.csv", index=False)

print("Saved: dataset_augmented.csv")
print("Original size:", len(df))
print("Augmented size:", len(df_aug))
