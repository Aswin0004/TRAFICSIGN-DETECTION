import pandas as pd # type: ignore
import os
import cv2 # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import plot_model  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import matplotlib.pyplot as plt


# === CONFIGURATION ===
IMG_SIZE = 64
TRAIN_CSV = 'Train.csv'
TEST_CSV = 'Test.csv'
META_CSV = 'Meta.csv'
TRAIN_FOLDER = 'Train'
TEST_FOLDER = 'Test'

# === STEP 1: LOAD DATA ===
print("[INFO] Loading training data...")
train_df = pd.read_csv(TRAIN_CSV)

X, y = [], []
for idx, row in train_df.iterrows():
    img_path = row['Path']  # Path already has Train/XX/...
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] Could not read image: {img_path}")
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X.append(img)
    y.append(row['ClassId'])

X = np.array(X) / 255.0
y = to_categorical(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 2: BUILD CNN MODEL ===
print("[INFO] Building model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === PLOT MODEL ===
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
print("[INFO] Model plot saved as 'model_plot.png'")

# === STEP 3: TRAIN MODEL ===
print("[INFO] Training model...")
history=model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# === STEP 4: SAVE MODEL ===
model.save('traffic_sign_model.h5')
print("[INFO] Model saved as traffic_sign_model.h5")

# === STEP 5: EVALUATE ON TEST SET ===
print("[INFO] Evaluating on test set...")
test_df = pd.read_csv(TEST_CSV)

test_images = []
test_paths = []

for idx, row in test_df.iterrows():
    img_path = row['Path']  # Path already has Test/XX/...
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] Could not read test image: {img_path}")
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    test_images.append(img)
    test_paths.append(img_path)


X_test = np.array(test_images) / 255.0
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# === STEP 6: MAP CLASS IDS TO NAMES (Meta.csv) ===
meta_df = pd.read_csv(META_CSV)
id_to_sign = dict(zip(meta_df['ClassId'], meta_df['SignName']))

print("\n[INFO] Sample Predictions:")
for path, pred in zip(test_paths[:10], predicted_classes[:10]):
    print(f"{path} -> {id_to_sign.get(pred, 'Unknown')}")
# ========== PLOT ==========
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
