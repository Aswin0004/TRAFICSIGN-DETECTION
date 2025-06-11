import cv2 # type: ignore
import numpy as np
import pandas as pd # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# === CONFIGURATION ===
IMG_SIZE = 64
MODEL_PATH = 'traffic_sign_model.h5'
META_CSV = 'Meta.csv'

# === LOAD MODEL ===
model = load_model(MODEL_PATH)

# === LOAD CLASS ID TO SIGN NAME MAPPING ===
meta_df = pd.read_csv(META_CSV)
id_to_sign = dict(zip(meta_df['ClassId'], meta_df['SignName']))

# === PREDICT FUNCTION ===
def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)  # (1, 64, 64, 3)

    predictions = model.predict(img_input)
    predicted_class = np.argmax(predictions)
    sign_name = id_to_sign.get(predicted_class, 'Unknown')

    print(f"[RESULT] {image_path} -> Class ID: {predicted_class}, Sign: {sign_name}")

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    # Replace this with your actual test image path
    test_image_path = '/Users/aswin/Documents/trafic sign /test_images/4.png'
    predict_image(test_image_path)
