import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# U-Net Model Definition
def build_unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    # Bottleneck
    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)

    # Decoder
    u5 = UpSampling2D()(c4)
    u5 = Concatenate()([u5, c3])
    c5 = Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)

    u6 = UpSampling2D()(c5)
    u6 = Concatenate()([u6, c2])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = UpSampling2D()(c6)
    u7 = Concatenate()([u7, c1])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    outputs = Conv2D(1, 1, activation='sigmoid')(c7)

    model = Model(inputs, outputs)
    return model

# Data Loading
IMG_SIZE = (256, 256)

img_dir = "data/Training/Fire"
mask_dir = "data/Resized_Masks"

X = []
Y = []

print("Loading data...")
for fname in tqdm(os.listdir(img_dir)):
    if not fname.endswith(".jpg"):
        continue

    img_path = os.path.join(img_dir, fname)
    mask_path = os.path.join(mask_dir, fname.replace(".jpg", ".png"))

    if not os.path.exists(mask_path):
        print(f"Bỏ qua {fname} vì không tìm thấy mask.")
        continue

    try:
        img = Image.open(img_path).resize(IMG_SIZE)
        mask = Image.open(mask_path).resize(IMG_SIZE)

        img = np.array(img) / 255.0
        mask = np.array(mask.convert("L")) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        X.append(img)
        Y.append(mask)
    except Exception as e:
        print(f" Lỗi khi load {fname}: {e}")

X = np.array(X)
Y = np.array(Y)

print(f"Loaded {len(X)} samples")

# Train/Test Split 
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

# Build & Train U-Net 
model = build_unet(input_size=(256, 256, 3))
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

print("Training U-Net model...")
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=8)

# Save Model 
os.makedirs("models", exist_ok=True)
model.save("models/segmenter.h5")
print("Segmentation model saved to models/segmenter.h5")
