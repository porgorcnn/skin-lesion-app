import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

IMG_HEIGHT, IMG_WIDTH = 224, 224
# --- แก้ไขส่วนนี้ ---
NUM_CLASSES = 4  # <--- แก้ไขจาก 2 เป็น 4
CLASS_NAMES = ["Melanocytic_Benign", "Melanocytic_Malignant", "Nonmelanocytic_Benign", "Nonmelanocytic_Malignant"] # <--- แก้ไขชื่อคลาส
MODEL_WEIGHTS_PATH = 'wait/efficientnet_weights_4class_best.h5' # <--- เพิ่ม Path ไปยังไฟล์โมเดล

def build_and_load_model():
    print("📦 Loading EfficientNetB0 model structure...")
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    # บรรทัดนี้จะใช้ NUM_CLASSES ที่เราแก้เป็น 4 โดยอัตโนมัติ
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    try:
        model.load_weights(MODEL_WEIGHTS_PATH)
        print("✅ Weights loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        return None

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(model, image_array):
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[0][predicted_index]
    return predicted_class, confidence, predictions[0]