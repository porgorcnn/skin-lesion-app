import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = 4
MODEL_WEIGHTS_PATH = 'wait/efficientnet_weights_4class_quicktest.h5'

print("🔧 Rebuilding EfficientNetB0 structure...")
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

print(f"📥 Loading weights from: {MODEL_WEIGHTS_PATH}")
try:
    model.load_weights(MODEL_WEIGHTS_PATH)
    print("✅ Weights loaded successfully! ✅")
except Exception as e:
    print(f"❌ Failed to load weights: {e}")
