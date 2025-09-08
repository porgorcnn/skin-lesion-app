import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# 🔧 กำหนด path
train_dir = 'D:/non funcion2/dataset_4class/train'
val_dir = 'D:/non funcion2/dataset_4class/test'
    MODEL_SAVE_PATH = 'wait/efficientnet_weights_4class_V3.h5'

# ⚙️ พารามิเตอร์
IMG_SIZE = (160, 160)
BATCH_SIZE = 2
NUM_CLASSES = 4
EPOCHS = 20
LEARNING_RATE = 1e-4

# ✅ Data Generator
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("CLASS INDICES:", train_gen.class_indices)

# ✅ คำนวณ class weights
labels = train_gen.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print("CLASS WEIGHTS:", class_weights)

# ✅ โหลด EfficientNetB0 + fine-tune
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# ❄️ Freeze บาง layer แล้ว fine-tune บางส่วน
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

# ✅ Compile
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=2, factor=0.2, min_lr=1e-6, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max', save_weights_only=True,
                    verbose=1)
]

# ✅ Train
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks
)

print("✅ Training complete. Best weights saved to:", MODEL_SAVE_PATH)
