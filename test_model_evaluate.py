# test_model_evaluate.py

from model import build_and_load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

# พารามิเตอร์
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
val_dir = 'D:/non funcion2/dataset_4class/test'

# โหลดข้อมูล
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# โหลดและ compile โมเดล
model = build_and_load_model()
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

# ประเมินผล
loss, acc = model.evaluate(val_gen)
print(f"\n📊 Accuracy on validation set: {acc:.2%}")
