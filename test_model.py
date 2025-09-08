import os
from model import build_and_load_model, preprocess_image, predict_image

# 🔧 ตั้งชื่อโฟลเดอร์ที่เก็บภาพทดสอบ
test_folder = 'test_images'

# โหลดโมเดล
model = build_and_load_model()

# อ่านไฟล์ทั้งหมดในโฟลเดอร์
image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# เช็คว่ามีรูปไหม
if not image_files:
    print("❌ ไม่พบภาพในโฟลเดอร์ test_images/")
    exit()

# ทำนายภาพทีละรูป
for image_name in image_files:
    image_path = os.path.join(test_folder, image_name)
    image_array = preprocess_image(image_path)

    predicted_class, confidence, all_probs = predict_image(model, image_array)

    print(f"\n🖼️ Image: {image_name}")
    print(f"✅ Predicted Class: {predicted_class}")
    print(f"✅ Confidence: {confidence:.2%}")
    print(f"✅ All Probabilities: {[f'{p:.2%}' for p in all_probs]}")
