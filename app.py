# D:\non funcion2\app.py

import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
# 1. เพิ่มการ import CLASS_NAMES จาก model
from model import build_and_load_model, preprocess_image, predict_image, CLASS_NAMES

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

print("📦 Loading AI model...")
model = build_and_load_model()
if model is None:
    print("❌ Failed to load model. Exiting.")
    exit()
print("✅ AI Model loaded successfully.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_array = preprocess_image(filepath)
            predicted_class, confidence, predictions = predict_image(model, img_array)

            # 2. สร้างตัวแปรเพื่อจับคู่ชื่อคลาสกับ %
            all_scores = zip(CLASS_NAMES, predictions)

            # 3. ส่งตัวแปร all_scores ไปให้หน้าเว็บ
            return render_template('index.html',
                                   filename=filename,
                                   predicted_class=predicted_class,
                                   confidence=f"{confidence * 100:.2f}%",
                                   all_scores=all_scores)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)