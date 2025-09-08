import os
import shutil
import pandas as pd

# === CONFIG PATHS ===
RAW_IMAGE_ROOT = r"D:\DERM12345_dataset_raw"
TRAIN_CSV = os.path.join(RAW_IMAGE_ROOT, "derm12345_metadata_train.csv")
TEST_CSV = os.path.join(RAW_IMAGE_ROOT, "derm12345_metadata_test.csv")
OUTPUT_ROOT = r"D:\non funcion2\dataset_4class"

    # === MAPPING: subclass → 4-class ===
    map_dict = {
        # ✅ Melanocytic_Benign
        'acb': 'Melanocytic_Benign',
        'acd': 'Melanocytic_Benign',
        'ajb': 'Melanocytic_Benign',
        'ajd': 'Melanocytic_Benign',
        'ak': 'Melanocytic_Benign',
        'alm': 'Melanocytic_Benign',
        'angk': 'Melanocytic_Benign',
        'anm': 'Melanocytic_Benign',
        'bd': 'Melanocytic_Benign',
        'bdb': 'Melanocytic_Benign',
        'jb': 'Melanocytic_Benign',
        'jd': 'Melanocytic_Benign',
        'lk': 'Melanocytic_Benign',

        # ✅ Melanocytic_Malignant
        'bcc': 'Melanocytic_Malignant',
        'mel': 'Melanocytic_Malignant',
        'lmm': 'Melanocytic_Malignant',
        'nmm': 'Melanocytic_Malignant',  # หากมี
        'mcb': 'Melanocytic_Malignant',  # หากเป็น Melanoma cutaneous based
        'lmm': 'Melanocytic_Malignant',

        # ✅ Nonmelanocytic_Benign
        'cb': 'Nonmelanocytic_Benign',
        'ccb': 'Nonmelanocytic_Benign',
        'ccd': 'Nonmelanocytic_Benign',
        'cd': 'Nonmelanocytic_Benign',
        'ch': 'Nonmelanocytic_Benign',
        'cjb': 'Nonmelanocytic_Benign',
        'db': 'Nonmelanocytic_Benign',
        'ha': 'Nonmelanocytic_Benign',
        'isl': 'Nonmelanocytic_Benign',
        'ks': 'Nonmelanocytic_Benign',
        'la': 'Nonmelanocytic_Benign',
        'lm': 'Nonmelanocytic_Benign',
        'ls': 'Nonmelanocytic_Benign',
        'mpd': 'Nonmelanocytic_Benign',
        'pg': 'Nonmelanocytic_Benign',
        'rd': 'Nonmelanocytic_Benign',
        'sa': 'Nonmelanocytic_Benign',
        'sk': 'Nonmelanocytic_Benign',
        'sl': 'Nonmelanocytic_Benign',
        'srjd': 'Nonmelanocytic_Benign',

        # ✅ Nonmelanocytic_Malignant
        'dfsp': 'Nonmelanocytic_Malignant',
        'scc': 'Nonmelanocytic_Malignant',
        'df': 'Nonmelanocytic_Malignant',  # หากยืนยันว่า malignant variant
    }


# === สร้างโฟลเดอร์ปลายทาง ===
def make_output_dirs():
    for split in ["train", "test"]:
        for label in set(map_dict.values()):
            os.makedirs(os.path.join(OUTPUT_ROOT, split, label), exist_ok=True)

# === ค้นหารูปภาพใน RAW_IMAGE_ROOT ทั้งหมด ===
def index_all_images(root):
    indexed = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in [".jpg", ".jpeg", ".png"]:
                indexed[name] = os.path.join(dirpath, f)
    return indexed

# === คัดลอกภาพตามคลาส ===
def map_images(df, split, all_images):
    df["label_4class"] = df["label"].map(map_dict)
    df = df.dropna(subset=["label_4class"])
    not_found = []

    for _, row in df.iterrows():
        image_id = row["image_id"]
        label = row["label_4class"]

        if image_id in all_images:
            src = all_images[image_id]
            dst = os.path.join(OUTPUT_ROOT, split, label, os.path.basename(src))
            shutil.copy(src, dst)
        else:
            not_found.append(image_id)

    print(f"✅ {split.upper()}: คัดลอก {len(df) - len(not_found)} ภาพ | ❌ ไม่พบ {len(not_found)} ภาพ")
    if not_found:
        print("   เช่น:", not_found[:5])

# === เริ่มทำงาน ===
if __name__ == "__main__":
    make_output_dirs()
    all_images = index_all_images(RAW_IMAGE_ROOT)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    map_images(train_df, "train", all_images)
    map_images(test_df, "test", all_images)
