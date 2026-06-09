import shutil
from pathlib import Path

SRC_MODEL = Path("artifacts/model_trainer/model.keras")
SRC_LABELS = Path("artifacts/model_trainer/labels.json")
SRC_TRAIN = Path("artifacts/data_ingestion/preprocessed/train")
REQUIRED = Path("required")
SAMPLE = REQUIRED / "sample"

REQUIRED.mkdir(exist_ok=True)
SAMPLE.mkdir(exist_ok=True)

shutil.copy2(SRC_MODEL, REQUIRED / "model.keras")
print(f"Copied model.keras -> {REQUIRED / 'model.keras'}")

shutil.copy2(SRC_LABELS, REQUIRED / "labels.json")
print(f"Copied labels.json -> {REQUIRED / 'labels.json'}")

class_folders = sorted(SRC_TRAIN.iterdir(), key=lambda p: p.name)
for class_folder in class_folders:
    if not class_folder.is_dir():
        continue
    images = list(class_folder.glob("*.png"))
    if not images:
        print(f"WARNING: no images found in {class_folder}")
        continue
    dest_dir = SAMPLE / class_folder.name
    dest_dir.mkdir(exist_ok=True)
    shutil.copy2(images[0], dest_dir / images[0].name)

print(f"\nDone. {len(list(SAMPLE.iterdir()))} class folders created in {SAMPLE}")
