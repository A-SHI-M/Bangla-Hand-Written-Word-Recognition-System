# Bangla Handwritten Word Recognition System

A deep learning pipeline for recognizing handwritten Bangla characters using a custom CNN trained on the BanglaLekha-Isolated dataset, with a Streamlit web app for real-time prediction.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Dataset Download](#dataset-download)
3. [Folder Structure Setup](#folder-structure-setup)
4. [Installing Dependencies](#installing-dependencies)
5. [Dataset Preparation](#dataset-preparation)
6. [Preprocessing](#preprocessing)
7. [Model Architecture](#model-architecture)
8. [Training Process](#training-process)
9. [MLflow Tracking](#mlflow-tracking)
10. [Word Segmentation Strategy](#word-segmentation-strategy)
11. [Streamlit UI](#streamlit-ui)
12. [Docker](#docker)
13. [Limitations](#limitations)
14. [Possible Improvements](#possible-improvements)

---

## Environment Setup

A batch script named `EnvironmentSetup.bat` creates a Python virtual environment named `BanglaLekha` using **Python 3.11.9**.

```cmd
EnvironmentSetup.bat
```

---

## Dataset Download

This project uses the **BanglaLekha-Isolated** dataset from Mendeley. A batch script named `DatasetDownload.bat` downloads and extracts it automatically.

```cmd
DatasetDownload.bat
```

---

## Folder Structure Setup

A script named `template.py` creates the required project directory structure and supporting files.

```cmd
python template.py
```

---

## Installing Dependencies

With the `BanglaLekha` environment active:

```cmd
pip install -r requirements.txt
```

---

## Dataset Preparation

The BanglaLekha-Isolated dataset contains **~166,000 PNG images** across **84 classes** of handwritten Bangla characters (vowels, consonants, digits, and compound characters), with approximately 1,975 images per class organized in numbered folders (`1` to `84`).

The data ingestion stage:
- Scans all 84 class folders and builds an image index
- Applies a **stratified 80/20 train/test split** so every class is proportionally represented in both sets (~132,884 train / ~33,221 test)
- Preprocesses and saves images into a class-labelled folder structure ready for training

```cmd
python src/banglaOCR/pipeline/stage_01_data_ingestion.py
```

---

## Preprocessing

Each image is preprocessed during data ingestion:

1. **Grayscale conversion** — strips colour to a single channel
2. **Resize to 32×32** — standardizes spatial dimensions using LANCZOS resampling
3. **Saved as PNG** — stored in `artifacts/data_ingestion/preprocessed/train/<class>/` and `test/<class>/`

Pixel **normalization** (`÷ 255`) is not applied at this stage — it is handled on-the-fly by the Keras data loader during training via a `Rescaling(1/255)` layer.

---

## Model Architecture

A custom CNN named `bangla_ocr_cnn` is built for 32×32 grayscale input with 84 output classes.

| Block | Layers | Output Shape |
|---|---|---|
| Input | — | 32×32×1 |
| Block 1 | Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25) | 16×16×32 |
| Block 2 | Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25) | 8×8×64 |
| Block 3 | Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.25) | 4×4×128 |
| Head | Flatten → Dense(256) → BN → ReLU → Dropout(0.5) → Dense(84, softmax) | 84 |

- **BatchNormalization** stabilizes and accelerates training
- **Dropout** reduces overfitting
- **Softmax** output gives a probability distribution over 84 classes

---

## Training Process

```cmd
python src/banglaOCR/pipeline/stage_02_model_trainer.py
```

- **Optimizer** — Adam (`lr=0.001`)
- **Loss** — Sparse Categorical Crossentropy
- **EarlyStopping** — stops training if `val_loss` does not improve for 5 epochs; restores best weights automatically
- **ReduceLROnPlateau** — halves the learning rate if `val_loss` stalls for 3 consecutive epochs (min `1e-6`)
- **ModelCheckpoint** — saves only the epoch with the highest `val_accuracy` to `artifacts/model_trainer/model.keras`
- **Resume support** — if `model.keras` and `training_state.json` already exist, training resumes from the last completed epoch instead of starting from scratch
- **Label mapping** — `artifacts/model_trainer/labels.json` maps each model output index to its class folder name; saved before training begins

To run the full pipeline (data ingestion + training) in sequence:

```cmd
python train.py
```

---

## MLflow Tracking

Every training run is logged to the `bangla-ocr-cnn` MLflow experiment.

| What | Details |
|---|---|
| Parameters | `image_size`, `batch_size`, `epochs`, `learning_rate`, `num_classes`, `initial_epoch` |
| Per-epoch metrics | `loss`, `accuracy`, `val_loss`, `val_accuracy` |
| Summary metrics | `best_val_accuracy`, `best_val_loss` |
| Artifacts | `model.keras`, `labels.json` |

```cmd
mlflow ui
```

Open `http://127.0.0.1:5000` to compare runs and view metric curves.

---

## Word Segmentation Strategy

When the user draws a word on the canvas, the app segments it into individual characters:

1. **Grayscale conversion** — RGBA canvas image is converted to grayscale
2. **Thresholding** — pixels darker than 200 are marked as ink, producing a binary image
3. **Dilation** — a 5×5 kernel merges disconnected strokes of the same character (e.g. a consonant and its matra)
4. **Contour detection** — external contours are found on the dilated image
5. **Noise filtering** — bounding boxes smaller than 100px² are discarded
6. **Left-to-right sorting** — bounding boxes are sorted by x-coordinate to preserve character order
7. **Cropping** — each character is cropped from the original grayscale image with 4px padding and passed to the model

---

## Streamlit UI

```cmd
streamlit run app.py
```

Open `http://localhost:8501` in a browser.

1. Draw a Bangla word on the white canvas using the mouse
2. Click **Predict**
3. The app segments the word into characters, predicts each one, and displays:
   - Thumbnails of each segmented character with predicted class and confidence
   - The recognized word as a row of sample images from the predicted classes
   - A confidence score bar for each character

---

## Docker

Ensure training is complete and `artifacts/model_trainer/` exists before building.

```cmd
docker build -t bangla-ocr .
```

```cmd
docker run -p 8501:8501 bangla-ocr
```

Open `http://localhost:8501` in a browser.

---

## Limitations

- **Small input resolution (32×32)** — fine details of visually similar Bangla characters may be lost, reducing accuracy on hard classes
- **No data augmentation** — the model has not seen rotated, shifted, or scaled variations during training, so it may struggle with non-standard handwriting styles
- **Simple contour-based segmentation** — touching or overlapping characters (common in cursive or ligature-heavy writing) may not segment correctly
- **No Unicode mapping** — the app displays class folder numbers rather than actual Bangla Unicode characters
- **Isolated character dataset** — the model is trained on isolated characters, so recognition quality on connected or natural handwriting may be lower

---

## Possible Improvements

- **Data augmentation** — add random rotation, translation, and zoom during training to improve generalization across handwriting styles
- **Larger input size** — increase `IMAGE_SIZE` to 64 in `params.yaml` and re-run data ingestion for richer spatial features
- **Unicode label mapping** — add a JSON file mapping class folder numbers to Bangla Unicode codepoints for meaningful character display
- **Advanced segmentation** — use projection profiles or a dedicated segmentation model to handle touching and ligated characters
- **Transfer learning** — fine-tune a pretrained model (e.g. EfficientNet) with RGB input for potentially higher accuracy
- **Confidence threshold** — reject predictions below a minimum confidence and flag them as unrecognized rather than returning a wrong character
