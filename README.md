# Bangla-Hand-Written-Word-Recognition-System

## Python Environment Setup

To ensure dependency consistency and avoid package conflicts, a separate Python virtual environment is created. For this purpose, a batch script file named `EnvironmentSetup.bat` has been developed. This script automatically creates a Python virtual environment named `BanglaLekha` using **Python 3.11.9** and activates the environment for further setup.

### Run the script

Run the following command in command prompt to setup the Python environment:

```cmd
EnvironmentSetup.bat
```

## Dataset Download

This project uses the **BanglaLekha-Isolated** dataset from Mendeley. To simplify the dataset setup process, a batch script file named `DatasetDownload.bat` has been developed. This script automatically downloads and extracts the dataset required for the project.

### Run the Script

Run the following command in Command Prompt to download and prepare the dataset:

```cmd
DatasetDownload.bat
```

## Folder Structure Setup

To maintain a clean and organized project structure, a Python script named `template.py` has been developed to automatically create the required files and folders for the project. This script initializes the necessary directory structure and supporting files needed for development.

### Run the Script

Run the following command in Command Prompt to create the required project structure:

```cmd
python template.py
```

## Installing Dependencies

With the `BanglaLekha` virtual environment active, install all required packages using:

```cmd
pip install -r requirements.txt
```

### Key dependencies

| Package | Version | Purpose |
|---|---|---|
| `tensorflow` | 2.20.0 | Model training and inference |
| `numpy` | 2.0.2 | Numerical operations |
| `pandas` | 2.2.2 | Data handling |
| `scikit-learn` | 1.6.1 | Train/test split |
| `Pillow` | 10.4.0 | Image loading and preprocessing |
| `mlflow` | 3.12.0 | Experiment tracking |
| `streamlit` | 1.38.0 | Web application |

> The `-e .` entry at the bottom of `requirements.txt` installs the `banglaOCR` source package in editable mode so all pipeline imports resolve correctly.

---

## Data Ingestion

The data ingestion stage loads the **BanglaLekha-Isolated** dataset, preprocesses every image, and saves the results in a folder structure ready for CNN training.

### What it does

1. **Scans** `Dataset/BanglaLekha-Isolated/Images/` — 84 class folders, ~1975 PNG images each (~166k total)
2. **Splits** the data into 80% train (~132,884 images) and 20% test (~33,221 images) using a **stratified** split, ensuring all 84 classes are proportionally represented in both sets
3. **Preprocesses** each image: converts to grayscale → resizes to 32×32 pixels
4. **Saves** the preprocessed images into a class-labelled folder structure:

```
artifacts/data_ingestion/preprocessed/
  train/
    0/   (~1,580 images)
    1/
    ...
    83/
  test/
    0/   (~395 images)
    ...
    83/
```

> Pixel normalization (dividing by 255) is applied during training, not here.

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | 32 | Width and height to resize images to |
| `TEST_SIZE` | 0.2 | Fraction of data reserved for the test set |
| `RANDOM_STATE` | 42 | Seed for reproducible splitting |

All parameters are defined in [`params.yaml`](params.yaml).

### Run data ingestion

```cmd
python src/banglaOCR/pipeline/stage_01_data_ingestion.py
```

---

## Model Training

The model training stage builds and trains a custom CNN on the preprocessed images, logs everything to MLflow, and saves the best model and class label mapping.

### CNN Architecture

Input: `(32, 32, 1)` — 32×32 grayscale single-channel image

| Block | Layers | Output Shape |
|---|---|---|
| Block 1 | Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25) | 16×16×32 |
| Block 2 | Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25) | 8×8×64 |
| Block 3 | Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.25) | 4×4×128 |
| Head | Flatten → Dense(256) → BN → ReLU → Dropout(0.5) → Dense(84, softmax) | 84 |

### Training setup

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Early Stopping | Patience 5 on `val_loss`, restores best weights |
| LR Scheduler | ReduceLROnPlateau — halves LR after 3 stagnant epochs |
| Model Checkpoint | Saves best `val_accuracy` epoch only |

### MLflow tracking

Every training run logs to the `bangla-ocr-cnn` experiment:

- **Parameters** — `image_size`, `batch_size`, `epochs`, `learning_rate`, `num_classes`
- **Metrics per epoch** — `loss`, `accuracy`, `val_loss`, `val_accuracy`
- **Summary metrics** — `best_val_accuracy`, `best_val_loss`
- **Artifacts** — `model.keras`, `labels.json`

To view the MLflow UI after training:

```cmd
mlflow ui
```

Then open `http://127.0.0.1:5000` in a browser.

### Output artifacts

| File | Description |
|---|---|
| `artifacts/model_trainer/model.keras` | Best model weights (highest `val_accuracy`) |
| `artifacts/model_trainer/labels.json` | Maps model output index to class folder name |

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `EPOCHS` | 50 | Maximum training epochs |
| `LEARNING_RATE` | 0.001 | Initial Adam learning rate |
| `BATCH_SIZE` | 32 | Images per training batch |
| `NUM_CLASSES` | 84 | Number of Bangla character classes |

All parameters are defined in [`params.yaml`](params.yaml).

### Run model training

```cmd
python src/banglaOCR/pipeline/stage_02_model_trainer.py
```

---

## Run the full pipeline

To run data ingestion and model training in sequence:

```cmd
python train.py
```
