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

## Data Ingestion

Loads the BanglaLekha-Isolated dataset, converts images to grayscale and resizes them to 32×32, then splits into 80% train and 20% test sets and saves the preprocessed images to `artifacts/data_ingestion/preprocessed/`.

```cmd
python src/banglaOCR/pipeline/stage_01_data_ingestion.py
```

## Model Training

Trains a custom CNN on the preprocessed images for 84-class Bangla character recognition. Saves the best model to `artifacts/model_trainer/model.keras` and class label mapping to `artifacts/model_trainer/labels.json`.

Every training run is tracked with MLflow under the `bangla-ocr-cnn` experiment. It logs parameters, per-epoch metrics, and artifacts. To view the results, run:

```cmd
mlflow ui
```

Then open `http://127.0.0.1:5000` in a browser.

```cmd
python src/banglaOCR/pipeline/stage_02_model_trainer.py
```

## Run the Full Pipeline

```cmd
python train.py
```

## Prediction UI

A Streamlit web app with a drawing canvas for real-time Bangla handwritten word recognition. Draw a word on the canvas, click **Predict**, and the app will segment each character, predict it using the trained model, and display the recognized word as sample images from the predicted classes alongside confidence scores.

```cmd
streamlit run app.py
```

Then open `http://localhost:8501` in a browser.
