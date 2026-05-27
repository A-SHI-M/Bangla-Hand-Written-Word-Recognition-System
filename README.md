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
