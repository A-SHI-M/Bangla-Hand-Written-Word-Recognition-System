from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path
    labels_path: Path
    train_preprocessed_dir: Path
    test_preprocessed_dir: Path
    image_size: int
    num_classes: int
    batch_size: int
    epochs: int
    learning_rate: float


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_dir: Path
    train_preprocessed_dir: Path
    test_preprocessed_dir: Path
    image_size: int
    test_size: float
    random_state: int
