from banglaOCR.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from banglaOCR.utils import read_yaml, create_directories
from banglaOCR.entity import DataIngestionConfig, ModelTrainerConfig
from pathlib import Path


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        data_config = self.config.data_ingestion
        create_directories([config.root_dir])
        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            labels_path=Path(config.labels_path),
            train_preprocessed_dir=Path(data_config.train_preprocessed_dir),
            test_preprocessed_dir=Path(data_config.test_preprocessed_dir),
            image_size=self.params.IMAGE_SIZE,
            num_classes=self.params.NUM_CLASSES,
            batch_size=self.params.BATCH_SIZE,
            epochs=self.params.EPOCHS,
            learning_rate=self.params.LEARNING_RATE,
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_dir=Path(config.source_dir),
            train_preprocessed_dir=Path(config.train_preprocessed_dir),
            test_preprocessed_dir=Path(config.test_preprocessed_dir),
            image_size=self.params.IMAGE_SIZE,
            test_size=self.params.TEST_SIZE,
            random_state=self.params.RANDOM_STATE,
        )
