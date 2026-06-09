import mlflow

from banglaOCR import logger
from banglaOCR.config.configuration import ConfigurationManager


RUN_ID = "d11cf4e7296b4388928d1c8d6d81af3a"

if __name__ == "__main__":
    config = ConfigurationManager()
    model_trainer_config = config.get_model_trainer_config()

    with mlflow.start_run(run_id=RUN_ID):
        mlflow.log_artifact(str(model_trainer_config.trained_model_path))
        mlflow.log_artifact(str(model_trainer_config.labels_path))
        logger.info(f"Artifacts logged to run {RUN_ID}.")
