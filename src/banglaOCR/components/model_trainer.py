import json

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers

from banglaOCR import logger
from banglaOCR.entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def build_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(self.config.image_size, self.config.image_size, 1))

        # Block 1
        x = layers.Conv2D(32, 3, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(32, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

        # Block 2
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

        # Block 3
        x = layers.Conv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

        # Classifier head
        x = layers.Flatten()(x)
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.config.num_classes, activation="softmax")(x)

        return tf.keras.Model(inputs, outputs, name="bangla_ocr_cnn")

    def get_data_loaders(self):
        AUTOTUNE = tf.data.AUTOTUNE
        normalize = layers.Rescaling(1.0 / 255)

        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.train_preprocessed_dir,
            image_size=(self.config.image_size, self.config.image_size),
            color_mode="grayscale",
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=42,
        )
        test_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.test_preprocessed_dir,
            image_size=(self.config.image_size, self.config.image_size),
            color_mode="grayscale",
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        train_ds = train_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE)

        train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
        test_ds = test_ds.cache().prefetch(AUTOTUNE)

        return train_ds, test_ds

    def train(self):
        model = self.build_model()
        model.summary(print_fn=logger.info)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.config.trained_model_path),
                monitor="val_accuracy",
                save_best_only=True,
            ),
        ]

        train_ds, test_ds = self.get_data_loaders()

        mlflow.set_experiment("bangla-ocr-cnn")

        with mlflow.start_run():
            mlflow.log_params({
                "image_size": self.config.image_size,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "num_classes": self.config.num_classes,
            })

            logger.info("Starting model training...")
            history = model.fit(
                train_ds,
                validation_data=test_ds,
                epochs=self.config.epochs,
                callbacks=callbacks,
            )

            for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                history.history["loss"],
                history.history["accuracy"],
                history.history["val_loss"],
                history.history["val_accuracy"],
            )):
                mlflow.log_metrics(
                    {"loss": loss, "accuracy": acc, "val_loss": val_loss, "val_accuracy": val_acc},
                    step=epoch,
                )

            best_val_acc = max(history.history["val_accuracy"])
            best_val_loss = min(history.history["val_loss"])
            mlflow.log_metrics({"best_val_accuracy": best_val_acc, "best_val_loss": best_val_loss})

            mlflow.log_artifact(str(self.config.trained_model_path))
            mlflow.log_artifact(str(self.config.labels_path))

            logger.info(f"MLflow run logged — best val_accuracy: {best_val_acc:.4f}")

        logger.info(f"Best model saved to {self.config.trained_model_path}")

        label_map = {str(i): name for i, name in enumerate(train_ds.class_names)}
        with open(self.config.labels_path, "w") as f:
            json.dump(label_map, f, indent=4)
        logger.info(f"Label mapping saved to {self.config.labels_path}")

        return history
