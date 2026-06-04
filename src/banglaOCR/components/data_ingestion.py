from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from banglaOCR import logger
from banglaOCR.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.image_size = config.image_size
        self.test_size = config.test_size
        self.random_state = config.random_state

    def load_data(self) -> pd.DataFrame:
        """Walk source_dir and build a DataFrame of (filepath, label) for every image."""
        source_dir = Path(self.config.source_dir)
        records = []

        class_folders = sorted(source_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name)
        for class_folder in class_folders:
            if not class_folder.is_dir():
                continue
            label = int(class_folder.name) - 1  # zero-indexed label
            for img_path in class_folder.glob("*.png"):
                records.append({"filepath": str(img_path), "label": label})

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} image records across {df['label'].nunique()} classes from {source_dir}")
        return df

    def preprocess_image(self, img_path: str) -> Image.Image:
        """Load a single image, convert to grayscale, and resize."""
        img = Image.open(img_path).convert("L")
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        return img

    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified split into train and test DataFrames."""
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df["label"],
        )
        logger.info(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def preprocess_and_save(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Preprocess every image in df and save to output_dir/<label>/<filename>.png"""
        total = len(df)
        for i, row in df.iterrows():
            label_dir = output_dir / str(row["label"])
            label_dir.mkdir(parents=True, exist_ok=True)

            img = self.preprocess_image(row["filepath"])
            filename = Path(row["filepath"]).name
            img.save(label_dir / filename)

            if (i + 1) % 10000 == 0:
                logger.info(f"Preprocessed {i + 1}/{total} images -> {output_dir}")

        logger.info(f"All {total} images saved to {output_dir}")

    def run(self) -> None:
        """End-to-end data ingestion: load → split → preprocess and save images."""
        logger.info("Starting data ingestion.")

        df = self.load_data()
        train_df, test_df = self.split_data(df)

        logger.info("Preprocessing and saving train images...")
        self.preprocess_and_save(train_df, Path(self.config.train_preprocessed_dir))

        logger.info("Preprocessing and saving test images...")
        self.preprocess_and_save(test_df, Path(self.config.test_preprocessed_dir))

        logger.info("Data ingestion complete.")
