import sys
import argparse
import mlflow
import warnings
import shutil
import os
import random

from pathlib import Path
from loguru import logger


logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings('ignore')

def split_dataset(input_dir, output_dir, val_ratio=0.2, seed=42):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'

    if output_dir.exists():
        shutil.rmtree(output_dir)
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    class_counts = {}

    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        images = list(class_dir.glob('*'))
        random.seed(seed)
        random.shuffle(images)

        split_idx = int(len(images) * (1 - val_ratio))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)

        for img_path in train_images:
            shutil.copy(img_path, train_dir / class_name / img_path.name)
        for img_path in val_images:
            shutil.copy(img_path, val_dir / class_name / img_path.name)

        class_counts[class_name] = {
            'total': len(images),
            'train': len(train_images),
            'val': len(val_images)
        }

    return class_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", type=str, default="raw_images", help="Directory with raw class folders")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output dataset root dir (with train/val)")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation set ratio")
    args = parser.parse_args()

    RAW_DIR = args.raw_data_dir
    OUT_DIR = args.output_dir
    VAL_RATIO = args.val_size

    logger.info(f"Starting data preprocessing...")
    logger.info(f"Source dir: {RAW_DIR} | Output dir: {OUT_DIR} | Val size: {VAL_RATIO}")

    class_stats = split_dataset(RAW_DIR, OUT_DIR, VAL_RATIO)

    total_images = sum(stats['total'] for stats in class_stats.values())
    mlflow.log_metric("total_images", total_images)
    mlflow.log_metric("num_classes", len(class_stats))

    for class_name, stats in class_stats.items():
        mlflow.log_metric(f"{class_name}_train", stats['train'])
        mlflow.log_metric(f"{class_name}_val", stats['val'])

    logger.info(f"Split complete. Stats per class: {class_stats}")
    logger.info(f"Data preprocessing finished.")
