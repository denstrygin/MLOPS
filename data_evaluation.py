import sys
import os
import argparse
import warnings
import logging
import mlflow
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings('ignore')
logging.getLogger('mlflow').setLevel(logging.ERROR)

if __name__ == "__main__":
    logger.info("Evaluation started")

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dataset", type=str, required=True, help="Path to validation dataset directory")
    args = parser.parse_args()
    eval_path = args.eval_dataset

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    val_dataset = datasets.ImageFolder(eval_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32)
    classes = val_dataset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run():
        # Загрузка последней версии модели
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions("LeopardsVsLionsClassifier", stages=["None", "Staging", "Production"])[0].version
        model_uri = f"models:/LeopardsVsLionsClassifier/{latest_version}"

        model = mlflow.pytorch.load_model(model_uri)
        model = model.to(device)
        model.eval()

        correct = total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        mlflow.log_metric("val_accuracy", accuracy)
        logger.success(f"Evaluation finished — Accuracy: {accuracy:.4f}")
