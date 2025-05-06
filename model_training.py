import os
import sys
import mlflow
import warnings
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from loguru import logger
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings('ignore')
logging.getLogger('mlflow').setLevel(logging.ERROR)
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

def save_prediction_examples(model, val_loader, classes, run_dir, device):
    model.eval()
    os.makedirs(run_dir, exist_ok=True)
    saved = {cls: False for cls in range(len(classes))}

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for i in range(len(inputs)):
                label = labels[i].item()
                if not saved[label]:
                    img = inputs[i].cpu().permute(1, 2, 0).numpy()
                    img = (img - img.min()) / (img.max() - img.min())
                    plt.imshow(img)
                    plt.title(f"True: {classes[label]} | Pred: {classes[preds[i]]}")
                    plt.axis('off')
                    path = os.path.join(run_dir, f"example_{classes[label]}.png")
                    plt.savefig(path)
                    plt.close()
                    saved[label] = True
            if all(saved.values()):
                return

if __name__ == "__main__":
    logger.info("Model training started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id
        run_id = run.info.run_id
        logger.info(f"Start MLflow run: {run_id}")

        # Загрузка данных
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
        val_dataset = datasets.ImageFolder("dataset/val", transform=transform)

        classes = train_dataset.classes

        last_tuning_run = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.mlflow.runName = 'Hyperparameters_Search' and status = 'FINISHED'",
            order_by=["start_time DESC"]
        ).iloc[0]

        params = {k.split("params.")[1]: v for k, v in last_tuning_run.items() if k.startswith("params.")}
        mlflow.log_params(params)

        lr = float(params["learning_rate"])
        batch_size = int(params["batch_size"])
        weight_decay = float(params["weight_decay"])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(3):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        logger.info("Saving prediction examples...")
        save_prediction_examples(model, val_loader, classes, "examples", device)
        mlflow.log_artifacts("examples", artifact_path="examples")

        mlflow.pytorch.log_model(model, artifact_path="model")
        logger.info("Model training finished")

        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "LeopardsVsLionsClassifier")
        logger.info("Model registered")
