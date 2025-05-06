import argparse
import sys
import warnings
import mlflow
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from loguru import logger

# Логирование
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)

def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
    val_dataset = datasets.ImageFolder("dataset/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, train_dataset.classes

# Объектив для Optuna
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    train_loader, val_loader, classes = get_data_loaders(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay
        })

        model.train()
        for epoch in range(3):
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

        val_accuracy = correct / total
        mlflow.log_metric("val_accuracy", val_accuracy)
        logger.info(f"Trial {trial.number}: val_accuracy={val_accuracy:.4f}")

    return 1 - val_accuracy  # Optuna минимизирует, поэтому 1 - acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args()

    logger.info(f"Hyperparameter tuning started with {args.n_trials} trials")

    with mlflow.start_run():
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.n_trials)

        best_trial = study.best_trial
        logger.info(f"Best trial params: {best_trial.params}")
        logger.info(f"Best trial accuracy: {1 - best_trial.value:.4f}")

        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_val_accuracy", 1 - best_trial.value)
